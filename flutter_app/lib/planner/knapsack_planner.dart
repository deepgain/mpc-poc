/// Dart port of `exercise_selection_algorithm/knapsack_planner.py`
/// (the planner on `develop` — replaces the deleted greedy `planner.py`).
///
/// Algorithm:
///   1. Compute MPC before via [DeepGain.predictMpc].
///   2. Phase 1 — pick the best main exercise (squat/bench/deadlift variants)
///      by stimulus_score = Σ(engagement × MPC_before) / Σ(engagement).
///   3. Phase 2 — 0/1 knapsack DP on the remaining time budget across
///      variation/isolation/core (max 2 core), with weight tuned by binary
///      search on [DeepGain.predictRir] to land near `targetRir`.
///   4. Phase 3 — constraint repair: if simulating the plan triggers
///      overfatigue, drop the worst offender and rerun DP.
///   5. Phase 4 — final MPC simulation + per-muscle constraint evaluation.
///
/// Weights and reps come exclusively from the model — no Epley/Brzycki.
library;

import 'dart:math' as math;

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/strength.dart';
import 'package:deepgain_app/inference/types.dart';

import 'exercise_block.dart';
import 'planner_meta.dart';

/// Order used by KnapsackPlanner._knapsackDp to sort the final selection.
const _typeOrder = ['compound', 'variation', 'isolation', 'core'];

class KnapsackPlanner {
  final DeepGain _deepgain;
  final StrengthPriors _strengthPriors;
  final PlannerMeta _meta;

  /// User's onboarding 1RMs — `{bench_press: kg, squat: kg, deadlift: kg}`
  /// or a 3-element list. Used to tune training weights.
  final dynamic strengthAnchors;

  /// Per-muscle [min_after, max_after] MPC bounds. Defaults to
  /// `PlannerMeta.targetZones` if null.
  final Map<String, List<double>> targetZones;
  final int restBetweenSetsSec;

  /// DP discretization granularity in seconds (60 = 1-minute buckets).
  /// Smaller = more precise plans, larger = faster DP.
  final int timeResolutionSec;

  /// Set of exercise_ids that are both in EXERCISE_META AND known to the model.
  final Set<String> _knownExercises;

  static const int fixedReps = 10;
  static const int _maxCoreCount = 2;
  static const int _binarySearchIterations = 12;
  static const int _repairMaxIterations = 5;

  KnapsackPlanner({
    required DeepGain deepgain,
    required StrengthPriors strengthPriors,
    required PlannerMeta meta,
    required this.strengthAnchors,
    Map<String, List<double>>? targetZones,
    this.restBetweenSetsSec = 120,
    this.timeResolutionSec = 60,
  })  : _deepgain = deepgain,
        _strengthPriors = strengthPriors,
        _meta = meta,
        targetZones = targetZones ?? meta.targetZones,
        _knownExercises =
            meta.exerciseMeta.keys.toSet().intersection(deepgain.exerciseToIdx.keys.toSet());

  /// Plan a session.
  ///
  /// [userHistory] is past sets used to compute MPC_before (and for the
  /// inner predict_mpc replay). [timeBudgetSec] is the session length;
  /// [targetRir] is the target reps-in-reserve (1=hard, 5=easy).
  /// [exclusions] removes specific exercises (e.g. injuries / no equipment).
  KnapsackPlan plan({
    required List<WorkoutSet> userHistory,
    int timeBudgetSec = 3600,
    int targetRir = 2,
    Set<String>? exclusions,
    DateTime? now,
  }) {
    final ts = now ?? DateTime.now();
    final excl = exclusions ?? const <String>{};

    // 1. MPC before training.
    final mpcBefore = _deepgain.predictMpc(
      history: userHistory,
      timestamp: ts,
      anchorsKg: _anchorsAsList(strengthAnchors),
      strengthPriors: _strengthPriors,
    );

    // 2. Main exercise.
    final main = _selectMainExercise(
      mpcState: mpcBefore,
      targetRir: targetRir,
      exclusions: excl,
      timeBudgetSec: timeBudgetSec,
    );

    final selected = <ExerciseBlock>[
      ?main,
    ];
    final remaining = timeBudgetSec - (main?.timeCostSec ?? 0);

    // 3. Knapsack DP on accessories.
    final accessoryExclusions = <String>{
      ...excl, ..._meta.mainExercises,
      ?main?.exerciseId,
    };
    final accessories = _buildCandidates(
      mpcState: mpcBefore,
      targetRir: targetRir,
      exclusions: accessoryExclusions,
      timeBudgetSec: remaining,
    );
    final accSelected = _knapsackDp(accessories, remaining);
    selected.addAll(accSelected);

    // 4. Constraint repair.
    final allCandidates = <ExerciseBlock>[?main, ...accessories];
    final repaired = _repairConstraints(
      selected: selected,
      candidates: allCandidates,
      userHistory: userHistory,
      mpcBefore: mpcBefore,
      timeBudgetSec: timeBudgetSec,
      now: ts,
    );

    // 5. Final simulation + evaluation.
    final mpcAfter = _simulateMpc(repaired, userHistory, ts);
    final eval = _evaluateConstraints(mpcBefore, mpcAfter);

    return KnapsackPlan(
      blocks: repaired,
      totalTimeSec: repaired.fold(0, (a, b) => a + b.timeCostSec),
      totalStimulus: repaired.fold(0.0, (a, b) => a + b.stimulusScore),
      mpcBefore: mpcBefore,
      mpcAfter: mpcAfter,
      constraintViolations: eval.violations,
      notes: eval.notes,
    );
  }

  // ── Main selection ────────────────────────────────────────────────────────

  ExerciseBlock? _selectMainExercise({
    required Mpc mpcState,
    required int targetRir,
    required Set<String> exclusions,
    required int timeBudgetSec,
  }) {
    ExerciseBlock? best;
    var bestScore = -1.0;
    // Sort to make iteration deterministic — when multiple mains tie on
    // stimulus_score (e.g. fresh user, all MPC = 1.0), the "best" must be
    // resolvable without depending on Set iteration order.
    //
    // ⚠ DIVERGES FROM UPSTREAM PYTHON. `knapsack_planner.py` (~line 397)
    // iterates `(MAIN_EXERCISES & known) - exclusions` directly, which is
    // a Python set with hash-based order. For Python:
    //   list(MAIN_EXERCISES) == ['low_bar_squat', 'sumo_deadlift', ...]
    // For Dart (sorted): ['bench_press', 'deadlift', ...]
    //
    // With strict-greater-than tiebreaking (`if score > best_score`), the
    // FIRST iterated tied main wins — so Python silently picks low_bar_squat
    // for any tied scenario, Dart picks bench_press. The Python golden
    // generator monkey-patches `_select_main_exercise` to also sort, so the
    // Dart parity tests pass. Upstream fix: add `sorted(...)` at line 394
    // of knapsack_planner.py — flagged to Miłosz.
    final available = _meta.mainExercises
        .intersection(_knownExercises)
        .difference(exclusions)
        .toList()
      ..sort();

    for (final exId in available) {
      // Skip if this exercise primarily targets a muscle that's already
      // fatigued past its target_min. Without this guard the DP can pick
      // e.g. close_grip_bench when chest is at 0.25, and the constraint
      // repair loop runs out of iterations before it can drop it.
      if (_primarilyTargetsFatiguedMuscle(exId, mpcState)) continue;

      final meta = _meta.exerciseMeta[exId]!;
      final timeCost = meta.sets * meta.setSec + (meta.sets - 1) * restBetweenSetsSec;
      if (timeCost > timeBudgetSec) continue;

      final tuned = _tuneWeightAndReps(exId, mpcState, targetRir);
      final score = _computeStimulus(exId, mpcState);
      if (score > bestScore) {
        bestScore = score;
        best = _buildBlock(exId, meta, meta.sets, tuned, score);
      }
    }
    return best;
  }

  /// `true` if [exerciseId] primarily targets at least one muscle whose
  /// current MPC is already below the configured target_min for that muscle.
  ///
  /// "Primarily" = engagement ratio ≥ 0.40 in the involvement matrix; a
  /// secondary muscle (0.20–0.39) doesn't trigger the guard because the
  /// added fatigue is small enough that the constraint-repair loop can
  /// still make a sensible plan around it.
  bool _primarilyTargetsFatiguedMuscle(String exerciseId, Mpc mpcState) {
    const primaryThreshold = 0.40;
    final inv = _meta.muscleInvolvement[exerciseId];
    if (inv == null || inv.isEmpty) return false;
    for (final entry in inv.entries) {
      if (entry.value < primaryThreshold) continue;
      final tMin = (targetZones[entry.key] ?? const [0.55, 0.85])[0];
      final mpc = mpcState[entry.key] ?? 1.0;
      if (mpc < tMin) return true;
    }
    return false;
  }

  // ── Candidate generation ──────────────────────────────────────────────────

  List<ExerciseBlock> _buildCandidates({
    required Mpc mpcState,
    required int targetRir,
    required Set<String> exclusions,
    required int timeBudgetSec,
  }) {
    final candidates = <ExerciseBlock>[];
    // Sort exercise IDs to match Python's `sorted(self._known_exercises)`.
    final ids = _knownExercises.toList()..sort();

    for (final exId in ids) {
      if (exclusions.contains(exId)) continue;
      // Same fatigue guard as _selectMainExercise — see that comment.
      if (_primarilyTargetsFatiguedMuscle(exId, mpcState)) continue;

      final meta = _meta.exerciseMeta[exId]!;
      final tuned = _tuneWeightAndReps(exId, mpcState, targetRir);

      var sets = meta.sets;
      var timeCost = sets * meta.setSec + (sets - 1) * restBetweenSetsSec;
      if (timeCost > timeBudgetSec) {
        sets = 1;
        timeCost = meta.setSec;
      }
      if (timeCost > timeBudgetSec) continue;

      final score = _computeStimulus(exId, mpcState);
      candidates.add(_buildBlock(exId, meta, sets, tuned, score));
    }

    // Cap core count at 2 — same drop policy as Python (preserve order).
    final filtered = <ExerciseBlock>[];
    var coreCount = 0;
    for (final b in candidates) {
      if (b.exType == 'core') {
        if (coreCount < _maxCoreCount) {
          filtered.add(b);
          coreCount++;
        }
      } else {
        filtered.add(b);
      }
    }

    // Greedy heuristic for DP: sort by value-density desc.
    // Dart's List.sort isn't stable, so we tag each item with its original
    // index and use that as a tiebreaker — required to match Python's
    // (stable) Timsort, otherwise ties (e.g. fresh user where all stimulus
    // scores are 1.0) cause divergent DP picks.
    final indexed = <_Indexed>[
      for (var i = 0; i < filtered.length; i++) _Indexed(i, filtered[i]),
    ];
    indexed.sort((a, b) {
      final da = a.block.stimulusScore / math.max(a.block.timeCostSec, 1);
      final db = b.block.stimulusScore / math.max(b.block.timeCostSec, 1);
      final cmp = db.compareTo(da);
      return cmp != 0 ? cmp : a.idx.compareTo(b.idx);
    });
    return indexed.map((e) => e.block).toList();
  }

  // ── Weight tuning via binary search on predict_rir ────────────────────────

  _Tuned _tuneWeightAndReps(String exId, Mpc mpcState, int targetRir) {
    final meta = _meta.exerciseMeta[exId]!;
    final reps = fixedReps;

    // Project 1RM via the model's anchor logic. If none, fall back to a
    // synthetic e1RM so binary search still runs. Mirrors Python's
    //   if e1rm is None: e1rm = BODYWEIGHT_FALLBACK_KG / max(intensity, 0.1)
    var e1rm = _strengthPriors.projectExercise1rmKg(
      exId, anchorValues: strengthAnchors,
    );
    e1rm ??= _meta.bodyweightFallbackKg / math.max(meta.intensity, 0.1);
    if (e1rm <= 0) {
      e1rm = _meta.bodyweightFallbackKg / math.max(meta.intensity, 0.1);
    }

    var wLow = math.max(5.0, e1rm * 0.30);
    var wHigh = math.min(500.0, e1rm * 0.95);
    var bestWeight = (e1rm * meta.intensity / 2.5).round() * 2.5;
    var bestPred = targetRir.toDouble();
    var bestDiff = double.infinity;

    for (var i = 0; i < _binarySearchIterations; i++) {
      final wMid = (wLow + wHigh) / 2;
      try {
        final pred = _deepgain.predictRir(
          state: mpcState, exercise: exId,
          weightKg: wMid, reps: reps,
          anchorsKg: _anchorsAsList(strengthAnchors),
        );
        final diff = (pred - targetRir).abs();
        if (diff < bestDiff) {
          bestDiff = diff;
          bestWeight = wMid;
          bestPred = pred;
        }
        if (pred > targetRir) {
          wLow = wMid;
        } else {
          wHigh = wMid;
        }
      } catch (_) {
        break;
      }
    }
    bestWeight = math.max(5.0, (bestWeight / 2.5).round() * 2.5);
    return _Tuned(weight: bestWeight, reps: reps, predictedRir: bestPred);
  }

  // ── Stimulus score ────────────────────────────────────────────────────────

  double _computeStimulus(String exId, Mpc mpcState) {
    final inv = _meta.muscleInvolvement[exId];
    if (inv == null || inv.isEmpty) return 0.0;
    var totalW = 0.0;
    for (final v in inv.values) {
      totalW += v;
    }
    if (totalW == 0.0) return 0.0;
    var num = 0.0;
    inv.forEach((muscle, ratio) {
      num += ratio * (mpcState[muscle] ?? 1.0);
    });
    return num / totalW;
  }

  // ── 0/1 Knapsack DP ───────────────────────────────────────────────────────

  /// Public for testing — same algorithm as the inner `_knapsackDp`.
  List<ExerciseBlock> knapsackDp(List<ExerciseBlock> candidates, int budgetSec) =>
      _knapsackDp(candidates, budgetSec);

  List<ExerciseBlock> _knapsackDp(List<ExerciseBlock> candidates, int budgetSec) {
    final R = timeResolutionSec;
    final C = budgetSec ~/ R;
    final n = candidates.length;
    if (C <= 0 || n == 0) return const [];

    int wOf(ExerciseBlock b) => math.max(1, (b.timeCostSec + R - 1) ~/ R);

    final dp = List<List<double>>.generate(
      n + 1, (_) => List<double>.filled(C + 1, 0.0),
    );
    for (var i = 1; i <= n; i++) {
      final item = candidates[i - 1];
      final w = wOf(item);
      for (var c = 0; c <= C; c++) {
        dp[i][c] = dp[i - 1][c];
        if (c >= w) {
          final alt = dp[i - 1][c - w] + item.stimulusScore;
          if (alt > dp[i][c]) dp[i][c] = alt;
        }
      }
    }

    final selected = <ExerciseBlock>[];
    var c = C;
    for (var i = n; i > 0; i--) {
      if (dp[i][c] != dp[i - 1][c]) {
        selected.add(candidates[i - 1]);
        c -= wOf(candidates[i - 1]);
      }
    }
    // Backtracking emits items in reverse order; reverse so that the
    // subsequent stable type-sort preserves the original DP ordering
    // within each type bucket (matches Python's selected.reverse()).
    final ordered = selected.reversed.toList();
    ordered.sort((a, b) {
      final ai = _typeIndex(a.exType);
      final bi = _typeIndex(b.exType);
      return ai.compareTo(bi);
    });
    return ordered;
  }

  static int _typeIndex(String t) {
    final i = _typeOrder.indexOf(t);
    return i < 0 ? 3 : i;
  }

  // ── Constraint repair ─────────────────────────────────────────────────────

  List<ExerciseBlock> _repairConstraints({
    required List<ExerciseBlock> selected,
    required List<ExerciseBlock> candidates,
    required List<WorkoutSet> userHistory,
    required Mpc mpcBefore,
    required int timeBudgetSec,
    required DateTime now,
  }) {
    var current = selected;
    var pool = candidates;
    for (var iter = 0; iter < _repairMaxIterations; iter++) {
      final mpcAfter = _simulateMpc(current, userHistory, now);
      final eval = _evaluateConstraints(mpcBefore, mpcAfter);
      final overfatigue = eval.violations.where((v) => v.contains('OVERFATIGUE')).toList();
      if (overfatigue.isEmpty) return current;

      final worst = _findWorstOffender(current, mpcBefore, mpcAfter);
      if (worst == null) break;

      current = current.where((b) => b.exerciseId != worst.exerciseId).toList();
      pool = pool.where((b) => b.exerciseId != worst.exerciseId).toList();
      current = _knapsackDp(pool, timeBudgetSec);
    }
    return current;
  }

  ExerciseBlock? _findWorstOffender(
    List<ExerciseBlock> selected,
    Mpc mpcBefore,
    Mpc mpcAfter,
  ) {
    ExerciseBlock? worst;
    var worstPenalty = 0.0;
    for (final block in selected) {
      final inv = _meta.muscleInvolvement[block.exerciseId] ?? const <String, double>{};
      var penalty = 0.0;
      inv.forEach((muscle, ratio) {
        final after = mpcAfter[muscle] ?? 1.0;
        final tMin = (targetZones[muscle] ?? const [0.55, 0.85])[0];
        if (after < tMin) penalty += (tMin - after) * ratio;
      });
      if (penalty > worstPenalty) {
        worstPenalty = penalty;
        worst = block;
      }
    }
    return worst;
  }

  // ── MPC simulation ────────────────────────────────────────────────────────

  Mpc _simulateMpc(
    List<ExerciseBlock> blocks,
    List<WorkoutSet> userHistory,
    DateTime now,
  ) {
    if (blocks.isEmpty) {
      return _deepgain.predictMpc(
        history: userHistory, timestamp: now,
        anchorsKg: _anchorsAsList(strengthAnchors),
        strengthPriors: _strengthPriors,
      );
    }
    final combined = List<WorkoutSet>.from(userHistory);
    var ts = now;
    for (final b in blocks) {
      final meta = _meta.exerciseMeta[b.exerciseId]!;
      combined.addAll(b.toWorkoutSets(ts, meta.setSec));
      ts = ts.add(Duration(seconds: b.timeCostSec));
    }
    return _deepgain.predictMpc(
      history: combined, timestamp: ts,
      anchorsKg: _anchorsAsList(strengthAnchors),
      strengthPriors: _strengthPriors,
    );
  }

  // ── Constraint evaluation ─────────────────────────────────────────────────

  /// Public for testing.
  ConstraintEvaluation evaluateConstraints(Mpc mpcBefore, Mpc mpcAfter) =>
      _evaluateConstraints(mpcBefore, mpcAfter);

  ConstraintEvaluation _evaluateConstraints(Mpc mpcBefore, Mpc mpcAfter) {
    final violations = <String>[];
    final notes = <String>[];
    final muscles = mpcAfter.keys.toList()..sort();
    for (final m in muscles) {
      final after = mpcAfter[m]!;
      final before = mpcBefore[m] ?? 1.0;
      final zone = targetZones[m] ?? const [0.55, 0.85];
      final tMin = zone[0];
      final tMax = zone[1];
      final wasWorked = after < before - 0.02;

      if (after < tMin) {
        final msg = '⚠ OVERFATIGUE  $m: MPC=${after.toStringAsFixed(2)} '
            '< min=${tMin.toStringAsFixed(2)}';
        violations.add(msg);
        notes.add(msg);
      } else if (wasWorked && after > tMax) {
        final msg = '⚠ UNDERFATIGUE $m: MPC=${after.toStringAsFixed(2)} '
            '> max=${tMax.toStringAsFixed(2)}';
        violations.add(msg);
        notes.add(msg);
      } else if (wasWorked) {
        notes.add('✓ OK           $m: MPC=${after.toStringAsFixed(2)} '
            'in [${tMin.toStringAsFixed(2)}, ${tMax.toStringAsFixed(2)}]');
      }
    }
    return ConstraintEvaluation(violations: violations, notes: notes);
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  ExerciseBlock _buildBlock(
    String exId, ExerciseMeta meta, int sets, _Tuned tuned, double score,
  ) {
    final timeCost = sets * meta.setSec + (sets - 1) * restBetweenSetsSec;
    final inv = _meta.muscleInvolvement[exId] ?? const <String, double>{};
    final sorted = inv.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final primary = <String>[];
    final secondary = <String>[];
    for (var i = 0; i < sorted.length; i++) {
      final e = sorted[i];
      if (i < 2 && e.value >= 0.40) {
        primary.add(e.key);
      } else if (i >= 2 && e.value >= 0.20) {
        secondary.add(e.key);
      }
    }
    return ExerciseBlock(
      exerciseId: exId, weightKg: tuned.weight, reps: tuned.reps,
      setsCount: sets, restSec: restBetweenSetsSec,
      predictedRir: tuned.predictedRir, stimulusScore: score,
      timeCostSec: timeCost, exType: meta.type,
      primaryMuscles: primary, secondaryMuscles: secondary,
    );
  }

  AnchorsKg? _anchorsAsList(dynamic raw) {
    if (raw == null) return null;
    return _strengthPriors.coerceAnchorValues(raw);
  }
}

class _Tuned {
  final double weight;
  final int reps;
  final double predictedRir;
  const _Tuned({required this.weight, required this.reps, required this.predictedRir});
}

class _Indexed {
  final int idx;
  final ExerciseBlock block;
  const _Indexed(this.idx, this.block);
}

class ConstraintEvaluation {
  final List<String> violations;
  final List<String> notes;
  const ConstraintEvaluation({required this.violations, required this.notes});
}
