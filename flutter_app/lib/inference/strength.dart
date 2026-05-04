/// Dart port of `strength_priors.py` (Michał).
///
/// 1:1 translation of the public functions used by the inference replay loop
/// and the planner. The `EXERCISE_STRENGTH_PRIORS` table itself isn't ported
/// as code — it's loaded from `assets/model/strength_priors.json`, generated
/// by `tools/tflite_export/gen_strength_assets.py`.
///
/// Verified against `assets/golden/strength_golden.json` in
/// `test/inference/strength_test.dart`.
library;

import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/services.dart' show rootBundle;

/// Three-anchor 1RM array, ordered as [bench_press, squat, deadlift].
typedef Anchors = List<double>;

class _ExercisePrior {
  final String anchorLift;
  final double ratioMean;
  final String exerciseFamily;
  const _ExercisePrior({
    required this.anchorLift,
    required this.ratioMean,
    required this.exerciseFamily,
  });
}

/// Loaded from `strength_priors.json`. Single instance per app session.
class StrengthPriors {
  static const List<String> anchorNames = ['bench_press', 'squat', 'deadlift'];

  final List<double> defaultAnchorsKg;
  final Map<String, _ExercisePrior> _priors;
  // Aliases per anchor name — accept e.g. config_1rm_bench_press as bench_press.
  final Map<String, List<String>> _aliases;

  // Constants — all match strength_priors.py defaults.
  final double defaultUpdateAlpha;
  final double defaultUpdateMaxRelativeChange;
  final double defaultUpdateMinRelativeLoad;
  final int defaultUpdateMaxReps;
  final double defaultUpdateMaxRir;
  final int defaultUpdateTopK;
  final double defaultSessionGapHours;

  StrengthPriors._({
    required this.defaultAnchorsKg,
    required Map<String, _ExercisePrior> priors,
    required Map<String, List<String>> aliases,
    required this.defaultUpdateAlpha,
    required this.defaultUpdateMaxRelativeChange,
    required this.defaultUpdateMinRelativeLoad,
    required this.defaultUpdateMaxReps,
    required this.defaultUpdateMaxRir,
    required this.defaultUpdateTopK,
    required this.defaultSessionGapHours,
  })  : _priors = priors,
        _aliases = aliases;

  static Future<StrengthPriors> load({
    String path = 'assets/model/strength_priors.json',
  }) async {
    final raw = jsonDecode(await rootBundle.loadString(path)) as Map<String, dynamic>;
    return StrengthPriors.fromJson(raw);
  }

  factory StrengthPriors.fromJson(Map<String, dynamic> raw) {
    final defaults = raw['default_anchor_values_kg'] as Map<String, dynamic>;
    final defaultAnchorsKg = anchorNames
        .map((n) => (defaults[n] as num).toDouble())
        .toList(growable: false);

    final priors = <String, _ExercisePrior>{};
    for (final entry in (raw['exercise_priors'] as Map<String, dynamic>).entries) {
      final v = entry.value as Map<String, dynamic>;
      priors[entry.key] = _ExercisePrior(
        anchorLift: v['anchor_lift'] as String,
        ratioMean: (v['ratio_mean'] as num).toDouble(),
        exerciseFamily: v['exercise_family'] as String,
      );
    }

    final aliases = <String, List<String>>{};
    for (final entry in (raw['anchor_aliases'] as Map<String, dynamic>).entries) {
      aliases[entry.key] = List<String>.from(entry.value as List);
    }

    final c = raw['constants'] as Map<String, dynamic>;
    return StrengthPriors._(
      defaultAnchorsKg: defaultAnchorsKg,
      priors: priors,
      aliases: aliases,
      defaultUpdateAlpha: (c['DEFAULT_UPDATE_ALPHA'] as num).toDouble(),
      defaultUpdateMaxRelativeChange:
          (c['DEFAULT_UPDATE_MAX_RELATIVE_CHANGE'] as num).toDouble(),
      defaultUpdateMinRelativeLoad:
          (c['DEFAULT_UPDATE_MIN_RELATIVE_LOAD'] as num).toDouble(),
      defaultUpdateMaxReps: (c['DEFAULT_UPDATE_MAX_REPS'] as num).toInt(),
      defaultUpdateMaxRir: (c['DEFAULT_UPDATE_MAX_RIR'] as num).toDouble(),
      defaultUpdateTopK: (c['DEFAULT_UPDATE_TOP_K'] as num).toInt(),
      defaultSessionGapHours: (c['DEFAULT_SESSION_GAP_HOURS'] as num).toDouble(),
    );
  }

  /// Anchor name (`bench_press`/`squat`/`deadlift`) for an exercise — null
  /// if no prior exists or the anchor lift is bodyweight-only.
  String? anchorNameFor(String exercise) {
    final p = _priors[exercise];
    if (p == null) return null;
    if (!anchorNames.contains(p.anchorLift)) return null;
    return p.anchorLift;
  }

  /// Per-exercise ratio relative to its anchor lift. Null if undefined.
  double? anchorRatioFor(String exercise) {
    final p = _priors[exercise];
    if (p == null) return null;
    if (!anchorNames.contains(p.anchorLift)) return null;
    if (!p.ratioMean.isFinite || p.ratioMean <= 0.0) return null;
    return p.ratioMean;
  }

  // ── Anchor normalization ────────────────────────────────────────────────

  /// Normalize anchor input from Map / List / null into a dense
  /// [bench, squat, deadlift] list.
  ///
  /// - Map keys can be the canonical name or any alias from
  ///   `anchor_aliases` (e.g. `config_1rm_bench_press`).
  /// - List entries are positional [bench, squat, deadlift]. Missing /
  ///   non-finite / non-positive entries fall back to [defaults].
  Anchors coerceAnchorValues(dynamic input, {Anchors? defaults}) {
    final base = List<double>.from(defaults ?? defaultAnchorsKg);
    if (input == null) return base;

    if (input is Map) {
      final out = List<double>.from(base);
      for (var i = 0; i < anchorNames.length; i++) {
        final name = anchorNames[i];
        final keys = _aliases[name] ?? <String>[name];
        for (final k in keys) {
          if (input.containsKey(k)) {
            final v = input[k];
            if (v is num) {
              final d = v.toDouble();
              if (d.isFinite && d > 0.0) out[i] = d;
            }
            break;
          }
        }
      }
      return out;
    }

    if (input is List) {
      final out = List<double>.from(base);
      final n = math.min(input.length, anchorNames.length);
      for (var i = 0; i < n; i++) {
        final v = input[i];
        if (v is num) {
          final d = v.toDouble();
          if (d.isFinite && d > 0.0) out[i] = d;
        }
      }
      return out;
    }

    return base;
  }

  /// Resolve anchors from explicit input, then from records, else defaults.
  /// Mirrors `strength_priors.resolve_anchor_values`.
  Anchors resolveAnchorValues({
    dynamic anchorValues,
    List<Map<String, dynamic>>? records,
    Anchors? defaults,
  }) {
    final base = List<double>.from(defaults ?? defaultAnchorsKg);
    if (anchorValues != null) {
      return coerceAnchorValues(anchorValues, defaults: base);
    }
    if (records != null) {
      for (final rec in records) {
        final resolved = coerceAnchorValues(rec, defaults: base);
        if (resolved.any((v) => v > 0.0)) return resolved;
      }
    }
    return base;
  }

  // ── Per-exercise 1RM projection ─────────────────────────────────────────

  /// Project an exercise-specific 1RM (kg) from the current anchor values.
  /// Returns null if the exercise has no anchor or projected value is
  /// non-positive / non-finite.
  double? projectExercise1rmKg(
    String exercise, {
    dynamic anchorValues,
    Anchors? defaults,
  }) {
    final anchors = coerceAnchorValues(anchorValues, defaults: defaults);
    final name = anchorNameFor(exercise);
    final ratio = anchorRatioFor(exercise);
    if (name == null || ratio == null) return null;
    final idx = anchorNames.indexOf(name);
    final projected = anchors[idx] * ratio;
    if (!projected.isFinite || projected <= 0.0) return null;
    return projected;
  }

  // ── Epley e1RM ──────────────────────────────────────────────────────────

  /// Estimate single-rep max from a completed set via Epley + RIR.
  /// Returns null for invalid input (non-positive weight/reps, non-finite, etc.).
  double? estimateE1rmCandidate(double weightKg, int reps, double rir) {
    if (!weightKg.isFinite || !rir.isFinite) return null;
    if (weightKg <= 0.0 || reps <= 0 || rir < 0.0) return null;
    final repsToFailure = reps + rir;
    if (repsToFailure <= 0.0) return null;
    final candidate = weightKg * (1.0 + repsToFailure / 30.0);
    if (!candidate.isFinite || candidate <= 0.0) return null;
    return candidate;
  }

  // ── Candidate scoring + collection ──────────────────────────────────────

  double _scoreUpdateCandidate(
    double reps,
    double rir,
    double relativeLoad, {
    required double minRelativeLoad,
    required double maxRir,
  }) {
    final loadScore = ((relativeLoad - minRelativeLoad) /
            math.max(1.0 - minRelativeLoad, 1e-6))
        .clamp(0.0, 1.0);
    final rirScore = ((maxRir + 1.0 - rir) / (maxRir + 1.0)).clamp(0.0, 1.0);
    final repsScore = (1.0 - (reps - 5.0).abs() / 7.0).clamp(0.25, 1.0);
    return 0.50 * loadScore + 0.30 * rirScore + 0.20 * repsScore;
  }

  /// One quality-filtered candidate from a completed set.
  /// Mirrors the dict produced by `collect_strength_update_candidates`.
  /// Public for tests and for the planner if it needs raw candidates.
  List<UpdateCandidate> collectStrengthUpdateCandidates(
    List<Map<String, dynamic>> completedSets, {
    dynamic anchors,
    double? minRelativeLoad,
    int? maxReps,
    double? maxRir,
  }) {
    final anchorsKg = resolveAnchorValues(anchorValues: anchors);
    final minLoad = minRelativeLoad ?? defaultUpdateMinRelativeLoad;
    final mReps = maxReps ?? defaultUpdateMaxReps;
    final mRir = maxRir ?? defaultUpdateMaxRir;

    final out = <UpdateCandidate>[];
    for (final entry in completedSets) {
      final exercise = entry['exercise'] as String? ?? '';
      final anchorName = anchorNameFor(exercise);
      final ratio = anchorRatioFor(exercise);
      if (anchorName == null || ratio == null) continue;

      final wRaw = entry['weight_kg'];
      final rRaw = entry['reps'];
      final rirRaw = entry['rir'];
      double weightKg;
      int reps;
      double rir;
      try {
        weightKg = (wRaw as num).toDouble();
        reps = (rRaw as num).toInt();
        rir = (rirRaw as num).toDouble();
      } catch (_) {
        continue;
      }

      if (weightKg <= 0.0 || reps <= 0 || reps > mReps || rir < 0.0 || rir > mRir) {
        continue;
      }

      final projected = projectExercise1rmKg(exercise, anchorValues: anchorsKg);
      if (projected == null || projected <= 0.0) continue;

      final relativeLoad = weightKg / projected;
      if (!relativeLoad.isFinite || relativeLoad < minLoad) continue;

      final e1rm = estimateE1rmCandidate(weightKg, reps, rir);
      if (e1rm == null) continue;

      final anchorCandidate = e1rm / ratio;
      if (!anchorCandidate.isFinite || anchorCandidate <= 0.0) continue;

      out.add(UpdateCandidate(
        exercise: exercise,
        anchorName: anchorName,
        weightKg: weightKg,
        reps: reps,
        rir: rir,
        relativeLoad: relativeLoad,
        projected1rm: projected,
        exerciseCandidate1rm: e1rm,
        anchorCandidate1rm: anchorCandidate,
        quality: _scoreUpdateCandidate(
          reps.toDouble(), rir, relativeLoad,
          minRelativeLoad: minLoad, maxRir: mRir,
        ),
        timestamp: entry['timestamp'],
      ));
    }
    return out;
  }

  // ── Anchor update (per-session EMA blend + clip) ────────────────────────

  /// Update [bench, squat, deadlift] anchors from a session of completed sets.
  /// Returns the new anchors; old anchors are NOT mutated.
  Anchors updateStrengthAnchors(
    dynamic anchors,
    List<Map<String, dynamic>> completedSets, {
    double? alpha,
    double? maxRelativeChange,
    double? minRelativeLoad,
    int? maxReps,
    double? maxRir,
    int? topKPerAnchor,
  }) {
    final current = resolveAnchorValues(anchorValues: anchors);
    final newAnchors = List<double>.from(current);

    final a = (alpha ?? defaultUpdateAlpha).clamp(0.0, 1.0);
    final maxRel = math.max(0.0, maxRelativeChange ?? defaultUpdateMaxRelativeChange);
    final topK = math.max(1, topKPerAnchor ?? defaultUpdateTopK);

    final candidates = collectStrengthUpdateCandidates(
      completedSets,
      anchors: current,
      minRelativeLoad: minRelativeLoad,
      maxReps: maxReps,
      maxRir: maxRir,
    );

    final grouped = <String, List<UpdateCandidate>>{
      for (final n in anchorNames) n: <UpdateCandidate>[],
    };
    for (final c in candidates) {
      grouped[c.anchorName]!.add(c);
    }

    for (var i = 0; i < anchorNames.length; i++) {
      final name = anchorNames[i];
      final group = grouped[name]!;
      if (group.isEmpty) continue;

      // Sort by quality desc, tie-break by relative_load desc.
      group.sort((x, y) {
        final q = y.quality.compareTo(x.quality);
        return q != 0 ? q : y.relativeLoad.compareTo(x.relativeLoad);
      });

      final selected = group.take(topK).toList(growable: false);
      var weightSum = 0.0;
      var weightedValueSum = 0.0;
      for (final c in selected) {
        final w = math.max(c.quality, 1e-6);
        weightSum += w;
        weightedValueSum += w * c.anchorCandidate1rm;
      }
      final sessionCandidate = weightedValueSum / weightSum;

      final oldValue = current[i];
      final blended = (1.0 - a) * oldValue + a * sessionCandidate;
      final lower = oldValue * (1.0 - maxRel);
      final upper = oldValue * (1.0 + maxRel);
      newAnchors[i] = blended.clamp(lower, upper);
    }

    return newAnchors;
  }

  // ── Anchor history reconstruction ───────────────────────────────────────

  bool isNewSession(DateTime? prev, DateTime? curr, {double? sessionGapHours}) {
    if (prev == null || curr == null) return false;
    final gap = curr.difference(prev).inMicroseconds / 3.6e9;
    if (!gap.isFinite) return false;
    return gap > (sessionGapHours ?? defaultSessionGapHours);
  }

  /// Replay [completedSets] applying per-session anchor updates at each
  /// session boundary. Returns:
  ///   - history[i] = anchors AT the moment of set i (before this session
  ///     could have updated them)
  ///   - finalAnchors: anchors after the trailing session is applied
  ///     (when [applyTrailingSession] is true)
  ///
  /// `predict_mpc` consumes `history[i]` as the strength_features input
  /// for set i when the model has `strength_feature_dim > 0`.
  AnchorHistory buildAnchorHistoryFromCompletedSets(
    dynamic anchors,
    List<Map<String, dynamic>> completedSets, {
    double? sessionGapHours,
    bool applyTrailingSession = true,
  }) {
    var current = resolveAnchorValues(anchorValues: anchors);
    final history = <Anchors>[];
    final pending = <Map<String, dynamic>>[];
    DateTime? previousTs;

    for (final entry in completedSets) {
      final ts = _parseTs(entry['timestamp']);
      if (pending.isNotEmpty &&
          isNewSession(previousTs, ts, sessionGapHours: sessionGapHours)) {
        current = resolveAnchorValues(
          anchorValues: updateStrengthAnchors(current, pending),
        );
        pending.clear();
      }
      history.add(List<double>.from(current));
      pending.add(entry);
      previousTs = ts;
    }

    var finalAnchors = List<double>.from(current);
    if (pending.isNotEmpty && applyTrailingSession) {
      finalAnchors = resolveAnchorValues(
        anchorValues: updateStrengthAnchors(current, pending),
      );
    }

    return AnchorHistory(history: history, finalAnchors: finalAnchors);
  }
}

DateTime? _parseTs(dynamic ts) {
  if (ts == null) return null;
  if (ts is DateTime) return ts;
  final s = ts.toString().trim();
  if (s.isEmpty) return null;
  try {
    return DateTime.parse(s.replaceFirst('Z', '+00:00'));
  } catch (_) {
    return null;
  }
}

class UpdateCandidate {
  final String exercise;
  final String anchorName;
  final double weightKg;
  final int reps;
  final double rir;
  final double relativeLoad;
  final double projected1rm;
  final double exerciseCandidate1rm;
  final double anchorCandidate1rm;
  final double quality;
  final dynamic timestamp;

  const UpdateCandidate({
    required this.exercise,
    required this.anchorName,
    required this.weightKg,
    required this.reps,
    required this.rir,
    required this.relativeLoad,
    required this.projected1rm,
    required this.exerciseCandidate1rm,
    required this.anchorCandidate1rm,
    required this.quality,
    required this.timestamp,
  });
}

class AnchorHistory {
  /// `history[i]` = anchors at the moment of set i (one entry per completed set).
  final List<Anchors> history;

  /// Anchors after the trailing session is applied (or after the last session
  /// boundary if `applyTrailingSession=false`).
  final Anchors finalAnchors;

  const AnchorHistory({required this.history, required this.finalAnchors});
}
