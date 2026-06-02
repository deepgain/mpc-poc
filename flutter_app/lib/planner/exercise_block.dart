/// Data structures returned by [KnapsackPlanner.plan].
///
/// 1:1 with `knapsack_planner.py`'s `ExerciseBlock` and `KnapsackPlan`.
library;

import 'package:deepgain_app/inference/types.dart';

class ExerciseBlock {
  final String exerciseId;

  /// Training weight in kg (multiple of 2.5 after binary-search tuning).
  final double weightKg;

  /// Reps per set. Fixed to 10 by current planner (KnapsackPlanner.fixedReps).
  final int reps;
  final int setsCount;

  /// Rest between sets in seconds.
  final int restSec;

  /// Predicted RIR from `predict_rir` for the chosen weight.
  final double predictedRir;

  /// Σ(engagement × MPC_before) / Σ(engagement) — the knapsack item value.
  final double stimulusScore;

  /// Total block time: setsCount × setSec + (setsCount - 1) × restSec.
  final int timeCostSec;

  /// 'compound' / 'variation' / 'isolation' / 'core'.
  final String exType;

  /// Top-2 most-engaged muscles with engagement ≥ 0.40.
  final List<String> primaryMuscles;

  /// Remaining engaged muscles with engagement ≥ 0.20.
  final List<String> secondaryMuscles;

  const ExerciseBlock({
    required this.exerciseId,
    required this.weightKg,
    required this.reps,
    required this.setsCount,
    required this.restSec,
    required this.predictedRir,
    required this.stimulusScore,
    required this.timeCostSec,
    required this.exType,
    required this.primaryMuscles,
    required this.secondaryMuscles,
  });

  Map<String, dynamic> toJson() => {
        'exercise_id': exerciseId,
        'weight_kg': weightKg,
        'reps': reps,
        'sets_count': setsCount,
        'rest_sec': restSec,
        'predicted_rir': predictedRir,
        'stimulus_score': stimulusScore,
        'time_cost_sec': timeCostSec,
        'ex_type': exType,
        'primary_muscles': primaryMuscles,
        'secondary_muscles': secondaryMuscles,
      };

  /// Generate per-set workout history entries for `predict_mpc` replay.
  /// Mirrors `ExerciseBlock.to_history_dicts` in knapsack_planner.py.
  List<WorkoutSet> toWorkoutSets(DateTime baseTs, int setSecPerSet) {
    final out = <WorkoutSet>[];
    var t = baseTs;
    for (var i = 0; i < setsCount; i++) {
      out.add(WorkoutSet(
        exercise: exerciseId,
        weightKg: weightKg,
        reps: reps,
        rir: predictedRir.round().clamp(0, 5).toDouble(),
        timestamp: t,
      ));
      t = t.add(Duration(seconds: setSecPerSet + restSec));
    }
    return out;
  }
}

class KnapsackPlan {
  final List<ExerciseBlock> blocks;
  final int totalTimeSec;
  final double totalStimulus;
  final Mpc mpcBefore;
  final Mpc mpcAfter;

  /// Per-muscle constraint violations (overfatigue / underfatigue messages).
  /// Empty list = plan respects all target zones.
  final List<String> constraintViolations;

  /// Full per-muscle status log (✓ OK, ⚠ OVERFATIGUE, ⚠ UNDERFATIGUE).
  final List<String> notes;

  const KnapsackPlan({
    required this.blocks,
    required this.totalTimeSec,
    required this.totalStimulus,
    required this.mpcBefore,
    required this.mpcAfter,
    required this.constraintViolations,
    required this.notes,
  });
}
