/// Core types for the DeepGain inference layer.
///
/// Mirrors the dict-based interface in `inference.py` (Michał) but with
/// proper Dart types. The field names match what Python expects so
/// the Phase-3 strength_priors port can hand the same maps around.
library;

class WorkoutSet {
  final String exercise;
  final double weightKg;
  final int reps;
  final double rir;
  final DateTime timestamp;

  const WorkoutSet({
    required this.exercise,
    required this.weightKg,
    required this.reps,
    required this.rir,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
        'exercise': exercise,
        'weight_kg': weightKg,
        'reps': reps,
        'rir': rir,
        'timestamp': timestamp.toIso8601String(),
      };

  factory WorkoutSet.fromJson(Map<String, dynamic> json) => WorkoutSet(
        exercise: json['exercise'] as String,
        weightKg: (json['weight_kg'] as num).toDouble(),
        reps: (json['reps'] as num).toInt(),
        rir: (json['rir'] as num).toDouble(),
        timestamp: DateTime.parse(json['timestamp'] as String),
      );
}

/// Map muscle_id → MPC in [0.1, 1.0]. 1.0 = fully recovered.
typedef Mpc = Map<String, double>;

/// Three values: [bench_press, squat, deadlift] in kg.
typedef AnchorsKg = List<double>;
