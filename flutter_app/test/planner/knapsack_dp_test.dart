/// Unit tests for the 0/1 knapsack DP and constraint evaluation.
///
/// These mirror `TestKnapsackDP` and `TestConstraintEvaluation` from
/// `exercise_selection_algorithm/test_knapsack_planner.py` (the Python tests
/// for the same module). No TFLite needed — the DP and constraint logic are
/// pure functions.
library;

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/strength.dart';
import 'package:deepgain_app/planner/exercise_block.dart';
import 'package:deepgain_app/planner/knapsack_planner.dart';
import 'package:deepgain_app/planner/planner_meta.dart';
import 'package:flutter_test/flutter_test.dart';

ExerciseBlock _block({
  required String id,
  required String type,
  required double score,
  required int timeSec,
  double weight = 100.0,
}) =>
    ExerciseBlock(
      exerciseId: id, weightKg: weight, reps: 10, setsCount: 3, restSec: 120,
      predictedRir: 2.0, stimulusScore: score, timeCostSec: timeSec,
      exType: type, primaryMuscles: const [], secondaryMuscles: const [],
    );

/// Build a planner with a no-op DeepGain mock so we can exercise the
/// pure-Dart parts (DP, constraint eval) without a real TFLite model.
KnapsackPlanner _planner({
  Map<String, List<double>>? targetZones,
  int restSec = 120,
  int resolution = 60,
}) {
  // Minimal mock model that never gets called by the DP / eval paths.
  final mockDg = DeepGain(
    exercises: const ['x'],
    exerciseToIdx: const {'x': 0},
    muscles: const ['chest', 'triceps'],
    involvement: const [[1.0, 0.5]],
    tau: const [16.0, 9.0],
    scales: const Scales(weight: 200.0, reps: 30.0, rir: 5.0, dt: 5.13),
    defaultAnchorsKg: const [100.0, 140.0, 180.0],
    fNet: (_, _, _, _, _, _) => Float32List.fromList([0.0, 0.0]),
    gNet: (_, _, _, _, _) => 0.5,
  );

  // Strength priors loaded from disk JSON (no rootBundle in test environment).
  final priors = StrengthPriors.fromJson(jsonDecode(
    File('assets/model/strength_priors.json').readAsStringSync(),
  ) as Map<String, dynamic>);

  // Empty planner meta is fine — DP doesn't read it for hand-crafted blocks.
  final meta = PlannerMeta(
    muscleInvolvement: const {},
    exerciseMeta: const {},
    mainExercises: const {},
    targetZones: const {},
    bodyweightFallbackKg: 40.0,
  );

  return KnapsackPlanner(
    deepgain: mockDg,
    strengthPriors: priors,
    meta: meta,
    strengthAnchors: const [100.0, 140.0, 180.0],
    targetZones: targetZones,
    restBetweenSetsSec: restSec,
    timeResolutionSec: resolution,
  );
}

void main() {
  group('knapsackDp', () {
    test('picks the higher-stimulus item when only one fits', () {
      final p = _planner();
      final picked = p.knapsackDp([
        _block(id: 'a', type: 'isolation', score: 0.5, timeSec: 600),
        _block(id: 'b', type: 'isolation', score: 0.9, timeSec: 600),
      ], 600);
      expect(picked.length, 1);
      expect(picked.first.exerciseId, 'b');
    });

    test('picks both when both fit', () {
      final p = _planner();
      final picked = p.knapsackDp([
        _block(id: 'a', type: 'compound', score: 0.5, timeSec: 600),
        _block(id: 'b', type: 'isolation', score: 0.9, timeSec: 600),
      ], 1500);
      expect(picked.map((b) => b.exerciseId).toSet(), {'a', 'b'});
    });

    test('zero budget → empty plan', () {
      final p = _planner();
      expect(p.knapsackDp([
        _block(id: 'a', type: 'isolation', score: 0.5, timeSec: 600),
      ], 0), isEmpty);
    });

    test('larger budget never gives lower total stimulus', () {
      final p = _planner();
      final cands = [
        _block(id: 'a', type: 'compound', score: 0.7, timeSec: 600),
        _block(id: 'b', type: 'isolation', score: 0.5, timeSec: 400),
        _block(id: 'c', type: 'isolation', score: 0.4, timeSec: 300),
      ];
      double sum(List<ExerciseBlock> bs) => bs.fold(0.0, (a, b) => a + b.stimulusScore);
      final small = sum(p.knapsackDp(cands, 600));
      final big = sum(p.knapsackDp(cands, 1800));
      expect(big, greaterThanOrEqualTo(small));
    });

    test('total time never exceeds budget', () {
      final p = _planner();
      final cands = [
        _block(id: 'a', type: 'isolation', score: 0.5, timeSec: 700),
        _block(id: 'b', type: 'isolation', score: 0.5, timeSec: 500),
        _block(id: 'c', type: 'isolation', score: 0.5, timeSec: 400),
      ];
      final picked = p.knapsackDp(cands, 1000);
      final total = picked.fold(0, (a, b) => a + b.timeCostSec);
      expect(total, lessThanOrEqualTo(1000));
    });

    test('each candidate is selected at most once', () {
      final p = _planner();
      final cands = [
        _block(id: 'a', type: 'compound', score: 0.5, timeSec: 300),
        _block(id: 'b', type: 'isolation', score: 0.5, timeSec: 300),
      ];
      final picked = p.knapsackDp(cands, 5000);
      final ids = picked.map((b) => b.exerciseId).toList();
      expect(ids.toSet().length, ids.length, reason: 'duplicates');
    });

    test('output sorted compound → variation → isolation → core', () {
      final p = _planner();
      final picked = p.knapsackDp([
        _block(id: 'core1', type: 'core', score: 0.3, timeSec: 200),
        _block(id: 'iso1', type: 'isolation', score: 0.4, timeSec: 200),
        _block(id: 'comp1', type: 'compound', score: 0.5, timeSec: 200),
        _block(id: 'var1', type: 'variation', score: 0.45, timeSec: 200),
      ], 1000);
      expect(picked.map((b) => b.exType).toList(),
          ['compound', 'variation', 'isolation', 'core']);
    });
  });

  group('evaluateConstraints', () {
    final mpcBefore = {
      'chest': 1.0, 'triceps': 1.0, 'lats': 1.0, 'quads': 1.0,
    };

    test('no violations when all worked muscles in zone', () {
      final p = _planner(targetZones: const {
        'chest': [0.50, 0.85],
        'triceps': [0.45, 0.80],
      });
      final after = {'chest': 0.70, 'triceps': 0.65, 'lats': 1.0, 'quads': 1.0};
      final eval = p.evaluateConstraints(mpcBefore, after);
      expect(eval.violations, isEmpty);
    });

    test('overfatigue detected (after < target_min)', () {
      final p = _planner(targetZones: const {'chest': [0.55, 0.85]});
      final after = {'chest': 0.30, 'triceps': 1.0, 'lats': 1.0, 'quads': 1.0};
      final eval = p.evaluateConstraints(mpcBefore, after);
      expect(eval.violations.length, 1);
      expect(eval.violations.first, contains('OVERFATIGUE'));
      expect(eval.violations.first, contains('chest'));
    });

    test('underfatigue when worked muscle stays above target_max', () {
      final p = _planner(targetZones: const {'chest': [0.55, 0.85]});
      // Worked = mpcAfter < mpcBefore - 0.02; here 1.0 → 0.95, worked, but above 0.85 max
      final after = {'chest': 0.95, 'triceps': 1.0, 'lats': 1.0, 'quads': 1.0};
      final eval = p.evaluateConstraints(mpcBefore, after);
      expect(eval.violations.length, 1);
      expect(eval.violations.first, contains('UNDERFATIGUE'));
    });

    test('no note for muscle that was not worked', () {
      final p = _planner(targetZones: const {'chest': [0.55, 0.85]});
      // chest unchanged → not worked → no note
      final after = {'chest': 1.0, 'triceps': 1.0, 'lats': 1.0, 'quads': 1.0};
      final eval = p.evaluateConstraints(mpcBefore, after);
      expect(eval.violations, isEmpty);
      expect(eval.notes.where((n) => n.contains('chest')), isEmpty);
    });
  });
}
