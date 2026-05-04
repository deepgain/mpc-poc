/// Unit tests for the Dart orchestration logic in `DeepGain`, with the
/// f_net / g_net calls mocked. Verifies recovery formula, multiplicative
/// MPC update, clamping, dt computation, history filtering/sorting, and
/// the unknown-exercise path.
///
/// The TFLite numerical parity itself is verified by:
///   1. `tools/tflite_export/parity_test.py` (Python ↔ TFLite, max diff 1.19e-7)
///   2. `integration_test/parity_test.dart` (Dart on-device, runs the same
///      golden fixtures through real TFLite)
///
/// This file deliberately avoids importing tflite_flutter so it runs on the
/// host without libtensorflowlite_c.
// ignore_for_file: unnecessary_underscores — mock callback signatures use
// multiple underscores for unused positional args; the Dart 3 lint wants
// `_, _, _` but that's harder to read in the same line as the real arg names.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:flutter_test/flutter_test.dart';

const _muscles = <String>['chest', 'triceps', 'lats'];

DeepGain _makeDeepGain({
  required FNetFn fNet,
  GNetFn? gNet,
  List<List<double>>? involvement,
  List<double>? tau,
}) {
  final exercises = ['bench_press', 'lat_pulldown'];
  return DeepGain(
    exercises: exercises,
    exerciseToIdx: {for (var i = 0; i < exercises.length; i++) exercises[i]: i},
    muscles: _muscles,
    involvement: involvement ??
        [
          [1.0, 0.5, 0.0], // bench_press: chest=1.0, triceps=0.5, lats=0.0
          [0.0, 0.3, 1.0], // lat_pulldown: chest=0.0, triceps=0.3, lats=1.0
        ],
    tau: tau ?? [16.0, 9.0, 13.0], // chest, triceps, lats — from FIXED_TAU
    scales: const Scales(weight: 200.0, reps: 30.0, rir: 5.0, dt: 5.13),
    defaultAnchorsKg: const [100.0, 140.0, 180.0],
    fNet: fNet,
    gNet: gNet ?? (_, __, ___, ____, _____) => 0.5,
  );
}

void main() {
  group('predictMpc', () {
    test('empty history → all muscles fresh at 1.0', () {
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            throw 'should not be called for empty history',
      );
      final mpc = dg.predictMpc(
        history: [],
        timestamp: DateTime(2026, 5, 1),
      );
      expect(mpc, {'chest': 1.0, 'triceps': 1.0, 'lats': 1.0});
    });

    test('single set updates mpc by (1 - involvement * drop)', () {
      var calls = 0;
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) {
          calls++;
          return Float32List.fromList([0.4, 0.4, 0.4]); // uniform drop
        },
      );
      final t = DateTime(2026, 5, 1, 10);
      final mpc = dg.predictMpc(
        history: [
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2, timestamp: t),
        ],
        timestamp: t, // query at the exact moment of the set → no recovery
      );
      // bench_press involvement = [1.0, 0.5, 0.0]
      // chest:    1.0 * (1 - 1.0 * 0.4) = 0.6
      // triceps:  1.0 * (1 - 0.5 * 0.4) = 0.8
      // lats:     1.0 * (1 - 0.0 * 0.4) = 1.0
      expect(calls, 1);
      expect(mpc['chest']!, closeTo(0.6, 1e-6));
      expect(mpc['triceps']!, closeTo(0.8, 1e-6));
      expect(mpc['lats']!, closeTo(1.0, 1e-6));
    });

    test('mpc is floored at 0.1 even when (1 - inv*drop) < 0.1', () {
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.99, 0.99, 0.99]),
      );
      final t = DateTime(2026, 5, 1, 10);
      final mpc = dg.predictMpc(
        history: [
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 0, timestamp: t),
        ],
        timestamp: t,
      );
      // chest: 1 * (1 - 1.0 * 0.99) = 0.01 → floored to 0.1
      expect(mpc['chest']!, closeTo(0.1, 1e-6));
    });

    test('recovery: mpc = 1 - (1-mpc)*exp(-dt/tau)', () {
      // After a set that drops chest to ~0.5, wait 16 hours (tau_chest = 16).
      // Recovery: 1 - (1 - 0.5)*exp(-16/16) = 1 - 0.5*exp(-1) ≈ 0.8161
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.5, 0.0, 0.0]), // only chest drops
      );
      final t0 = DateTime(2026, 5, 1, 10);
      final mpc = dg.predictMpc(
        history: [
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2, timestamp: t0),
        ],
        timestamp: t0.add(const Duration(hours: 16)),
      );
      final expected = 1.0 - (1.0 - 0.5) * math.exp(-16.0 / 16.0);
      expect(mpc['chest']!, closeTo(expected, 1e-6));
      // Untouched muscles fully recover (already 1.0, recovery is no-op)
      expect(mpc['triceps']!, closeTo(1.0, 1e-6));
    });

    test('history is sorted by timestamp before replay', () {
      // Two sets, given out of order. Result must equal in-order replay.
      final calls = <int>[];
      final dg = _makeDeepGain(
        fNet: (exIdx, _, __, ___, ____, _____) {
          calls.add(exIdx);
          return Float32List.fromList([0.1, 0.1, 0.1]);
        },
      );
      final t = DateTime(2026, 5, 1, 10);
      dg.predictMpc(
        history: [
          // Listed in REVERSE order; orchestrator must sort.
          WorkoutSet(exercise: 'lat_pulldown', weightKg: 70, reps: 8, rir: 2,
              timestamp: t.add(const Duration(minutes: 5))),
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2, timestamp: t),
        ],
        timestamp: t.add(const Duration(hours: 1)),
      );
      // bench_press idx=0 must be called BEFORE lat_pulldown idx=1
      expect(calls, [0, 1]);
    });

    test('sets after the query timestamp are excluded', () {
      var calls = 0;
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) {
          calls++;
          return Float32List.fromList([0.0, 0.0, 0.0]);
        },
      );
      final t = DateTime(2026, 5, 1, 10);
      dg.predictMpc(
        history: [
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2, timestamp: t),
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2,
              timestamp: t.add(const Duration(hours: 5))),
        ],
        timestamp: t.add(const Duration(hours: 2)), // before the second set
      );
      expect(calls, 1);
    });

    test('unknown exercises in history are silently skipped', () {
      var calls = 0;
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) {
          calls++;
          return Float32List.fromList([0.1, 0.1, 0.1]);
        },
      );
      final t = DateTime(2026, 5, 1, 10);
      dg.predictMpc(
        history: [
          WorkoutSet(exercise: 'made_up', weightKg: 80, reps: 5, rir: 2, timestamp: t),
          WorkoutSet(exercise: 'bench_press', weightKg: 80, reps: 5, rir: 2, timestamp: t),
        ],
        timestamp: t,
      );
      expect(calls, 1);
    });
  });

  group('predictRir', () {
    test('returns gNet output × RIR_SCALE, clamped to [0, 5]', () {
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.0, 0.0, 0.0]),
        gNet: (_, __, ___, ____, _____) => 0.5, // 0.5 × RIR_SCALE(5) = 2.5
      );
      final rir = dg.predictRir(
        state: {'chest': 0.9, 'triceps': 0.9, 'lats': 0.9},
        exercise: 'bench_press', weightKg: 100, reps: 5,
      );
      expect(rir, closeTo(2.5, 1e-6));
    });

    test('clamps gNet output > 1.0 to RIR=5.0', () {
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.0, 0.0, 0.0]),
        gNet: (_, __, ___, ____, _____) => 2.0, // 2.0 × 5 = 10 → clamped to 5
      );
      final rir = dg.predictRir(
        state: {'chest': 0.9}, exercise: 'bench_press', weightKg: 80, reps: 5,
      );
      expect(rir, closeTo(5.0, 1e-6));
    });

    test('throws ArgumentError for unknown exercise', () {
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.0, 0.0, 0.0]),
      );
      expect(
        () => dg.predictRir(
          state: {}, exercise: 'made_up', weightKg: 80, reps: 5,
        ),
        throwsArgumentError,
      );
    });

    test('missing muscles in state default to 1.0', () {
      Float32List? capturedMpc;
      final dg = _makeDeepGain(
        fNet: (_, __, ___, ____, _____, ______) =>
            Float32List.fromList([0.0, 0.0, 0.0]),
        gNet: (_, __, ___, mpc, _____) {
          capturedMpc = Float32List.fromList(mpc);
          return 0.4;
        },
      );
      dg.predictRir(
        state: {'chest': 0.5}, // triceps and lats missing
        exercise: 'bench_press', weightKg: 80, reps: 5,
      );
      expect(capturedMpc!.toList(), [0.5, 1.0, 1.0]);
    });
  });
}
