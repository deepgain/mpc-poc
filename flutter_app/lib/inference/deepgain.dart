/// Dart port of `inference.predict_mpc` / `predict_rir` (Michał).
///
/// 1:1 translation of `TFLiteOrchestrator` from
/// `tools/tflite_export/parity_test.py`. That Python file is the spec —
/// when it changes, this changes.
///
/// The TFLite model calls (`fNet`, `gNet`) are injected as functions so that
/// orchestration logic can be unit-tested with a mock interpreter on the host
/// (no libtensorflowlite_c needed). Real on-device wiring lives in
/// [DeepGain.fromAssets].
///
/// Phase 2 scope: fixed strength anchors per call. Phase 3 will replay
/// dynamic per-session 1RM updates from `strength_priors.py`.
library;

import 'dart:math' as math;
import 'dart:typed_data';

import 'model_assets.dart';
import 'strength.dart';
import 'types.dart';

/// f_net invocation contract: returns drop[15] for one set.
typedef FNetFn = Float32List Function(
  int exerciseIdx,
  double weightN,
  double repsN,
  double rirN,
  Float32List mpc,
  Float32List anchorsN,
);

/// g_net invocation contract: returns rir_norm scalar for one planned set.
typedef GNetFn = double Function(
  int exerciseIdx,
  double weightN,
  double repsN,
  Float32List mpcAll,
  Float32List anchorsN,
);

class DeepGain {
  final List<String> exercises;
  final Map<String, int> exerciseToIdx;
  final List<String> muscles;
  final List<List<double>> involvement;
  final List<double> tau;
  final Scales scales;
  final List<double> defaultAnchorsKg;

  /// Production wiring: holds the Interpreters directly. Calls happen inline
  /// in predictMpc / predictRir. This avoids a tflite_flutter quirk where
  /// runForMultipleInputs raises `Bad state: failed precondition` when called
  /// through a stored Dart closure indirection.
  final ModelAssets? _assets;

  /// Test wiring: pre-injected fNet / gNet implementations. When set, used
  /// instead of [_assets]. The orchestration tests use this to mock TFLite.
  final FNetFn? _fNetMock;
  final GNetFn? _gNetMock;

  /// Test constructor — pass mocks for fNet / gNet. Production code should
  /// use [DeepGain.fromAssets].
  DeepGain({
    required this.exercises,
    required this.exerciseToIdx,
    required this.muscles,
    required this.involvement,
    required this.tau,
    required this.scales,
    required this.defaultAnchorsKg,
    required FNetFn fNet,
    required GNetFn gNet,
  })  : _fNetMock = fNet,
        _gNetMock = gNet,
        _assets = null;

  /// Production constructor — wires the real TFLite interpreters.
  DeepGain._fromAssets(this._assets)
      : exercises = _assets!.exercises,
        exerciseToIdx = _assets.exerciseToIdx,
        muscles = _assets.muscles,
        involvement = _assets.involvement,
        tau = _assets.tau,
        scales = _assets.scales,
        defaultAnchorsKg = _assets.defaultAnchorsKg,
        _fNetMock = null,
        _gNetMock = null;

  factory DeepGain.fromAssets(ModelAssets assets) =>
      DeepGain._fromAssets(assets);

  /// Encode an int64 as 8 little-endian bytes — works around a bug in
  /// tflite_flutter 0.11.0 where `_convertElementToBytes` writes Int64 with
  /// `Endian.big`, garbling any non-zero index on little-endian hosts.
  /// (For value 0 the two encodings happen to match, which is why a smoke
  /// test against exerciseIdx=0 wouldn't catch it.)
  Uint8List _int64LE(int v) {
    final bd = ByteData(8);
    bd.setInt64(0, v, Endian.little);
    return bd.buffer.asUint8List();
  }

  /// Inline f_net invoke — bypasses closure indirection AND the broken
  /// Int64 byte conversion.
  Float32List _callFNet(int exerciseIdx, double weightN, double repsN,
      double rirN, Float32List mpc, Float32List anchorsN) {
    if (_fNetMock != null) {
      return _fNetMock(exerciseIdx, weightN, repsN, rirN, mpc, anchorsN);
    }
    final out = Float32List(muscles.length);
    _assets!.fNet.runForMultipleInputs(
      [
        _int64LE(exerciseIdx),
        Float32List.fromList([weightN]),
        Float32List.fromList([repsN]),
        Float32List.fromList([rirN]),
        mpc,
        anchorsN,
      ],
      {0: out},
    );
    return out;
  }

  double _callGNet(int exerciseIdx, double weightN, double repsN,
      Float32List mpcAll, Float32List anchorsN) {
    if (_gNetMock != null) {
      return _gNetMock(exerciseIdx, weightN, repsN, mpcAll, anchorsN);
    }
    final out = Float32List(1);
    _assets!.gNet.runForMultipleInputs(
      [
        _int64LE(exerciseIdx),
        Float32List.fromList([weightN]),
        Float32List.fromList([repsN]),
        mpcAll,
        anchorsN,
      ],
      {0: out},
    );
    return out[0];
  }

  /// Estimate Muscle Performance Capacity for all 15 muscles at [timestamp].
  ///
  /// Replays [history] through the model — recovery between sets, fatigue
  /// after each set, then a final recovery up to [timestamp].
  ///
  /// When [strengthPriors] is provided, anchors are dynamically updated per
  /// session via `buildAnchorHistoryFromCompletedSets` (mirrors what Python's
  /// `predict_mpc` does internally). Required for any model with
  /// `strength_feature_dim > 0` (Variant 2 onwards) — without it, anchors
  /// are pinned to [anchorsKg] for every set and outputs will diverge from
  /// the reference. For models with `strength_feature_dim = 0`, the static
  /// path is fine since anchors are unused by the network.
  ///
  /// Empty history → all muscles at 1.0 (fresh).
  /// Sets after [timestamp] are excluded. Unknown exercises are skipped.
  Mpc predictMpc({
    required List<WorkoutSet> history,
    required DateTime timestamp,
    AnchorsKg? anchorsKg,
    StrengthPriors? strengthPriors,
  }) {
    final initialAnchorsKg = anchorsKg ?? defaultAnchorsKg;
    final initialAnchorsN = Float32List.fromList(
      initialAnchorsKg.map((kg) => kg / scales.weight).toList(),
    );

    final valid = <_Set>[];
    final validRaw = <Map<String, dynamic>>[];
    for (final h in history) {
      if (h.timestamp.isAfter(timestamp)) continue;
      final idx = exerciseToIdx[h.exercise];
      if (idx == null) continue;
      valid.add(_Set(
        exerciseIdx: idx,
        weightN: h.weightKg / scales.weight,
        repsN: h.reps / scales.reps,
        rirN: h.rir / scales.rir,
        ts: h.timestamp,
      ));
      validRaw.add(h.toJson());
    }
    if (valid.isEmpty) {
      return {for (final m in muscles) m: 1.0};
    }

    // Sort both lists in lock-step by timestamp.
    final order = List<int>.generate(valid.length, (i) => i)
      ..sort((a, b) => valid[a].ts.compareTo(valid[b].ts));
    final sortedValid = [for (final i in order) valid[i]];
    final sortedRaw = [for (final i in order) validRaw[i]];

    // Per-set anchors. Without strengthPriors, every set gets initialAnchorsN.
    List<Float32List> perSetAnchorsN;
    if (strengthPriors != null) {
      final ah = strengthPriors.buildAnchorHistoryFromCompletedSets(
        initialAnchorsKg, sortedRaw,
      );
      perSetAnchorsN = [
        for (final a in ah.history)
          Float32List.fromList(
            a.map((kg) => kg / scales.weight).toList(),
          ),
      ];
    } else {
      perSetAnchorsN = List<Float32List>.filled(sortedValid.length, initialAnchorsN);
    }

    final m = muscles.length;
    final mpc = Float32List(m)..fillRange(0, m, 1.0);
    var prevTs = sortedValid.first.ts;

    for (var i = 0; i < sortedValid.length; i++) {
      final s = sortedValid[i];
      if (i > 0) {
        final dtH = s.ts.difference(prevTs).inMicroseconds / 3.6e9;
        _applyRecovery(mpc, dtH);
      }
      final inv = involvement[s.exerciseIdx];
      final drop = _callFNet(
        s.exerciseIdx, s.weightN, s.repsN, s.rirN, mpc, perSetAnchorsN[i],
      );
      for (var k = 0; k < m; k++) {
        var v = mpc[k] * (1.0 - inv[k] * drop[k]);
        if (v < 0.1) v = 0.1;
        mpc[k] = v;
      }
      prevTs = s.ts;
    }

    final dtFinal = timestamp.difference(prevTs).inMicroseconds / 3.6e9;
    _applyRecovery(mpc, dtFinal);

    return {for (var k = 0; k < m; k++) muscles[k]: mpc[k]};
  }

  /// Predict RIR for one planned set given current muscle [state].
  /// Throws [ArgumentError] if [exercise] is unknown to the model.
  double predictRir({
    required Mpc state,
    required String exercise,
    required double weightKg,
    required int reps,
    AnchorsKg? anchorsKg,
  }) {
    final exIdx = exerciseToIdx[exercise];
    if (exIdx == null) {
      throw ArgumentError.value(exercise, 'exercise', 'unknown to the model');
    }
    final anchors = anchorsKg ?? defaultAnchorsKg;
    final anchorsN = Float32List.fromList(
      anchors.map((kg) => kg / scales.weight).toList(),
    );

    final mpcAll = Float32List(muscles.length);
    for (var k = 0; k < muscles.length; k++) {
      mpcAll[k] = state[muscles[k]] ?? 1.0;
    }

    final rirNorm = _callGNet(exIdx, weightKg / scales.weight,
        reps / scales.reps, mpcAll, anchorsN);
    final scaled = rirNorm * scales.rir;
    return scaled.clamp(0.0, 5.0);
  }

  void _applyRecovery(Float32List mpc, double dtHours) {
    if (dtHours <= 0) return;
    for (var k = 0; k < mpc.length; k++) {
      mpc[k] = 1.0 - (1.0 - mpc[k]) * math.exp(-dtHours / tau[k]);
    }
  }
}

class _Set {
  final int exerciseIdx;
  final double weightN;
  final double repsN;
  final double rirN;
  final DateTime ts;

  const _Set({
    required this.exerciseIdx,
    required this.weightN,
    required this.repsN,
    required this.rirN,
    required this.ts,
  });
}
