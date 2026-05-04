/// Regression guard for tflite_flutter 0.11.0's Int64 endian bug.
///
/// `_convertElementToBytes` in
/// pkgs/tflite_flutter/lib/src/util/byte_conversion_utils.dart writes Int64
/// values as `Endian.big` instead of `Endian.little`. On macOS / iOS / Android
/// (all little-endian), any non-zero Int64 input becomes a huge garbage value:
///
///   value 6 → bytes [0,0,0,0,0,0,0,6] → reinterpreted as 6 << 56 ≈ 4.32e17
///
/// For an Embedding lookup that's catastrophic — TFLite raises
/// "gather_nd index out of bounds" or, more confusingly, "failed precondition"
/// with no further info. Value 0 is endian-symmetric so smoke tests against
/// idx=0 silently pass — we hit this when our model placed bench_press at
/// index 6 instead of 0.
///
/// Workaround in `lib/inference/deepgain.dart`: pre-encode int64 inputs as
/// Uint8List with Endian.little.
///
/// This test calls predictRir for every exercise (idx 0..33). If the bug
/// regresses (e.g. tflite_flutter releases a fixed version and we revert
/// the workaround prematurely), all non-zero indices will throw.
library;

import 'dart:convert';
import 'dart:io';

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late ModelAssets assets;
  late DeepGain dg;
  late Mpc fresh;

  setUpAll(() async {
    final dir = Directory('assets/model');
    Map<String, dynamic> j(String n) => jsonDecode(
        File('${dir.path}/$n').readAsStringSync()) as Map<String, dynamic>;
    List<dynamic> jl(String n) => jsonDecode(
        File('${dir.path}/$n').readAsStringSync()) as List<dynamic>;

    assets = ModelAssets.fromRaw(
      fNetBytes: File('${dir.path}/f_net.tflite').readAsBytesSync(),
      gNetBytes: File('${dir.path}/g_net.tflite').readAsBytesSync(),
      exercises: List<String>.from(jl('exercises.json')),
      muscles: List<String>.from(jl('muscles.json')),
      involvement: jl('involvement_matrix.json')
          .map((row) => List<double>.from(
              (row as List).map((v) => (v as num).toDouble())))
          .toList(growable: false),
      anchorRatios: (j('anchor_ratio_matrix.json')['ratios'] as List)
          .map((row) => List<double>.from(
              (row as List).map((v) => (v as num).toDouble())))
          .toList(growable: false),
      anchorAvailable: List<double>.from(
          (j('anchor_ratio_matrix.json')['available'] as List)
              .map((v) => (v as num).toDouble())),
      tau: List<double>.from(jl('fixed_tau.json').map((v) => (v as num).toDouble())),
      scales: Scales(
        weight: (j('scales.json')['WEIGHT_SCALE'] as num).toDouble(),
        reps: (j('scales.json')['REPS_SCALE'] as num).toDouble(),
        rir: (j('scales.json')['RIR_SCALE'] as num).toDouble(),
        dt: (j('scales.json')['DT_SCALE'] as num).toDouble(),
      ),
      defaultAnchorsKg: List<double>.from(
          (j('default_anchors_kg.json')['values_kg'] as List)
              .map((v) => (v as num).toDouble())),
    );
    dg = DeepGain.fromAssets(assets);
    fresh = {for (final m in assets.muscles) m: 1.0};
  });

  tearDownAll(() => assets.close());

  test('predictRir works for every exercise index 0..N-1', () {
    final failures = <String>[];
    for (var i = 0; i < assets.exercises.length; i++) {
      final ex = assets.exercises[i];
      try {
        final rir = dg.predictRir(
          state: fresh, exercise: ex, weightKg: 50.0, reps: 5,
        );
        if (rir.isNaN || rir < 0 || rir > 5) {
          failures.add('idx=$i ($ex): out-of-range RIR $rir');
        }
      } catch (e) {
        failures.add('idx=$i ($ex): $e');
      }
    }
    expect(failures, isEmpty,
        reason: 'tflite_flutter Int64 endian bug regression — '
            'see int64_endian_regression_test.dart header');
  });
}
