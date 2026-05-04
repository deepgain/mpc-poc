/// Host parity test — full TFLite pipeline vs golden fixtures.
///
/// Loads the real .tflite models via tflite_flutter and verifies that
/// `DeepGain.predictMpc` / `predictRir` reproduce Python's outputs from
/// `tools/tflite_export/gen_golden.py` within FP32 tolerance
/// (MPC ±1e-4, RIR ±1e-3 — same as the Python parity test).
///
/// Requires libtensorflowlite_c-mac.dylib at:
///   /Users/adev/flutter/bin/cache/artifacts/engine/resources/
/// (see test/README.md for build instructions). Without it, this test
/// fails at setUpAll with a dylib-not-found error.
///
/// The orchestration math is also covered by orchestration_test.dart with
/// mocked TFLite calls, so that file runs without the dylib.
library;

import 'dart:convert';
import 'dart:io';

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:flutter_test/flutter_test.dart';

const double kMpcTol = 1e-4;
const double kRirTol = 1e-3;

Future<ModelAssets> _loadFromDisk() async {
  final modelDir = Directory('assets/model');
  Map<String, dynamic> j(String name) => jsonDecode(
        File('${modelDir.path}/$name').readAsStringSync(),
      ) as Map<String, dynamic>;
  List<dynamic> jl(String name) => jsonDecode(
        File('${modelDir.path}/$name').readAsStringSync(),
      ) as List<dynamic>;

  final exercises = List<String>.from(jl('exercises.json'));
  final muscles = List<String>.from(jl('muscles.json'));
  final inv = jl('involvement_matrix.json')
      .map((row) => List<double>.from((row as List).map((v) => (v as num).toDouble())))
      .toList(growable: false);
  final anchor = j('anchor_ratio_matrix.json');
  final ratios = (anchor['ratios'] as List)
      .map((row) => List<double>.from((row as List).map((v) => (v as num).toDouble())))
      .toList(growable: false);
  final availability = List<double>.from(
      (anchor['available'] as List).map((v) => (v as num).toDouble()));
  final tau = List<double>.from(
      jl('fixed_tau.json').map((v) => (v as num).toDouble()));
  final scalesRaw = j('scales.json');
  final scales = Scales(
    weight: (scalesRaw['WEIGHT_SCALE'] as num).toDouble(),
    reps: (scalesRaw['REPS_SCALE'] as num).toDouble(),
    rir: (scalesRaw['RIR_SCALE'] as num).toDouble(),
    dt: (scalesRaw['DT_SCALE'] as num).toDouble(),
  );
  final defaults = j('default_anchors_kg.json');
  final defaultAnchorsKg = List<double>.from(
      (defaults['values_kg'] as List).map((v) => (v as num).toDouble()));

  return ModelAssets.fromRaw(
    fNetBytes: File('${modelDir.path}/f_net.tflite').readAsBytesSync(),
    gNetBytes: File('${modelDir.path}/g_net.tflite').readAsBytesSync(),
    exercises: exercises,
    muscles: muscles,
    involvement: inv,
    anchorRatios: ratios,
    anchorAvailable: availability,
    tau: tau,
    scales: scales,
    defaultAnchorsKg: defaultAnchorsKg,
  );
}

void main() {
  late ModelAssets assets;
  late DeepGain deepgain;
  final fixturePath = 'assets/golden/golden.json';
  final fixtures = jsonDecode(File(fixturePath).readAsStringSync()) as List;

  setUpAll(() async {
    assets = await _loadFromDisk();
    deepgain = DeepGain.fromAssets(assets);
  });

  tearDownAll(() {
    assets.close();
  });

  for (final scenarioRaw in fixtures) {
    final sc = scenarioRaw as Map<String, dynamic>;
    final name = sc['name'] as String;

    test('parity: $name', () {
      final history = (sc['history'] as List)
          .map((h) => WorkoutSet.fromJson(h as Map<String, dynamic>))
          .toList();
      final queryTs = DateTime.parse(sc['query_ts'] as String);
      final anchorsKg = List<double>.from(
          (sc['anchors_kg'] as List).map((v) => (v as num).toDouble()));
      final expectedMpc = (sc['expected_mpc'] as Map<String, dynamic>)
          .map((k, v) => MapEntry(k, (v as num).toDouble()));

      final actualMpc = deepgain.predictMpc(
        history: history, timestamp: queryTs, anchorsKg: anchorsKg,
      );

      for (final entry in expectedMpc.entries) {
        final actual = actualMpc[entry.key];
        expect(actual, isNotNull, reason: 'muscle ${entry.key} missing');
        expect(
          (actual! - entry.value).abs(),
          lessThan(kMpcTol),
          reason: 'muscle ${entry.key}: expected ${entry.value} got $actual',
        );
      }

      for (final r in (sc['rir_checks'] as List)) {
        final rmap = r as Map<String, dynamic>;
        final actualRir = deepgain.predictRir(
          state: actualMpc,
          exercise: rmap['exercise'] as String,
          weightKg: (rmap['weight_kg'] as num).toDouble(),
          reps: (rmap['reps'] as num).toInt(),
          anchorsKg: anchorsKg,
        );
        final expectedRir = (rmap['expected_rir'] as num).toDouble();
        expect(
          (actualRir - expectedRir).abs(),
          lessThan(kRirTol),
          reason:
              'rir(${rmap['exercise']}, ${rmap['weight_kg']}kg×${rmap['reps']}): '
              'expected $expectedRir got $actualRir',
        );
      }
    });
  }
}
