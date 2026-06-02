/// On-device parity test — full TFLite pipeline vs golden fixtures.
///
/// This test loads the real .tflite models via tflite_flutter and verifies
/// that DeepGain.predictMpc / predictRir reproduce Python's outputs from
/// `tools/tflite_export/gen_golden.py` within FP32 tolerance.
///
/// Run on a connected iOS simulator / Android device:
///   cd flutter_app
///   flutter test integration_test/parity_test.dart
///
/// Cannot run via plain `flutter test` because tflite_flutter needs the
/// platform's native tensorflowlite_c library, which only ships with the
/// app bundle on iOS / .so on Android. The orchestration math itself is
/// covered by `test/inference/orchestration_test.dart` which mocks the
/// f_net / g_net calls.
library;

import 'dart:convert';

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/strength.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

const double kMpcTol = 1e-4;
const double kRirTol = 1e-3;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late ModelAssets assets;
  late DeepGain deepgain;
  late StrengthPriors strengthPriors;
  late List<dynamic> fixtures;

  setUpAll(() async {
    assets = await ModelAssets.load();
    deepgain = DeepGain.fromAssets(assets);
    strengthPriors = await StrengthPriors.load();
    fixtures = jsonDecode(
      await rootBundle.loadString('assets/golden/golden.json'),
    ) as List;
  });

  tearDownAll(() {
    assets.close();
  });

  testWidgets('Dart ↔ Python parity across all golden scenarios',
      (tester) async {
    final failures = <String>[];

    for (final scenarioRaw in fixtures) {
      final sc = scenarioRaw as Map<String, dynamic>;
      final name = sc['name'] as String;

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
        strengthPriors: strengthPriors,
      );

      for (final entry in expectedMpc.entries) {
        final actual = actualMpc[entry.key];
        if (actual == null) {
          failures.add('$name: muscle ${entry.key} missing from output');
          continue;
        }
        final diff = (actual - entry.value).abs();
        if (diff >= kMpcTol) {
          failures.add(
            '$name muscle=${entry.key}: '
            'expected=${entry.value.toStringAsFixed(6)} '
            'actual=${actual.toStringAsFixed(6)} diff=${diff.toStringAsExponential(2)}',
          );
        }
      }

      for (final r in (sc['rir_checks'] as List)) {
        final rmap = r as Map<String, dynamic>;
        final exercise = rmap['exercise'] as String;
        final weightKg = (rmap['weight_kg'] as num).toDouble();
        final reps = (rmap['reps'] as num).toInt();
        final expectedRir = (rmap['expected_rir'] as num).toDouble();

        final actualRir = deepgain.predictRir(
          state: actualMpc,
          exercise: exercise, weightKg: weightKg, reps: reps,
          anchorsKg: anchorsKg,
        );
        final diff = (actualRir - expectedRir).abs();
        if (diff >= kRirTol) {
          failures.add(
            '$name rir($exercise, ${weightKg}kg×$reps): '
            'expected=${expectedRir.toStringAsFixed(4)} '
            'actual=${actualRir.toStringAsFixed(4)} diff=${diff.toStringAsExponential(2)}',
          );
        }
      }
    }

    if (failures.isNotEmpty) {
      fail('${failures.length} parity failures:\n  ${failures.join("\n  ")}');
    }
  });
}
