/// Loads the exported TFLite models + JSON constants from Flutter assets.
///
/// Single source of truth for indexes (exercise / muscle name ↔ idx),
/// normalization scales, recovery time constants, and the involvement +
/// anchor-ratio matrices. Everything here was produced by
/// `tools/tflite_export/export.py` and verified via the Python parity test.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

class Scales {
  final double weight; // WEIGHT_SCALE
  final double reps; // REPS_SCALE
  final double rir; // RIR_SCALE
  final double dt; // DT_SCALE (unused in Dart — we don't log/expm1 round-trip)

  const Scales({
    required this.weight,
    required this.reps,
    required this.rir,
    required this.dt,
  });
}

class ModelAssets {
  final Interpreter fNet;
  final Interpreter gNet;

  final List<String> exercises;
  final Map<String, int> exerciseToIdx;
  final List<String> muscles;
  final Map<String, int> muscleToIdx;

  /// Per-exercise per-muscle involvement weights, shape (numExercises, 15).
  /// Indexed as `involvement[exerciseIdx][muscleIdx]`.
  final List<List<double>> involvement;

  /// Per-exercise 1RM ratios for [bench, squat, deadlift]: shape (numExercises, 3).
  /// `available[i]` is 1.0 if the exercise has a defined anchor, else 0.0.
  final List<List<double>> anchorRatios;
  final List<double> anchorAvailable;

  /// Recovery time constants τ in hours, per muscle.
  final List<double> tau;

  final Scales scales;
  final List<double> defaultAnchorsKg; // [bench, squat, deadlift]

  ModelAssets._({
    required this.fNet,
    required this.gNet,
    required this.exercises,
    required this.exerciseToIdx,
    required this.muscles,
    required this.muscleToIdx,
    required this.involvement,
    required this.anchorRatios,
    required this.anchorAvailable,
    required this.tau,
    required this.scales,
    required this.defaultAnchorsKg,
  });

  static Future<ModelAssets> load({String dir = 'assets/model'}) async {
    Future<dynamic> j(String name) async =>
        jsonDecode(await rootBundle.loadString('$dir/$name'));

    final exercises = List<String>.from(await j('exercises.json') as List);
    final muscles = List<String>.from(await j('muscles.json') as List);

    final inv = (await j('involvement_matrix.json') as List)
        .map((row) => List<double>.from((row as List).map((v) => (v as num).toDouble())))
        .toList(growable: false);

    final anchor = await j('anchor_ratio_matrix.json') as Map<String, dynamic>;
    final ratios = (anchor['ratios'] as List)
        .map((row) => List<double>.from((row as List).map((v) => (v as num).toDouble())))
        .toList(growable: false);
    final availability = List<double>.from(
        (anchor['available'] as List).map((v) => (v as num).toDouble()));

    final tau = List<double>.from(
        (await j('fixed_tau.json') as List).map((v) => (v as num).toDouble()));

    final scalesRaw = await j('scales.json') as Map<String, dynamic>;
    final scales = Scales(
      weight: (scalesRaw['WEIGHT_SCALE'] as num).toDouble(),
      reps: (scalesRaw['REPS_SCALE'] as num).toDouble(),
      rir: (scalesRaw['RIR_SCALE'] as num).toDouble(),
      dt: (scalesRaw['DT_SCALE'] as num).toDouble(),
    );

    final defaults = await j('default_anchors_kg.json') as Map<String, dynamic>;
    final defaultAnchorsKg = List<double>.from(
        (defaults['values_kg'] as List).map((v) => (v as num).toDouble()));

    final fNetBytes = await rootBundle.load('$dir/f_net.tflite');
    final gNetBytes = await rootBundle.load('$dir/g_net.tflite');
    final fNet = Interpreter.fromBuffer(fNetBytes.buffer.asUint8List());
    final gNet = Interpreter.fromBuffer(gNetBytes.buffer.asUint8List());
    fNet.allocateTensors();
    gNet.allocateTensors();

    return ModelAssets._(
      fNet: fNet,
      gNet: gNet,
      exercises: exercises,
      exerciseToIdx: {for (var i = 0; i < exercises.length; i++) exercises[i]: i},
      muscles: muscles,
      muscleToIdx: {for (var i = 0; i < muscles.length; i++) muscles[i]: i},
      involvement: inv,
      anchorRatios: ratios,
      anchorAvailable: availability,
      tau: tau,
      scales: scales,
      defaultAnchorsKg: defaultAnchorsKg,
    );
  }

  /// Variant for tests: load from raw bytes instead of rootBundle.
  static ModelAssets fromRaw({
    required Uint8List fNetBytes,
    required Uint8List gNetBytes,
    required List<String> exercises,
    required List<String> muscles,
    required List<List<double>> involvement,
    required List<List<double>> anchorRatios,
    required List<double> anchorAvailable,
    required List<double> tau,
    required Scales scales,
    required List<double> defaultAnchorsKg,
  }) {
    final fNet = Interpreter.fromBuffer(fNetBytes);
    final gNet = Interpreter.fromBuffer(gNetBytes);
    fNet.allocateTensors();
    gNet.allocateTensors();
    return ModelAssets._(
      fNet: fNet,
      gNet: gNet,
      exercises: exercises,
      exerciseToIdx: {for (var i = 0; i < exercises.length; i++) exercises[i]: i},
      muscles: muscles,
      muscleToIdx: {for (var i = 0; i < muscles.length; i++) muscles[i]: i},
      involvement: involvement,
      anchorRatios: anchorRatios,
      anchorAvailable: anchorAvailable,
      tau: tau,
      scales: scales,
      defaultAnchorsKg: defaultAnchorsKg,
    );
  }

  void close() {
    fNet.close();
    gNet.close();
  }
}
