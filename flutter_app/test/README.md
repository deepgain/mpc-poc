# Test layout

Three tiers, intentionally split by what they verify and what they need
to run.

## Note on the tflite_flutter Int64 endian bug

`tflite_flutter` 0.11.0 has a confirmed bug in
`_convertElementToBytes` (lib/src/util/byte_conversion_utils.dart) — Int64
inputs are written with `Endian.big` instead of `Endian.little`, garbling
any non-zero index on little-endian hosts (i.e., everywhere we run).

`DeepGain` works around it by pre-encoding Int64 inputs as `Uint8List` with
explicit little-endian bytes. The `int64_endian_regression_test.dart`
exercises every exercise index 0..N-1 to catch regressions if the workaround
is ever removed before tflite_flutter ships a fix.

If you bump tflite_flutter, check the conversion code first and only drop
the workaround when the upstream fix lands.

## 1. `test/inference/orchestration_test.dart` — host, no dylib

Runs via plain `flutter test`. Verifies the **Dart orchestration math** —
recovery formula, multiplicative MPC update, 0.1 floor, history sorting,
exclusion of future-dated sets, the unknown-exercise skip path, and RIR
clamping. The TFLite calls are mocked, so no `libtensorflowlite_c` needed.

```bash
cd flutter_app
flutter test test/inference/orchestration_test.dart
```

## 2. `test/inference/parity_test.dart` — host, real TFLite

Runs via `flutter test`. Loads the real `.tflite` models and checks that
`DeepGain.predictMpc` / `predictRir` match Python's outputs within the
same tolerances the Python parity test uses (`MPC ±1e-4`, `RIR ±1e-3`).

**Requires** `libtensorflowlite_c-mac.dylib` at:

```
/Users/adev/flutter/bin/cache/artifacts/engine/resources/libtensorflowlite_c-mac.dylib
```

(or wherever `${Platform.resolvedExecutable}.parent.parent/resources/` resolves
to on your machine; tflite_flutter will print the exact expected path in
the dlopen error.)

To build the dylib (one-time, ~10–20 min):

```bash
git clone --depth 1 https://github.com/tensorflow/tensorflow.git ~/tflite-build
cd ~/tflite-build
mkdir build && cd build
cmake ../tensorflow/lite/c -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build . -j$(sysctl -n hw.ncpu) --config Release
mkdir -p /Users/adev/flutter/bin/cache/artifacts/engine/resources/
cp libtensorflowlite_c.dylib /Users/adev/flutter/bin/cache/artifacts/engine/resources/libtensorflowlite_c-mac.dylib
```

Then:

```bash
cd flutter_app
flutter test test/inference/parity_test.dart
```

## 3. `integration_test/parity_test.dart` — on-device

Same fixtures and tolerances as #2, but runs on a connected iOS simulator
or Android device where the native library ships in the app bundle. Use
this to validate that the same code that passes on host actually runs on
the target platform.

```bash
cd flutter_app
flutter test integration_test/parity_test.dart  # needs a device/simulator
```

## Where the parity chain ends up

```
inference.predict_mpc (Python)        ← model spec
        ↕  Phase 1: tools/tflite_export/parity_test.py  (max diff 1.19e-7)
TFLiteOrchestrator (NumPy + .tflite)
        ↕  Phase 2: test/inference/parity_test.dart     (host, real TFLite)
        ↕  Phase 2: integration_test/parity_test.dart   (device, real TFLite)
DeepGain.predictMpc (Dart + .tflite)
```

If a Phase-1 model bump changes outputs, regenerate the golden:

```bash
cd tools/tflite_export
.venv/bin/python export.py --checkpoint ../../deepgain_model_muscle_ord.pt
.venv/bin/python parity_test.py
.venv/bin/python gen_golden.py
cp assets/* ../../flutter_app/assets/model/
cd ../../flutter_app
flutter test test/inference/parity_test.dart  # or integration_test/
```
