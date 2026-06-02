# Keep TFLite + GPU delegate classes — referenced reflectively by the
# native libtensorflowlite_jni.so. Without these, R8 strips the GPU
# delegate types and `flutter build apk` fails with:
#   Missing class org.tensorflow.lite.gpu.GpuDelegateFactory$Options
# even though we don't use the GPU delegate ourselves.
-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.gpu.**

# Drift / sqlite_async use isolates — keep entry points so R8 doesn't
# strip the background isolate's runner.
-keep class * extends drift.** { *; }
-dontwarn drift.**
