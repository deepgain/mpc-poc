/// Root MaterialApp + onboarding gate. While the model is still loading we
/// show a splash; once ready, route to onboarding (no anchors yet) or home
/// (anchors persisted).
library;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'features/home/home_screen.dart';
import 'features/onboarding/onboarding_screen.dart';
import 'state/providers.dart';

class DeepGainApp extends StatelessWidget {
  const DeepGainApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DeepGain',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1E88E5),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1E88E5),
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const _AppGate(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class _AppGate extends ConsumerWidget {
  const _AppGate();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Wait for the model + assets to load — without these no screen can render.
    final gates = <(String, AsyncValue<Object?>)>[
      ('model', ref.watch(modelAssetsProvider)),
      ('strength priors', ref.watch(strengthPriorsProvider)),
      ('planner meta', ref.watch(plannerMetaProvider)),
      ('anchors', ref.watch(anchorsProvider)),
    ];

    // Surface a failed load instead of hanging on the splash forever. Without
    // this, any startup exception (e.g. an asset or the TFLite runtime failing
    // to initialise on a given device) is indistinguishable from "still
    // loading" — the spinner just spins. See onboarding/iPad infinite-spinner.
    for (final (name, value) in gates) {
      if (value case AsyncError(:final error, :final stackTrace)) {
        // Log full detail for diagnostics, but never surface a raw stacktrace
        // to the user — show a friendly, retryable screen instead.
        debugPrint('Startup load failed ($name): $error\n$stackTrace');
        return _StartupErrorScreen(
          onRetry: () {
            ref.invalidate(modelAssetsProvider);
            ref.invalidate(strengthPriorsProvider);
            ref.invalidate(plannerMetaProvider);
            ref.invalidate(anchorsProvider);
          },
        );
      }
    }

    final loaded = gates.every((g) => g.$2 is AsyncData);
    if (!loaded) {
      return const _SplashScreen();
    }

    final hasAnchors = ref.watch(anchorsProvider).value != null;
    return hasAnchors ? const HomeScreen() : const OnboardingScreen();
  }
}

class _StartupErrorScreen extends StatelessWidget {
  final VoidCallback onRetry;

  const _StartupErrorScreen({required this.onRetry});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.refresh, size: 48, color: scheme.primary),
                const SizedBox(height: 20),
                Text(
                  "Couldn't get things ready",
                  textAlign: TextAlign.center,
                  style: theme.textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text(
                  'Something went wrong while starting up. Please try again.',
                  textAlign: TextAlign.center,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
                const SizedBox(height: 28),
                FilledButton(
                  onPressed: onRetry,
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(52),
                  ),
                  child: const Text('Try again'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _SplashScreen extends StatelessWidget {
  const _SplashScreen();

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 24),
            Text('Loading model…',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500)),
          ],
        ),
      ),
    );
  }
}
