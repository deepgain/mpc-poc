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
    final assets = ref.watch(modelAssetsProvider);
    final priors = ref.watch(strengthPriorsProvider);
    final meta = ref.watch(plannerMetaProvider);
    final anchors = ref.watch(anchorsProvider);

    final loaded = assets is AsyncData &&
        priors is AsyncData &&
        meta is AsyncData &&
        anchors is AsyncData;

    if (!loaded) {
      return const _SplashScreen();
    }

    final hasAnchors = anchors.value != null;
    return hasAnchors ? const HomeScreen() : const OnboardingScreen();
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
