/// Onboarding — three big numbers (bench / squat / deadlift), defaults
/// 100 / 140 / 180 kg. "I don't know" toggle keeps defaults. Single CTA
/// writes the row to [Anchors] and the gate routes to home.
///
/// Designed for <60s to first plan generation.
library;

import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class OnboardingScreen extends ConsumerStatefulWidget {
  const OnboardingScreen({super.key});

  @override
  ConsumerState<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends ConsumerState<OnboardingScreen> {
  static const _defaultBench = 100.0;
  static const _defaultSquat = 140.0;
  static const _defaultDeadlift = 180.0;

  late final TextEditingController _benchCtl =
      TextEditingController(text: _defaultBench.toStringAsFixed(0));
  late final TextEditingController _squatCtl =
      TextEditingController(text: _defaultSquat.toStringAsFixed(0));
  late final TextEditingController _deadliftCtl =
      TextEditingController(text: _defaultDeadlift.toStringAsFixed(0));

  bool _dontKnow = false;
  bool _saving = false;

  @override
  void dispose() {
    _benchCtl.dispose();
    _squatCtl.dispose();
    _deadliftCtl.dispose();
    super.dispose();
  }

  double _parse(TextEditingController c, double fallback) {
    final v = double.tryParse(c.text.replaceAll(',', '.'));
    return (v != null && v > 0 && v < 1000) ? v : fallback;
  }

  Future<void> _start() async {
    setState(() => _saving = true);
    final db = ref.read(databaseProvider);
    final bench = _dontKnow ? _defaultBench : _parse(_benchCtl, _defaultBench);
    final squat = _dontKnow ? _defaultSquat : _parse(_squatCtl, _defaultSquat);
    final dead = _dontKnow
        ? _defaultDeadlift
        : _parse(_deadliftCtl, _defaultDeadlift);
    try {
      await db.setAnchors(benchKg: bench, squatKg: squat, deadliftKg: dead);

      // Verify the write actually committed — without this it's hard to tell
      // whether a stuck gate is "drift didn't write" vs "drift didn't notify".
      final readback = await db.getAnchors();
      if (readback == null) {
        throw StateError('setAnchors completed but row is missing');
      }

      // Refresh the gate. anchorsProvider is a FutureProvider so .future
      // returns the next computed value; awaiting guarantees the gate has
      // data BEFORE we clear the spinner, so the user never sees a brief
      // stale onboarding screen.
      ref.invalidate(anchorsProvider);
      await ref.read(anchorsProvider.future);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save: $e')),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Scaffold(
      // resizeToAvoidBottomInset: true is the default but we make the layout
      // scrollable anyway, since on small phones the form + keyboard overflows.
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            return SingleChildScrollView(
              padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
              child: ConstrainedBox(
                // Keep the layout at least as tall as the screen so the
                // "Start" button still pins to the bottom when there's room,
                // but lets it scroll when the keyboard is open.
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: IntrinsicHeight(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const SizedBox(height: 32),
                      Text(
                        'DeepGain',
                        style: theme.textTheme.displaySmall?.copyWith(
                          fontWeight: FontWeight.w700,
                          color: scheme.primary,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'What can you lift today?',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 40),
                      Opacity(
                        opacity: _dontKnow ? 0.4 : 1.0,
                        child: AbsorbPointer(
                          absorbing: _dontKnow,
                          child: Column(
                            children: [
                              _NumberField(
                                label: 'Bench press',
                                suffix: 'kg',
                                controller: _benchCtl,
                              ),
                              const SizedBox(height: 16),
                              _NumberField(
                                label: 'Squat',
                                suffix: 'kg',
                                controller: _squatCtl,
                              ),
                              const SizedBox(height: 16),
                              _NumberField(
                                label: 'Deadlift',
                                suffix: 'kg',
                                controller: _deadliftCtl,
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      SwitchListTile.adaptive(
                        contentPadding: EdgeInsets.zero,
                        value: _dontKnow,
                        onChanged: (v) => setState(() => _dontKnow = v),
                        title: Text(
                          "I don't know my 1RMs",
                          style: theme.textTheme.bodyLarge,
                        ),
                        subtitle: Text(
                          'Use defaults, you can update later',
                          style: theme.textTheme.bodySmall,
                        ),
                      ),
                      const Spacer(),
                      const SizedBox(height: 16),
                      FilledButton(
                        onPressed: _saving ? null : _start,
                        style: FilledButton.styleFrom(
                          minimumSize: const Size.fromHeight(56),
                          textStyle: theme.textTheme.titleMedium,
                        ),
                        child: _saving
                            ? const SizedBox(
                                height: 24, width: 24,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Text('Start'),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Your data stays on this device. No account needed.',
                        textAlign: TextAlign.center,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        ),
      ),
    );
  }
}

class _NumberField extends StatelessWidget {
  final String label;
  final String suffix;
  final TextEditingController controller;

  const _NumberField({
    required this.label,
    required this.suffix,
    required this.controller,
  });

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: controller,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      inputFormatters: [
        FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
      ],
      decoration: InputDecoration(
        labelText: label,
        suffixText: suffix,
        border: const OutlineInputBorder(),
      ),
      style: const TextStyle(fontSize: 22, fontWeight: FontWeight.w600),
    );
  }
}
