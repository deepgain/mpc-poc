/// Body map — front + back human silhouettes painted with CustomPainter.
///
/// The silhouette is a single smooth bezier outline (head + gingerbread
/// body), drawn with a soft drop shadow and a subtle gradient fill so it
/// reads as a real figure rather than a stack of rectangles. Muscle groups
/// are anatomically-placed shapes, *clipped to the body*, filled with the
/// same red/orange/green recovery colour used on the muscle list. Tap a
/// lit-up region → drills into the per-muscle MPC chart.
library;

import 'package:deepgain_app/features/body_map/muscle_detail_screen.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class BodyMapScreen extends ConsumerWidget {
  const BodyMapScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mpcAsync = ref.watch(mpcProvider);

    return mpcAsync.when(
      loading: () => const Center(child: CircularProgressIndicator()),
      error: (e, _) => Center(child: Text('$e')),
      data: (mpc) {
        // Bottom padding clears the FAB ("Plan a session", ~64dp tall +
        // 16dp margin) AND the Android system gesture inset that
        // NavigationBar applies via SafeArea above.
        return SingleChildScrollView(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 140),
          child: Column(
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(child: _BodyFigure(mpc: mpc, isFront: true)),
                  const SizedBox(width: 8),
                  Expanded(child: _BodyFigure(mpc: mpc, isFront: false)),
                ],
              ),
              const SizedBox(height: 20),
              const _Legend(),
            ],
          ),
        );
      },
    );
  }
}

/// One anatomical region: a muscle id + the shapes that paint/hit-test it.
class _Region {
  final String muscle;
  final List<Path> shapes;
  const _Region(this.muscle, this.shapes);
}

class _BodyFigure extends StatelessWidget {
  final Mpc mpc;
  final bool isFront;
  const _BodyFigure({required this.mpc, required this.isFront});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Column(
      children: [
        Text(
          isFront ? 'Front' : 'Back',
          style: theme.textTheme.titleSmall?.copyWith(
            color: scheme.onSurfaceVariant,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.2,
          ),
        ),
        const SizedBox(height: 10),
        AspectRatio(
          aspectRatio: 0.46,
          child: LayoutBuilder(
            builder: (context, c) {
              final size = Size(c.maxWidth, c.maxHeight);
              final regions =
                  isFront ? _frontRegions(size) : _backRegions(size);
              return GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTapUp: (details) {
                  final p = details.localPosition;
                  for (final r in regions) {
                    if (r.shapes.any((s) => s.contains(p))) {
                      Navigator.of(context).push(MaterialPageRoute<void>(
                        builder: (_) => MuscleDetailScreen(muscle: r.muscle),
                      ));
                      return;
                    }
                  }
                },
                child: CustomPaint(
                  size: size,
                  painter: _BodyPainter(
                    mpc: mpc,
                    regions: regions,
                    scheme: scheme,
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }
}

// ── Geometry helpers (normalised [0,1] → pixel) ──────────────────────────────

Rect _rel(Size s, double x, double y, double w, double h) =>
    Rect.fromLTWH(x * s.width, y * s.height, w * s.width, h * s.height);

Path _rrect(Size s, double x, double y, double w, double h, double r) => Path()
  ..addRRect(RRect.fromRectAndRadius(_rel(s, x, y, w, h), Radius.circular(r)));

Path _ellipse(Size s, double x, double y, double w, double h) =>
    Path()..addOval(_rel(s, x, y, w, h));

/// Mirror a normalised rect across the vertical centre (x → 1 - x - w).
Path _rrectMirror(Size s, double x, double y, double w, double h, double r) =>
    _rrect(s, 1 - x - w, y, w, h, r);

Path _ellipseMirror(Size s, double x, double y, double w, double h) =>
    _ellipse(s, 1 - x - w, y, w, h);

// ── The silhouette outline ───────────────────────────────────────────────────

/// Right-half perimeter vertices (neck top → crotch), normalised. The left
/// half is this list mirrored about x = 0.5 and reversed, so the whole body
/// is one symmetric, closed curve. Proportions are deliberately masculine:
/// broad shoulders, a wide straight waist, and hips narrower than the
/// shoulders (no hourglass taper).
const List<Offset> _rightHalf = [
  Offset(0.538, 0.108), // neck, top
  Offset(0.556, 0.162), // neck base / trap inner
  Offset(0.702, 0.168), // trapezius slope (broad)
  Offset(0.794, 0.208), // shoulder / deltoid peak (wide)
  Offset(0.830, 0.276), // outer deltoid
  Offset(0.822, 0.368), // outer elbow
  Offset(0.798, 0.458), // outer forearm
  Offset(0.770, 0.508), // hand, outer
  Offset(0.718, 0.484), // hand, inner
  Offset(0.706, 0.376), // inner forearm
  Offset(0.696, 0.270), // inner upper arm
  Offset(0.668, 0.234), // armpit
  Offset(0.646, 0.338), // ribs (little taper)
  Offset(0.632, 0.434), // waist (broad, straight)
  Offset(0.654, 0.486), // hip
  Offset(0.676, 0.530), // hip, outer (narrower than shoulder)
  Offset(0.666, 0.646), // thigh
  Offset(0.638, 0.744), // knee
  Offset(0.632, 0.862), // calf
  Offset(0.616, 0.966), // ankle, outer
  Offset(0.606, 0.998), // foot, outer
  Offset(0.548, 0.998), // foot, inner
  Offset(0.548, 0.930), // ankle, inner
  Offset(0.556, 0.782), // inner calf
  Offset(0.534, 0.632), // inner thigh
  Offset(0.504, 0.566), // crotch
];

/// Head: a slightly squared oval on top of the neck. Drawn as its own shape;
/// the body's neck overlaps its base, hiding the lower arc for a clean join.
Path _headPath(Size s) =>
    _rrect(s, 0.420, 0.000, 0.160, 0.128, 0.058 * s.width);

/// Smooth closed curve through [pts] (anchors as quadratic control points,
/// curve passes through edge midpoints). Produces clean tangents everywhere.
Path _smoothClosed(List<Offset> pts, Size s) {
  final path = Path();
  if (pts.isEmpty) return path;
  Offset mid(Offset a, Offset b) =>
      Offset((a.dx + b.dx) / 2 * s.width, (a.dy + b.dy) / 2 * s.height);
  final start = mid(pts.last, pts.first);
  path.moveTo(start.dx, start.dy);
  for (var i = 0; i < pts.length; i++) {
    final curr = pts[i];
    final next = pts[(i + 1) % pts.length];
    final m = mid(curr, next);
    path.quadraticBezierTo(curr.dx * s.width, curr.dy * s.height, m.dx, m.dy);
  }
  path.close();
  return path;
}

Path _bodyPath(Size s) {
  final pts = <Offset>[..._rightHalf];
  // Left half: mirror + reverse, skip the shared crotch & neck endpoints.
  for (var i = _rightHalf.length - 2; i >= 0; i--) {
    final p = _rightHalf[i];
    pts.add(Offset(1 - p.dx, p.dy));
  }
  return _smoothClosed(pts, s);
}

// ── Muscle region layouts ────────────────────────────────────────────────────
//
// Shapes are deliberately a touch larger than the limb; they're clipped to the
// body outline, so generous bounds simply fill the limb edge-to-edge.

List<_Region> _frontRegions(Size s) => [
      _Region('chest', [
        _rrect(s, 0.502, 0.182, 0.148, 0.104, 14),
        _rrectMirror(s, 0.502, 0.182, 0.148, 0.104, 14),
      ]),
      _Region('anterior_delts', [
        _ellipse(s, 0.668, 0.184, 0.120, 0.092),
        _ellipseMirror(s, 0.668, 0.184, 0.120, 0.092),
      ]),
      // Side delts: the outer shoulder cap, clipped to the body's widest point.
      _Region('lateral_delts', [
        _ellipse(s, 0.766, 0.196, 0.092, 0.106),
        _ellipseMirror(s, 0.766, 0.196, 0.092, 0.106),
      ]),
      _Region('biceps', [
        _rrect(s, 0.700, 0.276, 0.124, 0.150, 18),
        _rrectMirror(s, 0.700, 0.276, 0.124, 0.150, 18),
      ]),
      _Region('abs', [_rrect(s, 0.424, 0.296, 0.152, 0.142, 16)]),
      // Inner-thigh strip, sitting in the gap between the two quads.
      _Region('adductors', [_rrect(s, 0.470, 0.556, 0.060, 0.168, 14)]),
      _Region('quads', [
        _rrect(s, 0.534, 0.540, 0.158, 0.200, 24),
        _rrectMirror(s, 0.534, 0.540, 0.158, 0.200, 24),
      ]),
    ];

List<_Region> _backRegions(Size s) => [
      _Region('rear_delts', [
        _ellipse(s, 0.668, 0.184, 0.120, 0.092),
        _ellipseMirror(s, 0.668, 0.184, 0.120, 0.092),
      ]),
      // Side delts: the outer shoulder cap, clipped to the body's widest point.
      _Region('lateral_delts', [
        _ellipse(s, 0.766, 0.196, 0.092, 0.106),
        _ellipseMirror(s, 0.766, 0.196, 0.092, 0.106),
      ]),
      _Region('rhomboids', [_rrect(s, 0.388, 0.178, 0.224, 0.106, 16)]),
      _Region('lats', [
        _rrect(s, 0.500, 0.272, 0.156, 0.156, 18),
        _rrectMirror(s, 0.500, 0.272, 0.156, 0.156, 18),
      ]),
      _Region('triceps', [
        _rrect(s, 0.700, 0.276, 0.124, 0.150, 18),
        _rrectMirror(s, 0.700, 0.276, 0.124, 0.150, 18),
      ]),
      _Region('erectors', [_rrect(s, 0.454, 0.336, 0.092, 0.136, 12)]),
      _Region('glutes', [
        _rrect(s, 0.500, 0.452, 0.162, 0.106, 22),
        _rrectMirror(s, 0.500, 0.452, 0.162, 0.106, 22),
      ]),
      _Region('hamstrings', [
        _rrect(s, 0.534, 0.564, 0.158, 0.182, 22),
        _rrectMirror(s, 0.534, 0.564, 0.158, 0.182, 22),
      ]),
      _Region('calves', [
        _rrect(s, 0.548, 0.782, 0.116, 0.164, 18),
        _rrectMirror(s, 0.548, 0.782, 0.116, 0.164, 18),
      ]),
    ];

class _BodyPainter extends CustomPainter {
  final Mpc mpc;
  final List<_Region> regions;
  final ColorScheme scheme;

  _BodyPainter({
    required this.mpc,
    required this.regions,
    required this.scheme,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final body = _bodyPath(size);
    final head = _headPath(size);

    final outline = Paint()
      ..color = scheme.outline.withValues(alpha: 0.85)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.2
      ..strokeJoin = StrokeJoin.round;

    // Body fill: gentle top-to-bottom gradient for a hint of volume.
    final fill = Paint()
      ..style = PaintingStyle.fill
      ..shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [
          Color.alphaBlend(
              scheme.surfaceContainerHighest.withValues(alpha: 0.9),
              scheme.surface),
          scheme.surfaceContainerLow,
        ],
      ).createShader(Offset.zero & size);

    // Soft drop shadows → the figure lifts off the background.
    canvas.drawShadow(head, Colors.black.withValues(alpha: 0.35), 6, false);
    canvas.drawShadow(body, Colors.black.withValues(alpha: 0.35), 7, false);

    // Head first; the body's neck is drawn on top and hides its lower arc.
    canvas.drawPath(head, fill);
    canvas.drawPath(head, outline);

    canvas.drawPath(body, fill);

    // Muscle fills, clipped to the body so nothing spills past the outline.
    // Each region is also stroked so adjacent groups (e.g. adductors next to
    // quads) stay visually distinct instead of merging into one colour blob.
    final muscleEdge = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4
      ..strokeJoin = StrokeJoin.round
      ..color = scheme.surface.withValues(alpha: 0.55);
    canvas.save();
    canvas.clipPath(body);
    for (final region in regions) {
      final v = mpc[region.muscle] ?? 1.0;
      final paint = Paint()
        ..style = PaintingStyle.fill
        ..color = _mpcColor(v).withValues(alpha: 0.92);
      for (final shape in region.shapes) {
        canvas.drawPath(shape, paint);
        canvas.drawPath(shape, muscleEdge);
      }
    }
    canvas.restore();

    // Outline on top so muscle edges tuck cleanly under the silhouette.
    canvas.drawPath(body, outline);

    // A couple of definition lines (centre seam) for a more anatomical read.
    final seam = Paint()
      ..color = scheme.outline.withValues(alpha: 0.28)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2
      ..strokeJoin = StrokeJoin.round;
    canvas.save();
    canvas.clipPath(body);
    final cx = 0.5 * size.width;
    canvas.drawLine(Offset(cx, 0.170 * size.height),
        Offset(cx, 0.430 * size.height), seam); // torso centre
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _BodyPainter old) =>
      old.mpc != mpc || old.regions != regions;
}

Color _mpcColor(double v) {
  if (v < 0.55) return Colors.red.shade400;
  if (v < 0.85) return Colors.orange.shade400;
  return Colors.green.shade400;
}

class _Legend extends StatelessWidget {
  const _Legend();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    Widget chip(Color c, String label) => Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 14,
              height: 14,
              decoration: BoxDecoration(
                color: c.withValues(alpha: 0.92),
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            const SizedBox(width: 6),
            Text(label, style: theme.textTheme.bodySmall),
          ],
        );
    return Wrap(
      alignment: WrapAlignment.center,
      spacing: 16,
      runSpacing: 8,
      children: [
        chip(Colors.green.shade400, 'Fresh (≥0.85)'),
        chip(Colors.orange.shade400, 'Working (0.55–0.85)'),
        chip(Colors.red.shade400, 'Fatigued (<0.55)'),
      ],
    );
  }
}
