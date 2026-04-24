"""Sanity check inference — czy strength anchors faktycznie personalizują predykcje."""

from datetime import datetime, timedelta

from inference import (
    load_model,
    predict_mpc,
    predict_rir,
    project_exercise_1rm,
)
from strength_priors import update_strength_anchors


def sep(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


model = load_model("deepgain_model_best.pt")
print(f"Model loaded. strength_feature_dim = {model.strength_feature_dim}")


# ─── TEST 1 ────────────────────────────────────────────────────────────────
sep("TEST 1: Ten sam set, dwie różne osoby")
print("Oczekiwanie: silny ma więcej RIR niż słaby przy tym samym ciężarze.\n")

strong = {"bench_press": 140, "squat": 200, "deadlift": 240}
weak   = {"bench_press":  60, "squat":  80, "deadlift": 100}
fresh  = {m: 1.0 for m in ["chest", "triceps", "anterior_delts", "quads", "glutes", "hamstrings",
                           "lats", "biceps", "rhomboids", "lateral_delts", "rear_delts",
                           "erectors", "adductors", "calves", "abs"]}

for ex, w, r in [("bench_press", 80, 5), ("squat", 100, 5), ("deadlift", 150, 5)]:
    rir_strong = predict_rir(model, fresh, ex, w, r, strength_anchors=strong)
    rir_weak   = predict_rir(model, fresh, ex, w, r, strength_anchors=weak)
    print(f"  {ex:15s} {w:3.0f}kg x{r}: silny RIR={rir_strong:.2f} | słaby RIR={rir_weak:.2f} | Δ={rir_strong-rir_weak:+.2f}")


# ─── TEST 2 ────────────────────────────────────────────────────────────────
sep("TEST 2: Ten sam user, rosnący ciężar")
print("Oczekiwanie: im więcej kg, tym mniejsze RIR.\n")

for w in [40, 60, 80, 100, 120]:
    rir = predict_rir(model, fresh, "bench_press", w, 5, strength_anchors=strong)
    print(f"  bench {w:3.0f}kg x5 (1RM=140): RIR={rir:.2f}")


# ─── TEST 3 ────────────────────────────────────────────────────────────────
sep("TEST 3: Efekt zmęczenia — MPC klatki obniżona")
print("Oczekiwanie: zmęczona klatka → niższe RIR niż świeża.\n")

tired_chest = dict(fresh)
tired_chest["chest"] = 0.5
tired_chest["triceps"] = 0.6
tired_chest["anterior_delts"] = 0.5

rir_fresh = predict_rir(model, fresh, "bench_press", 80, 5, strength_anchors=strong)
rir_tired = predict_rir(model, tired_chest, "bench_press", 80, 5, strength_anchors=strong)
print(f"  bench 80kg x5 silny, świeży:       RIR={rir_fresh:.2f}")
print(f"  bench 80kg x5 silny, zmęczona klatka: RIR={rir_tired:.2f}")
print(f"  Δ = {rir_fresh - rir_tired:+.2f} (powinno być dodatnie)")


# ─── TEST 4 ────────────────────────────────────────────────────────────────
sep("TEST 4: Projekcja 1RM z anchors")
print("Oczekiwanie: projected 1RM zgodne z ratio_mean z readme.\n")

for ex in ["incline_bench", "ohp", "dips", "leg_press", "rdl", "pull_up", "plank"]:
    p_strong = project_exercise_1rm(ex, strength_anchors=strong)
    p_weak   = project_exercise_1rm(ex, strength_anchors=weak)
    p_strong = f"{p_strong:.1f}kg" if p_strong else "—"
    p_weak   = f"{p_weak:.1f}kg" if p_weak else "—"
    print(f"  {ex:20s}: silny={p_strong:>10s} | słaby={p_weak:>10s}")


# ─── TEST 5 ────────────────────────────────────────────────────────────────
sep("TEST 5: predict_mpc z realną historią + anchors")
print("Oczekiwanie: MPC klatki spada po bench, recovery widoczny po czasie.\n")

now = datetime(2026, 4, 24, 10, 0)
history = [
    {"exercise": "bench_press", "weight_kg": 100, "reps": 5, "rir": 2,
     "timestamp": (now - timedelta(minutes=30)).isoformat(),
     "config_1rm_bench_press": 140, "config_1rm_squat": 200, "config_1rm_deadlift": 240},
    {"exercise": "bench_press", "weight_kg": 100, "reps": 5, "rir": 1,
     "timestamp": (now - timedelta(minutes=25)).isoformat()},
    {"exercise": "bench_press", "weight_kg": 100, "reps": 4, "rir": 0,
     "timestamp": (now - timedelta(minutes=20)).isoformat()},
]

mpc_now = predict_mpc(model, history, now.isoformat())
print("  MPC tuż po 3 setach bench:")
for m in ["chest", "triceps", "anterior_delts", "quads", "biceps"]:
    print(f"    {m:18s} = {mpc_now[m]:.3f}")

mpc_24h = predict_mpc(model, history, (now + timedelta(hours=24)).isoformat())
print("\n  MPC po 24h recovery:")
for m in ["chest", "triceps", "anterior_delts", "quads", "biceps"]:
    print(f"    {m:18s} = {mpc_24h[m]:.3f}")


# ─── TEST 6 ────────────────────────────────────────────────────────────────
sep("TEST 6: Dynamic anchor update — user się wzmocnił")
print("Oczekiwanie: update_strength_anchors podnosi anchors po dobrej sesji.\n")

initial = {"bench_press": 100, "squat": 150, "deadlift": 180}
heavy_session = [
    {"exercise": "bench_press", "weight_kg": 95, "reps": 3, "rir": 1,
     "timestamp": datetime(2026, 4, 23, 18, 0)},
    {"exercise": "bench_press", "weight_kg": 92, "reps": 4, "rir": 0,
     "timestamp": datetime(2026, 4, 23, 18, 10)},
    {"exercise": "squat",       "weight_kg": 140, "reps": 3, "rir": 1,
     "timestamp": datetime(2026, 4, 23, 18, 25)},
    {"exercise": "deadlift",    "weight_kg": 170, "reps": 5, "rir": 1,
     "timestamp": datetime(2026, 4, 23, 18, 40)},
]

updated = update_strength_anchors(initial, heavy_session)
print("  Anchor          | przed | po    | Δ")
for i, name in enumerate(["bench_press", "squat", "deadlift"]):
    print(f"  {name:15s} | {initial[name]:>5.1f} | {updated[i]:>5.1f} | {updated[i] - initial[name]:+.2f}")


# ─── TEST 7 ────────────────────────────────────────────────────────────────
sep("TEST 7: Cold start (brak anchors) vs explicit anchors")
print("Oczekiwanie: bez anchors → domyślne mediany populacyjne, z explicit → personalizacja.\n")

rir_default = predict_rir(model, fresh, "bench_press", 80, 5)
rir_strong  = predict_rir(model, fresh, "bench_press", 80, 5, strength_anchors=strong)
rir_weak    = predict_rir(model, fresh, "bench_press", 80, 5, strength_anchors=weak)
print(f"  bench 80kg x5, defaults:        RIR={rir_default:.2f}")
print(f"  bench 80kg x5, explicit silny:  RIR={rir_strong:.2f}")
print(f"  bench 80kg x5, explicit słaby:  RIR={rir_weak:.2f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
