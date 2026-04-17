# Notatka dla Michała (po aktualizacji datasetu i generatora)

Czesc,
ponizej masz konkret, co teraz jest juz zrobione po stronie danych/generatora i co warto poprawic po stronie modelu/wykresow przed kolejnym retreningiem.

## 1) Co jest teraz spójne i zamkniete

- Mamy jeden aktualny zestaw danych:
  - `training_data_michal_full.csv` (pelny)
  - `training_data_train.csv` (train)
  - `training_data_val.csv` (val/test)
- Split ma pelne pokrycie cwiczen: 34/34 cwiczen jest w train i 34/34 w val.
- Generator i train czytaja te same source-of-truth:
  - `exercise_muscle_order.yaml`
  - `exercise_muscle_weights_scaled.csv`
- Aktywny schemat to 15 miesni i 34 cwiczenia (A/B sa tu spojne).

Uwaga techniczna: `exercise_muscle_weights_scaled.csv` ma jeszcze kolumny historyczne (`upper_traps`, `brachialis`), ale train/generator ich nie uzywaja, bo ladują tylko 15 aktywnych miesni.

## 2) Transformacja nowych wag (co zostalo uzyte)

Wagi w `exercise_muscle_weights_scaled.csv` sa po transformacji:

1. Agregacja wybranych grup miesniowych z surowego CSV:
   - chest = klatka gora + klatka srodek/dol
   - erectors = prostownik ledzwiowy + multifidus
   - glutes = posladkowy wielki + posladkowy sredni
   - abs = prosty brzucha + skosny zew + skosny wew
2. Dla grup agregowanych zastosowano faktor zalezny od liczby skladowych.
3. Potem clamp do >= 0.
4. Global clipping do p99.
5. Na koncu normalizacja per cwiczenie do [0, 1].

W praktyce train i generator biora te wartosci bezposrednio z tego pliku.

## 3) Kolejnosc miesni vs nowe wagi

Tak: kolejnosc miesni w YAML jest dostosowana do nowych wag (ranking zostal poprawiony pod wartosci transformed).

To ma znaczenie glownie dla ordinal penalty i diagnostyki orderingu.

## 4) Co nie gralo na wykresach i co poprawic

### Heatmapy fatigue (najwazniejsze)

Problem nie jest jeden:

- Heatmapa jest liczona przy stalym RIR=2 i MPC=1.0.
- Zakres osi jest szeroki i staly (`20-200 kg`, `1-15 reps`) dla wszystkich cwiczen.
- To wrzuca duzo punktow poza realny rozklad danych dla konkretnego cwiczenia (off-manifold).

Efekt: mapa moze wygladac niefizjologicznie mimo poprawnego treningu.

### Co z tym zrobic (konkretny plan)

1. Ograniczyc zakres per cwiczenie, nie globalnie:
   - weight: np. p5-p95 z `training_data_train.csv` dla danego cwiczenia,
   - reps: np. p5-p95 dla danego cwiczenia.
2. Dodac maske in-distribution:
   - punkty poza gestym obszarem danych wyszarzyc/hatchowac,
   - nie interpretowac ich fizjologicznie.
3. Dodatkowo rozwazyc os `%e1RM` zamiast czystego kg dla czytelniejszej interpretacji miedzy userami.

To jest poprawka warstwy diagnostycznej (charting), niekoniecznie samego treningu modelu.

## 5) Co koniecznie zrobic przed retreningiem modelu

1. Dodac minimum-drop penalty (zeby ograniczyc muscle collapse na miesniach wtornych).
2. Zaktualizowac `inference.py` do 15 miesni / 34 cwiczen.
   - teraz inference jest jeszcze na starym schemacie (to jest niespojne z A/B).
3. Poprawic `eval_ordering.py`:
   - obecnie ma hardkodowane (legacy) wagi i przez to moze raportowac mylacy ordering,
   - musi korzystac z tych samych wag co train/generator, tj. z `exercise_muscle_weights_scaled.csv`,
   - najlepiej: dodac loader `_load_scaled_weights` analogiczny do `train.py` i budowac probe/order bezposrednio z transformed CSV.
4. Puscic ablacjeki:
   - z/bez ordinal penalty,
   - z/bez minimum-drop,
   - kilka embedding dims.
5. Sprawdzic po treningu:
   - per-exercise MAE,
   - ordering,
   - czy wtórne miesnie nie zapadaja sie do ~0.

## 6) Jak szybko sprawdzic zgodnosc wartosci liczbowych

Przyklad `spoto_press` (powinno wyjsc identycznie):

- z CSV:
  - chest = 1.0
  - anterior_delts = 0.503196
  - triceps = 0.693386

Sprawdz to jednym snippetem (max_abs_diff powinno byc 0.0):

```python
import pandas as pd
import generate_training_data as g

ex = "spoto_press"
weights = g._load_scaled_weights()
d_gen = weights[ex]
row = pd.read_csv("exercise_muscle_weights_scaled.csv").set_index("exercise_id").loc[ex]
d_csv = {m: float(row[m]) for m in g.ALL_MUSCLES if float(row[m]) > 0.0}
keys = sorted(set(d_gen) | set(d_csv))
mx = max(abs(d_gen.get(k, 0.0) - d_csv.get(k, 0.0)) for k in keys)
print(d_gen)
print(d_csv)
print("max_abs_diff:", mx)
```

## 7) Czy temat dotyczy osoby C?

- Niespojnosc `inference.py` (stary schemat miesni/cwiczen) to glownie temat C / deployment runtime.
- To nie psuje samego treningu i wykresow z `train.py` (one ida po A/B i nowym schemacie).
- Ale przed oddaniem dalej inference trzeba koniecznie zsynchronizowac.

---

Jesli chcesz, moge dopisac od razu checklistę "Definition of Done" na retrening (co musi przejsc, z jakimi progami), zeby nie wracac do tego w petli.
