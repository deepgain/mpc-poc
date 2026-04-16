# README dla Michala

Ten folder po czyszczeniu zawiera tylko CSV przekazywane dalej:
- Ratios - Arkusz1.csv
- exercise_muscle_weights_scaled.csv
- training_data_train.csv
- training_data_val.csv

## Jak powstaly nowe liczby w exercise_muscle_weights_scaled.csv
1. Bazujemy na Ratios - Arkusz1.csv oraz exercise_muscle_order.yaml.
2. Dla miesni agregowanych liczymy sume skladowych:
- chest = Klatka (Gora) + Klatka (Srodek/Dol)
- erectors = Prostownik Ledzwiowy + Multifidus
- glutes = Posladkowy Wielki + Posladkowy Sredni
- abs = Prosty Brzucha + Skosny zewnetrzny + Skosny wewnetrzny
3. Tylko dla miesni agregowanych stosujemy faktor:
- faktor = 0.5 + 0.1 * x
- x = liczba miesni skladowych
- chest/erectors/glutes: x=2 -> faktor=0.7
- abs: x=3 -> faktor=0.8
4. Miesnie nieagregowane sa przepisane 1:1 (bez dodatkowego mnoznika).
5. Potem: clamp do >=0, globalny clipping p99, normalizacja per cwiczenie do skali [0,1].

## Co zrobic przed przetrenowaniem modelu
1. Usunac z calego projektu miesnie nieuzywane: upper_traps oraz brachialis.
2. Po usunieciu tych 2 miesni przetrenowac model od zera (zmienia sie wymiar wejsc/embeddingow).
3. Sprawdzic zgodnosc list miesni i kolejnosci indeksow we wszystkich miejscach:
- train.py
- generate_training_data.py
- exercise_muscle_order.yaml
- exercise_muscle_weights_scaled.csv
4. Wygenerowac ponownie dataset train/val po zmianie schematu miesni.

## Dodatkowa notka
Poniewaz zmienila sie metoda skalowania/agregacji, stare checkpointy i porownania metryk nie sa 1:1 porownywalne z nowa wersja danych.
