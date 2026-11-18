# Botting2026-Aimlabs

# AimLab Color Bot

Projekt polega na implementacji bota do gry **Aim Lab**, zorientowanego na minimalizację opóźnień. Przechwytywanie wycinka ekranu (ROI) odbywa się natywnie w C++ z użyciem DirectX 11, natomiast detekcja celów i sterowanie ruchem myszy działają w Pythonie.

## Architektura systemu

1. **Percepcja (C++ / DXGI Desktop Duplication):** Przechwytywanie klatek bezpośrednio z GPU i wycinanie obszaru zainteresowania (ROI) jeszcze przed kopiowaniem do RAM.
2. **Most danych (pybind11):** Udostępnianie bufora klatki w trybie Zero-Copy (ten sam proces, brak narzutu serializacji).
3. **Detekcja i logika (Python / OpenCV / NumPy):** Konwersja BGRA → HSV, progowanie koloru celów, wyznaczanie środków ciężkości i wybór najbliższego celu względem środka ekranu.
4. **Akcja (Python / WinAPI):** Relatywny ruch myszy.
