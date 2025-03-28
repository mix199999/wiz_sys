import cv2
import numpy as np
import os

# Konfiguracja parametrów
MIN_CONTOUR_AREA = 3000   # Minimalna powierzchnia konturu 
KERNEL_SIZE = (5, 5)      # Rozmiar kernela morfologicznego
HISTORY = 500            # Liczba klatek do zapamiętania przez odejmowanie tła
VAR_THRESHOLD = 30       # Czułość na zmiany w tle 
MOG_SHADOWS = False       # Włącz/wyłącz detekcję cieni w MOG2
TRACKED_LIFETIME = 1.5     # Czas życia obiektów w sekundach 
# Pozycja linii liczenia
MIDDLE_AREA_TOP_RATIO = 0.8
MIDDLE_AREA_LEFT_RATIO = 0.3
MIDDLE_AREA_RIGHT_RATIO = 0.68

# Ścieżki do plików
INPUT_VIDEO_PATH = "resources/videos/zd4.mp4"
OUTPUT_VIDEO_PATH = "output/videos/cars_detected.mp4"

# Tworzenie katalogów wyjściowych
os.makedirs("output/videos", exist_ok=True)
os.makedirs("output/images", exist_ok=True)

# Inicjalizacja wideo
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Nie udało się otworzyć wideo.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Obliczenie pozycji linii liczenia
line_y = int(frame_height * MIDDLE_AREA_TOP_RATIO)
line_left = int(frame_width * MIDDLE_AREA_LEFT_RATIO)
line_right = int(frame_width * MIDDLE_AREA_RIGHT_RATIO)

# Inicjalizacja zapisu wideo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height), isColor=True)

# Inicjalizacja algorytmu odejmowania tła
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=HISTORY, varThreshold=VAR_THRESHOLD, detectShadows=MOG_SHADOWS)

# Zmienna do liczenia pojazdów oraz lista śledzonych obiektów
car_count = 0
tracked_objects = {}
frame_index = 0

# Ustalanie klatek do zapisania (początek, środek, koniec)
save_frames = [0, frame_count // 2, frame_count - 1]
saved_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1

    # Zastosowanie algorytmu odejmowania tła
    fgmask = bg_subtractor.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 150, 255, cv2.THRESH_BINARY)  # Zmniejszony próg

    # Operacje morfologiczne
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

    # Znalezienie konturów
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            
            # Rysowanie prostokąta i środka obiektu
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Detekcja przecięcia linii liczenia
            if (y < line_y <= y + h) and (line_left <= cx <= line_right):
                if not any(abs(cx - obj[0]) < 40 and abs(cy - obj[1]) < 40 for obj in tracked_objects):
                    tracked_objects[(cx, cy)] = frame_index  # Rejestracja pojazdu
                    car_count += 1

    # Usuwanie starych wpisów z listy śledzonych obiektów
    tracked_objects = {k: v for k, v in tracked_objects.items() if frame_index - v < fps * TRACKED_LIFETIME}

    # Rysowanie linii liczenia
    cv2.line(frame, (line_left, line_y), (line_right, line_y), (255, 0, 0), 2)
    cv2.putText(frame, f"Count: {car_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Zapisywanie klatek w ustalonych momentach
    if frame_index in save_frames and saved_frames < 3:
        cv2.imwrite(f"output/images/frame_{saved_frames+1}.png", frame)
        cv2.imwrite(f"output/images/mask_{saved_frames+1}.png", fgmask)
        saved_frames += 1

    # Zapis przetworzonej klatki do wideo
    out.write(frame)

# Zwolnienie zasobów
cap.release()
out.release()

print(f"Przetwarzanie zakończone. Wykryto łącznie {car_count} pojazdów.")
