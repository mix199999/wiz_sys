import cv2
import numpy as np
import os

# Ścieżki do plików
input_video_path = "resources/videos/zd3.mp4"
trajectory_output_path = "output/videos/trajectory_farneback.jpg"

# Tworzenie katalogu wyjściowego
os.makedirs("output/videos", exist_ok=True)

# Wczytanie wideo
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Nie udało się otworzyć wideo: {input_video_path}")
    exit()

# Definicja zakresu koloru w przestrzeni HSV (dla koloru R77 G183 B153)
lower_bound = np.array([72, 144, 117])  # Dolna granica HSV
upper_bound = np.array([110, 179, 187])  # Górna granica HSV

# Pobranie pierwszej klatki
ret, frame = cap.read()
if not ret or frame is None:
    print("Nie udało się wczytać pierwszej klatki wideo.")
    cap.release()
    exit()

# Konwersja pierwszej klatki do HSV i wykrycie obiektu na podstawie koloru
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

# Znajdowanie konturów w masce
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(contours) == 0:
    print("Nie znaleziono obiektu o podanym kolorze.")
    cap.release()
    exit()

# Wybór największego konturu (zakładamy, że to obiekt)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Rysowanie prostokąta wokół obiektu
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Lista przechowująca trajektorię ruchu
trajectory_points = [(x + w // 2, y + h // 2)]

# Konwersja pierwszej klatki do skali szarości
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Przetwarzanie każdej klatki
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Koniec wideo lub błąd odczytu.")
        break

    # Konwersja bieżącej klatki do HSV i wykrywanie obiektu
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Znajdowanie konturów w masce
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center = (x + w // 2, y + h // 2)
        trajectory_points.append(center)

    # Rysowanie trajektorii na bieżącej klatce
    for i in range(1, len(trajectory_points)):
        cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

    # Wyświetlanie wyniku
    cv2.imshow("Optical Flow Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rysowanie trajektorii na pierwszej klatce i zapis
for i in range(1, len(trajectory_points)):
    cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

cv2.imwrite(trajectory_output_path, frame)
print(f"Trajektoria zapisana w pliku {trajectory_output_path}.")

# Zapewnia, że okno z wideo nie zamknie się automatycznie
print("Naciśnij dowolny klawisz, aby zamknąć okno...")
while True:
    cv2.imshow("Optical Flow Tracking", frame)
    if cv2.waitKey(0) & 0xFF:
        break

cap.release()
cv2.destroyAllWindows()