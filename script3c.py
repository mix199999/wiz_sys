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

# Pobranie pierwszej klatki i konwersja do skali szarości
ret, prev_frame = cap.read()
if not ret or prev_frame is None:
    print("Nie udało się wczytać pierwszej klatki wideo.")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Umożliwienie ręcznego zaznaczenia obiektu do śledzenia (np. mężczyzny w seledynowej koszulce)
x, y, w, h = cv2.selectROI("Select Object", prev_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Object")

# Rysowanie prostokąta wokół zaznaczonego obiektu
cv2.rectangle(prev_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Lista przechowująca trajektorię ruchu
trajectory_points = [(x + w // 2, y + h // 2)]

# Przetwarzanie każdej klatki
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Koniec wideo lub błąd odczytu.")
        break

    # Konwersja bieżącej klatki do skali szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obliczanie optycznego przepływu za pomocą algorytmu Farnebacka
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Śledzenie środka prostokąta na podstawie przepływu optycznego
    center_x, center_y = trajectory_points[-1]
    dx, dy = flow[center_y, center_x]
    new_center = (int(center_x + dx), int(center_y + dy))
    trajectory_points.append(new_center)

    # Rysowanie trajektorii na bieżącej klatce
    for i in range(1, len(trajectory_points)):
        cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

    # Aktualizacja poprzedniej klatki
    prev_gray = gray

    # Wyświetlanie wyniku
    cv2.imshow("Optical Flow Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rysowanie trajektorii na pierwszej klatce i zapis
for i in range(1, len(trajectory_points)):
    cv2.line(prev_frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 255), 2)

cv2.imwrite(trajectory_output_path, prev_frame)
print(f"Trajektoria zapisana w pliku {trajectory_output_path}.")

cap.release()
cv2.destroyAllWindows()