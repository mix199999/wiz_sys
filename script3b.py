import cv2
import os

# Ścieżki do plików
input_video_path = "resources/videos/zd3.mp4"
trajectory_output_path = "output/videos/trajectory.jpg"

# Tworzenie katalogu wyjściowego
os.makedirs("output/videos", exist_ok=True)

# Wczytanie wideo
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Nie udało się otworzyć wideo: {input_video_path}")
    exit()

# Pobranie pierwszej klatki
ret, frame = cap.read()
if not ret or frame is None:
    print("Nie udało się wczytać pierwszej klatki wideo.")
    cap.release()
    exit()

# Ustaw prostokąt obejmujący obiekt do śledzenia
bbox = cv2.selectROI("Wybierz obiekt do śledzenia", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Inicjalizacja algorytmu śledzenia
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Lista przechowująca punkty trajektorii
trajectory_points = []
last_valid_frame = frame.copy()  # Przechowuje ostatnią poprawną klatkę

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Koniec wideo lub błąd odczytu.")
        break

    # Aktualizacja śledzenia
    success, bbox = tracker.update(frame)
    if success:
        # Wyznacz środek prostokąta
        x, y, w, h = [int(v) for v in bbox]
        center = (x + w // 2, y + h // 2)
        trajectory_points.append(center)

        # Rysuj prostokąt na obiekcie
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Rysuj trajektorię
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

        # Zapisz ostatnią poprawną klatkę
        last_valid_frame = frame.copy()

    # Opcjonalnie pokaż wynik na żywo
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sprawdzenie, czy mamy poprawną klatkę do zapisania
if last_valid_frame is not None and len(trajectory_points) > 0:
    # Rysowanie trajektorii na ostatniej poprawnej klatce
    for i in range(1, len(trajectory_points)):
        cv2.line(last_valid_frame, trajectory_points[i - 1], trajectory_points[i], (0, 255, 0), 2)

    # Zapis obrazu z trajektorią
    cv2.imwrite(trajectory_output_path, last_valid_frame)
    print(f"Trajektoria zapisana w pliku {trajectory_output_path}.")
else:
    print("Błąd: Brak poprawnej klatki lub punktów trajektorii do zapisania.")

cap.release()
cv2.destroyAllWindows()