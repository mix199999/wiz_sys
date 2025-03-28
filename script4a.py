import cv2
import numpy as np
import os

# Parametry konfiguracyjne
FLOW_THRESHOLD = 2.0  # Próg dla wartości przepływu optycznego

# Tworzenie folderów na wyniki
os.makedirs("output/videos", exist_ok=True)
os.makedirs("output/images", exist_ok=True)

# Ścieżki do plików
input_video_path = "resources/videos/zd4.mp4"
output_diff_video_path = "output/videos/diff_output.mp4"
output_avg_image_path = "output/images/avg_background.png"

# Wczytanie wideo
cap = cv2.VideoCapture(input_video_path)

# Pobranie właściwości wideo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Inicjalizacja zapisu wideo różnicowego
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_diff = cv2.VideoWriter(output_diff_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

# Inicjalizacja zmiennych do uśredniania
avg_frame = np.zeros((frame_height, frame_width, 3), dtype=np.float32)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if ret else None

frame_index = 0

while ret:
    # Dodawanie klatki do uśredniania
    avg_frame += prev_frame.astype(np.float32)
    
    # Wczytanie kolejnej klatki
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konwersja na skalę szarości
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Obliczenie różnicy między kolejnymi klatkami
    diff_frame = cv2.absdiff(prev_gray, gray)
    
    # Zapis klatki różnicowej
    out_diff.write(diff_frame)
    
    # Aktualizacja poprzedniej klatki
    prev_gray = gray.copy()
    prev_frame = frame.copy()
    
    frame_index += 1

# Uśrednianie tła
avg_frame /= frame_index
avg_frame = np.uint8(avg_frame)  # Konwersja do uint8

# Zapis uśrednionego tła
cv2.imwrite(output_avg_image_path, avg_frame)

# Zwolnienie zasobów
cap.release()
out_diff.release()

print(f"Zakończono przetwarzanie. Wygenerowano klatki różnicowe oraz obraz uśredniony tła.")
