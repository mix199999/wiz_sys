import cv2
import os

# Ścieżki do plików
input_video_path = "resources/videos/zd3.mp4"
output_video_path = "output/videos/zd3_edges.mp4"
frames_output_dir = "output/videos/frames"

# Tworzenie katalogu na klatki
os.makedirs(frames_output_dir, exist_ok=True)

# Wczytanie wideo
cap = cv2.VideoCapture(input_video_path)

# Pobranie parametrów wideo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definicja kodera i zapis filmu
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=False)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konwersja do skali szarości i detekcja krawędzi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Zapis klatek co 30 klatek (przykładowe klatki do sprawozdania)
    if frame_count % 30 == 0 and frame_count <= 90:
        cv2.imwrite(f"{frames_output_dir}/original_{frame_count}.jpg", frame)
        cv2.imwrite(f"{frames_output_dir}/edges_{frame_count}.jpg", edges)

    # Zapis wideo z krawędziami
    out.write(edges)
    
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print("Wideo z krawędziami zapisane oraz wybrane klatki zapisane w folderze 'videos/frames'.")
