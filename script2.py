import cv2
import numpy as np
import matplotlib.pyplot as plt

# PARAMETRY DO EDYCJI
MIN_NEIGHBORS = 9  # Im wyższa wartość, tym mniej fałszywych twarzy (zalecane 8-10)
SCALE_FACTOR = 1.25  # Jak szybko zmniejszamy obraz przy skanowaniu (zalecane 1.2-1.3)
CLUSTER_THRESHOLD = 30  # Jak blisko siebie wykryte twarze są łączone (zalecane 20-30)
MIN_FACE_SIZE = (50, 50)  # Minimalny rozmiar wykrywanej twarzy

# Wczytaj obraz
image_path = "resources/images/zd2.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Dodanie marginesu, aby twarze nie znikały podczas rotacji
border_size = 100
image_with_border = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

# Klasyfikator do wykrywania twarzy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Funkcja do obrotu obrazu i przeliczania pozycji wykrytych twarzy
def detect_faces_in_rotations(image, step=5):
    original_h, original_w = image.shape[:2]
    detected_faces = []

    for angle in range(0, 360, step):
        # Obracanie obrazu
        center = (original_w // 2, original_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (original_w, original_h))

        # Wykrywanie twarzy
        gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=SCALE_FACTOR, minNeighbors=MIN_NEIGHBORS, minSize=MIN_FACE_SIZE)

        # Transformacja pozycji twarzy na oryginalny obraz
        for (x, y, w, h) in faces:
            rotated_point = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)
            rotated_point = np.append(rotated_point, np.ones((1, 1)), axis=1)
            inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
            original_point = np.dot(inverse_matrix, rotated_point.T).T
            detected_faces.append(tuple(original_point[0]))

    # Usuwanie duplikatów twarzy
    final_faces = cluster_faces(detected_faces)
    return final_faces

# Funkcja do grupowania twarzy (eliminuje duplikaty)
def cluster_faces(faces):
    if not faces:
        return []

    clustered_faces = []
    faces = np.array(faces)

    while len(faces) > 0:
        base_face = faces[0]
        distances = np.linalg.norm(faces - base_face, axis=1)
        close_points = faces[distances < CLUSTER_THRESHOLD]

        # Średnia pozycja grupy twarzy
        mean_x = int(np.mean(close_points[:, 0]))
        mean_y = int(np.mean(close_points[:, 1]))
        clustered_faces.append((mean_x, mean_y))

        # Usuwanie już przetworzonych punktów
        faces = faces[distances >= CLUSTER_THRESHOLD]

    return clustered_faces

# Wykrywanie twarzy
detected_faces = detect_faces_in_rotations(image_with_border)

# Rysowanie wykrytych twarzy
for (x, y) in detected_faces:
    x, y = int(x - border_size), int(y - border_size)  # Przesunięcie do oryginalnych wymiarów
    cv2.drawMarker(image_rgb, (x, y), (255, 0, 0), cv2.MARKER_CROSS, thickness=2)

# Wyświetlenie obrazu
plt.imshow(image_rgb)
plt.axis("off")
plt.title(f"Wykryte twarze: {len(detected_faces)}")
plt.show()
