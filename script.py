import cv2
import matplotlib.pyplot as plt


image_path = "resources/images/zd1.jpg"
image = cv2.imread(image_path)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Obraz po odczytniu
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykrywanie twarzy
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


for (x, y, w, h) in faces:
    center = (x + w // 2, y + h // 2)
    radius = w // 2
    cv2.circle(image_rgb, center, radius, (255, 0, 0), 3)  

# Wyświetlenie obrazu z zaznaczonymi twarzami
plt.imshow(image_rgb)
plt.axis("off")
plt.title(f"Wykryto twarzy: {len(faces)}")
plt.show()


# Obrót 
rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

faces_rotated = face_cascade.detectMultiScale(rotated_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)
for (x, y, w, h) in faces_rotated:
    center = (x + w // 2, y + h // 2)
    radius = w // 2
    cv2.circle(rotated_image_rgb, center, radius, (255, 0, 0), 3)

plt.imshow(rotated_image_rgb)
plt.axis("off")
plt.title(f"Po obrocie wykryto twarzy: {len(faces_rotated)}")
plt.show()

