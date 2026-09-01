import cv2
import sys
import os


def detect_faces(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    image = cv2.imread(image_path)

    if image is None:
        print("❌ Could not read the image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    print("\n================================")
    print("   HH GOA 2026 - TASK 3")
    print("================================")
    print(f"Input image: {image_path}")
    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        print("❌ No face detected.")
    else:
        print("✅ Face detected successfully!")

        for i, (x, y, w, h) in enumerate(faces, start=1):
            print(f"   Face {i}: x={x}, y={y}, width={w}, height={h}")

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

        output_path = "detected_faces.jpg"
        cv2.imwrite(output_path, image)

        print(f"📁 Result saved as: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python app.py <image_path>")
        print("\nExample:")
        print("python app.py sample.jpg")
    else:
        detect_faces(sys.argv[1])
