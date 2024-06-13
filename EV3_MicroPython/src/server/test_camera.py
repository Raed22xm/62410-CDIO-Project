import cv2

def test_camera():
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    if not cap.isOpened():
        print("Camera not found")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Camera Test', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()
