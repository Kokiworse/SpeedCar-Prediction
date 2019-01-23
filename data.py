import cv2
import numpy as np

PATH = "C:\\Users\\koki\\Desktop\\python\\comma\\speedchallenge\\data\\test.mp4"

cap = cv2.VideoCapture(PATH)
dataset = []

if __name__ == "__main__":
    while 1:
        ret, frame = cap.read()
        if ret is False:
            np.savez("testGray.npz", np.array(dataset))
            exit(0)
        # cv2.imshow('frame',frame)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (150, 150))
        dataset.append(gray_image)
        print(gray_image.shape)
        print(len(dataset))

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    cap.release()
    cv2.destroyAllWindows()
