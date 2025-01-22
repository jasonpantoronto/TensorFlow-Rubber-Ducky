import cv2
import numpy as np
import tensorflow as tf
from model import load_model  # Assuming load_model is a function in model.py
from utils.data_preprocessing import preprocess_frame  # Assuming preprocess_frame is defined in data_preprocessing.py

class RubberDuckPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, frame):
        processed_frame = preprocess_frame(frame)
        predictions = self.model.predict(np.expand_dims(processed_frame, axis=0))
        return predictions

def main():
    model_path = 'path/to/your/model.h5'  # Update with your model path
    predictor = RubberDuckPredictor(model_path)

    cap = cv2.VideoCapture(0)  # Start video capture from the camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = predictor.predict(frame)
        # Assuming a threshold for detection
        if predictions[0][1] > 0.5:  # Adjust index based on your model's output
            cv2.putText(frame, "Rubber Duck Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Rubber Duck", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Rubber Duck Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()