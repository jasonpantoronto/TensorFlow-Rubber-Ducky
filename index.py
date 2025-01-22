class ImageData:
    def __init__(self, image_path: str, label: str):
        self.image_path = image_path
        self.label = label

class ModelPrediction:
    def __init__(self, label: str, confidence: float):
        self.label = label
        self.confidence = confidence

class DatabaseEntry:
    def __init__(self, id: int, image_path: str, prediction: ModelPrediction):
        self.id = id
        self.image_path = image_path
        self.prediction = prediction