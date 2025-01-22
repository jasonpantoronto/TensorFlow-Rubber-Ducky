from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    image_path = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    label = Column(String, nullable=False)

class Database:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_prediction(self, image_path, confidence, label):
        session = self.Session()
        prediction = Prediction(image_path=image_path, confidence=confidence, label=label)
        session.add(prediction)
        session.commit()
        session.close()

    def get_predictions(self):
        session = self.Session()
        predictions = session.query(Prediction).all()
        session.close()
        return predictions

    def clear_predictions(self):
        session = self.Session()
        session.query(Prediction).delete()
        session.commit()
        session.close()