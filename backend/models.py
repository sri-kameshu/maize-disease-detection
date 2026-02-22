from sqlalchemy import Column, Integer, String, Float
from database import Base

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    disease = Column(String)
    severity_percent = Column(Float)
    severity_level = Column(String)
    timestamp = Column(String)