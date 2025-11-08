from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_doctor = db.Column(db.Boolean, default=False)
    
    # Relationship with patient history
    histories = db.relationship('PatientHistory', backref='user', lazy=True)

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_age = db.Column(db.Integer)
    patient_gender = db.Column(db.String(10))
    image_filename = db.Column(db.String(200))
    cnn_prediction = db.Column(db.String(100))
    cnn_confidence = db.Column(db.Float)
    cancer_stage = db.Column(db.String(100))
    ml_predictions = db.Column(db.Text)
    ensemble_predictions = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'patient_name': self.patient_name,
            'patient_age': self.patient_age,
            'patient_gender': self.patient_gender,
            'cnn_prediction': self.cnn_prediction,
            'cnn_confidence': self.cnn_confidence,
            'cancer_stage': self.cancer_stage,
            'ml_predictions': json.loads(self.ml_predictions) if self.ml_predictions else {},
            'ensemble_predictions': json.loads(self.ensemble_predictions) if self.ensemble_predictions else {},
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'notes': self.notes
        }