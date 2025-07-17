import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
import joblib

class NeuralEnsembleBooster:
    def __init__(self):
        self.neural_models = []
        self.label_encoder = LabelEncoder()
        
    def create_deep_model(self, input_dim, num_classes):
        model = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_neural_ensemble(self, X, y):
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        num_classes = len(self.label_encoder.classes_)
        
        # Create 3 different neural architectures
        for i in range(3):
            model = self.create_deep_model(X.shape[1], num_classes)
            
            # Train with different strategies
            if i == 0:
                # Standard training
                model.fit(X, y_encoded, epochs=50, batch_size=32, verbose=0)
            elif i == 1:
                # With data augmentation (noise)
                X_aug = X + np.random.normal(0, 0.01, X.shape)
                model.fit(X_aug, y_encoded, epochs=50, batch_size=32, verbose=0)
            else:
                # Longer training with smaller learning rate
                model.optimizer.learning_rate = 0.0001
                model.fit(X, y_encoded, epochs=100, batch_size=16, verbose=0)
            
            self.neural_models.append(model)
        
        return self.neural_models
    
    def predict_ensemble(self, X):
        predictions = []
        for model in self.neural_models:
            pred = model.predict(X, verbose=0)
            predictions.append(pred)
        
        # Average ensemble predictions
        avg_pred = np.mean(predictions, axis=0)
        return self.label_encoder.inverse_transform(np.argmax(avg_pred, axis=1))

# Integration with existing model
def enhance_with_neural(X, y):
    booster = NeuralEnsembleBooster()
    booster.train_neural_ensemble(X, y)
    joblib.dump(booster, 'models/neural_booster.pkl')
    return booster
