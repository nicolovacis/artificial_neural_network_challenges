import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Model:
    def __init__(self):
        self.model = tf.keras.models.load_model("weights.keras")

    def predict(self, X):
        """
        Predict labels using test-time augmentation (TTA).

        X: numpy array of shape (n_samples, 96, 96, 3).
        y_true: numpy array of true labels.

        The function returns predicted labels after TTA.
        """

        # Perform Test-Time Augmentation
        def augment_images(X):
            """
            Apply random augmentations to the input images.
            """
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
            )
            return np.stack([datagen.random_transform(image) for image in X])

        # Apply TTA (augment N times and average the predictions)
        n_augmentations = 5
        all_preds = []

        for _ in range(n_augmentations):
            augmented_X = augment_images(X)
            preds = self.model.predict(augmented_X)
            all_preds.append(preds)

        # Average predictions across all augmentations
        averaged_preds = np.mean(all_preds, axis=0)

        # Convert averaged predictions to class labels
        predicted_labels = np.argmax(averaged_preds, axis=1)

        return predicted_labels
