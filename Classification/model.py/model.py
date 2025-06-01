import numpy as np
import tensorflow as tf

class Model:
  def __init__(self):
        self.model = tf.keras.models.load_model('weights.keras')

  def predict(self, X):
        """
        Predict the labels corresponding to the input X. Note that X is a numpy
        array of shape (n_samples, 96, 96, 3) and the output should be a numpy
        array of shape (n_samples,). Therefore, outputs must no be one-hot
        encoded.

        The following is an example of a prediction from the pre-trained model
        loaded in the __init__ method.
        """
        # Perform prediction
        preds = self.model.predict(X)

        # Convert predictions to class labels by taking the argmax of each prediction
        predicted_labels = np.argmax(preds, axis=1)

        return predicted_labels