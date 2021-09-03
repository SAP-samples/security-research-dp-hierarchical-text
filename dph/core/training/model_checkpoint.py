import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
from transformers import TFPreTrainedModel

from dph.utils.saver import Saver


class ModelCheckpoint(Callback):

    def __init__(self, freq, saver: Saver):
        super().__init__()
        self.freq = freq
        self.saver = saver
        self.saved_models = 0

    def on_epoch_begin(self, epoch, logs=None):
        # Save the model on epoch begin so that when training ends, the model is not saved twice
        if self.freq > 0 and epoch > 0 and epoch % self.freq == 0:
            self.save_model(identifier=epoch)

    def on_train_end(self, logs=None):
        self.save_model(identifier='end')

    def save_model(self, identifier):
        save_path = self.saver.get_subdirectory('target_model/' + str(identifier))
        if isinstance(self.model, TFPreTrainedModel):
            self.model.save_pretrained(save_path)
        else:
            assert isinstance(self.model, tf.keras.models.Model)
            self.model.save(save_path)

        self.saved_models += 1
