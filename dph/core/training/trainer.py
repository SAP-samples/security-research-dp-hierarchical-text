import os
import numpy as np
import time

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent

from dph.core.modeling.model_factory import ModelFactory
from dph.core.parameters.parameters import Parameters, DPParameters
from dph.core.training.model_checkpoint import ModelCheckpoint
from dph.core.training.train_loop import TrainLoop
from dph.utils.saver import Saver
import tensorflow as tf
import pandas as pd


class Trainer:

    @staticmethod
    def train(train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              p: Parameters,
              saver: Saver):
        """
        Training method.
        """
        strategy, model, optimizer, loss_dict, metrics = ModelFactory().build_model(p)

        if hasattr(p, "steps_per_epoch") and hasattr(p, "val_steps") and hasattr(p, "train_size"):
            train_dataset = train_dataset \
                .shuffle(p.train_size, seed=p.seed).batch(int(p.batch_size_train), drop_remainder=True)
            val_dataset = val_dataset.batch(p.batch_size_val)

            callbacks = []
            if p.patience > 0:
                callbacks.append(tf.keras.callbacks.EarlyStopping(patience=p.patience, restore_best_weights=True))
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=saver.log_dir, histogram_freq=1))
            callbacks.append(tf.keras.callbacks.CSVLogger(filename=os.path.join(saver.log_dir, "log.csv")))
            if hasattr(p, "save_model_freq"):
                callbacks.append(ModelCheckpoint(freq=p.save_model_freq, saver=saver))

            # Save parameters
            tf.print(p.get_parameters())
            saver.save_dict(p.get_parameters(), "parameters")

            # Train and evaluate using tf.keras.Model.fit()
            start = time.time()

            grad_file_path = (saver.log_dir + "/gradient_norms.txt") if hasattr(p, "write_grads") else None
            train_loop = TrainLoop(strategy,
                                   model, optimizer, loss_dict,
                                   metrics,
                                   train_dataset, p.num_epochs, p.steps_per_epoch, p.batch_size_train,
                                   val_dataset,
                                   p.val_steps,
                                   callbacks=callbacks,
                                   grad_file_path=grad_file_path)

            train_loop.start()

            history = train_loop.history

            end = time.time()
            print("\n")

            if len(history.history) > 0:
                best_epoch = int(np.argmin(history.history['val_loss'])) + 1
                training = {"time": int(end - start),
                            "n_epochs": int(len(history.history['loss'])),
                            "best_epoch": best_epoch}

                if grad_file_path is not None:
                    gradients = pd.read_csv(grad_file_path)
                    gradients_median = float(gradients.median())
                    training['gradients_median'] = gradients_median

                if isinstance(p, DPParameters) and p.sigma > 0:
                    epsilon, delta = get_dp_guarantee(p.train_size, p.batch_size_train, best_epoch,
                                                      p.sigma / p.clipnorm)
                    training['epsilon'] = epsilon
                    training['delta'] = delta

                # Save training metrics
                saver.save_dict(training, "training")

            return model, history


def get_dp_guarantee(training_size, batch_size, n_epochs, noise_multiplier):
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))

    sampling_probability = batch_size / training_size
    steps = n_epochs * training_size // batch_size

    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=noise_multiplier,
                      steps=steps,
                      orders=orders)

    delta = 1 / training_size
    epsilon = get_privacy_spent(orders, rdp, target_delta=delta)[0]

    return epsilon, delta
