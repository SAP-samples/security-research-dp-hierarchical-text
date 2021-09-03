import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from dph.mia.attacker.attack_model import AttackModel
from dph.mia.attack_parameters import AttackParameters
from dph.mia.features.features import Features
from dph.utils.saver import Saver


class Attacker:
    def __init__(self, features: Features, parameters: AttackParameters, saver: Saver):
        self.p = parameters
        self.saver = saver

        self.train_dataset, self.val_dataset, self.test_dataset = features.prepare_datasets()
        self.dummy_features = features.dummy_features

    def train(self):
        # mirrored_strategy = tf.distribute.MirroredStrategy()
        # with mirrored_strategy.scope():
        model = AttackModel(self.dummy_features, self.p)

        train_dataset = self.train_dataset

        class TensorBoardWithGradients(tf.keras.callbacks.TensorBoard):
            def _log_gradients(self, epoch):
                writer = self._get_writer(self._train_run_name)

                with writer.as_default(), tf.GradientTape() as g:
                    # here we use train data to calculate the gradients
                    attack_features, member = list(train_dataset.batch(100).take(1))[0]

                    _y_pred = self.model(attack_features)  # forward-propagation
                    loss = self.model.compiled_loss(y_true=member, y_pred=_y_pred)  # calculate loss
                    gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

                    # In eager mode, grad does not have name, so we get names from model.trainable_weights
                    for weight, grad in zip(self.model.trainable_weights, gradients):
                        tf.summary.histogram(weight.name.replace(':', '_') + '_grads', data=grad, step=epoch)

                writer.flush()

            def on_epoch_end(self, epoch, logs=None):
                # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
                # but we do need to run the original on_epoch_end, so here we use the super function.
                super(TensorBoardWithGradients, self).on_epoch_end(epoch, logs=logs)

                if self.histogram_freq and epoch % self.histogram_freq == 0:
                    self._log_gradients(epoch)

        callbacks = []
        if self.p.patience > 0:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.p.patience,
                                                              restore_best_weights=True))
        if self.p.log_gradients:
            callbacks.append(TensorBoardWithGradients(log_dir=self.saver.log_dir, histogram_freq=1))
        else:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=self.saver.log_dir, histogram_freq=1))
        callbacks.append(tf.keras.callbacks.CSVLogger(filename=os.path.join(self.saver.log_dir, "log_attack.csv")))

        # Train and evaluate using tf.keras.Model.fit()
        start = time.time()

        history = model.fit(
            self.train_dataset.batch(self.p.batch_size_train).prefetch(tf.data.experimental.AUTOTUNE),
            validation_data=self.val_dataset.batch(self.p.batch_size_val).prefetch(
                tf.data.experimental.AUTOTUNE),
            epochs=self.p.num_epochs,
            callbacks=callbacks
        )

        # model.save(self.saver.get_subdirectory('attack_model'))

        end = time.time()
        print("\n")

        if len(history.history) > 0:
            training = {"time": int(end - start),
                        "n_epochs": int(len(history.history['loss'])),
                        "best_epoch": int(np.argmin(history.history['val_loss'])) + 1}

            # Save training metrics
            self.saver.save_dict(training, "attacker_training")

        return model, history

    def evaluate(self, attack_model: tf.keras.Model):
        val_dataset_batched = self.test_dataset.batch(self.p.batch_size_val).prefetch(tf.data.experimental.AUTOTUNE)

        y_pred = attack_model.predict(val_dataset_batched)

        y_true = np.array(list(val_dataset_batched.unbatch().map(lambda attack_features, member: member)))
        self.saver.save_dict({'y_pred': y_pred.tolist(),
                              'y_true': y_true.tolist()}, "y_pred_true")

        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        self.saver.save_dict({'auc': auc,
                              'fpr': fpr.tolist(),
                              'tpr': tpr.tolist(),
                              'thresholds': thresholds.tolist()}, "roc_curve")

        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
        average_precision = metrics.average_precision_score(y_true, y_pred)
        self.saver.save_dict({'precision': precision.tolist(),
                              'recall': recall.tolist(),
                              'thresholds': thresholds.tolist()}, "precision_recall_curve")

        matrix = metrics.confusion_matrix(y_true, [1 if prediction >= 0.5 else 0 for prediction in y_pred])
        cm = pd.DataFrame(matrix)
        cm.to_csv(self.saver.log_dir + '/confusion_matrix.csv')

        tn, fp, fn, tp = matrix.ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        advantage = tpr - fpr

        self.saver.save_dict({'accuracy': float(accuracy),
                              'advantage': float(advantage),
                              'auc': float(auc),
                              'average_precision': float(average_precision)
                              },
                             "combined_metrics")

    def analyse(self, attack_model: tf.keras.Model):
        try:
            val_dataset_batched = self.test_dataset.batch(self.p.batch_size_val).prefetch(tf.data.experimental.AUTOTUNE)

            y_pred = attack_model.predict(val_dataset_batched)
            y_pred = [1 if prediction >= 0.5 else 0 for prediction in y_pred]

            predicted_lvl0 = ""
            predicted_lvl1 = ""
            predicted_lvl2 = ""

            correctly_predicted_lvl0 = ""
            correctly_predicted_lvl1 = ""
            correctly_predicted_lvl2 = ""

            falsely_predicted_lvl0 = ""
            falsely_predicted_lvl1 = ""
            falsely_predicted_lvl2 = ""

            for index, instance in enumerate(val_dataset_batched.unbatch()):
                attack_features, member = instance
                predicted_lvl0 += str(int(tf.argmax(attack_features['label0']))) + "\n"
                predicted_lvl1 += str(int(tf.argmax(attack_features['label1']))) + "\n"
                predicted_lvl2 += str(int(tf.argmax(attack_features['label2']))) + "\n"
                if y_pred[index] == member:
                    correctly_predicted_lvl0 += str(int(tf.argmax(attack_features['label0']))) + "\n"
                    correctly_predicted_lvl1 += str(int(tf.argmax(attack_features['label1']))) + "\n"
                    correctly_predicted_lvl2 += str(int(tf.argmax(attack_features['label2']))) + "\n"
                else:
                    falsely_predicted_lvl0 += str(int(tf.argmax(attack_features['label0']))) + "\n"
                    falsely_predicted_lvl1 += str(int(tf.argmax(attack_features['label1']))) + "\n"
                    falsely_predicted_lvl2 += str(int(tf.argmax(attack_features['label2']))) + "\n"

            self.saver.save_string(predicted_lvl0, "predicted_lvl0")
            self.saver.save_string(predicted_lvl1, "predicted_lvl1")
            self.saver.save_string(predicted_lvl2, "predicted_lvl2")

            self.saver.save_string(correctly_predicted_lvl0, "correctly_predicted_lvl0")
            self.saver.save_string(correctly_predicted_lvl1, "correctly_predicted_lvl1")
            self.saver.save_string(correctly_predicted_lvl2, "correctly_predicted_lvl2")

            self.saver.save_string(falsely_predicted_lvl0, "falsely_predicted_lvl0")
            self.saver.save_string(falsely_predicted_lvl1, "falsely_predicted_lvl1")
            self.saver.save_string(falsely_predicted_lvl2, "falsely_predicted_lvl2")
        except Exception:
            pass
