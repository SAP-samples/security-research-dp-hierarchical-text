from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
from anytree import AnyNode
from scipy.special import softmax
from tqdm.auto import tqdm

from dph.core.evaluation.metrics import global_accuracy, global_measures, hierarchical_measures
from dph.core.parameters.parameters import Parameters
from dph.utils.saver import Saver


class Evaluator:

    @staticmethod
    def evaluate(saver: Saver,
                 split: str,
                 labels: list,
                 model: tf.keras.Model,
                 test_dataset: tf.data.Dataset,
                 hierarchy: AnyNode,
                 p: Parameters):
        """
        Evaluates a model using the given test dataset and saves the results with help of the given saver object.

        :param saver: Saver object to save evaluation results
        :param split: split name
        :param labels: Labels
        :param model:
        :param test_dataset:
        :param hierarchy: Hierarchy structure for evaluation
        :param p:
        """
        evaluation = {}
        confusion_matrices = pd.ExcelWriter(saver.log_dir + f'/confusion_matrices_{split}.xlsx')

        test_dataset = test_dataset.batch(p.batch_size_val)

        # Predict dataset
        level_predictions = model.predict(test_dataset.map(lambda inputs, _: inputs))[0]
        level_predictions = [(levels, predictions) for levels, predictions in level_predictions.items()]
        level_predictions = sorted(level_predictions, key=lambda levels: levels[0])  # sort by levels

        # Save Confusion Matrix
        pickle.dump(level_predictions, open(saver.log_dir + f'/level_predictions{split}.p', 'wb'))

        y_pred, y_pred_orig = Evaluator.calculate_y_pred(hierarchy, labels, level_predictions)

        # Calculate y_true and loss
        y_true = {}
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        losses = {}
        for level_num, level_prediction in enumerate(level_predictions):
            y_true_one_hot = list(test_dataset.unbatch().map(lambda _, l: l[level_num]))
            losses[level_num] = loss_object(y_true_one_hot, level_predictions[level_num][1])
            y_true[level_num] = [labels[level_num][int(np.argmax(one_hot))] if np.sum(one_hot) > 0 else 'nan'
                                 for one_hot in y_true_one_hot]

        try:
            # Per-Level metrics
            for level_num, level_prediction in enumerate(level_predictions):
                cm = pd.DataFrame(confusion_matrix(y_true[level_num], y_pred[level_num], labels[level_num]),
                                  index=labels[level_num],
                                  columns=labels[level_num])
                cm.to_excel(confusion_matrices, sheet_name="level_" + str(level_prediction[0]))

                # Calculate Accuracy to validate used predictions
                metrics = {
                    "loss": float(losses[level_num]),
                    "accuracy": float(accuracy_score(y_true[level_num], y_pred[level_num])),
                    "accuracy_orig": float(accuracy_score(y_true[level_num], y_pred_orig[level_num])),
                    "weighted_precision": float(
                        precision_score(y_true[level_num], y_pred[level_num], average='weighted')),
                    "weighted_precision_orig": float(
                        precision_score(y_true[level_num], y_pred_orig[level_num], average='weighted')),
                    "weighted_recall": float(recall_score(y_true[level_num], y_pred[level_num], average='weighted')),
                    "weighted_recall_orig": float(
                        recall_score(y_true[level_num], y_pred_orig[level_num], average='weighted')),
                    "weighted_f1": float(f1_score(y_true[level_num], y_pred[level_num], average='weighted')),
                    "weighted_f1_orig": float(f1_score(y_true[level_num], y_pred_orig[level_num], average='weighted'))
                }

                evaluation["level_" + str(level_prediction[0])] = metrics

            # Global Metrics
            evaluation["loss"] = float(sum(losses.values()))
            evaluation["global_accuracy"] = global_accuracy(y_true, y_pred)
            evaluation["global_measures_weighted"] = global_measures(y_true, y_pred, 'weighted')
            evaluation["global_measures_macro"] = global_measures(y_true, y_pred, 'macro')
            evaluation["hierarchical_measures"] = hierarchical_measures(hierarchy, y_true, y_pred)
        except ValueError:
            pass

        tf.print(evaluation)
        saver.save_dict(evaluation, f"metrics_{split}")
        confusion_matrices.save()

    @staticmethod
    def calculate_y_pred(hierarchy, labels, level_predictions):
        # Calculate y_pred with Postprocessing
        y_pred = dict([(level_num, []) for level_num, _ in enumerate(level_predictions)])
        nan_node = AnyNode(id='nan')
        level_softmaxs = [(level_prediction[0], [softmax(logits) for logits in level_prediction[1]])
                          for level_prediction in level_predictions]
        for idx in tqdm(range(len(level_softmaxs[0][1]))):
            def get_best_node(parent, level_softmaxs_slice, depth, probability=1):
                level_num = depth - len(level_softmaxs_slice)
                level_softmax = level_softmaxs_slice[0][1]

                # Find possible children
                children_dict = dict([(child.id, child) for child in parent.children])
                if "nan" in labels[level_num]:
                    children_dict['nan'] = nan_node

                # Find indices of possible children
                indices_by_idd = dict([(idd, index) for index, idd in enumerate(labels[level_num])])
                indices_possible = np.array(
                    [indices_by_idd[idd] for idd in sorted(children_dict.keys())])
                # Find child with highest probability
                best_indices = indices_possible[np.argsort(level_softmax[idx][indices_possible])]
                y_pred_idx_best, y_pred_idx_prob_best = nan_node, 0

                if len(level_softmaxs_slice) == 1:
                    # recursion base case
                    y_pred_idx_best = children_dict[labels[level_num][best_indices[-1]]]
                    y_pred_idx_prob_best = level_softmax[idx][best_indices[-1]]
                else:
                    for i in range(min(len(children_dict), 10)):
                        y_pred_idx_cand = children_dict[labels[level_num][best_indices[-i]]]
                        y_pred_idx_prob_cand = level_softmax[idx][best_indices[-i]]
                        y_pred_idx_cand, y_pred_idx_prob_cand \
                            = get_best_node(y_pred_idx_cand, level_softmaxs_slice[1:], depth,
                                            y_pred_idx_prob_cand)

                        if y_pred_idx_prob_cand > y_pred_idx_prob_best:
                            y_pred_idx_best, y_pred_idx_prob_best = y_pred_idx_cand, y_pred_idx_prob_cand

                if y_pred_idx_best == nan_node:
                    y_pred_idx_best = parent
                return y_pred_idx_best, y_pred_idx_prob_best * probability

            y_pred_idx, _ = get_best_node(hierarchy, level_softmaxs, len(level_softmaxs))
            path = list(y_pred_idx.path)[1:]
            y_pred_list = [node.id for node in path]
            y_pred_list = y_pred_list + ['nan'] * (len(y_pred) - len(y_pred_list))

            for level_num in range(len(y_pred)):
                y_pred[level_num].append(y_pred_list[level_num])

        # calculate y_pred as it is put out by the classifier
        y_pred_orig = {}
        for level_num, level_prediction in enumerate(level_predictions):
            y_pred_orig[level_num] = [labels[level_num][i] for i in
                                      np.argmax(level_prediction[1], axis=1)]

        return y_pred, y_pred_orig
