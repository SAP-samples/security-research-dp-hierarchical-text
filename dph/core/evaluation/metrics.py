from anytree import PreOrderIter, AnyNode
from pathlib import Path
import subprocess

from sklearn.metrics import *


def global_accuracy(y_true: dict, y_pred: dict):
    correct_global = 0

    for idx in range(len(y_pred[0])):
        if all([y_pred[level][idx] == y_true[level][idx] for level in range(len(y_pred))]):
            correct_global += 1

    return correct_global / len(y_pred[0])


def global_measures(y_true: dict, y_pred: dict, average='weighted'):
    y_true_concat = []
    y_pred_concat = []

    for idx in range(len(y_pred[0])):
        y_true_concat.append(" ".join([y_true[level][idx] for level in range(len(y_pred))]))
        y_pred_concat.append(" ".join([y_pred[level][idx] for level in range(len(y_pred))]))

    return_dict = {
        "accuracy": float(accuracy_score(y_true_concat, y_pred_concat)),
        "precision": float(precision_score(y_true_concat, y_pred_concat, average=average)),
        "recall": float(recall_score(y_true_concat, y_pred_concat, average=average)),
        "f1": float(f1_score(y_true_concat, y_pred_concat, average=average)),
    }

    return return_dict


def hierarchical_measures(hierarchy: AnyNode, y_true, y_pred):
    eval_files_dir = Path("./tools/HEMKit/eval_files")
    eval_files_dir.mkdir(parents=True, exist_ok=True)
    for file in eval_files_dir.glob('*'):
        if file.is_file():
            file.unlink()

    node_dict = dict()

    hierarchy_file_name = "hierarchy.txt"
    with eval_files_dir.joinpath(hierarchy_file_name).open(mode="w") as hierarchy_file:
        for node in PreOrderIter(hierarchy):
            if node in node_dict:
                print(node.id, node.name)
            node_dict[node] = len(node_dict) + 1
            if node.parent is not None:
                assert node.parent.id != node.id
                assert node_dict[node.parent] != node_dict[node]
                hierarchy_file.write(str(node_dict[node.parent]) + " " + str(node_dict[node]) + "\n")

    def write_categories(file_name, y):
        with eval_files_dir.joinpath(file_name).open(mode="w") as cat_file:
            for idx in range(len(y[0])):
                cat_node = hierarchy
                for level in range(len(y)):
                    if y[level][idx] != 'nan':
                        children_dict = dict([(child.id, child) for child in cat_node.children])
                        if y[level][idx] in children_dict:
                            cat_node = children_dict[y[level][idx]]
                        else:
                            print("Warning: test labels not consistent with class hierarchy!")

                cat_file.write(str(node_dict[cat_node]) + "\n")

    true_cat_file_name = "true_cat.txt"
    write_categories(true_cat_file_name, y_true)
    pred_cat_file_name = "pred_cat.txt"
    write_categories(pred_cat_file_name, y_pred)

    result = subprocess.run(['../bin/HEMKit',
                             hierarchy_file_name, true_cat_file_name, pred_cat_file_name,
                             str(len(node_dict)), "5"],
                            stdout=subprocess.PIPE, cwd=str(eval_files_dir))
    output = str(result.stdout, encoding='utf-8')
    split_lines = [line.split("=") for line in output.split("\n")]

    return_dict = dict([(line[0].strip(), float(line[1])) for line in split_lines if len(line) == 2])

    return return_dict
