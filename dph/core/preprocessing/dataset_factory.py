import tensorflow as tf


class DatasetFactory:

    @staticmethod
    def create_dataset(ids, masks, labels_list):
        def gen():
            for i in range(len(ids)):
                yield (
                    {
                        "input_ids": ids[i],
                        "attention_mask": masks[i] if masks is not None else [0]
                    },
                    tuple([labels[i] for labels in labels_list])
                )

        # noinspection PyTypeChecker
        return tf.data.Dataset.from_generator(
            gen,
            (
                {
                    "input_ids": tf.int32,
                    "attention_mask": tf.int32
                },
                tuple([tf.int32 for _ in labels_list])
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None])
                },
                tuple([tf.TensorShape([None]) for _ in labels_list])
            )
        )
