import tensorflow as tf
from dph.core.modeling.dp_optimizer import DPAdamGaussianOptimizer
from tqdm.auto import tqdm
from dph.core.training.history import History


class TrainLoop:
    def __init__(self,
                 strategy: tf.distribute.Strategy,
                 model: tf.keras.models.Model, optimizer, loss_dict: dict,
                 metrics: dict,
                 train_dataset: tf.data.Dataset, num_epochs, steps_per_epoch, batch_size_train,
                 val_dataset: tf.data.Dataset, val_steps,
                 callbacks: list,
                 grad_file_path: str):

        self.strategy = strategy
        self.model = model
        self.optimizer = optimizer

        self.compute_loss = get_compute_loss(strategy, loss_dict)

        self.history = History(metrics)

        self.train_dataset = train_dataset
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size_train = batch_size_train

        self.val_dataset = val_dataset
        self.val_steps = val_steps

        self.callbacks = callbacks

        self.grad_file_path = grad_file_path

    def train_step(self, batch):
        inputs, labels = batch

        with tf.GradientTape(persistent=True) as gradient_tape:
            predictions = self.model(inputs, training=True)[0]
            per_example_loss = self.compute_loss(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.batch_size_train)

            if isinstance(self.optimizer, tf.keras.optimizers.Optimizer):
                with gradient_tape.stop_recording():
                    gradients = gradient_tape.gradient(loss, self.model.trainable_variables)
                # gradients = [(tf.clip_by_norm(grad, 1.0)) for grad in gradients]
                grads_and_vars = zip(gradients, self.model.trainable_variables)

            elif isinstance(self.optimizer, tf.compat.v1.train.Optimizer):
                assert isinstance(self.optimizer, DPAdamGaussianOptimizer)
                # noinspection PyArgumentList
                grads_and_vars = self.optimizer.compute_gradients(lambda: per_example_loss,
                                                                  self.model.trainable_variables,
                                                                  gradient_tape=gradient_tape,
                                                                  grad_file_path=self.grad_file_path)
            else:
                raise ValueError("Unexpected Optimizer class!")

        # When apply_gradients is called within a distribution strategy scope, its behavior is modified!
        self.optimizer.apply_gradients(grads_and_vars)

        self.history.update_metrics(labels, predictions)
        return loss

    def test_step(self, batch):
        inputs, labels = batch

        predictions = self.model(inputs, training=False)[0]

        per_example_loss = self.compute_loss(labels, predictions)

        self.history.update_loss(per_example_loss)
        self.history.update_metrics(labels, predictions)

    @tf.function
    def distributed_train_step(self, batch):
        per_replica_losses = self.strategy.run(self.train_step, args=(batch,))
        loss_step = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        self.history.update_loss(loss_step)

    @tf.function
    def distributed_test_step(self, dataset_inputs):
        self.strategy.run(self.test_step, args=(dataset_inputs,))

    def start(self):
        train_dist_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        val_dist_dataset = self.strategy.experimental_distribute_dataset(self.val_dataset)

        self.model.stop_training = False

        for callback in self.callbacks:
            callback.set_model(self.model)
            callback.on_train_begin()

        for epoch in range(self.num_epochs):
            if self.model.stop_training:
                break

            tf.print('=' * 50, f"EPOCH {epoch}", '=' * 50)

            [callback.on_epoch_begin(epoch=epoch, logs=self.history.history) for callback in self.callbacks]

            # TRAIN LOOP
            for i, batch in tqdm(enumerate(train_dist_dataset), total=self.steps_per_epoch):
                self.distributed_train_step(batch)
                if i % 50 == 0 or i == (self.steps_per_epoch - 1):
                    self.history.print_current_results()

            self.history.save_current_results(epoch)
            self.history.reset_current()

            # TEST LOOP
            for batch in tqdm(val_dist_dataset, total=self.val_steps):
                self.distributed_test_step(batch)

            self.history.save_current_results(epoch, val=True)
            self.history.reset_current()
            self.history.print_summary()

            [callback.on_epoch_end(epoch, logs=self.history.get_logs()) for callback in self.callbacks]

        [callback.on_train_end() for callback in self.callbacks]


def get_compute_loss(strategy, loss_dict: dict):
    with strategy.scope():
        def compute_loss(labels, predictions):
            per_example_loss = tf.zeros(labels[0].shape[0])

            for index, (level, level_prediction) in enumerate(predictions.items()):
                loss_object = loss_dict[level]
                assert isinstance(loss_object, tf.keras.losses.Loss)
                per_example_loss += loss_object(labels[index], level_prediction)
            return per_example_loss

        return compute_loss
