import tensorflow as tf


class History:
    def __init__(self, metrics: dict):
        self.output_names = list(metrics.keys())
        self.metrics = metrics
        self.metrics['loss'] = [tf.keras.metrics.Mean('loss')]

        self.history = {}

    def reset_current(self):
        for output in self.metrics.keys():
            metrics = self.metrics[output]
            assert isinstance(metrics, list)
            for metric in metrics:
                assert isinstance(metric, tf.keras.metrics.Metric)
                metric.reset_states()

    def update_metrics(self, labels, predictions):
        #if isinstance(predictions[0], tuple):
        if isinstance(predictions, dict):
            predictions = [prediction for level_name, prediction in predictions.items()]
        for i, output_name in enumerate(self.output_names):
            metrics = self.metrics[output_name]
            assert isinstance(metrics, list)
            for metric in metrics:
                assert isinstance(metric, tf.keras.metrics.Metric)
                metric.update_state(labels[i], predictions[i])

    def update_loss(self, loss):
        metrics = self.metrics['loss']
        assert isinstance(metrics, list)
        for metric in metrics:
            assert isinstance(metric, tf.keras.metrics.Mean)
            metric.update_state(loss)

    def save_current_results(self, epoch, val=False):
        results = self.get_current_results(val=val)
        for key, value in results.items():
            self.history.setdefault(key, []).append(value)
            assert len(self.history[key]) == epoch + 1

    def print_summary(self):
        out = str()
        for key in sorted(self.history.keys()):
            out += f"{key}: {self.history[key][-1]:.4f} - "
        tf.print(out)

    def print_current_results(self):
        out = str()
        results = self.get_current_results()
        for key in sorted(results):
            out += f"{key}: {results[key]:.4f} - "
        tf.print(out)

    def get_current_results(self, val=False):
        results = dict()
        prefix = 'val_' if val else ''
        for output_name in self.metrics.keys():
            metrics = self.metrics[output_name]
            assert isinstance(metrics, list)
            for metric in metrics:
                assert isinstance(metric, tf.keras.metrics.Metric)
                key = prefix
                key += (output_name + '_') if output_name != 'loss' else ''
                key += metric.name
                results[key] = float(metric.result())
        return results

    def get_logs(self):
        return {k: v[-1] for k, v in self.history.items()}
