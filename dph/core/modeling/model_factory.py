import math
from functools import partialmethod

from tensorflow_privacy import DPAdamGaussianOptimizer

from dph.core.modeling.custom_models import *
from dph.core.parameters.parameters import DPParameters, Parameters, AdamWParameters
from transformers import AutoConfig, create_optimizer, AdamWeightDecay, BertConfig, RobertaConfig


class ModelFactory:
    @staticmethod
    def build_model(p: Parameters):
        if hasattr(p, "num_classes") \
                and hasattr(p, "train_size") \
                and hasattr(p, "val_size") \
                and hasattr(p, "levels") \
                and hasattr(p, "model_path"):

            config = ModelFactory.get_config(p)

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                p.strategy = strategy.__class__.__name__

                # Model Topology Definition
                config.num_level_classes = list(zip(p.levels, p.num_classes))
                # noinspection PyTypeChecker
                model = ModelFactory.get_hierarchical_classifier(config, p)

                loss_dict = dict([
                    (name, tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                   reduction=tf.keras.losses.Reduction.NONE))
                    for name in p.levels])
                metrics = dict([
                    (name, [tf.keras.metrics.CategoricalAccuracy()])
                    for name in p.levels])

                if hasattr(p, "random_init"):
                    ModelFactory.random_init(model, p)

                if hasattr(p, "only_dense"):
                    model.bert.trainable = False

                p.steps_per_epoch = p.train_size // p.batch_size_train + 1
                p.val_steps = p.val_size // p.batch_size_val + 1

                # Prepare training
                optimizer = ModelFactory.get_optimizer(p)

            return strategy, model, optimizer, loss_dict, metrics

    @staticmethod
    def get_hierarchical_classifier(config, p):
        if isinstance(config, RobertaConfig):
            return TFRobertaForHierarchicalClassification.from_pretrained(
                p.model_path,
                config=config,
                from_pt=p.model_path.startswith("./")
            )
        if isinstance(config, BertConfig):
            return TFBertForHierarchicalClassification.from_pretrained(
                p.model_path,
                config=config,
                from_pt=p.model_path.startswith("./")
            )
        if p.model_path == 'WordCnn':
            return WordCnnForHierarchicalClassification(config)
        if p.model_path == 'BasicClf':
            return BasicHierarchicalTextClassifier(config)
        raise NotImplementedError

    @staticmethod
    def get_config(p):
        if hasattr(p, 'embeddings_id'):
            pretrained_embeddings = p.pretrained_embeddings
            p.vocab_size = pretrained_embeddings.vocab_size
            p.embedding_size = pretrained_embeddings.embedding_size

            class ConfigMock:
                pass

            config = ConfigMock()

            config.text_length = p.text_length
            config.vocab_size = p.vocab_size
            config.embedding_size = p.embedding_size
            config.pretrained_embeddings = pretrained_embeddings.embeddings

            del p.pretrained_embeddings

            if hasattr(p, "hidden_dropout_prob"):
                config.hidden_dropout_prob = 0

        elif hasattr(p, "hidden_dropout_prob"):
            config = AutoConfig.from_pretrained(p.model_path, hidden_dropout_prob=p.hidden_dropout_prob,
                                                output_hidden_states=True, output_attentions=True)
        else:
            config = AutoConfig.from_pretrained(p.model_path, output_hidden_states=True, output_attentions=True)
        return config

    @staticmethod
    def random_init(model, p: Parameters):
        print("starting random init")
        # noinspection PyUnresolvedReferences
        if p.random_init == "xavier":
            initializer = tf.keras.initializers.GlorotNormal()
            for w in model.bert.weights:
                w_pre = w.numpy()  # pretrained weights
                w_random = initializer(shape=w_pre.shape)
                w.assign(w_random)
        else:
            p.rand_stddev = 0.1
            for w in model.bert.weights:
                w_pre = w.numpy()  # pretrained weights
                w_random = tf.random.normal(w_pre.shape, stddev=p.rand_stddev, dtype=w_pre.dtype).numpy()
                w.assign(w_random)

    @staticmethod
    def get_optimizer(p):
        if isinstance(p, AdamWParameters):
            # noinspection PyUnresolvedReferences
            warmup_steps = p.steps_per_epoch // 5
            # noinspection PyUnresolvedReferences
            total_steps = p.steps_per_epoch * p.num_epochs - warmup_steps
            AdamWeightDecay.apply_gradients = partialmethod(AdamWeightDecay.apply_gradients, clip_norm=1.0)
            optimizer = create_optimizer(p.learning_rate, num_train_steps=total_steps, num_warmup_steps=warmup_steps)
        else:
            # Parameters according to google-research/bert/optimization.py
            p.beta_1 = 0.9
            p.beta_2 = 0.999
            p.epsilon = 1e-6

            if isinstance(p, DPParameters):
                assert p.batch_size_train % p.microbatch_size == 0, \
                    "batch_size_train should be divisible by microbatch_size!"
                p.num_microbatches = p.batch_size_train / p.microbatch_size
                tf.print(f"Setting num_microbatches to {p.num_microbatches}.")

                assert p.num_microbatches % p.num_gpu == 0, "num_microbatches should be divisible by number of GPUs!"
                p.num_microbatches_per_gpu = int(p.num_microbatches / p.num_gpu)

                def sigma_n(sigma_1, num_gpu):
                    return math.sqrt(math.pow(sigma_1, 2) / num_gpu)

                p.sigma = p.noise_multiplier * p.clipnorm
                p.sigma_n = sigma_n(p.sigma, p.num_gpu)

                optimizer = DPAdamGaussianOptimizer(learning_rate=p.learning_rate,
                                                    beta1=p.beta_1,
                                                    beta2=p.beta_2,
                                                    epsilon=p.epsilon,
                                                    l2_norm_clip=p.clipnorm,
                                                    noise_multiplier=p.sigma_n / p.clipnorm,
                                                    num_microbatches=p.num_microbatches_per_gpu)
            else:
                p.clipnorm = 1.0
                p.clipnorm = "unsupported"
                optimizer = tf.keras.optimizers.Adam(learning_rate=p.learning_rate,
                                                     beta_1=p.beta_1,
                                                     beta_2=p.beta_2,
                                                     epsilon=p.epsilon)
        return optimizer
