from dph.core.experiment import Experiment
from dph.mia.attack_parameters import AttackParameters
from dph.mia.attacker.attacker import Attacker
from dph.mia.features.features import Features
from dph.mia.features.analyze.visualize_loss import visualize_loss
from dph.utils.saver import Saver
from dph.utils.set_seed import set_seed


class AttackExperiment:
    def __init__(self,
                 p: AttackParameters,
                 target_experiment: Experiment = None):

        self.p = p
        self.target_experiment = target_experiment
        if self.target_experiment is not None:
            self.p.seed = self.target_experiment.p.seed

        self.saver = Saver(p.experiment_name)

        set_seed(p.seed)

    def start(self):
        if self.p.target_model_path is None:
            # save model at end of training
            self.target_experiment.p.save_model_freq = 0
            # Run target model training
            self.target_experiment.start()

            self.p.target_model_path = self.target_experiment.saver.log_dir

        features = Features(self.p, self.saver)
        self.saver.save_dict(self.p.get_parameters(), "attack_parameters")

        # Evaluate Membership Inference Risk
        if self.p.visualize_loss:
            visualize_loss(features, self.saver.log_dir + "/attackloss")

        if self.p.train_attacker:
            attacker = Attacker(features, self.p, self.saver)
            attack_model, attack_history = attacker.train()
            attacker.evaluate(attack_model)
            attacker.analyse(attack_model)

            return attack_history

        else:
            return features
