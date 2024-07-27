import os
import logging
from torch import nn
from pathlib import Path


abs_path = Path(os.path.abspath(__file__)).parent

def set_logging():
    # set logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
    # Disable c10d logging
    logging.getLogger('c10d').setLevel(logging.ERROR)


from lightning.pytorch.loggers import TensorBoardLogger


class Logger(TensorBoardLogger):
    def __init__(self, model, version="v1", *args, **kwargs):
        super().__init__(abs_path, name=f"lightning_logs/{version}", *args, **kwargs)
        self._log_graph = True
        self.model = model
        self.model.set_custom_logger(self)

    def log_model_arch(self):
        # Log the model architecture at the start of training
        model_info = "<br>".join([f"{name}: {str(el)}" for name, el in self.model.get_elements().items()])
        self.experiment.add_text('Model Architecture', model_info, 0)

    def log_params(self):
        # Log hyperparameters as text
        hp_text = "<br>".join([f"{key}: {value}" for key, value in vars(self.model.params).items()])
        self.experiment.add_text('Hyperparameters', hp_text, 0)


