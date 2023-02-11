import sys
from timeit import default_timer as timer
import tensorflow as tf


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to note training and prediction time
    """

    def __init__(self, ):
        super(TimingCallback, self).__init__()
        self.train_start_time = None
        self.train_end_time = None
        self.prediction_time = []
        self.prediction_start_time = None

    def on_train_begin(self, logs: dict):
        tf.print("Training starting time noted.", output_stream=sys.stdout)
        self.train_start_time = timer()

    def on_train_end(self, logs: dict):
        tf.print("Training ending time noted.", output_stream=sys.stdout)
        self.train_end_time = timer()

    def on_test_batch_begin(self, batch: int, logs: dict):
        tf.print(
            f"For batch:{batch} prediction start time noted.",
            output_stream=sys.stdout
        )
        self.prediction_start_time = timer()

    def on_test_batch_end(self, batch: int, logs: dict):
        tf.print(
            f"For batch:{batch} prediction end time noted.",
            output_stream=sys.stdout
        )
        self.prediction_time.append(timer() - self.prediction_start_time)
