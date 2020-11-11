from pathlib import Path
import torch as th
import pandas as pd

# All of this can be moved to the test loop of pytorch lightning itself
# Decided to make the inference separate from the training loops


class ResultWriter():
    def __init__(self, trained_model, hparams):
        self.model = trained_model
        self.hparams = hparams

    def load_data(self):
        input_file = Path(self.hparams.data_folder).joinpath(self.hparams.test_input).as_posix()
        df = pd.read_csv(input_file)
        x = df.values
        return x

    def test_model(self):
        test_data = self.load_data()
        output = []

        for line in test_data:
            x = th.tensor(line[:10], dtype=th.float)
            y = self.model(x.view(-1, 10))
            output.append(y.item())

        self.save_output(test_data, output)

    def save_output(self, input, output):
        output_file = Path(self.hparams.data_folder).joinpath(self.hparams.test_output).as_posix()

        df = pd.DataFrame(input)
        df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']
        df['Y'] = output

        df.to_csv(output_file, index=False)
