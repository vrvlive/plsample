import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from collections import OrderedDict


class PlDNNModule(pl.LightningModule):
    def __init__(self, hparams):
        super(PlDNNModule, self).__init__()
        self.hparams = hparams
        self.layers = nn.Sequential(OrderedDict(self.get_fclayer_list([10, 512, 256, 128, 64, 32, 16])))

    # creates a list of hidden layers with given number of neuron in each layer and connects it to the output layer.
    # Relu is used as the activation funtion. No activation function is applied for the last layer output
    def get_fclayer_list(self, hidden_layers, outputs=1):
        input_layers, output_layers = hidden_layers[:-1], hidden_layers[1:]
        layers = []
        for i, (l1, l2) in enumerate(zip(input_layers, output_layers)):
            layers.append((f'fc{i}', nn.Linear(l1, l2)))
            layers.append((f'relu{i}', nn.ReLU()))
        layers.append(('fc_out', nn.Linear(output_layers[-1], outputs)))
        return layers

    def forward(self, x):
        x = self.layers(x)
        return x

    # def configure_optimizers(self):
    #    return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2, min_lr=1e-7, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.l1_loss(logits, y)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log('val_loss', loss)
        return {'val_loss', loss}

    def validation_epoch_end(self, outputs):
        avg_val_loss = th.stack([list(x)[0] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)

    def test_step(self, batch, batch_idx):
        loss, _ = self.shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        return {'test_loss': loss}

    def test_epoch_end_removed(self, outputs):
        avg_loss = th.stack([x['test_loss'] for x in outputs]).mean()
        self.log('test_loss', avg_loss)
        return {'avg_test_loss': avg_loss}
