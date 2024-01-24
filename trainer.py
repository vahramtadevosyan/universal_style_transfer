import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import MSELoss, DataParallel

from model import Encoder, Decoder
from dataloader import get_dataloader


class Trainer:
    def __init__(self, config, depth):
        assert 1 <= depth <= 5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._seed_everything(config['seed'])

        self.depth = int(depth)
        self.encoder = Encoder(depth, load_weights=True).to(self.device)
        self.decoder = Decoder(depth, load_weights=False).to(self.device)
        self.encoder.eval()

        if torch.cuda.device_count() > 1:
            self.decoder = DataParallel(self.decoder)

        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.lr = config['learning_rate']
        self.val_interval = config['validation_interval']
        self.lambda_value = config['lambda']
        
        self.train_dataloader = get_dataloader(
            root_dir=config['train_data_path'],
            batch_size=self.batch_size,
            max_side=config['max_side'],
            max_data=config['max_data'],
        )
        self.val_dataloader = get_dataloader(
            root_dir=config['val_data_path'],
            batch_size=self.batch_size,
            max_side=config['max_side'],
            is_validation=True,
        )

        self.criterion = MSELoss()
        self.optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)

        self.best_val_loss = float('inf')
        self.checkpoint_dir = config['checkpoint_path']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f'decoder{self.depth}.pth')
        
        print(f'Starting the training with resolution {config["max_side"]} on {self.device} device...')
        print(f'Decoder depth: {self.depth}')
        print(f'Training data: {config["train_data_path"]}')
        print(f'Validation data: {config["val_data_path"]}\n')


    def train(self):
        for epoch in range(self.num_epochs):
            reconstruction_train_loss = 0.
            cycle_consistency_train_loss = 0.
            total_train_loss = 0.

            # Training
            self.decoder.train()
            for inputs in tqdm(self.train_dataloader):
                # Forward pass
                inputs = inputs.to(self.device)
                encoded_inputs = self.encoder(inputs)
                reconstructed_outputs = self.decoder(encoded_inputs)
                encoded_outputs = self.encoder(reconstructed_outputs)

                # Compute loss
                reconstruction_loss = self.criterion(reconstructed_outputs, inputs)
                cycle_consistency_loss = self.criterion(encoded_inputs, encoded_outputs)
                total_loss = reconstruction_loss + self.lambda_value * cycle_consistency_loss

                reconstruction_train_loss += reconstruction_loss.item()
                cycle_consistency_train_loss += cycle_consistency_loss.item()
                total_train_loss += total_loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Validation
            if (epoch + 1) % self.val_interval == 0:
                self.decoder.eval()
                val_loss = self._validate()

                reconstruction_train_loss /= len(self.train_dataloader)
                cycle_consistency_train_loss /= len(self.train_dataloader)
                total_train_loss /= len(self.train_dataloader)

                log = f'Epoch [{epoch+1}/{self.num_epochs}],\t'
                log += f'Reconstruction Loss: {reconstruction_train_loss:.4f},\t'
                log += f'Cycle Consistency Loss: {cycle_consistency_train_loss:.4f},\t'
                log += f'Total Loss: {total_train_loss:.4f},\tVal Loss: {val_loss:.4f}'
                print(log)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.decoder.state_dict(), self.checkpoint_path)

    def _validate(self):
        total_val_loss = 0.
        with torch.no_grad():
            for inputs in tqdm(tqdm(self.val_dataloader)):
                # Forward pass
                inputs = inputs.to(self.device)
                encoded_inputs = self.encoder(inputs)
                reconstructed_outputs = self.decoder(encoded_inputs)
                encoded_outputs = self.encoder(reconstructed_outputs)

                # Compute loss
                reconstruction_loss = self.criterion(reconstructed_outputs, inputs)
                cycle_consistency_loss = self.criterion(encoded_inputs, encoded_outputs)
                total_loss = reconstruction_loss + self.lambda_value * cycle_consistency_loss
                total_val_loss += total_loss.item()

        avg_val_loss = total_val_loss / len(self.val_dataloader)
        return avg_val_loss

    def _seed_everything(self, seed):
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

