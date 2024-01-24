import torch
import torch.nn as nn

class no_def_model(nn.Module):
    def __init__(self, nb_classes):
        super(no_def_model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=1,
                padding='same',
            ),
            nn.BatchNorm1d(32),
            # nn.ELU(alpha=1.0),
            nn.ReLU(),
            nn.Conv1d(32, 32, 8, 1, padding='same'),
            nn.BatchNorm1d(32),
            # nn.ELU(alpha=1.0),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 4),
            nn.Dropout(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 8, 1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 8, 1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 4),
            nn.Dropout(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 8, 1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, 1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 4),
            nn.Dropout(0.1),
        )

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 8, 1, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 8, 1, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(8, 4, 4),
            nn.Dropout(0.1),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5120, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.out = nn.Sequential(
            nn.Linear(512, nb_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output, x

