import torch
from torch import nn

class DenseLayer1D(nn.Module):
    def __init__(self, in_channels, growth_rate, dilation):
        super().__init__()
        self.features = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=in_channels, out_channels=growth_rate,
                       kernel_size=3, padding =dilation, dilation=dilation, bias = False)
        )

    def forward(self, x):

        new_features = self.features(x)
        output = torch.cat([x, new_features], dim=1)

        return output


class DenseBlock1D(nn.Module):
    def __init__(self, growth_rate=32):
        super().__init__()
        self.first_conv = nn.Conv1d(
        in_channels=2,
        out_channels=growth_rate,
        kernel_size=3,
        padding=1,
        dilation=1,
        bias=False)
        
        self.layers = nn.ModuleList()
        dilations = [2, 4, 8, 16, 32, 64, 1]
        in_channels = 2 + growth_rate
        for dilation in dilations:
            model = DenseLayer1D(in_channels, growth_rate, dilation)
            self.layers.append(model)
            in_channels += growth_rate

    def forward(self, x):

        first_features = self.first_conv(x)
        x = torch.cat([x, first_features], dim=1)
        for layer in self.layers:
            x = layer(x)

        return x

class EyeKnowYouTooDense(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dense_block = DenseBlock1D(growth_rate=32)
        self.embedding_head = nn.Sequential(
            nn.BatchNorm1d(258),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(258,128),
            )
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.dense_block(x)
        embeddings = self.embedding_head(x)
        logits = self.classifier(embeddings)

        return embeddings, logits


            
class EyeKnowYouTooSimple(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #Feature extractor
        self.features = nn.Sequential(
            nn.Conv1d(in_channels = 2, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),

            nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        #Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        embeddings = self.features(x)
        logits = self.classifier(embeddings)

        return embeddings, logits

