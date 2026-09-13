class AttentiveCQTCNNBiLSTMEncoder(nn.Module):
    """CQT local CNN followed by global bidirectional temporal modeling."""

    def __init__(
        self,
        embedding_dim=256,
        lstm_hidden_size=128,
        lstm_layers=1,
        lstm_dropout=0.20,
    ):
        super().__init__()
        if lstm_layers < 1:
            raise ValueError(f"lstm_layers must be at least 1, got {lstm_layers}.")

        # Keep the CQT CNN channels, pooling and dropout identical to experiment 15.
        self.features = nn.Sequential(
            ConvBNReLU(1, 32, dropout=0.05),
            ConvBNReLU(32, 32, dropout=0.05),
            nn.MaxPool2d(kernel_size=2),
            ConvBNReLU(32, 64, dropout=0.10),
            ConvBNReLU(64, 64, dropout=0.10),
            nn.MaxPool2d(kernel_size=2),
            ConvBNReLU(64, 128, dropout=0.10),
            ConvBNReLU(128, 128, dropout=0.10),
            nn.MaxPool2d(kernel_size=2),
            ConvBNReLU(128, 256, dropout=0.10),
            ConvBNReLU(256, 256, dropout=0.10),
            nn.MaxPool2d(kernel_size=(2, 1)),
        )

        # Mean/max frequency summaries form a 512-D sequence of about 27 steps.
        cnn_temporal_dim = 256 * 2
        self.bilstm = nn.LSTM(
            input_size=cnn_temporal_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        bilstm_dim = lstm_hidden_size * 2
        self.temporal_attention = nn.Sequential(
            nn.Linear(bilstm_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(bilstm_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, inputs):
        feature_map = self.features(inputs)
        frequency_mean = torch.mean(feature_map, dim=2)
        frequency_max = torch.amax(feature_map, dim=2)
        cnn_sequence = torch.cat((frequency_mean, frequency_max), dim=1).transpose(1, 2)

        temporal_sequence, _ = self.bilstm(cnn_sequence)
        attention_logits = self.temporal_attention(temporal_sequence)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled_feature = torch.sum(attention_weights * temporal_sequence, dim=1)
        return self.embedding_projection(pooled_feature)