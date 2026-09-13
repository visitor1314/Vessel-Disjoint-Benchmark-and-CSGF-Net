def channel_shuffle(inputs, groups=2):
    """Mix channel groups before the time/frequency split."""
    batch_size, channels, frequency_bins, time_steps = inputs.shape
    if channels % groups != 0:
        raise ValueError(f"channels={channels} must be divisible by groups={groups}.")
    channels_per_group = channels // groups
    shuffled = inputs.reshape(
        batch_size,
        groups,
        channels_per_group,
        frequency_bins,
        time_steps,
    )
    shuffled = shuffled.transpose(1, 2).contiguous()
    return shuffled.reshape(batch_size, channels, frequency_bins, time_steps)


class AdaptiveResidualNormalization(nn.Module):
    """Learn a balance between batch-normalized and untouched features."""

    def __init__(self, channels):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(channels)
        # sigmoid(0) = 0.5: begin with equal normalized/residual contributions.
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * self.batch_norm(inputs) + (1.0 - alpha) * inputs


class TFSCBlock(nn.Module):
    """Time-frequency separate convolution with axis context and ARN."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        if out_channels % 2 != 0:
            raise ValueError(f"TFSC out_channels must be even, got {out_channels}.")

        branch_channels = out_channels // 2
        self.channel_projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # 3 x 1 follows adjacent Mel bins; 1 x 3 follows adjacent time frames.
        self.frequency_depthwise = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            groups=branch_channels,
            bias=False,
        )
        self.frequency_pointwise = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.time_depthwise = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=(1, 3),
            padding=(0, 1),
            groups=branch_channels,
            bias=False,
        )
        self.time_pointwise = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.adaptive_norm = AdaptiveResidualNormalization(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, inputs):
        projected = channel_shuffle(self.channel_projection(inputs), groups=2)
        frequency_input, time_input = torch.chunk(projected, chunks=2, dim=1)

        frequency_local = self.frequency_depthwise(frequency_input)
        # Pool over frequency, then broadcast global temporal context back to all bins.
        frequency_context = torch.mean(frequency_local, dim=2, keepdim=True)
        frequency_context = self.frequency_pointwise(frequency_context)
        frequency_output = frequency_input + frequency_context.expand_as(frequency_input)

        time_local = self.time_depthwise(time_input)
        # Pool over time, then broadcast the stable spectral context back to all frames.
        time_context = torch.mean(time_local, dim=3, keepdim=True)
        time_context = self.time_pointwise(time_context)
        time_output = time_input + time_context.expand_as(time_input)

        fused = torch.cat((frequency_output, time_output), dim=1)
        return self.dropout(F.relu(self.adaptive_norm(fused), inplace=True))


class AttentiveTFSCEncoder(nn.Module):
    """Log-Mel encoder built from four stages and nine TFSC blocks."""

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, groups=6, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            TFSCBlock(48, 48, dropout=0.05),
            TFSCBlock(48, 48, dropout=0.05),
            nn.MaxPool2d(kernel_size=2),
            TFSCBlock(48, 64, dropout=0.08),
            TFSCBlock(64, 64, dropout=0.08),
            nn.MaxPool2d(kernel_size=2),
            TFSCBlock(64, 96, dropout=0.10),
            TFSCBlock(96, 96, dropout=0.10),
            nn.MaxPool2d(kernel_size=(2, 1)),
            TFSCBlock(96, 128, dropout=0.10),
            TFSCBlock(128, 128, dropout=0.10),
            TFSCBlock(128, 128, dropout=0.10),
        )

        temporal_dim = 128 * 2
        self.temporal_attention = nn.Sequential(
            nn.Linear(temporal_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.embedding_projection = nn.Sequential(
            nn.Linear(temporal_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, inputs):
        feature_map = self.features(self.stem(inputs))
        frequency_mean = torch.mean(feature_map, dim=2)
        frequency_max = torch.amax(feature_map, dim=2)
        temporal_sequence = torch.cat((frequency_mean, frequency_max), dim=1).transpose(1, 2)

        attention_logits = self.temporal_attention(temporal_sequence)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled_feature = torch.sum(attention_weights * temporal_sequence, dim=1)
        return self.embedding_projection(pooled_feature)