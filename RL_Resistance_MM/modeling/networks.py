"""networks.py — Network architecture definitions for RL Resistance MM.

Single source of truth for all network classes.
Both the training notebook and demo/run_agent.py import from here.

Hierarchy:
  DecomposedQNetwork (abstract base)
    ├── DQN_V1              — 3-layer CNN + 2-layer FC, LazyBatchNorm throughout
    ├── DQN_V1_Mini         — 3-layer CNN + 1-layer FC (smaller debug network)
    ├── DQN_MultiBranch_Mini — GoogLeNet-style: stem + 2 Inception blocks + FC(256)
    └── DQN_AnyNet_Mini     — ResNeXt-style: stem + 2 stages × 2 blocks + FC(256)

To add a new architecture: subclass DecomposedQNetwork, implement forward(),
then register it in experiment._NETWORK_REGISTRY.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class DecomposedQNetwork(nn.Module):
    """Abstract base for Q-functions decomposed into independent per-action heads.

    All subclasses share the same interface:
      - __init__(num_outputs, stack_size)
      - forward(image: (B, stack_size, H, W)) -> (B, num_outputs)

    Training only propagates loss through heads whose action was active in that frame.
    """

    def __init__(self, num_outputs: int, stack_size: int):
        super().__init__()
        self.num_outputs = num_outputs
        self.stack_size = stack_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DQN_V1(DecomposedQNetwork):
    """3-layer CNN + 2-layer FC with LazyBatchNorm.

    Input:  (B, stack_size, 84, 84) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        # --- 3-layer CNN for visual + temporal features ---
        # Input: stack_size x 84 x 84
        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=1), nn.LazyBatchNorm2d(),  # -> 32 x 20 x 20
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 9 x 9
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 7 x 7
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- State embedding ---
        self.state_fc = nn.Sequential(
            nn.LazyLinear(4096), nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(256), nn.LazyBatchNorm1d(),
            nn.ReLU(),
        )

        # --- Per-action Q heads (binary keys + mouse axes) ---
        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)             # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)      # (B, 3136)
        state = self.state_fc(x)        # (B, 256)
        return self.action_heads(state) # (B, num_outputs)


class DQN_V1_Mini(DecomposedQNetwork):
    """Mini debug CNN for iteration.

    Input:  (B, stack_size, 84, 84) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        # --- 3-layer CNN for visual + temporal features ---
        # Input: stack_size x 84 x 84
        self.cnn = nn.Sequential(
            nn.Conv2d(stack_size, 32, kernel_size=8, stride=4, padding=1), nn.LazyBatchNorm2d(),  # -> 32 x 20 x 20
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 9 x 9
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.LazyBatchNorm2d(),          # -> 64 x 7 x 7
            nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- State embedding ---
        self.state_fc = nn.Sequential(
            nn.LazyLinear(256), nn.LazyBatchNorm1d(),
            nn.ReLU(),
        )

        # --- Per-action Q heads (binary keys + mouse axes) ---
        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn(image)             # (B, 64, 7, 7)
        x = x.view(x.size(0), -1)      # (B, 3136)
        state = self.state_fc(x)        # (B, 256)
        return self.action_heads(state) # (B, num_outputs)

### ResNeXt related architecture
# To test power of Residual Blocks

class ResNeXtBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1) # KEY: IMAGE DIMENSIONS STAY CONSTANT WITHIN BLOCK (AS WELL AS BETWEEN BLOCKS)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        if use_1x1conv:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

class DQN_AnyNet(DecomposedQNetwork):
    def __init__(self, num_outputs, arch, stem_channels):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i,s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.LazyLinear(num_outputs)
        ))

    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU()
        )
    
    def stage(self, depth, num_channels, groups, bot_mul):
        blk = []
        for i in range(depth):
            if i == 0:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv=True, strides=2)) # 1st block uses downsampling on separate 1x1 kernel
            else:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul))
        return nn.Sequential(*blk)

    def forward(self, image: torch.Tensor):
        return self.net(image)


class DQN_AnyNet_Mini(DecomposedQNetwork):
    """Mini ResNeXt-based network: stem + 2 stages × 2 ResNeXtBlocks + FC(256).

    Roughly matches DQN_V1_Mini in parameter count (~280K at 224×224, stack=20).

    Input:  (B, stack_size, 224, 224) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        # stem: halve spatial dims, project to 32 channels
        self.stem = nn.Sequential(
            nn.LazyConv2d(32, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU(),
        )

        # stage1: 32 → 128ch, stride=2 on first block  (bot=64, conv2 groups=8)
        self.stage1 = nn.Sequential(
            ResNeXtBlock(128, groups=8, bot_mul=0.5, use_1x1conv=True, strides=2),
            ResNeXtBlock(128, groups=8, bot_mul=0.5),
        )

        # stage2: 128 → 256ch, stride=2 on first block  (bot=128, conv2 groups=16)
        self.stage2 = nn.Sequential(
            ResNeXtBlock(256, groups=8, bot_mul=0.5, use_1x1conv=True, strides=2),
            ResNeXtBlock(256, groups=8, bot_mul=0.5),
        )

        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        self.state_fc = nn.Sequential(
            nn.LazyLinear(256), nn.LazyBatchNorm1d(), nn.ReLU(),
        )

        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.stem(image)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.pool(x)
        state = self.state_fc(x)
        return self.action_heads(state)


### arch example:
# depths, channels = (4,6), (32,80)
# super().__init__(
#     (4, 32, groups, bot_mul),
#         (6, 80, groups, bot_mul)),
#         stem_channels, lr, num_classes
# )

### Multi-branch/GoogLeNet related architecture
# To test power of multi branch CNNs
class Inception(nn.Module):
    # c1--c4 specify the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1,b2,b3,b4), dim=1)


class DQN_MultiBranch(DecomposedQNetwork):
    def b1(): # block 1
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2():
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b3():
        return nn.Sequential(Inception(64, (96, 128), (16,32), 32),
            Inception(128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b4():
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
            Inception(160, (112, 224), (24,64), 64),
            Inception(128, (128, 2560), (24, 64), 64),
            Inception(112, (144, 288), (32, 64), 64),
            Inception(256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b5():
        return nn.Sequential(Inception(256, (160,320), (32, 128), 128),
                            Inception(384, (192, 384), (48, 128), 128),
                            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    def __init__(self, num_classes):
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(), self.b5(), nn.LazyLinear(num_classes))

    def forward(self, image: torch.Tensor):
        return self.net(image)


class DQN_MultiBranch_Mini(DecomposedQNetwork):
    """Mini GoogLeNet-style network: stem + 2 Inception blocks + FC(256).

    Roughly matches DQN_V1_Mini in parameter count (~259K at 224×224, stack=20).

    Input:  (B, stack_size, 224, 224) stacked grayscale frames
    Output: (B, num_outputs) Q-value score per action dimension
    """

    def __init__(self, num_outputs: int, stack_size: int = 4):
        super().__init__(num_outputs, stack_size)

        self.net = nn.Sequential(
            # Block 1 (stem): large-kernel conv + pool  → 56×56 × 32ch
            nn.LazyConv2d(32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 2: bottleneck pair + pool  → 28×28 × 64ch
            nn.LazyConv2d(32, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 3: small Inception  → 128ch, then pool → 14×14
            Inception(32, (32, 64), (8, 16), 16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Block 4: larger Inception  → 256ch at 14×14
            Inception(64, (64, 128), (16, 32), 32),
            # Global average pool
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
        )

        self.state_fc = nn.Sequential(
            nn.LazyLinear(256), nn.LazyBatchNorm1d(), nn.ReLU(),
        )

        self.action_heads = nn.Linear(256, num_outputs)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.net(image)
        state = self.state_fc(x)
        return self.action_heads(state)