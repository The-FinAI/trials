import gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MLP(BaseFeaturesExtractor):
    """The basic feature extractor network that adopting MLP."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        asset_num: int = 60,
        time_step: int = 1200,
        input_feature: int = 4,
        hidden_dim: int = 32,
        num_heads: int = 2,
        asset_attention: bool = False,
        **kwargs
    ):
        # Flatten all assets representations as N x M
        super(MLP, self).__init__(
            observation_space,
            asset_num * hidden_dim,
        )
        self.asset_num = asset_num  # N
        self.time_step = time_step  # T
        self.input_feature = input_feature  # F
        self.hidden_dim = hidden_dim  # M
        self.num_heads = num_heads
        self.asset_attention = asset_attention
        if asset_attention:
            self.asset_level_attention = nn.MultiheadAttention(
                self.hidden_dim, self.num_heads, batch_first=True
            )

        self.fnn = nn.Sequential(
            nn.Linear(
                self.time_step * self.input_feature,
                self.hidden_dim,
            ),
            nn.ReLU(),
        )
        self.out = nn.Flatten()

    def forward(self, observations, attention_output=False) -> th.Tensor:
        # Observations as (B x N x T x F) where F is the input feature dim
        asset_features = observations["assets"].reshape(
            -1, self.asset_num, self.time_step * self.input_feature
        )  # N x (T x F)
        output = self.fnn(asset_features)
        if self.asset_attention:
            # B x N x H
            output, asset_attn = self.asset_level_attention(
                output, output, output
            )
        else:
            asset_attn = None
        output = self.out(output)
        if attention_output:
            return output, None, asset_attn

        return output


class TemporalAttention(nn.Module):
    """The attention mechanism for temporal information"""

    def __init__(self, hidden_dim, asset_num, **kwargs):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.fnn = nn.Sequential(
            nn.Linear(
                4 * self.hidden_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(self, batch_size, present_output, history):
        """Attention over history with present output

        :params batch_size: int
        :params present_output: (B x N x 1 x 2H)
        :params history: (B x N x (T - 1) x 2H)
        """
        # (B x N) x 1 x (T - 1)
        attention_scores = th.bmm(present_output, history.permute(0, 2, 1))
        attention_scores = attention_scores / np.sqrt(present_output.shape[-1])

        # (B x N) x 1 x 2H
        attention_output = th.bmm(attention_scores, history)

        # B x N x 4H
        attention_output = th.cat(
            [present_output, attention_output], -1
        ).reshape(batch_size, self.asset_num, 4 * self.hidden_dim)

        # B x N x H
        attention_output = self.fnn(attention_output)

        return attention_output, attention_scores


class GRU(BaseFeaturesExtractor):
    """The extractor network that adopting GRU."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        asset_num: int = 60,
        time_step: int = 1200,
        input_feature: int = 4,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.5,
        asset_attention: bool = False,
        **kwargs
    ):
        # Flatten all assets representations as N x M
        super(GRU, self).__init__(
            observation_space,
            asset_num * hidden_dim,
        )
        self.asset_num = asset_num  # N
        self.time_step = time_step  # T
        self.input_feature = input_feature  # F
        self.hidden_dim = hidden_dim  # M
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.asset_attention = asset_attention
        self.rnn = th.nn.GRU(
            self.input_feature,
            self.hidden_dim,
            self.num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attention = TemporalAttention(
            self.hidden_dim, self.asset_num
        )
        if asset_attention:
            self.asset_level_attention = nn.MultiheadAttention(
                self.hidden_dim, self.num_heads, batch_first=True
            )
        self.fnn = nn.Flatten()

    def forward(self, observations, attention_output=False) -> th.Tensor:
        # Observations as (B x N x T x F) where F is the input feature dim
        asset_features = observations["assets"].reshape(
            -1, self.time_step, self.input_feature
        )  # (B x N) x T x F
        batch_size = asset_features.shape[0] // self.asset_num  # B

        output, _ = self.rnn(asset_features)

        # B x N x H
        output, temporal_attn = self.temporal_attention(
            batch_size, output[:, -1].unsqueeze(1), output[:, :-1]
        )

        if self.asset_attention:
            # B x N x H
            output, asset_attn = self.asset_level_attention(
                output, output, output
            )
        else:
            asset_attn = None

        # B x (N x H)
        output = self.fnn(output)

        if attention_output:
            return output, temporal_attn, asset_attn

        return output


FEATURE_EXTRACTORS = {
    "mlp": MLP,
    "gru": GRU,
}


class TradingLSTM(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super(TradingLSTM, self).__init__(observation_space, features_dim)
        self.feature_size = features_dim
        asset_position_space: gym.spaces.Box = observation_space["position"]
        self.num_layers = num_layers
        self.num_asset_position = int(
            asset_position_space.high[0] - asset_position_space.low[0] + 1
        )
        net_value_space: gym.spaces.Box = observation_space["net_value"]
        self.max_len = net_value_space.shape[0]

        # asset_x: 1
        # asset_y: 1
        rnn_input_size = 8

        self.rnn = th.nn.LSTM(
            rnn_input_size,
            features_dim,
            num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations["mask_len"].size(0)
        device = observations["mask_len"].device
        # [batch_size, max_len, input_size]
        input_features = []

        input_features.extend(
            [
                observations["asset_x"].unsqueeze(2),
                observations["asset_y"].unsqueeze(2),
            ]
        )

        input_features.append(observations["net_value"].unsqueeze(2))

        position = one_hot(
            observations["position"].type(th.int64),
            num_classes=self.num_asset_position,
        ).to(device)
        input_features.append(position)

        hold_indicator = one_hot(
            observations["hold_indicator"].type(th.int64), num_classes=2
        )
        input_features.append(hold_indicator)

        asset_input = th.concat(input_features, dim=2)

        packed_input = pack_padded_sequence(
            input=asset_input,
            lengths=observations["mask_len"].squeeze(1).to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )

        asset_output, (_, _) = self.rnn(
            packed_input,
            (
                th.zeros(
                    self.num_layers,
                    batch_size,
                    self.feature_size,
                    device=device,
                ),
                th.zeros(
                    self.num_layers,
                    batch_size,
                    self.feature_size,
                    device=device,
                ),
            ),
        )

        # [batch_size, max_len, feature_dim]
        asset_output, _ = pad_packed_sequence(asset_output, batch_first=True)

        # Select the output of the current step as the representation of the paired asset
        # [batch_size, 1, feature_dim]
        curr_index = (
            observations["mask_len"]
            .unsqueeze(2)
            .expand(-1, -1, asset_output.size(2))
            - 1
        )
        # [batch_size, feature_dim]
        final_output = th.gather(
            input=asset_output, dim=1, index=curr_index.type(th.int64)
        ).squeeze(1)

        return final_output


class FlattenInput(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        **kwargs
    ):
        super(FlattenInput, self).__init__(
            observation_space,
            features_dim,
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        device = observations["asset_x"].device
        return th.concat(
            [
                observations["asset_x"],
                observations["asset_y"],
                observations["net_value"][:, -1:],
                th.nn.functional.one_hot(
                    observations["position"][:, -1].type(th.int64),
                    num_classes=3,
                )
                .to(device)
                .type(th.float),
            ],
            1,
        )


TRADING_FEATURE_EXTRACTORS = {"lstm": TradingLSTM, "mlp": FlattenInput}
