from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    MultiCategoricalDistribution,
)
from torch import nn


class PairSelectionDistribution(MultiCategoricalDistribution):
    """Custom distribution to allow directly logit output from proxy network"""

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        The layer that directly take the output as the distribution:
        The proxy network should be the same as the logits (flattened)
        You can then get probabilities using a softmax on each sub-space.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Identity(latent_dim, sum(self.action_dims))
        return action_logits


class PairSelectionCateDistribution(CategoricalDistribution):
    """Custom distribution to allow directly logit output from proxy network"""

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        The layer that directly take the output as the distribution:
        The proxy network should be the same as the logits (flattened)
        You can then get probabilities using a softmax on each sub-space.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Identity(latent_dim, self.action_dim)
        return action_logits
