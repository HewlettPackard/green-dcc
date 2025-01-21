# # custom_model.py

# # custom_model.py

# import torch
# from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
# from ray.rllib.utils.annotations import override
# from ray.rllib.models.preprocessors import get_preprocessor
# from ray.rllib.models.modelv2 import restore_original_dimensions


# class CustomTorchModel(TorchModelV2, torch.nn.Module):
#     """
#     A custom Torch model for RLlib's PPO algorithm that includes both
#     actor (policy) and critic (value function) components.
#     """

#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         torch.nn.Module.__init__(self)

#         # Extract hidden layer sizes and activation function from config
#         hidden_sizes = model_config.get("fcnet_hiddens", [64, 64])
#         activation = model_config.get("fcnet_activation", "relu")

#         # Define shared hidden layers
#         layers = []
#         input_size = get_preprocessor(obs_space)(obs_space).size
#         for size in hidden_sizes:
#             layers.append(torch.nn.Linear(input_size, size))
#             if activation.lower() == "relu":
#                 layers.append(torch.nn.ReLU())
#             elif activation.lower() == "tanh":
#                 layers.append(torch.nn.Tanh())
#             else:
#                 raise ValueError(f"Unsupported activation: {activation}")
#             input_size = size

#         self.shared_layers = torch.nn.Sequential(*layers)

#         # Define actor (policy) output layer
#         self.policy_output = torch.nn.Linear(input_size, num_outputs)

#         # Extract policy_type from custom_options
#         policy_type = "low_level" if action_space.shape == (1,) else "high_level"  # Default to 'low_level'

#         # Initialize policy output layer based on policy_type
#         if policy_type == "high_level":
#             # Initialize weights to 0 and bias to 0 for high_level_policy
#             torch.nn.init.zeros_(self.policy_output.weight)
#             torch.nn.init.constant_(self.policy_output.bias, 0.0) # No transfers between datacenters
#         elif policy_type == "low_level":
#             # Initialize weights to 0 and bias to 1 for low_level_policies
#             torch.nn.init.zeros_(self.policy_output.weight)
#             torch.nn.init.constant_(self.policy_output.bias, 1.0) # Compute all of the workload
#         else:
#             raise ValueError(f"Unknown policy_type: {policy_type}")
#         # Define critic (value function) output layer
#         self.value_output = torch.nn.Linear(input_size, 1)

#         # Initialize critic output layer (optional customization)
#         torch.nn.init.xavier_uniform_(self.value_output.weight)
#         torch.nn.init.constant_(self.value_output.bias, 0.0)

#     @override(TorchModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         """
#         Forward pass for the model.
#         """
#         # Pass observations through shared layers
#         x = self.shared_layers(input_dict["obs"].float())

#         # Compute policy logits
#         policy_logits = self.policy_output(x)

#         # Store the output for the policy
#         self._value_out = self.value_output(x).squeeze(1)

#         return policy_logits, state

#     @override(TorchModelV2)
#     def value_function(self):
#         """
#         Returns the value function output.
#         """
#         return self._value_out

# custom_model.py

import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.modelv2 import restore_original_dimensions


class CustomTorchModel(TorchModelV2, torch.nn.Module):
    """
    A custom Torch model for RLlib's PPO algorithm that includes completely separate
    actor (policy) and critic (value function) networks.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)

        # Extract hidden layer sizes and activation function from config
        hidden_sizes = model_config.get("fcnet_hiddens", [64, 64])
        activation = model_config.get("fcnet_activation", "relu")

        # Define shared layers for actor (policy) network
        self.actor_layers = self.build_network(obs_space, hidden_sizes, activation, num_outputs)

        # Define shared layers for critic (value function) network
        self.critic_layers = self.build_network(obs_space, hidden_sizes, activation, 1)

        # Initialize the policy output layer differently for high and low-level agents
        policy_type = "low_level" if action_space.shape == (1,) else "high_level"  # Default to 'low_level'
        self.initialize_actor_output(policy_type)

    def build_network(self, obs_space, hidden_sizes, activation, output_size):
        """
        Builds a feedforward network with the specified hidden layers and output size.
        Includes normalization layers (LayerNorm) for stability.
        """
        layers = []
        input_size = get_preprocessor(obs_space)(obs_space).size

        for size in hidden_sizes:
            layers.append(torch.nn.Linear(input_size, size))
            layers.append(torch.nn.LayerNorm(size))  # Add LayerNorm after each hidden layer

            # Apply the selected activation function
            if activation.lower() == "relu":
                layers.append(torch.nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            input_size = size

        # Add the output layer (either for policy or value function)
        layers.append(torch.nn.Linear(input_size, output_size))
        return torch.nn.Sequential(*layers)

    def initialize_actor_output(self, policy_type):
        """
        Initializes the weights of the policy output layer depending on the policy type.
        """
        # Access the final linear layer in the actor network
        policy_output_layer = self.actor_layers[-1]

        if policy_type == "high_level":
            # Initialize weights to 0 and bias to 0 for high_level_policy
            torch.nn.init.zeros_(policy_output_layer.weight)
            torch.nn.init.constant_(policy_output_layer.bias, 0.0)  # No transfers between datacenters
        elif policy_type == "low_level":
            # Initialize weights to 0 and bias to 1 for low_level_policies
            torch.nn.init.zeros_(policy_output_layer.weight)
            torch.nn.init.constant_(policy_output_layer.bias, 1.0)  # Compute all of the workload
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for the model: computes both the policy logits and value function.
        """
        # Pass observations through actor (policy) network
        policy_logits = self.actor_layers(input_dict["obs"].float())

        # Pass observations through critic (value function) network
        value_out = self.critic_layers(input_dict["obs"].float()).squeeze(1)

        # Store the value function output for later use
        self._value_out = value_out

        return policy_logits, state

    @override(TorchModelV2)
    def value_function(self):
        """
        Returns the value function output.
        """
        return self._value_out
