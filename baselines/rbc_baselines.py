import numpy as np

class RBCBaselines:
    
    def __init__(self, env):
        self.env = env
        self.datacenters = env.datacenters
        self.datacenter_ids = list(self.datacenters.keys())

    def fair_workload_distribution(self):
        """Distribute workloads in a way that minimizes deviation across datacenters."""
        
        # Get current workloads
        workloads = [self.datacenters[dc].workload_m.get_current_workload() for dc in self.datacenters]
        
        # Calculate average workload
        avg_workload = sum(workloads) / len(workloads)

        actions = np.zeros(3)

        # Transfer workload to datacenters with workloads below the average
        for i in range(len(workloads)):
            for j in range(len(workloads)):
                if i != j:
                    workload_to_move = workloads[i] - avg_workload
                    if workload_to_move > 0:
                        if i == 0 and j == 1:
                            actions[0] = workload_to_move / workloads[i]
                        elif i == 0 and j == 2:
                            actions[1] = workload_to_move / workloads[i]
                        elif i == 1 and j == 0:
                            actions[0] = -workload_to_move / workloads[i]
                        elif i == 1 and j == 2:
                            actions[2] = workload_to_move / workloads[i]
                        elif i == 2 and j == 0:
                            actions[1] = -workload_to_move / workloads[i]
                        elif i == 2 and j == 1:
                            actions[2] = -workload_to_move / workloads[i]

        actions = np.clip(actions, -1, 1)
        
        return actions

    def multi_step_greedy(self):
        """Perform multi-step greedy workload transfers based on carbon intensity."""
        
        # Step 1: Rank data centers by their carbon intensity
        hier_obs = self.env.get_original_observation()
        carbon_intensities = [hier_obs[dc]['ci'][0] for dc in self.env.datacenters]
        ranked_indices = np.argsort(carbon_intensities)
        
        # Initialize the action array
        actions = np.zeros(3)
        
        # Step 2: First transfer - Highest to Lowest carbon intensity
        highest_idx = ranked_indices[-1]
        lowest_idx = ranked_indices[0]
        actions = self._compute_greedy_action(actions, highest_idx, lowest_idx, factor=1)

        # Step 3: Second transfer - Second Highest to Lowest carbon intensity
        second_highest_idx = ranked_indices[-2]
        actions = self._compute_greedy_action(actions, second_highest_idx, lowest_idx, factor=0.99)
        
        # Step 4: Third transfer - Highest to Second Lowest carbon intensity
        second_lowest_idx = ranked_indices[1]
        actions = self._compute_greedy_action(actions, highest_idx, second_lowest_idx, factor=0.98)
        
        return actions

    def _compute_greedy_action(self, actions, sender_idx, receiver_idx, factor=1.0):
        """Helper function to compute greedy action between two datacenters."""
        if sender_idx == 0 and receiver_idx == 1:
            actions[0] = 1.0*factor  # Transfer from DC1 to DC2
        elif sender_idx == 0 and receiver_idx == 2:
            actions[1] = 1.0*factor  # Transfer from DC1 to DC3
        elif sender_idx == 1 and receiver_idx == 0:
            actions[0] = -1.0*factor  # Transfer from DC2 to DC1
        elif sender_idx == 1 and receiver_idx == 2:
            actions[2] = 1.0*factor  # Transfer from DC2 to DC3
        elif sender_idx == 2 and receiver_idx == 0:
            actions[1] = -1.0*factor  # Transfer from DC3 to DC1
        elif sender_idx == 2 and receiver_idx == 1:
            actions[2] = -1.0*factor  # Transfer from DC3 to DC2
            
        return actions