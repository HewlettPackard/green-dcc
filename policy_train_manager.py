# policy_train_manager.py

class PolicyTrainManager:
    def __init__(self):
        self.current_iteration = 0

    def policies_to_train(self, policy_id, batch=None):
        if self.current_iteration < 100:
            # First 100 iterations: train only low-level policies
            print(f'Training only low-level policies. Current iteration: {self.current_iteration}')
            return policy_id in ['DC1_ls_policy', 'DC2_ls_policy', 'DC3_ls_policy']
        elif self.current_iteration < 200:
            # Next 100 iterations: train only high-level policy
            print(f'Training only high-level policy. Current iteration: {self.current_iteration}')
            return policy_id == 'high_level_policy'
        else:
            print(f'Training all policies. Current iteration: {self.current_iteration}')
            # After that, train all policies
            return True  # Train all policies

    def update_iteration(self, iteration):
        self.current_iteration = iteration
