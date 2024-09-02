import numpy as np

class RBCBaselines:
    
    def __init__(self, env):
        self.env = env
        self.datacenters = env.datacenters
        self.datacenter_ids = list(self.datacenters.keys())

    def equal_workload_distribution(self):
        """Distribute the workload equally between datacenters."""
        
        # Step 1: Calculate the total workload across all datacenters
        total_workload = sum([self.datacenters[dc].workload_m.get_current_workload() for dc in self.datacenters])
        
        # Step 2: Calculate the target workload for each datacenter
        num_datacenters = len(self.datacenters)
        target_workload = total_workload / num_datacenters
        
        # Step 3: Calculate workload differences
        workload_diffs = []
        for dc, workload in self.datacenters.items():
            workload_diffs.append(workload.workload_m.get_current_workload() - target_workload)
        
        # # Step 4: Create action array to balance the workload
        # actions = np.zeros(3)
        
        # # Transfer between DC1 and DC2
        # if workload_diffs[0] > 0 and workload_diffs[1] < 0:
        #     actions[0] = min(workload_diffs[0], abs(workload_diffs[1]))
        # elif workload_diffs[0] < 0 and workload_diffs[1] > 0:
        #     actions[0] = -min(abs(workload_diffs[0]), workload_diffs[1])
        
        # # Transfer between DC1 and DC3
        # if workload_diffs[0] > 0 and workload_diffs[2] < 0:
        #     actions[1] = min(workload_diffs[0], abs(workload_diffs[2]))
        # elif workload_diffs[0] < 0 and workload_diffs[2] > 0:
        #     actions[1] = -min(abs(workload_diffs[0]), workload_diffs[2])
        
        # # Transfer between DC2 and DC3
        # if workload_diffs[1] > 0 and workload_diffs[2] < 0:
        #     actions[2] = min(workload_diffs[1], abs(workload_diffs[2]))
        # elif workload_diffs[1] < 0 and workload_diffs[2] > 0:
        #     actions[2] = -min(abs(workload_diffs[1]), workload_diffs[2])
        
        # # Normalize actions to be within the expected range [-1, 1]
        # actions = actions / np.max(np.abs(actions)) if np.max(np.abs(actions)) > 0 else actions
        
        # Step 4: Create action array to balance the workload
        actions = np.zeros(3)
        
        
        # Transfer between DC1 and DC2
        if workload_diffs[0] > 0 and workload_diffs[1] < 0:
            actions[0] = min(workload_diffs[0], abs(workload_diffs[1])) / self.datacenters[self.datacenter_ids[0]].workload_m.get_current_workload()
        elif workload_diffs[0] < 0 and workload_diffs[1] > 0:
            actions[0] = -min(abs(workload_diffs[0]), workload_diffs[1]) / self.datacenters[self.datacenter_ids[1]].workload_m.get_current_workload()
        
        # Transfer between DC1 and DC3
        if workload_diffs[0] > 0 and workload_diffs[2] < 0:
            actions[1] = min(workload_diffs[0], abs(workload_diffs[2])) / self.datacenters[self.datacenter_ids[0]].workload_m.get_current_workload()
        elif workload_diffs[0] < 0 and workload_diffs[2] > 0:
            actions[1] = -min(abs(workload_diffs[0]), workload_diffs[2]) / self.datacenters[self.datacenter_ids[2]].workload_m.get_current_workload()
        
        # Transfer between DC2 and DC3
        if workload_diffs[1] > 0 and workload_diffs[2] < 0:
            actions[2] = min(workload_diffs[1], abs(workload_diffs[2])) / self.datacenters[self.datacenter_ids[1]].workload_m.get_current_workload()
        elif workload_diffs[1] < 0 and workload_diffs[2] > 0:
            actions[2] = -min(abs(workload_diffs[1]), workload_diffs[2]) / self.datacenters[self.datacenter_ids[2]].workload_m.get_current_workload()
        
        
        # Ensure actions are within the expected range [-1, 1]
        actions = np.clip(actions, -1, 1)
        
        return actions
