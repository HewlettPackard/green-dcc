from harl.common.base_logger import BaseLogger

import numpy as np

class DCRLLogger(BaseLogger):
    def get_task_name(self):
        """Specific implementation to return the task name based on the environment args."""
        return f"{self.env_args['location']}-discrete"

    def episode_init(self, episode):
        """Initialize metrics at the beginning of each episode."""
        super().episode_init(episode)
        self.metrics = {
            "net_energy_sum": 0,
            "ite_power_sum": 0,
            'ct_power_sum': 0,
            'chiller_power_sum': 0,
            "hvac_power_sum": 0,
            "CO2_footprint_sum": 0,
            "water_usage": 0,
            "step_count": 0,
            "load_left": 0,
            "ls_tasks_in_queue": 0,
            "ls_tasks_dropped": 0,
            "instantaneous_net_energy": [],
            'hvac_power_on_used': [],
            'PUE': 0,
        }
        self.is_off_policy = False


    def eval_init(self):
        """Initialize metrics at the beginning of each episode."""
        super().eval_init()
        self.eval_metrics = {
            "net_energy_sum": 0,
            "ite_power_sum": 0,
            'ct_power_sum': 0,
            'chiller_power_sum': 0,
            "hvac_power_sum": 0,
            "CO2_footprint_sum": 0,
            "water_usage": 0,
            "step_count": 0,
            "load_left": 0,
            "ls_tasks_in_queue": 0,
            "ls_tasks_dropped": 0,
            "instantaneous_net_energy": [],
            'hvac_power_on_used': [],
            'PUE': 0,
        }
        self.is_off_policy = False
        
    def eval_init_off_policy(self, total_num_steps):
        """Initialize metrics at the beginning of each episode."""
        super().eval_init_off_policy(total_num_steps)
        self.eval_metrics = {
            "net_energy_sum": 0,
            "ite_power_sum": 0,
            'ct_power_sum': 0,
            'chiller_power_sum': 0,
            "hvac_power_sum": 0,
            "CO2_footprint_sum": 0,
            "water_usage": 0,
            "step_count": 0,
            "load_left": 0,
            "ls_tasks_in_queue": 0,
            "ls_tasks_dropped": 0,
            "instantaneous_net_energy": [],
            'hvac_power_on_used': [],
            'PUE': 0,
        }
        self.is_off_policy = True
        
    def per_step(self, data):
        """Capture and update metrics per step."""
        super().per_step(data)
        obs, _, rewards, dones, infos, _, _, _, _, _, _ = data
        dones_env = np.all(dones, axis=1)
        
        for i in range(len(infos)):  # Assuming infos are structured with one dict per environment
            self.metrics["net_energy_sum"] += infos[i][2].get("bat_total_energy_with_battery_KWh", 0)
            self.metrics["CO2_footprint_sum"] += infos[i][2].get("bat_CO2_footprint", 0)
            self.metrics["water_usage"] += infos[i][1].get("dc_water_usage", 0)
            self.metrics["load_left"] += infos[i][0].get("ls_unasigned_day_load_left", 0)
            self.metrics["ls_tasks_in_queue"] += infos[i][0].get("ls_tasks_in_queue", 0)
            self.metrics["ls_tasks_dropped"] += infos[i][0].get("ls_tasks_dropped", 0)
            self.metrics["ite_power_sum"] += infos[i][1].get("dc_ITE_total_power_kW", 0)
            self.metrics['ct_power_sum'] += infos[i][1].get("dc_CT_total_power_kW", 0)  # Added
            self.metrics['chiller_power_sum'] += infos[i][1].get("dc_Compressor_total_power_kW", 0)  # Added
            self.metrics["hvac_power_sum"] += infos[i][1].get("dc_HVAC_total_power_kW", 0)
            
            if infos[i][1].get("dc_HVAC_total_power_kW", 0)> 0:
                self.metrics["hvac_power_on_used"].append(infos[i][1].get("dc_HVAC_total_power_kW", 0))

            self.metrics["step_count"] += 1

    def eval_per_step(self, eval_data):
        """Capture and update metrics per step during evaluation."""
        super().eval_per_step(eval_data)
        _, _, _, _, eval_infos, _ = eval_data
        
        for i in range(len(eval_infos)):  # Assuming eval_infos are structured with one dict per environment
            self.eval_metrics["net_energy_sum"] += eval_infos[i][2].get("bat_total_energy_with_battery_KWh", 0)
            self.eval_metrics["CO2_footprint_sum"] += eval_infos[i][2].get("bat_CO2_footprint", 0)
            self.eval_metrics["water_usage"] += eval_infos[i][1].get("dc_water_usage", 0)
            self.eval_metrics["load_left"] += eval_infos[i][0].get("ls_unasigned_day_load_left", 0)
            self.eval_metrics["ls_tasks_in_queue"] += eval_infos[i][0].get("ls_tasks_in_queue", 0)
            self.eval_metrics["ls_tasks_dropped"] += eval_infos[i][0].get("ls_tasks_dropped", 0)
            self.eval_metrics["ite_power_sum"] += eval_infos[i][1].get("dc_ITE_total_power_kW", 0)
            self.eval_metrics["ct_power_sum"] += eval_infos[i][1].get("dc_CT_total_power_kW", 0)  # Added
            self.eval_metrics["chiller_power_sum"] += eval_infos[i][1].get("dc_Compressor_total_power_kW", 0)  # Added
            self.eval_metrics["hvac_power_sum"] += eval_infos[i][1].get("dc_HVAC_total_power_kW", 0)

            if eval_infos[i][1].get("dc_HVAC_total_power_kW", 0)> 0:
                self.eval_metrics["hvac_power_on_used"].append(eval_infos[i][1].get("dc_HVAC_total_power_kW", 0))
            
            self.eval_metrics["step_count"] += 1


    def episode_log(self, actor_train_infos, critic_train_info, actor_buffer, critic_buffer):
        """Calculate and log metrics at the end of the episode."""
        super().episode_log(actor_train_infos, critic_train_info, actor_buffer, critic_buffer)
        
        if self.metrics["step_count"] > 0:
            average_net_energy = self.metrics["net_energy_sum"] / self.metrics["step_count"]
            average_ite_power = self.metrics["ite_power_sum"] / self.metrics["step_count"]
            average_ct_power = self.metrics["ct_power_sum"] / self.metrics["step_count"]  # Added
            average_chiller_power = self.metrics["chiller_power_sum"] / self.metrics["step_count"]  # Added
            average_hvac_power = self.metrics["hvac_power_sum"] / self.metrics["step_count"]

            average_CO2_footprint = self.metrics["CO2_footprint_sum"] / self.metrics["step_count"]
            total_water_usage = self.metrics["water_usage"]
            total_load_left = self.metrics["load_left"]
            total_tasks_in_queue = self.metrics["ls_tasks_in_queue"]
            total_tasks_dropped = self.metrics["ls_tasks_dropped"]
        else:
            average_net_energy = 0
            average_ite_power = 0
            average_hvac_power = 0
            average_CO2_footprint = 0
            total_water_usage = 0
            total_load_left = 0
            total_tasks_in_queue = 0
            total_tasks_dropped = 0
        
        if len(self.metrics["hvac_power_on_used"]) > 0:
            self.writter.add_scalar("metrics/Average HVAC Power on use", np.mean(self.metrics["hvac_power_on_used"]), self.total_num_steps)
            self.writter.add_scalar("metrics/Max HVAC Power on use", np.max(self.metrics["hvac_power_on_used"]), self.total_num_steps)
            self.writter.add_scalar("metrics/Percentile 90% HVAC Power on use", np.percentile(self.metrics["hvac_power_on_used"], 90), self.total_num_steps)
            
        # Log the calculated metrics to TensorBoard or similar
        self.writter.add_scalar("metrics/Average Net Energy", average_net_energy, self.total_num_steps)
        self.writter.add_scalar("metrics/Average ITE Power", average_ite_power, self.total_num_steps)
        self.writter.add_scalar("metrics/Average CT Power", average_ct_power, self.total_num_steps)  # Added
        self.writter.add_scalar("metrics/Average Chiller Power", average_chiller_power, self.total_num_steps)  # Added
        self.writter.add_scalar("metrics/Average HVAC Power", average_hvac_power, self.total_num_steps)

        PUE = 1 + average_hvac_power/average_ite_power
        self.writter.add_scalar("metrics/Average PUE", PUE, self.total_num_steps)

        self.writter.add_scalar("metrics/Average CO2 Footprint", average_CO2_footprint, self.total_num_steps)
        self.writter.add_scalar("metrics/Total Water Usage", total_water_usage, self.total_num_steps)
        self.writter.add_scalar("metrics/Total Tasks in Queue", total_tasks_in_queue, self.total_num_steps)
        self.writter.add_scalar("metrics/Total Tasks Dropped", total_tasks_dropped, self.total_num_steps)

        # Optionally, log to the console or to a file
        print(f"Episode {self.episode}: Avg Net Energy={average_net_energy:.3f}, Avg CO2={average_CO2_footprint:.3f}, Water Usage={total_water_usage:.3f}")
        print(f"Tasks in Queue={total_tasks_in_queue:.3f}, Tasks Dropped={total_tasks_dropped:.3f}")

        # Reset metrics for the next episode
        self.metrics = {
            "net_energy_sum": 0,
            "ite_power_sum": 0,
            'ct_power_sum': 0,
            'chiller_power_sum': 0,
            "hvac_power_sum": 0,
            "CO2_footprint_sum": 0,
            "water_usage": 0,
            "step_count": 0,
            "load_left": 0,
            "ls_tasks_in_queue": 0,
            "ls_tasks_dropped": 0,
            "instantaneous_net_energy": [],
            'hvac_power_on_used': []
        }

    def eval_log(self, eval_episode):
        """Log evaluation information at the end of an evaluation session."""
        super().eval_log(eval_episode)
        
        if self.eval_metrics["step_count"] > 0:
            average_net_energy = self.eval_metrics["net_energy_sum"] / self.eval_metrics["step_count"]
            average_ite_power = self.eval_metrics["ite_power_sum"] / self.eval_metrics["step_count"]
            average_ct_power = self.eval_metrics["ct_power_sum"] / self.eval_metrics["step_count"]  # Added
            average_chiller_power = self.eval_metrics["chiller_power_sum"] / self.eval_metrics["step_count"]  # Added
            average_hvac_power = self.eval_metrics["hvac_power_sum"] / self.eval_metrics["step_count"]

            average_CO2_footprint = self.eval_metrics["CO2_footprint_sum"] / self.eval_metrics["step_count"]
            total_water_usage = self.eval_metrics["water_usage"]
            total_load_left = self.eval_metrics["load_left"]
            total_tasks_in_queue = self.eval_metrics["ls_tasks_in_queue"]
            total_tasks_dropped = self.eval_metrics["ls_tasks_dropped"]
        else:
            average_net_energy = 0
            average_ite_power = 0
            average_hvac_power = 0
            average_CO2_footprint = 0
            total_water_usage = 0
            total_load_left = 0
            total_tasks_in_queue = 0
            total_tasks_dropped = 0
        
        if len(self.eval_metrics["hvac_power_on_used"]) > 0:
            self.writter.add_scalar("eval_metrics/Average HVAC Power on use", np.mean(self.eval_metrics["hvac_power_on_used"]), self.total_num_steps)
            self.writter.add_scalar("eval_metrics/Max HVAC Power on use", np.max(self.eval_metrics["hvac_power_on_used"]), self.total_num_steps)
            self.writter.add_scalar("eval_metrics/Percentile 90% HVAC Power on use", np.percentile(self.eval_metrics["hvac_power_on_used"], 90), self.total_num_steps)

        # Log the calculated eval_metrics to TensorBoard or similar
        self.writter.add_scalar("eval_metrics/Average Net Energy", average_net_energy, self.total_num_steps)
        self.writter.add_scalar("eval_metrics/Average ITE Power", average_ite_power, self.total_num_steps)
        self.writter.add_scalar("eval_metrics/Average CT Power", average_ct_power, self.total_num_steps)  # Added
        self.writter.add_scalar("eval_metrics/Average Chiller Power", average_chiller_power, self.total_num_steps)  # Added
        self.writter.add_scalar("eval_metrics/Average HVAC Power", average_hvac_power, self.total_num_steps)
        
        PUE = 1 + average_hvac_power/average_ite_power
        self.writter.add_scalar("eval_metrics/Average PUE", PUE, self.total_num_steps)
        
        self.writter.add_scalar("eval_metrics/Average CO2 Footprint", average_CO2_footprint, self.total_num_steps)
        self.writter.add_scalar("eval_metrics/Total Water Usage", total_water_usage, self.total_num_steps)
        self.writter.add_scalar("eval_metrics/Total Tasks in Queue", total_tasks_in_queue, self.total_num_steps)
        self.writter.add_scalar("eval_metrics/Total Tasks Dropped", total_tasks_dropped, self.total_num_steps)

        # Optionally, log to the console or to a file
        if self.is_off_policy:
            self.episode = (int(self.total_num_steps)//self.algo_args["train"]["episode_length"]//self.algo_args["train"]["n_rollout_threads"])
        print(f"Episode {self.episode}: Avg Net Energy={average_net_energy:.3f}, Avg CO2={average_CO2_footprint:.3f}, Water Usage={total_water_usage:.3f}")
        print(f"Tasks in Queue={total_tasks_in_queue:.3f}, Tasks Dropped={total_tasks_dropped:.3f}")

        # Reset metrics for the next episode
        self.eval_metrics = {
            "net_energy_sum": 0,
            "ite_power_sum": 0,
            'ct_power_sum': 0,
            'chiller_power_sum': 0,
            "hvac_power_sum": 0,
            "CO2_footprint_sum": 0,
            "water_usage": 0,
            "step_count": 0,
            "load_left": 0,
            "ls_tasks_in_queue": 0,
            "ls_tasks_dropped": 0,
            "instantaneous_net_energy": [],
            'hvac_power_on_used': []
        }


    def close(self):
        """Close the logger."""
        super().close()
