#%%
import pickle

# Load the VecNormalize .pkl file
with open("/lustre/guillant/green-dcc/vec_normalize_56JL3O.pkl", "rb") as file:
    vec_normalize_data = pickle.load(file)

# Extract observation and reward normalization stats
obs_mean = vec_normalize_data.obs_rms.mean
obs_std = vec_normalize_data.obs_rms.var ** 0.5  # Standard deviation is the square root of variance
reward_mean = vec_normalize_data.ret_rms.mean
reward_std = vec_normalize_data.ret_rms.var ** 0.5

print(f"Observation Mean: {obs_mean}, Observation Std: {obs_std}")
print(f"Reward Mean: {reward_mean}, Reward Std: {reward_std}")


#%%


#%%