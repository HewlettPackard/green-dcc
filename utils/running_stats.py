import numpy as np

# --- Running Stats ---
class RunningStats:
    # (Identical to the corrected RunningStats class defined previously)
    def __init__(self, shape=(), eps=1e-5): self.mean=np.zeros(shape,dtype=np.float64); self.var=np.ones(shape,dtype=np.float64); self.count=eps
    def update(self, x):
        x=np.asarray(x,dtype=np.float64)
        if x.ndim==0: x=x[np.newaxis]
        if x.shape[0]==0: return
        batch_mean, batch_var = np.mean(x,axis=0), np.var(x,axis=0); batch_count=x.shape[0]
        delta=batch_mean-self.mean; tot_count=self.count+batch_count; new_mean=self.mean+delta*batch_count/tot_count
        m_a=self.var*self.count; m_b=batch_var*batch_count; m2=m_a+m_b+np.square(delta)*self.count*batch_count/tot_count
        new_var=m2/tot_count; self.mean, self.var, self.count = new_mean, new_var, tot_count
    def normalize(self, x): x=np.asarray(x); std=np.sqrt(np.maximum(self.var,1e-6)); return (x-self.mean)/std
    def get_state(self): return {'mean':self.mean, 'var':self.var, 'count':self.count}
    def set_state(self, state): self.mean, self.var, self.count = state['mean'], state['var'], state['count']
