Benchmarking Algorithms
-----------------------

Sustain-Cluster currently employs the Soft Actor–Critic (SAC) algorithm, an off‐policy actor–critic method that maximizes a stochastic policy’s entropy‐augmented return. SAC alternates between two interleaved updates:

**Policy (Actor) Update**

.. math::
   :label: sac-policy-update

   J_{\pi}(\phi)
   = \mathbb{E}_{s_t \sim \mathcal{D},\,a_t \sim \pi_\phi}
     \Bigl[\,
       \alpha\,\log \pi_\phi(a_t\mid s_t)
       \;-\;
       Q_\theta(s_t, a_t)
     \Bigr] \,.

Here:

- :math:`\pi_\phi(a\mid s)` denotes the stochastic policy parameterized by :math:`\phi`.
- :math:`Q_\theta(s,a)` is the soft Q-function parameterized by :math:`\theta`.
- :math:`\alpha > 0` is the temperature coefficient balancing exploration (via entropy) and exploitation.
- :math:`\mathcal{D}` is the replay buffer of past transitions.

Minimizing :math:`J_{\pi}` encourages the policy to choose actions that both achieve high soft‐Q values and maintain high entropy, yielding robust exploration.

**Q-Function (Critic) Update**

.. math::
   :label: sac-q-update

   J_{Q}(\theta)
   = \mathbb{E}_{(s_t,a_t)\sim \mathcal{D}}
     \Bigl[\,
       \tfrac{1}{2}\bigl(Q_\theta(s_t,a_t) - y_t\bigr)^{2}
     \Bigr]

where the soft‐Bellman backup target is

.. math::

   y_t 
   = 
   r(s_t,a_t) 
   \;+\; 
   \gamma\,
   \mathbb{E}_{%
     \substack{s_{t+1}\sim p(\cdot\mid s_t,a_t)\\a_{t+1}\sim \pi_\phi}}
   \Bigl[
     Q_{\bar\theta}(s_{t+1},a_{t+1})
     \;-\;
     \alpha\,\log \pi_\phi(a_{t+1}\mid s_{t+1})
   \Bigr] \,.

Here:

- :math:`r(s_t,a_t)` is the immediate reward at step :math:`t`.
- :math:`\gamma \in [0,1)` is the discount factor.
- :math:`Q_{\bar\theta}` is a target network with parameters :math:`\bar\theta`, updated via Polyak averaging to stabilize training.
- :math:`p(s_{t+1}\mid s_t,a_t)` is the environment’s transition probability.

By fitting :math:`Q_\theta` to these targets, the critic learns to approximate the entropy‐regularized state‐action value. The temperature term :math:`\alpha` again trades off between reward maximization and policy entropy.

Overall, SAC proceeds by sampling minibatches from the replay buffer, performing a gradient descent step on the critic loss :math:`J_Q`, then updating the policy parameters :math:`\phi` to minimize :math:`J_{\pi}`, and finally updating the target network parameters :math:`\bar\theta` towards :math:`\theta`. This off‐policy, entropy‐regularized framework yields both sample efficiency and stable learning.

We are also extending Sustain-Cluster to support on‐policy methods such as Advantage Actor–Critic (A2C), which is currently a work in progress. In future releases, we plan to integrate additional algorithms from the Ray RLlib ecosystem—see the `Ray RLlib documentation <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_—to enable a broader and more rigorous benchmarking suite.
