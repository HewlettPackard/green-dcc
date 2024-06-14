===============
Getting Started
===============

1. **Setup Configuration**

Customize the :code:`dc_config_dc<N>.json` file (where :code:`N=1,2,..` represent the data center identifier) to specify the environment settings for each of the data centers. To learn more about the DC parameters you can customize, check :ref:`dcconf_ref`

2. **Environment Configuration**

The main environment for wrapping the environments is :code:`hierarchical_env.py`, which reads configurations from the :code:`EnvConfig` class and manages the external data sources for each of the data centers using managers for weather, carbon intensity, and workload. For instructions how to customize the enviroment configuration, check :ref:`mainconf_ref`

3. **Train Example:**

.. Specify :code:`location` inside :code:`harl.configs.envs_cfgs.sustaindc.yaml`. Specify other algorithm hyperparameteres in :code:`harl.configs.algos_cfgs.happo.yaml`. User can also specify the choice of reinforcement learning vs baseline agents in the :code:`happo.yaml`
   
To run a basic experiment, use the following command:
   
.. code-block:: bash
      
      python train_truly_hierarchical.py

This will start a simulation with the default configuration. The results will be saved in :code:`results/` output directory.

4. **Running in background mode**

If you want to run the |F| framework in background mode use:

.. code-block:: bash

    nohup python PYTHON_SCRIPT > OUTPUT_FILE.txt  &

where :code:`PYTHON_SCRIPT` is the script you want to run (e.g., :code:`train_truly_hierarchical.py`) and :code:`OUTPUT_FILE` is the name of the file that will contain the output (e.g. :code:`latest_experiment_output`)

5. **Monitor the results**

To visualize the experiments while they are running, you can launch TensorBoard. Open a new terminal, navigate to the "code"`results/` directory, and run the following command:

Example:

.. code-block:: bash

    tensorboard --logdir ./test

This will start a TensorBoard server, and you can view the experiment visualizations by opening a web browser and navigating to :code:`http://localhost:6006`