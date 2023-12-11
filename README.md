# rl-agent-self-driving-carla
Training a Reinforcement Learning Agent to Drive a Car Autonomously in CARLA Simulator. Repository for term project of the MSML-642 Robotics class.

**Running the Agent**

1. Install CARLA 0.9.15 [available at: https://github.com/carla-simulator/carla/releases/tag/0.9.15/]
2. Create .carla virtual environment using conda [Follow instructions in virtual_environment.md]
3. Run CarlaUE4.exe in the command line
4. In another command line window run `$python carla_dqn_trainer_v6.py`. Make sure that `carla_gym_v6` is installed in the current working directory.
5. If the simulation crashes, copy the latest file with the netkors' weights into a fila named 'state_dict_experiment_%' (where % is the experiment ID or number), then specify the path to the both the target network as well as the policy network weigths in the `carla_dqn_trainer_continue_v6.py` file. Next, run `$python carla_dqn_trainer_continue_v6.py` to continue training
6. To analyze the experiment's results save the output in the command line window into a text file. Use the `trainingAnalysis.ipynb` notebook to run tests abot the experiment
