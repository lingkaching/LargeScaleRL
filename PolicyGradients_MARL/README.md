# Zone-Based-Multiagent-Pathfinding


## Requirement for running the code

* Python 3 (>=3.6)
* Tensorflow (1.7)
* Numpy (1.13)
* Scipy (1.3)
* Pathos
* [Networkx](https://networkx.org/) (>=2.4)

## Running the code
```
cd PythonScripts 
python DCRL.py -t 500 -b 8 -i 500 -d 0.99 -l 0.001 -z 5x5 -m landmark -n 6 -k 5 -s 0 -x 0 -r 1 -q 25 -f 100 -p 50
python VPG.py -t 500 -b 8 -i 500 -d 0.99 -l 0.001 -z 5x5 -m landmark -n 6 -k 5 -s 0 -x 0 -r 1 -q 25 -f 100 -p 50
python HA.py -t 500 -b 8 -i 500 -d 0.99 -l 0.001 -z 5x5 -m landmark -n 6 -k 5 -s 0 -x 0 -r 1 -q 25 -f 100 -p 50
```
* Command line flags/options:
  * t: time steps in every training episode;
  * b: batch size (we are using batch traning);
  * i: the number of iterations;
  * d: discount factor;
  * l: learning rate for updating parameters of policy network;
  * z: size of map e.g., 5x5, 10x10;
  * m: type pf map e.g., open, landmark;
  * n: the number of agents in training;
  * k: the number of landmarks (it should be zero if the type of map is open);
  * s: instance ID, each instance has a different configuration e.g., start zones and goal zones for agents;
  * x: run of each instance;
  * r: per time step reward;
  * q: reward for visiting a landmark for the first time;
  * f: final reward for reaching an agent's destination;
  * p: penalty for capacity violation
* All log files (Tensorboard event, model, and agents' paths) will be automatically saved in */PythonScripts/log/* folder after training is done.
  