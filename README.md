# Maximum A Posteriori Inverse Kinematics
This repository is the result of work done in the DTU course 'Project Work in Artificial Intelligence & Data'. In the project, 2 models for generating human-like motion were constructed, both utilizing MAP to solve the inverse kinematics problem. The first model, constructs normal distributions over the dataset in order to form a prior distribution over human motion. Secondly, we have a time dependent prior modelled using a Hidden Markov model, essentially acting as a Gaussian Mixture Model with dynamic, time dependent weights.


### How to generate a series of motions:
In /Generate/Generate.py, select one of the models available in inverse_kinematics/models/. Then, select a motion to get the joint positions from which the rest of the motion is generated. Running the script will then generate a motion sequence, which will be saved as a .amc file.

### References
amc_parser.py and 3Dviewer.py were kindly borrowed from https://github.com/CalciferZh/AMCParser and later rewritten in order to be compatible with pyTorch - based gradient descent.





