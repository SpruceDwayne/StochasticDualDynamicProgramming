# StochasticDualDynamicProgramming
This is a repo where I will store code used for my master thesis on Stochastic Dual Dynamic Programming (SDDP)
It includes a solver for Standard SDDP and allows for the use of markov-chain driven uncertainty as well as risk averse formulations using AVaR. It does not include statistical stopping rules, but only heuristic stopping rules. 

Future work to be done on this package includes
-Adjusting the heuristic stopping rules to include absolute stalling of the lower bound
-Make all example scripts work (lots of changes were made in the final weeks of the project to investigate complexity)
-Computational optimisation of the code (forgetting cuts, iniitalize fewer jump models maybe etc.)
-Furher relaxationas (integrality restrictions, DecisionDependentUncertainty, etc.)

