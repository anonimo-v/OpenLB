# Laminoboost 

A computational laminography reconstruction framework with remote memoization.

## Status
The Laminoboost project is currently available as partial open source for implementation details check, The complete implementation, testing code, model training, and synthetic data will be open sourced within one month of acceptance


## Convergence
![Convergence](./images/curve.png)

In the convergence verification under the (1k,1k,1k) dataset with tau = 0.95, we consider that the convergence is reached at 80 iterations. Laminoboost achieves an average performance improvement of 52.8%. On 1K dataset, that is 36% with a convergence error of less than 5.3%. For the same loss level, an additional 6 iterations is required, which still results in a 32.4% performance improvement over computational method.


## Bandwidth 
![Bandwidth](./images/bandwidth_vs_node_number.png)





