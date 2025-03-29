# Laminoboost 

A computational laminography reconstruction framework with remote memoization.

## Status
The Laminoboost project is currently available as partial open source for implementation details check, The complete implementation, testing code, model training, and synthetic data will be open sourced within one month of acceptance


## Convergence
![Convergence](./images/curve.png)

In the convergence verification under the (1k,1k,1k) dataset, we consider that the convergence is reached at 80 iterations. Laminoboost achieves an average performance improvement of 58% with a convergence error of less than 5.3%. For the same loss, an additional 6 iterations is required, which still results in a 53.4% improvement over computation method.


## bandwidth utilization 
![Bandwidth](./images/bandwidth_vs_node_number.png)
