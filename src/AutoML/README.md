# AutoML Module

This module provides automated machine learning capabilities for the AiDotNet framework.

## Overview

The AutoML module includes several key components:

### Core Classes

1. **AutoMLModelBase<T, TInput, TOutput>** - Base class for all AutoML implementations
   - Generic implementation supporting any numeric type T
   - Supports various input/output data types
   - Provides common functionality for hyperparameter search

2. **BayesianOptimizationAutoML<T, TInput, TOutput>** - Bayesian optimization for hyperparameter search
   - Intelligent search using Gaussian processes
   - Efficient exploration/exploitation balance
   - Suitable for expensive model evaluations

3. **GridSearchAutoML<T, TInput, TOutput>** - Exhaustive grid search
   - Systematic exploration of parameter space
   - Guarantees finding the best combination within the grid
   - Best for small parameter spaces

4. **RandomSearchAutoML<T, TInput, TOutput>** - Random hyperparameter search
   - Simple but effective baseline method
   - Often outperforms grid search for high-dimensional spaces
   - Configurable number of trials

5. **NeuralArchitectureSearch<T>** - Automated neural network architecture design
   - Multiple search strategies: Evolutionary, Reinforcement Learning, Gradient-based, Random
   - Supports various layer types and configurations
   - Production-ready implementation

### Supporting Classes

- **HyperparameterSpace** - Defines the search space for hyperparameters
- **ParameterRange** - Represents a range of values for a hyperparameter
- **TrialResult** - Stores results from individual trials
- **ArchitectureCandidate<T>** - Represents a candidate neural architecture
- **LayerConfiguration<T>** - Configuration for individual layers
- **SearchSpace<T>** - Search space for neural architecture search

## Key Features

1. **Generic Implementation**: All classes use generic types for maximum flexibility
2. **Production Ready**: Proper error handling, logging, and async support
3. **Extensible**: Easy to add new search algorithms
4. **Type Safe**: Uses Vector<T> instead of raw arrays where appropriate
5. **Well Organized**: Each class in its own file for maintainability

## Usage Example

```csharp
// Bayesian optimization example
var automl = new BayesianOptimizationAutoML<double, double[][], double[]>(
    numInitialPoints: 10,
    explorationWeight: 2.0
);

// Define search space
var searchSpace = new Dictionary<string, ParameterRange>();
searchSpace["learning_rate"] = new ParameterRange
{
    Type = ParameterType.Continuous,
    MinValue = 0.0001,
    MaxValue = 0.1,
    LogScale = true
};

automl.SetSearchSpace(searchSpace);
automl.SetOptimizationMetric(MetricType.Accuracy, maximize: true);

// Run search
var bestModel = await automl.SearchAsync(
    trainInputs, trainTargets,
    valInputs, valTargets,
    TimeSpan.FromHours(1)
);
```

## Future Enhancements

- Add more neural architecture search strategies
- Implement multi-objective optimization
- Add distributed search capabilities
- Enhance early stopping mechanisms
- Add more sophisticated acquisition functions for Bayesian optimization