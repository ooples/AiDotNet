# AiDotNet Online Learning Module

The Online Learning module provides implementations of various algorithms for learning from streaming data. These algorithms can process data incrementally, making them suitable for scenarios where data arrives continuously or when the full dataset cannot fit in memory.

## Features

- **Incremental Learning**: Process data one sample or batch at a time
- **Memory Efficient**: No need to store entire dataset
- **Adaptive**: Many algorithms can adapt to changing data distributions (concept drift)
- **Generic Implementation**: All algorithms support any numeric type (double, float, decimal, etc.)

## Available Algorithms

### Classification Algorithms

1. **OnlinePerceptron** - Simple linear classifier
   - Fast and memory efficient
   - Best for linearly separable data

2. **OnlineSVM** - Support Vector Machine
   - Supports both linear and kernel variants
   - Good for high-dimensional data

3. **AROW** - Adaptive Regularization of Weights
   - Maintains confidence bounds on weights
   - Robust to label noise

4. **ConfidenceWeighted** - Gaussian confidence modeling
   - Models uncertainty in weight estimates
   - Good for data with varying confidence levels

### Regression Algorithms

1. **PassiveAggressiveRegressor** - Aggressive updates on errors
   - Epsilon-insensitive loss
   - Good for noisy data

2. **OnlineSGDRegressor** - Stochastic Gradient Descent
   - Supports various loss functions
   - Includes momentum and learning rate schedules

3. **FTRL** - Follow-The-Regularized-Leader
   - Excellent for high-dimensional sparse data
   - Produces sparse models via L1 regularization

### Tree-Based Algorithms

1. **HoeffdingTree** - Incremental decision tree
   - Uses statistical bounds for splitting decisions
   - Memory efficient tree construction

2. **OnlineRandomForest** - Ensemble of Hoeffding trees
   - Better accuracy than single trees
   - Handles complex patterns

### Ensemble Methods

1. **OnlineBagging** - Generic bagging for any base learner
   - Uses Poisson sampling for online bootstrap
   - Reduces variance of base models

### Probabilistic Models

1. **OnlineNaiveBayes** - Incremental Naive Bayes
   - Supports both Gaussian and Multinomial variants
   - Good for text classification and probabilistic predictions

### Clustering

1. **OnlineKMeans** - Incremental K-means clustering
   - Supports both sequential and mini-batch updates
   - Adaptive learning rates

## Usage Examples

### Basic Classification
```csharp
// Create a perceptron for 2D classification
var perceptron = new OnlinePerceptron<double>(numFeatures: 2);

// Train on streaming data
var input = new Vector<double>(new[] { 1.5, 2.0 });
var label = 1.0; // Positive class
perceptron.UpdateOne(input, label);

// Make predictions
var prediction = perceptron.PredictOne(input);
```

### Regression with Options
```csharp
// Configure SGD regressor
var options = new OnlineModelOptions<double>
{
    LearningRate = 0.01,
    Alpha = 0.001, // L2 regularization
    Momentum = 0.9
};

var regressor = new OnlineSGDRegressor<double>(numFeatures: 3, options);

// Batch update
var inputs = new Matrix<double>(new double[,] 
{
    { 1.0, 2.0, 3.0 },
    { 4.0, 5.0, 6.0 }
});
var targets = new Vector<double>(new[] { 10.0, 20.0 });

regressor.UpdateBatch(inputs, targets);
```

### Adaptive Learning
```csharp
// AROW adapts to uncertainty in data
var arow = new AROW<double>(numFeatures: 5);

// The algorithm maintains confidence bounds
// and adapts to noisy or changing data
foreach (var (input, label) in dataStream)
{
    arow.UpdateOne(input, label);
}
```

### High-Dimensional Sparse Data
```csharp
// FTRL is excellent for sparse, high-dimensional data
var ftrl = new FTRL<double>(numFeatures: 10000, new OnlineModelOptions<double>
{
    Alpha = 1.0, // L1 regularization for sparsity
    Beta = 1.0   // L2 regularization
});

// Produces sparse models automatically
```

### Online Clustering
```csharp
// Cluster streaming data into 5 groups
var kmeans = new OnlineKMeans<double>(numFeatures: 3, k: 5);

// Update with each new data point
foreach (var dataPoint in stream)
{
    kmeans.UpdateOne(dataPoint, 0); // Target ignored for clustering
    var cluster = kmeans.PredictOne(dataPoint);
}
```

## Model Options

All models support configuration through `OnlineModelOptions<T>`:

```csharp
var options = new OnlineModelOptions<T>
{
    // Learning parameters
    LearningRate = 0.01,
    LearningRateSchedule = "constant", // or "invscaling", "adaptive"
    PowerT = 0.5, // For invscaling schedule
    
    // Regularization
    Alpha = 0.0001, // L1/L2 regularization strength
    Beta = 0.0001,  // Additional L2 for FTRL
    C = 1.0,        // SVM/PA regularization
    
    // Momentum
    Momentum = 0.9,
    
    // Algorithm-specific
    Epsilon = 0.1,  // PA insensitive loss
    Eta = 0.9,      // CW confidence parameter
    R = 1.0,        // AROW regularization
    
    // Tree parameters
    MinSamplesSplit = 20,
    MaxDepth = 10,
    SplitConfidence = 0.95,
    TieThreshold = 0.05,
    
    // General
    MaxIterations = 1000,
    Tolerance = 1e-4,
    RandomSeed = 42
};
```

## Best Practices

1. **Choose the right algorithm**:
   - Linear data → Perceptron, Linear SVM
   - Noisy data → AROW, PassiveAggressive
   - Sparse data → FTRL
   - Complex patterns → Trees, Ensembles
   - Probabilistic → Naive Bayes

2. **Tune hyperparameters**:
   - Start with default values
   - Adjust learning rate based on convergence
   - Use regularization to prevent overfitting

3. **Handle concept drift**:
   - Use adaptive algorithms (AROW, CW)
   - Monitor performance over time
   - Consider forgetting mechanisms

4. **Memory management**:
   - Trees and ensembles use more memory
   - Linear models are most efficient
   - Consider mini-batch updates for large streams

## Advanced Features

### Custom Kernels (SVM)
```csharp
var kernel = new GaussianKernel<double>(gamma: 0.1);
var svm = new OnlineSVM<double>(numFeatures: 2, kernel);
```

### Ensemble of Any Model
```csharp
var ensemble = new OnlineBagging<double, OnlineSGDRegressor<double>>(
    numFeatures: 4,
    numEstimators: 10,
    baseModelFactory: () => new OnlineSGDRegressor<double>(4)
);
```

### Model Persistence
```csharp
// Get model parameters
var parameters = model.GetParameters();

// Save to file (using your preferred serialization)
SaveToFile(parameters);

// Restore model
var restoredModel = new OnlinePerceptron<double>(2);
restoredModel.SetParameters(parameters);
```

## Performance Considerations

- **Linear models** (Perceptron, PA, SGD) are fastest
- **Tree-based models** have higher computational cost
- **Kernel methods** scale with number of support vectors
- **Batch updates** are more efficient than single updates
- **Sparse operations** in FTRL optimize high-dimensional data

## References

- Perceptron: Rosenblatt, F. (1958)
- Online SVM: Bordes et al. (2005)
- AROW: Crammer et al. (2009)
- Confidence Weighted: Dredze et al. (2008)
- Passive-Aggressive: Crammer et al. (2006)
- FTRL: McMahan et al. (2013)
- Hoeffding Tree: Domingos & Hulten (2000)