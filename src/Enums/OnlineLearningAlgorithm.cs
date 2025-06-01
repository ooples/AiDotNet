namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of online learning algorithm.
/// </summary>
public enum OnlineLearningAlgorithm
{
    /// <summary>
    /// Stochastic Gradient Descent - Basic online learning algorithm.
    /// </summary>
    StochasticGradientDescent,
    
    /// <summary>
    /// Online Perceptron - Linear classifier for binary classification.
    /// </summary>
    Perceptron,
    
    /// <summary>
    /// Passive-Aggressive algorithm - Large margin online learning.
    /// </summary>
    PassiveAggressive,
    
    /// <summary>
    /// Online Support Vector<double> Machine.
    /// </summary>
    OnlineSVM,
    
    /// <summary>
    /// Adaptive Regularization of Weights (AROW).
    /// </summary>
    AROW,
    
    /// <summary>
    /// Confidence-Weighted learning.
    /// </summary>
    ConfidenceWeighted,
    
    /// <summary>
    /// Online Gradient Boosting.
    /// </summary>
    OnlineGradientBoosting,
    
    /// <summary>
    /// Hoeffding Tree (Very Fast Decision Tree).
    /// </summary>
    HoeffdingTree,
    
    /// <summary>
    /// Online Random Forest.
    /// </summary>
    OnlineRandomForest,
    
    /// <summary>
    /// Online Bagging.
    /// </summary>
    OnlineBagging,
    
    /// <summary>
    /// Adaptive Random Forest with drift detection.
    /// </summary>
    AdaptiveRandomForest,
    
    /// <summary>
    /// Online K-Means clustering.
    /// </summary>
    OnlineKMeans,
    
    /// <summary>
    /// Sequential K-Means clustering.
    /// </summary>
    SequentialKMeans,
    
    /// <summary>
    /// Online Neural Network with backpropagation.
    /// </summary>
    OnlineNeuralNetwork,
    
    /// <summary>
    /// Follow-The-Regularized-Leader algorithm.
    /// </summary>
    FTRL,
    
    /// <summary>
    /// Online LASSO regression.
    /// </summary>
    OnlineLASSO,
    
    /// <summary>
    /// Online Ridge regression.
    /// </summary>
    OnlineRidge,
    
    /// <summary>
    /// Online Elastic Net regression.
    /// </summary>
    OnlineElasticNet,
    
    /// <summary>
    /// Incremental Principal Component Analysis.
    /// </summary>
    IncrementalPCA,
    
    /// <summary>
    /// Online Independent Component Analysis.
    /// </summary>
    OnlineICA,
    
    /// <summary>
    /// Streaming Linear Discriminant Analysis.
    /// </summary>
    StreamingLDA,
    
    /// <summary>
    /// Online Naive Bayes classifier.
    /// </summary>
    OnlineNaiveBayes,
    
    /// <summary>
    /// Mondrian Forest - Online random forest variant.
    /// </summary>
    MondrianForest,
    
    /// <summary>
    /// Online Gaussian Process.
    /// </summary>
    OnlineGaussianProcess,
    
    /// <summary>
    /// Budgeted Stochastic Gradient Descent.
    /// </summary>
    BudgetedSGD,
    
    /// <summary>
    /// Online Learning with Kernels.
    /// </summary>
    OnlineKernelLearning
}