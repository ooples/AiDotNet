namespace AiDotNet.Enums
{
    /// <summary>
    /// Hyperparameter tuning strategies for AutoML optimization.
    /// </summary>
    public enum HyperparameterTuningStrategy
    {
        /// <summary>
        /// Grid search - exhaustive search over specified parameter values.
        /// </summary>
        GridSearch,

        /// <summary>
        /// Random search - random sampling from parameter distributions.
        /// </summary>
        RandomSearch,

        /// <summary>
        /// Bayesian optimization - probabilistic model-based optimization.
        /// </summary>
        BayesianOptimization,

        /// <summary>
        /// Hyperband - bandit-based approach for hyperparameter optimization.
        /// </summary>
        Hyperband,

        /// <summary>
        /// BOHB - Bayesian Optimization and HyperBand combined strategy.
        /// </summary>
        BOHB
    }
}
