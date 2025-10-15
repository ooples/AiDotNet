namespace AiDotNet.Pipeline
{
    using System.Collections.Generic;
    using AiDotNet.Enums;

    /// <summary>
    /// Configuration settings for training machine learning models in a pipeline.
    /// </summary>
    public class ModelTrainingConfig
    {
        /// <summary>
        /// Gets or sets the type of machine learning model to train.
        /// </summary>
        public ModelType ModelType { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of training epochs.
        /// </summary>
        public int Epochs { get; set; } = 100;

        /// <summary>
        /// Gets or sets the maximum number of training iterations.
        /// </summary>
        public int MaxIterations { get; set; } = 1000;

        /// <summary>
        /// Gets or sets the learning rate for gradient-based optimization.
        /// </summary>
        public double LearningRate { get; set; } = 0.01;

        /// <summary>
        /// Gets or sets the batch size for mini-batch training.
        /// </summary>
        public int BatchSize { get; set; } = 32;

        /// <summary>
        /// Gets or sets the convergence tolerance threshold.
        /// </summary>
        public double Tolerance { get; set; } = 1e-6;

        /// <summary>
        /// Gets or sets whether to use early stopping during training.
        /// </summary>
        public bool UseEarlyStopping { get; set; } = false;

        /// <summary>
        /// Gets or sets the validation split ratio (0.0 to 1.0) for early stopping.
        /// </summary>
        public double ValidationSplit { get; set; } = 0.2;

        /// <summary>
        /// Gets or sets the random seed for reproducibility.
        /// </summary>
        public int? RandomSeed { get; set; }

        /// <summary>
        /// Gets or sets whether to enable verbose logging during training.
        /// </summary>
        public bool Verbose { get; set; } = false;

        /// <summary>
        /// Gets or sets the optimizer type to use for training.
        /// </summary>
        public OptimizerType Optimizer { get; set; } = OptimizerType.Adam;

        /// <summary>
        /// Gets or sets custom hyperparameters for the model.
        /// </summary>
        public Dictionary<string, object> Hyperparameters { get; set; } = new Dictionary<string, object>();
    }
}
