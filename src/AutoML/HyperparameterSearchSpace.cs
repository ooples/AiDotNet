namespace AiDotNet.AutoML
{
    using System.Collections.Generic;
    using AiDotNet.Enums;

    /// <summary>
    /// Defines the search space for hyperparameter optimization in AutoML.
    /// </summary>
    public class HyperparameterSearchSpace
    {
        /// <summary>
        /// Gets or sets the learning rate search range.
        /// </summary>
        public (double Min, double Max) LearningRateRange { get; set; } = (1e-4, 1e-1);

        /// <summary>
        /// Gets or sets the batch size options to try.
        /// </summary>
        public List<int> BatchSizes { get; set; } = new List<int> { 16, 32, 64, 128 };

        /// <summary>
        /// Gets or sets the number of layers range for neural networks.
        /// </summary>
        public (int Min, int Max) LayersRange { get; set; } = (1, 10);

        /// <summary>
        /// Gets or sets the number of units per layer range.
        /// </summary>
        public (int Min, int Max) UnitsPerLayerRange { get; set; } = (32, 512);

        /// <summary>
        /// Gets or sets the dropout rate range.
        /// </summary>
        public (double Min, double Max) DropoutRange { get; set; } = (0.0, 0.5);

        /// <summary>
        /// Gets or sets the activation functions to try.
        /// </summary>
        public List<ActivationFunction> ActivationFunctions { get; set; } = new List<ActivationFunction>
        {
            ActivationFunction.ReLU,
            ActivationFunction.Tanh,
            ActivationFunction.Sigmoid
        };

        /// <summary>
        /// Gets or sets the optimizer types to try.
        /// </summary>
        public List<OptimizerType> OptimizerTypes { get; set; } = new List<OptimizerType>
        {
            OptimizerType.Adam,
            OptimizerType.SGD,
            OptimizerType.RMSProp
        };

        /// <summary>
        /// Gets or sets the regularization strength range.
        /// </summary>
        public (double Min, double Max) RegularizationRange { get; set; } = (1e-6, 1e-2);

        /// <summary>
        /// Gets or sets the momentum range for optimizers that support it.
        /// </summary>
        public (double Min, double Max) MomentumRange { get; set; } = (0.5, 0.99);

        /// <summary>
        /// Gets or sets custom hyperparameter ranges.
        /// </summary>
        public Dictionary<string, object> CustomParameters { get; set; } = new Dictionary<string, object>();

        /// <summary>
        /// Gets or sets whether to use logarithmic scale for learning rate search.
        /// </summary>
        public bool UseLogScaleForLearningRate { get; set; } = true;

        /// <summary>
        /// Gets or sets the maximum number of epochs for training during search.
        /// </summary>
        public int MaxEpochs { get; set; } = 100;

        /// <summary>
        /// Gets or sets whether to enable early stopping during hyperparameter search.
        /// </summary>
        public bool EnableEarlyStopping { get; set; } = true;

        /// <summary>
        /// Gets or sets the patience for early stopping.
        /// </summary>
        public int EarlyStoppingPatience { get; set; } = 5;
    }
}
