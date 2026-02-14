using AiDotNet.Enums;
using AiDotNet.LossFunctions;

namespace AiDotNet.Training.Factories;

/// <summary>
/// Factory for creating loss function instances from <see cref="LossType"/> enum values.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This factory creates loss functions based on a simple name or enum value.
/// You don't need to know the exact class name or constructor details - just specify the type
/// and optional parameters.
/// </para>
/// </remarks>
public static class LossFunctionFactory<T>
{
    /// <summary>
    /// Creates a loss function of the specified type with default parameters.
    /// </summary>
    /// <param name="lossType">The type of loss function to create.</param>
    /// <returns>An <see cref="ILossFunction{T}"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the loss type is not supported for parameterless creation.</exception>
    public static ILossFunction<T> Create(LossType lossType)
    {
        return Create(lossType, null);
    }

    /// <summary>
    /// Creates a loss function of the specified type with optional parameters.
    /// </summary>
    /// <param name="lossType">The type of loss function to create.</param>
    /// <param name="parameters">Optional dictionary of named parameters (e.g., "delta" for Huber loss).</param>
    /// <returns>An <see cref="ILossFunction{T}"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the loss type is not supported or required parameters are missing.</exception>
    public static ILossFunction<T> Create(LossType lossType, Dictionary<string, object>? parameters)
    {
        return lossType switch
        {
            LossType.MeanSquaredError => new MeanSquaredErrorLoss<T>(),
            LossType.MeanAbsoluteError => new MeanAbsoluteErrorLoss<T>(),
            LossType.RootMeanSquaredError => new RootMeanSquaredErrorLoss<T>(),
            LossType.Huber => new HuberLoss<T>(GetDoubleParam(parameters, "delta", 1.0)),
            LossType.CrossEntropy => new CrossEntropyLoss<T>(),
            LossType.BinaryCrossEntropy => new BinaryCrossEntropyLoss<T>(),
            LossType.CategoricalCrossEntropy => new CategoricalCrossEntropyLoss<T>(),
            LossType.SparseCategoricalCrossEntropy => new SparseCategoricalCrossEntropyLoss<T>(),
            LossType.Focal => new FocalLoss<T>(
                GetDoubleParam(parameters, "gamma", 2.0),
                GetDoubleParam(parameters, "alpha", 0.25)),
            LossType.Hinge => new HingeLoss<T>(),
            LossType.SquaredHinge => new SquaredHingeLoss<T>(),
            LossType.LogCosh => new LogCoshLoss<T>(),
            LossType.Quantile => new QuantileLoss<T>(GetDoubleParam(parameters, "quantile", 0.5)),
            LossType.Poisson => new PoissonLoss<T>(),
            LossType.KullbackLeiblerDivergence => new KullbackLeiblerDivergence<T>(),
            LossType.CosineSimilarity => new CosineSimilarityLoss<T>(),
            LossType.Contrastive => new ContrastiveLoss<T>(GetDoubleParam(parameters, "margin", 1.0)),
            LossType.Triplet => new TripletLoss<T>(GetDoubleParam(parameters, "margin", 1.0)),
            LossType.Dice => new DiceLoss<T>(),
            LossType.Jaccard => new JaccardLoss<T>(),
            LossType.ElasticNet => new ElasticNetLoss<T>(
                GetDoubleParam(parameters, "l1Ratio", 0.5),
                GetDoubleParam(parameters, "alpha", 0.01)),
            LossType.Exponential => new ExponentialLoss<T>(),
            LossType.ModifiedHuber => new ModifiedHuberLoss<T>(),
            LossType.Charbonnier => new CharbonnierLoss<T>(GetDoubleParam(parameters, "epsilon", 1e-6)),
            LossType.MeanBiasError => new MeanBiasErrorLoss<T>(),
            LossType.Wasserstein => new WassersteinLoss<T>(),
            LossType.Margin => new MarginLoss<T>(
                GetDoubleParam(parameters, "mPlus", 0.9),
                GetDoubleParam(parameters, "mMinus", 0.1),
                GetDoubleParam(parameters, "lambda", 0.5)),
            LossType.CTC => new CTCLossAdapter<T>(
                GetIntParam(parameters, "numClasses", 10),
                GetIntParam(parameters, "blankIndex", 0)),
            LossType.NoiseContrastiveEstimation => new NoiseContrastiveEstimationLoss<T>(
                GetIntParam(parameters, "numNoiseSamples", 10)),
            LossType.OrdinalRegression => new OrdinalRegressionLoss<T>(
                GetIntParam(parameters, "numClasses", 5)),
            LossType.WeightedCrossEntropy => new WeightedCrossEntropyLoss<T>(),
            LossType.ScaleInvariantDepth => new ScaleInvariantDepthLoss<T>(
                GetDoubleParam(parameters, "lambda", 0.5)),
            LossType.Quantum => new QuantumLoss<T>(),
            _ => throw new ArgumentException($"Unsupported loss type: {lossType}. " +
                "PerceptualLoss, RealESRGANLoss, and RotationPredictionLoss require specialized constructors " +
                "and cannot be created via the factory.")
        };
    }

    /// <summary>
    /// Creates a loss function by parsing the name string to a <see cref="LossType"/> enum value.
    /// </summary>
    /// <param name="name">The name of the loss function type (case-insensitive).</param>
    /// <param name="parameters">Optional dictionary of named parameters.</param>
    /// <returns>An <see cref="ILossFunction{T}"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when the name does not match any known loss type.</exception>
    public static ILossFunction<T> Create(string name, Dictionary<string, object>? parameters = null)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Loss function name cannot be null or empty.", nameof(name));
        }

        if (!Enum.TryParse<LossType>(name, ignoreCase: true, out var lossType))
        {
            throw new ArgumentException(
                $"Unknown loss function name: '{name}'. Valid names are: {string.Join(", ", Enum.GetNames(typeof(LossType)))}",
                nameof(name));
        }

        return Create(lossType, parameters);
    }

    /// <summary>
    /// Gets a double parameter from the dictionary with a fallback default value.
    /// </summary>
    private static double GetDoubleParam(Dictionary<string, object>? parameters, string key, double defaultValue)
    {
        if (parameters is null || !parameters.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        return value switch
        {
            double d => d,
            float f => f,
            int i => i,
            long l => l,
            string s when double.TryParse(s, out var parsed) => parsed,
            _ => Convert.ToDouble(value)
        };
    }

    /// <summary>
    /// Gets an integer parameter from the dictionary with a fallback default value.
    /// </summary>
    private static int GetIntParam(Dictionary<string, object>? parameters, string key, int defaultValue)
    {
        if (parameters is null || !parameters.TryGetValue(key, out var value))
        {
            return defaultValue;
        }

        return value switch
        {
            int i => i,
            long l => (int)l,
            double d => (int)d,
            float f => (int)f,
            string s when int.TryParse(s, out var parsed) => parsed,
            _ => Convert.ToInt32(value)
        };
    }
}
