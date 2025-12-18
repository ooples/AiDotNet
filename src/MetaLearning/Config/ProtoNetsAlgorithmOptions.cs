using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for the Prototypical Networks (ProtoNets) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// ProtoNets is a metric-based meta-learning algorithm that learns a feature space
/// where classification can be performed by computing distances to prototype representations.
/// This configuration controls all aspects of the algorithm's behavior.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how ProtoNets learns and classifies:
///
/// Key parameters:
/// - <b>DistanceFunction:</b> How to measure similarity (Euclidean, Cosine, Mahalanobis)
/// - <b>Temperature:</b> Controls confidence of predictions (lower = more confident)
/// - <b>NormalizeFeatures:</b> Whether to normalize features to unit length
/// - <b>UseAttentionMechanism:</b> Whether to weight important features more
/// - <b>UseAdaptiveClassScaling:</b> Whether to use different scales for different classes
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Attention mechanisms for adaptive prototype computation
/// - Learnable distance metrics with class-specific scaling
/// - Feature normalization for stable training
/// - Temperature scaling for calibrated predictions
/// - Support for multiple distance functions (Euclidean, Cosine, Mahalanobis)
/// </para>
/// </remarks>
public class ProtoNetsAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the neural network used as feature encoder.
    /// </summary>
    /// <value>
    /// The feature encoder that maps inputs to the metric space where prototypes are computed.
    /// This is the only learnable component of ProtoNets.
    /// </value>
    /// <remarks>
    /// The feature encoder is typically a CNN for images or an RNN for sequences.
    /// It should be powerful enough to extract discriminative features.
    /// </remarks>
    public INeuralNetwork<T>? FeatureEncoder { get; set; }

    /// <summary>
    /// Gets or sets the distance function to use for measuring similarity.
    /// </summary>
    /// <value>
    /// The distance function used to compute distances between query features and class prototypes.
    /// Default is Euclidean.
    /// </value>
    /// <remarks>
    /// <b>Distance Functions:</b>
    /// - <b>Euclidean:</b> Standard L2 distance, good for most tasks
    /// - <b>Cosine:</b> Measures angular similarity, good for text/normalized features
    /// - <b>Mahalanobis:</b> Accounts for feature correlations, learnable
    /// </remarks>
    public DistanceFunction DistanceFunction { get; set; } = DistanceFunction.Euclidean;

    /// <summary>
    /// Gets or sets the temperature scaling factor for softmax probabilities.
    /// </summary>
    /// <value>
    /// Temperature value > 0. Lower values make predictions more confident (sharper distribution),
    /// higher values make predictions less confident (softer distribution).
    /// Default is 1.0 (no scaling).
    /// </value>
    /// <remarks>
    /// <b>Temperature effects:</b>
    /// - 0.1: Very confident predictions, may overfit
    /// - 0.5: Moderately confident, good for clean data
    /// - 1.0: Standard softmax, balanced approach
    /// - 2.0: Less confident, good for noisy data
    /// - 5.0: Very soft predictions, good for uncertainty
    ///
    /// Temperature is learned during training if LearnTemperature is enabled.
    /// </remarks>
    public double Temperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to learn the temperature parameter during training.
    /// </summary>
    /// <value>
    /// If true, temperature is treated as a learnable parameter optimized during training.
    /// If false, temperature is fixed to the initial value.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// Learning temperature can improve calibration but adds a learnable parameter.
    /// Consider disabling for very few-shot scenarios where overfitting is a concern.
    /// </remarks>
    public bool LearnTemperature { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to normalize feature vectors to unit length.
    /// </summary>
    /// <value>
    /// If true, features are L2-normalized before computing distances.
    /// This can improve stability and make Euclidean distance equivalent to cosine distance.
    /// Default is true.
    /// </value>
    /// <remarks>
    /// <b>Benefits of normalization:</b>
    /// - Removes scale variations between features
    /// - Makes distances more interpretable
    /// - Improves numerical stability
    /// - Makes Euclidean and cosine distance more comparable
    ///
    /// Disable only if feature scale carries important information.
    /// </remarks>
    public bool NormalizeFeatures { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use attention mechanism for prototype computation.
    /// </summary>
    /// <value>
    /// If true, applies learnable attention weights when computing class prototypes.
    /// This allows the model to focus on more informative features.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// <b>Attention mechanism:</b>
    /// - Learns weights for each feature dimension
    /// - Emphasizes features that are more discriminative
    /// - Reduces impact of noisy or irrelevant features
    /// - Adds computational overhead
    ///
    /// Enable for complex datasets with feature redundancy.
    /// </remarks>
    public bool UseAttentionMechanism { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use adaptive class-specific distance scaling.
    /// </summary>
    /// <value>
    /// If true, learns a scaling factor for each class to account for class-specific
    /// variations in feature distributions.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// <b>Adaptive scaling helps when:</b>
    /// - Classes have different intra-class variances
    /// - Some classes are more spread out in feature space
    /// - Classes have different numbers of examples
    /// - Feature distributions vary significantly between classes
    ///
    /// Each class gets its own learnable scaling factor.
    /// </remarks>
    public bool UseAdaptiveClassScaling { get; set; } = false;

    /// <summary>
    /// Gets or sets the scaling factor for Mahalanobis distance.
    /// </summary>
    /// <value>
    /// Scaling factor applied to Mahalanobis distance computation.
    /// Only used when DistanceFunction is Mahalanobis.
    /// Default is 1.0.
    /// </value>
    /// <remarks>
    /// This is a simplified implementation of Mahalanobis distance.
    /// A full implementation would learn a covariance matrix.
    /// The scaling factor compensates for this simplification.
    /// </remarks>
    public double MahalanobisScaling { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the feature dimension for the encoded representations.
    /// </summary>
    /// <value>
    /// Dimension of the feature space where prototypes are computed.
    /// Higher dimensions can capture more information but risk overfitting.
    /// Default is 128.
    /// </value>
    /// <remarks>
    /// <b>Dimension guidelines:</b>
    /// - 32-64: Very few-shot, prevent overfitting
    /// - 128-256: Standard few-shot scenarios
    /// - 512-1024: Complex tasks with more examples
    /// - 2048+: High-dimensional data (e.g., embeddings)
    ///
    /// Should be less than number of examples in support set.
    /// </remarks>
    public int FeatureDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the L2 regularization coefficient for feature encoder.
    /// </summary>
    /// <value>
    /// Coefficient for L2 regularization on encoder parameters.
    /// Higher values prevent overfitting but may underfit.
    /// Default is 1e-4.
    /// </value>
    /// <remarks>
    /// <b>Regularization guidance:</b>
    /// - 1e-6: Very little regularization (large datasets)
    /// - 1e-4: Standard regularization (balanced)
    /// - 1e-3: Strong regularization (small datasets)
    /// - 1e-2: Very strong regularization (prevent overfitting)
    ///
    /// Monitor validation loss to adjust appropriately.
    /// </remarks>
    public double L2Regularization { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate for the feature encoder.
    /// </summary>
    /// <value>
    /// Dropout rate between 0.0 and 1.0 applied to encoder layers.
    /// Helps prevent overfitting in few-shot scenarios.
    /// Default is 0.1 (10% dropout).
    /// </value>
    /// <remarks>
    /// <b>Dropout recommendations:</b>
    /// - 0.0: No dropout (very few-shot scenarios)
    /// - 0.1-0.2: Light dropout (standard few-shot)
    /// - 0.3-0.5: Heavy dropout (risk of underfitting)
    /// - >0.5: Very heavy dropout (rarely used)
    ///
    /// Use less dropout with fewer examples per class.
    /// </remarks>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use episode difficulty curriculum learning.
    /// </summary>
    /// <value>
    /// If true, starts with easier episodes and gradually increases difficulty.
    /// Easier episodes have more distinct classes and clearer features.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// <b>Curriculum learning benefits:</b>
    /// - Stabilizes early training
    /// - Learns basic discriminations first
    /// - Gradually handles harder distinctions
    /// - Can improve final performance
    ///
    /// Useful for complex datasets with many similar classes.
    /// </remarks>
    public bool UseCurriculumLearning { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial difficulty level for curriculum learning.
    /// </summary>
    /// <value>
    /// Starting difficulty from 0.0 (easiest) to 1.0 (hardest).
    /// Only used when UseCurriculumLearning is true.
    /// Default is 0.3 (start moderately easy).
    /// </value>
    public double InitialDifficulty { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets the final difficulty level for curriculum learning.
    /// </summary>
    /// <value>
    /// Final difficulty from 0.0 (easiest) to 1.0 (hardest).
    /// Only used when UseCurriculumLearning is true.
    /// Default is 1.0 (end at full difficulty).
    /// </value>
    public double FinalDifficulty { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of training steps to ramp up difficulty.
    /// </summary>
    /// <value>
    /// Number of episodes over which to increase from InitialDifficulty to FinalDifficulty.
    /// Only used when UseCurriculumLearning is true.
    /// Default is 1000 episodes.
    /// </value>
    public int CurriculumSteps { get; set; } = 1000;

    /// <summary>
    /// Creates a default ProtoNets configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on the original ProtoNets paper:
    /// - Feature encoder: CNN with 128-dim embeddings
    /// - Distance function: Euclidean
    /// - Temperature: 1.0 (standard softmax)
    /// - Feature normalization: Enabled
    /// - Regularization: L2 with coefficient 1e-4
    /// - Dropout: 0.1 for few-shot regularization
    /// </remarks>
    public ProtoNetsAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // ProtoNets doesn't need inner loop steps
    }

    /// <summary>
    /// Creates a ProtoNets configuration with custom values.
    /// </summary>
    /// <param name="featureEncoder">The neural network feature encoder.</param>
    /// <param name="distanceFunction">The distance function to use.</param>
    /// <param name="temperature">Temperature for softmax scaling.</param>
    /// <param name="learnTemperature">Whether to learn temperature.</param>
    /// <param name="normalizeFeatures">Whether to normalize features.</param>
    /// <param name="useAttentionMechanism">Whether to use attention.</param>
    /// <param name="useAdaptiveClassScaling">Whether to use class-specific scaling.</param>
    /// <param name="featureDimension">Dimension of feature space.</param>
    /// <param name="l2Regularization">L2 regularization coefficient.</param>
    /// <param name="dropoutRate">Dropout rate.</param>
    /// <param name="useCurriculumLearning">Whether to use curriculum learning.</param>
    /// <param name="innerLearningRate">Learning rate for feature encoder.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public ProtoNetsAlgorithmOptions(
        INeuralNetwork<T> featureEncoder,
        DistanceFunction distanceFunction = DistanceFunction.Euclidean,
        double temperature = 1.0,
        bool learnTemperature = false,
        bool normalizeFeatures = true,
        bool useAttentionMechanism = false,
        bool useAdaptiveClassScaling = false,
        int featureDimension = 128,
        double l2Regularization = 1e-4,
        double dropoutRate = 0.1,
        bool useCurriculumLearning = false,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        FeatureEncoder = featureEncoder;
        DistanceFunction = distanceFunction;
        Temperature = temperature;
        LearnTemperature = learnTemperature;
        NormalizeFeatures = normalizeFeatures;
        UseAttentionMechanism = useAttentionMechanism;
        UseAdaptiveClassScaling = useAdaptiveClassScaling;
        FeatureDimension = featureDimension;
        L2Regularization = l2Regularization;
        DropoutRate = dropoutRate;
        UseCurriculumLearning = useCurriculumLearning;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // ProtoNets doesn't use inner loop
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public override bool IsValid()
    {
        // Check base class validation
        if (!base.IsValid())
            return false;

        // Check temperature
        if (Temperature <= 0.0 || Temperature > 100.0)
            return false;

        // Check feature dimension
        if (FeatureDimension <= 0 || FeatureDimension > 10000)
            return false;

        // Check regularization coefficients
        if (L2Regularization < 0.0 || L2Regularization > 1.0)
            return false;

        // Check dropout rate
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        // Check curriculum learning parameters
        if (UseCurriculumLearning)
        {
            if (InitialDifficulty < 0.0 || InitialDifficulty > 1.0)
                return false;
            if (FinalDifficulty < 0.0 || FinalDifficulty > 1.0)
                return false;
            if (InitialDifficulty > FinalDifficulty)
                return false;
            if (CurriculumSteps <= 0)
                return false;
        }

        // Check Mahalanobis scaling if using Mahalanobis distance
        if (DistanceFunction == DistanceFunction.Mahalanobis)
        {
            if (MahalanobisScaling <= 0.0 || MahalanobisScaling > 100.0)
                return false;
        }

        // Check feature encoder
        if (FeatureEncoder == null)
            return false;

        return true;
    }

    /// <summary>
    /// Gets the current difficulty level for curriculum learning.
    /// </summary>
    /// <param name="currentEpisode">Current episode number.</param>
    /// <returns>Difficulty value between InitialDifficulty and FinalDifficulty.</returns>
    public double GetCurrentDifficulty(int currentEpisode)
    {
        if (!UseCurriculumLearning || currentEpisode >= CurriculumSteps)
        {
            return FinalDifficulty;
        }

        // Linear interpolation between initial and final difficulty
        double progress = (double)currentEpisode / CurriculumSteps;
        return InitialDifficulty + progress * (FinalDifficulty - InitialDifficulty);
    }
}