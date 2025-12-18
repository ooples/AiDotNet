using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for the Relation Networks algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Relation Networks learn a neural network to measure similarity between examples,
/// rather than using fixed distance functions. This configuration controls all aspects
/// of the algorithm's behavior including the relation module architecture.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how Relation Networks learns to compare examples:
///
/// Key parameters:
/// - <b>RelationType:</b> Architecture for comparing features (Concatenate, Convolution, Attention)
/// - <b>AggregationMethod:</b> How to combine scores from multiple support examples
/// - <b>FeatureEncoderL2Reg:</b> Regularization for feature encoder (prevents overfitting)
/// - <b>RelationModuleL2Reg:</b> Regularization for relation module (prevents overfitting)
/// - <b>UseMultiHeadRelation:</b> Whether to use multiple relation heads (ensemble)
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Multiple relation module architectures (MLP, CNN, Attention, Transformer)
/// - Multi-head relation networks for capturing diverse similarity patterns
/// - Learnable feature transformations and attention mechanisms
/// - Curriculum learning on relation complexity
/// - Support for both few-shot and many-shot scenarios
/// </para>
/// </remarks>
public class RelationNetworkAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the neural network used as feature encoder.
    /// </summary>
    /// <value>
    /// The feature encoder that extracts representations from inputs.
    /// This network is trained jointly with the relation module.
    /// </value>
    /// <remarks>
    /// The feature encoder learns to extract features that are useful for the relation module.
    /// Common choices include CNNs for images, RNNs for sequences, or Transformers for text.
    /// </remarks>
    public INeuralNetwork<T>? FeatureEncoder { get; set; }

    /// <summary>
    /// Gets or sets the relation module for computing similarity.
    /// </summary>
    /// <value>
    /// The relation module that takes combined features and outputs a similarity score.
    /// This is the learnable component that replaces fixed distance functions.
    /// </value>
    /// <remarks>
    /// The relation module can be:
    /// - MLP for concatenated features
    /// - CNN for stacked features
    /// - Attention mechanism for learned weighting
    /// - Transformer for complex interactions
    /// </remarks>
    public IRelationModule<T>? RelationModule { get; set; }

    /// <summary>
    /// Gets or sets the type of relation module architecture.
    /// </summary>
    /// <value>
    /// The architecture used for comparing feature representations.
    /// Default is Concatenate.
    /// </value>
    /// <remarks>
    /// <b>Relation Module Types:</b>
    /// - <b>Concatenate:</b> Simple MLP on concatenated features (fast, baseline)
    /// - <b>Convolution:</b> 2D CNN on stacked features (good for spatial data)
    /// - <b>Attention:</b> Learned attention between features (interpretable)
    /// - <b>Transformer:</b> Full transformer for complex interactions (powerful)
    /// </remarks>
    public RelationModuleType RelationType { get; set; } = RelationModuleType.Concatenate;

    /// <summary>
    /// Gets or sets the method for aggregating multiple relation scores.
    /// </summary>
    /// <value>
    /// How to combine scores from multiple support examples per class.
    /// Default is Mean.
    /// </value>
    /// <remarks>
    /// <b>Aggregation Methods:</b>
    /// - <b>Mean:</b> Average all scores (stable, simple)
    /// - <b>Max:</b> Take maximum score (focus on best match)
    /// - <b>Attention:</b> Learn attention weights (adaptive)
    /// - <b>LearnedWeighting:</b> Learn fixed weights (efficient)
    /// </remarks>
    public AggregationMethod AggregationMethod { get; set; } = AggregationMethod.Mean;

    /// <summary>
    /// Gets or sets the dimension for feature concatenation.
    /// </summary>
    /// <value>
    /// The dimension along which to concatenate features.
    /// Usually -1 (last dimension) for neural networks.
    /// Default is -1.
    /// </value>
    public int ConcatenationDimension { get; set; } = -1;

    /// <summary>
    /// Gets or sets whether to apply learned feature transformation.
    /// </summary>
    /// <value>
    /// If true, applies a learnable linear transformation to features
    /// before computing relations.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// Feature transformations can help align features for better relation computation.
    /// This is especially useful when features come from different modalities.
    /// </remarks>
    public bool ApplyFeatureTransform { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use multi-head relation modules.
    /// </summary>
    /// <value>
    /// If true, uses multiple relation modules in parallel (like multi-head attention).
    /// Each head can learn different types of similarities.
    /// Default is false.
    /// </value>
    /// <remarks>
    /// Multi-head relation networks can capture:
    /// - Different similarity notions (shape, color, texture)
    /// - Different abstraction levels (fine-grained vs coarse)
    /// - Complementary relations for better performance
    /// </remarks>
    public bool UseMultiHeadRelation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of heads for multi-head relation.
    /// </summary>
    /// <value>
    /// Number of parallel relation modules when UseMultiHeadRelation is true.
    /// Each head has its own parameters.
    /// Default is 4.
    /// </value>
    public int NumHeads { get; set; } = 4;

    /// <summary>
    /// Gets or sets the L2 regularization coefficient for the feature encoder.
    /// </summary>
    /// <value>
    /// Regularization strength for feature encoder parameters.
    /// Higher values prevent overfitting but may underfit.
    /// Default is 1e-4.
    /// </value>
    public double FeatureEncoderL2Reg { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the L2 regularization coefficient for the relation module.
    /// </summary>
    /// <value>
    /// Regularization strength for relation module parameters.
    /// The relation module can easily overfit with few examples.
    /// Default is 1e-3.
    /// </value>
    public double RelationModuleL2Reg { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate for the relation module.
    /// </summary>
    /// <value>
    /// Dropout rate between 0.0 and 1.0 applied to relation module layers.
    /// Helps prevent overfitting in few-shot scenarios.
    /// Default is 0.1.
    /// </value>
    public double RelationModuleDropout { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the hidden dimension for MLP relation modules.
    /// </summary>
    /// <value>
    /// Size of hidden layers in MLP relation modules.
    /// Only used when RelationType is Concatenate.
    /// Default is 256.
    /// </value>
    public int MLPHiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of layers in MLP relation modules.
    /// </summary>
    /// <value>
    /// Number of hidden layers in MLP relation modules.
    /// Deeper networks can capture more complex relations.
    /// Default is 2.
    /// </value>
    public int MLPNumLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of convolution filters for CNN relation modules.
    /// </summary>
    /// <value>
    /// Number of filters in each convolution layer.
    /// Only used when RelationType is Convolution.
    /// Default is 64.
    /// </value>
    public int CNNNumFilters { get; set; } = 64;

    /// <summary>
    /// Gets or sets the kernel size for CNN relation modules.
    /// </summary>
    /// <value>
    /// Size of convolution kernels.
    /// Larger kernels capture larger context.
    /// Default is 3.
    /// </value>
    public int CNNKernelSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of attention heads for attention relation modules.
    /// </summary>
    /// <value>
    /// Number of attention heads in attention-based relation modules.
    /// Different heads can focus on different aspects.
    /// Default is 8.
    /// </value>
    public int AttentionNumHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the dimension of attention keys/values.
    /// </summary>
    /// <value>
    /// Dimension size for attention mechanisms.
    /// Must be divisible by AttentionNumHeads.
    /// Default is 64.
    /// </value>
    public int AttentionDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets whether to use curriculum learning on relation complexity.
    /// </summary>
    /// <value>
    /// If true, starts with simpler relation tasks and gradually increases complexity.
    /// Helps stabilize training of complex relation modules.
    /// Default is false.
    /// </value>
    public bool UseRelationCurriculum { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial relation complexity for curriculum learning.
    /// </summary>
    /// <value>
    /// Starting complexity from 0.0 (simplest) to 1.0 (full complexity).
    /// Only used when UseRelationCurriculum is true.
    /// Default is 0.3.
    /// </value>
    public double InitialRelationComplexity { get; set; } = 0.3;

    /// <summary>
    /// Gets or sets whether to use relational bias in the relation module.
    /// </summary>
    /// <value>
    /// If true, adds learnable bias terms that depend on example positions.
    /// Can help the network learn positional relationships.
    /// Default is false.
    /// </value>
    public bool UseRelationalBias { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to normalize relation scores.
    /// </summary>
    /// <value>
    /// If true, applies layer normalization to relation scores.
    /// Can improve training stability.
    /// Default is true.
    /// </value>
    public bool NormalizeRelationScores { get; set; } = true;

    /// <summary>
    /// Gets or sets the temperature for relation score softmax.
    /// </summary>
    /// <value>
    /// Temperature value for controlling the sharpness of the distribution.
    /// Lower values make predictions more confident.
    /// Default is 1.0.
    /// </value>
    public double RelationTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use residual connections in relation module.
    /// </summary>
    /// <value>
    /// If true, adds residual connections to help gradient flow.
    /// Useful for deep relation modules.
    /// Default is false.
    /// </value>
    public bool UseResidualConnections { get; set; } = false;

    /// <summary>
    /// Creates a default Relation Networks configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on the original Relation Networks paper:
    /// - Relation type: Concatenate with MLP
    /// - Aggregation: Mean of support examples
    /// - MLP: 2 hidden layers with 256 units
    /// - L2 regularization: 1e-4 for encoder, 1e-3 for relation module
    /// - Dropout: 0.1 for regularization
    /// </remarks>
    public RelationNetworkAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // Relation Networks don't need inner loop
    }

    /// <summary>
    /// Creates a Relation Networks configuration with custom values.
    /// </summary>
    /// <param name="featureEncoder">The neural network feature encoder.</param>
    /// <param name="relationModule">The relation module for computing similarities.</param>
    /// <param name="relationType">Type of relation module architecture.</param>
    /// <param name="aggregationMethod">Method for aggregating multiple scores.</param>
    /// <param name="useMultiHeadRelation">Whether to use multi-head relation.</param>
    /// <param name="numHeads">Number of heads for multi-head relation.</param>
    /// <param name="featureEncoderL2Reg">L2 regularization for feature encoder.</param>
    /// <param name="relationModuleL2Reg">L2 regularization for relation module.</param>
    /// <param name="relationModuleDropout">Dropout rate for relation module.</param>
    /// <param name="innerLearningRate">Learning rate for optimization.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public RelationNetworkAlgorithmOptions(
        INeuralNetwork<T> featureEncoder,
        IRelationModule<T> relationModule,
        RelationModuleType relationType = RelationModuleType.Concatenate,
        AggregationMethod aggregationMethod = AggregationMethod.Mean,
        bool useMultiHeadRelation = false,
        int numHeads = 4,
        double featureEncoderL2Reg = 1e-4,
        double relationModuleL2Reg = 1e-3,
        double relationModuleDropout = 0.1,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        FeatureEncoder = featureEncoder;
        RelationModule = relationModule;
        RelationType = relationType;
        AggregationMethod = aggregationMethod;
        UseMultiHeadRelation = useMultiHeadRelation;
        NumHeads = numHeads;
        FeatureEncoderL2Reg = featureEncoderL2Reg;
        RelationModuleL2Reg = relationModuleL2Reg;
        RelationModuleDropout = relationModuleDropout;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // Relation Networks don't use inner loop
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

        // Check feature encoder
        if (FeatureEncoder == null)
            return false;

        // Check relation module
        if (RelationModule == null)
            return false;

        // Check aggregation parameters
        if (UseMultiHeadRelation && NumHeads <= 0)
            return false;

        // Check regularization parameters
        if (FeatureEncoderL2Reg < 0.0 || FeatureEncoderL2Reg > 1.0)
            return false;

        if (RelationModuleL2Reg < 0.0 || RelationModuleL2Reg > 1.0)
            return false;

        if (RelationModuleDropout < 0.0 || RelationModuleDropout >= 1.0)
            return false;

        // Check MLP parameters
        if (RelationType == RelationModuleType.Concatenate)
        {
            if (MLPHiddenDimension <= 0 || MLPHiddenDimension > 10000)
                return false;
            if (MLPNumLayers <= 0 || MLPNumLayers > 10)
                return false;
        }

        // Check CNN parameters
        if (RelationType == RelationModuleType.Convolution)
        {
            if (CNNNumFilters <= 0 || CNNNumFilters > 1000)
                return false;
            if (CNNKernelSize <= 0 || CNNKernelSize > 11)
                return false;
        }

        // Check Attention parameters
        if (RelationType == RelationModuleType.Attention || RelationType == RelationModuleType.Transformer)
        {
            if (AttentionNumHeads <= 0 || AttentionNumHeads > 32)
                return false;
            if (AttentionDimension <= 0 || AttentionDimension > 1024)
                return false;
            if (AttentionDimension % AttentionNumHeads != 0)
                return false; // Must be divisible for proper attention
        }

        // Check curriculum parameters
        if (UseRelationCurriculum)
        {
            if (InitialRelationComplexity < 0.0 || InitialRelationComplexity > 1.0)
                return false;
        }

        // Check temperature
        if (RelationTemperature <= 0.0 || RelationTemperature > 100.0)
            return false;

        return true;
    }

    /// <summary>
    /// Gets the current relation complexity for curriculum learning.
    /// </summary>
    /// <param name="currentEpisode">Current episode number.</param>
    /// <returns>Complexity value between 0.0 and 1.0.</returns>
    public double GetCurrentRelationComplexity(int currentEpisode)
    {
        if (!UseRelationCurriculum || currentEpisode >= NumEpisodes)
        {
            return 1.0; // Full complexity
        }

        // Linear interpolation from initial to full complexity
        double progress = (double)currentEpisode / NumEpisodes;
        return InitialRelationComplexity + progress * (1.0 - InitialRelationComplexity);
    }
}