using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Conditional Neural Adaptive Processes (CNAPs).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CNAPs extend Neural Processes by conditioning on task-specific context points
/// and learning to adapt to new tasks with fast weight generation. The model combines
/// representation learning with meta-learning for efficient adaptation.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how CNAPs learns to adapt:
///
/// Key parameters:
/// - <b>RepresentationDimension:</b> Size of task-specific representations
/// - <b>UseAttention:</b> Whether to use attention for context aggregation
/// - <b>NumTransformerBlocks:</b> Number of transformer layers in encoder
/// - <b>PredictUncertainty:</b> Whether to predict uncertainty along with outputs
/// - <b>SupportSetSize:</b> Number of examples used for adaptation
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Multi-head attention for context point processing
/// - Transformer blocks for representation learning
/// - Fast weight generation and normalization
/// - Uncertainty prediction with heteroscedastic output
/// - Adaptive task representations
/// - Hierarchical meta-learning
/// </para>
/// </remarks>
public class CNAPAlgorithmOptions<T, TInput, TOutput> : MetaLearningOptions<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the input data dimension.
    /// </summary>
    /// <value>
    /// Dimensionality of input features.
    /// Must match the data being processed.
    /// Default is 128.
    /// </value>
    public int InputDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the output data dimension.
    /// </summary>
    /// <value>
    /// Dimensionality of output predictions.
    /// For classification, this equals number of classes.
    /// Default is 10.
    /// </value>
    public int OutputDimension { get; set; } = 10;

    /// <summary>
    /// Gets or sets the context set size.
    /// </summary>
    /// <value>
    /// Number of context points to use for encoding tasks.
    /// Larger contexts provide more information but increase computation.
    /// Default is 10.
    /// </value>
    public int ContextSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets the hidden layer dimension.
    /// </summary>
    /// <value>
    /// Dimension of hidden layers in encoder/decoder.
    /// Affects model capacity.
    /// Default is 256.
    /// </value>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the task representation dimension.
    /// </summary>
    /// <value>
    /// Dimension of the task-specific representation vector.
    /// Encodes task information for adaptation.
    /// Default is 128.
    /// </value>
    public int RepresentationDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the feed-forward dimension in transformer blocks.
    /// </summary>
    /// <value>
    /// Dimension of feed-forward layers in transformers.
    /// Usually 4x the hidden dimension.
    /// Default is 1024.
    /// </value>
    public int FeedForwardDimension { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the adaptation network hidden dimension.
    /// </summary>
    /// <value>
    /// Size of hidden layers in the adaptation network.
    /// Controls complexity of fast weight generation.
    /// Default is 512.
    /// </value>
    public int AdaptationHiddenDimension { get; set; } = 512;

    /// <summary>
    /// Gets or sets the number of attention heads.
    /// </summary>
    /// <value>
    /// Number of parallel attention mechanisms.
    /// Must divide the hidden dimension evenly.
    /// Default is 8.
    /// </value>
    public int NumAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the attention dimension per head.
    /// </summary>
    /// <value>
    /// Dimension of each attention head.
    /// Default is 64.
    /// </value>
    public int AttentionDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the number of transformer blocks.
    /// </summary>
    /// <value>
    /// Number of transformer encoder blocks to use.
    /// More blocks allow for more complex representations.
    /// Default is 6.
    /// </value>
    public int NumTransformerBlocks { get; set; } = 6;

    /// <summary>
    /// Gets or sets the number of decoder layers.
    /// </summary>
    /// <value>
    /// Number of layers in the decoder network.
    /// Affects prediction complexity.
    /// Default is 3.
    /// </value>
    public int NumDecoderLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of adaptation network layers.
    /// </summary>
    /// <value>
    /// Number of layers in the fast weight generation network.
    /// Default is 3.
    /// </value>
    public int NumAdaptationLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the support set size for adaptation.
    /// </summary>
    /// <value>
    /// Number of examples to use for fast adaptation.
    /// Typical few-shot scenarios use 5 or 10.
    /// Default is 5.
    /// </value>
    public int SupportSetSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use attention for context aggregation.
    /// </summary>
    /// <value>
    /// If true, uses multi-head attention for combining context points.
    /// If false, uses simple mean pooling.
    /// Default is true.
    /// </value>
    public bool UseAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to predict uncertainty.
    /// </summary>
    /// <value>
    /// If true, predicts mean and variance for each output.
    /// Enables heteroscedastic uncertainty estimation.
    /// Default is false.
    /// </value>
    public bool PredictUncertainty { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to normalize fast weights.
    /// </summary>
    /// <value>
    /// If true, applies weight normalization to generated fast weights.
    /// Helps maintain training stability.
    /// Default is true.
    /// </value>
    public bool NormalizeFastWeights { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>
    /// If true, applies layer normalization in transformer blocks.
    /// Default is true.
    /// </value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>
    /// Dropout rate for regularization (0.0 to 1.0).
    /// Applied to transformer and decoder layers.
    /// Default is 0.1.
    /// </value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the fast weight regularization coefficient.
    /// </summary>
    /// <value>
    /// L2 regularization strength for generated fast weights.
    /// Prevents extreme fast weight values.
    /// Default is 1e-4.
    /// </value>
    public double FastWeightRegularization { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the weight for adaptation loss.
    /// </summary>
    /// <value>
    /// Multiplier for the adaptation regularization term.
    /// Balances task loss with adaptation regularization.
    /// Default is 1.0.
    /// </value>
    public double AdaptationWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the weight for uncertainty loss.
    /// </summary>
    /// <value>
    /// Multiplier for uncertainty regularization term.
    /// Only used when PredictUncertainty is true.
    /// Default is 0.1.
    /// </value>
    public double UncertaintyWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the scale for fast weights.
    /// </summary>
    /// <value>
    /// Scaling factor applied to normalized fast weights.
    /// Controls the magnitude of weight updates.
    /// Default is 0.1.
    /// </value>
    public double FastWeightScale { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the learning rate for fast adaptation.
    /// </summary>
    /// <value>
    /// Learning rate used for updating fast weights.
    /// Separate from meta-learning rate.
    /// Default is 0.01.
    /// </value>
    public double FastLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the number of fast adaptation steps.
    /// </summary>
    /// <value>
    /// Number of gradient steps on support set for fine-tuning.
    /// 0 means no fine-tuning, only use generated fast weights.
    /// Default is 0.
    /// </value>
    public int FastAdaptationSteps { get; set; } = 0;

    /// <summary>
    /// Gets or sets the maximum context points.
    /// </summary>
    /// <value>
    /// Maximum number of context points to process.
    /// Helps with memory management for large tasks.
    /// Default is 100.
    /// </value>
    public int MaxContextPoints { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to use hierarchical representations.
    /// </summary>
    /// <value>
    /// If true, builds hierarchical task representations.
    /// Can capture more complex task structure.
    /// Default is false.
    /// </value>
    public bool UseHierarchicalRepresentation { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of hierarchy levels.
    /// </summary>
    /// <value>
    /// Number of levels in hierarchical representation.
    /// Only used when UseHierarchicalRepresentation is true.
    /// Default is 3.
    /// </value>
    public int NumHierarchyLevels { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use residual connections.
    /// </summary>
    /// <value>
    /// If true, adds residual connections in transformer blocks.
    /// Helps with gradient flow in deep networks.
    /// Default is true.
    /// </value>
    public bool UseResidualConnections { get; set; } = true;

    /// <summary>
    /// Gets or sets the temperature for attention.
    /// </summary>
    /// <value>
    /// Temperature parameter for attention softmax.
    /// Lower values create sharper attention.
    /// Default is 1.0.
    /// </value>
    public double AttentionTemperature { get; set; } = 1.0;

    /// <summary>
    /// Creates a default CNAP configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on original CNAP paper:
    /// - Hidden dimension: 256
    /// - Representation dimension: 128
    /// - 6 transformer blocks
    /// - Multi-head attention with 8 heads
    /// - Support set size: 5
    /// </remarks>
    public CNAPAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 1; // CNAP uses fast weights instead of inner loop
    }

    /// <summary>
    /// Creates a CNAP configuration with custom values.
    /// </summary>
    /// <param name="inputDimension">Input data dimension.</param>
    /// <param name="outputDimension">Output data dimension.</param>
    /// <param name="contextSize">Context set size.</param>
    /// <param name="hiddenDimension">Hidden layer dimension.</param>
    /// <param name="representationDimension">Task representation dimension.</param>
    /// <param name="useAttention">Whether to use attention.</param>
    /// <param name="numTransformerBlocks">Number of transformer blocks.</param>
    /// <param name="supportSetSize">Support set size for adaptation.</param>
    /// <param name="predictUncertainty">Whether to predict uncertainty.</param>
    /// <param name="fastWeightRegularization">Fast weight regularization.</param>
    /// <param name="innerLearningRate">Meta-learning rate.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public CNAPAlgorithmOptions(
        int inputDimension = 128,
        int outputDimension = 10,
        int contextSize = 10,
        int hiddenDimension = 256,
        int representationDimension = 128,
        bool useAttention = true,
        int numTransformerBlocks = 6,
        int supportSetSize = 5,
        bool predictUncertainty = false,
        double fastWeightRegularization = 1e-4,
        double innerLearningRate = 0.001,
        int numEpisodes = 10000)
    {
        InputDimension = inputDimension;
        OutputDimension = outputDimension;
        ContextSize = contextSize;
        HiddenDimension = hiddenDimension;
        RepresentationDimension = representationDimension;
        UseAttention = useAttention;
        NumTransformerBlocks = numTransformerBlocks;
        SupportSetSize = supportSetSize;
        PredictUncertainty = predictUncertainty;
        FastWeightRegularization = fastWeightRegularization;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = 1; // CNAP doesn't use traditional inner loop
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

        // Check dimensions
        if (InputDimension <= 0 || InputDimension > 10000)
            return false;

        if (OutputDimension <= 0 || OutputDimension > 1000)
            return false;

        if (ContextSize <= 0 || ContextSize > MaxContextPoints)
            return false;

        if (HiddenDimension <= 0 || HiddenDimension > 2048)
            return false;

        if (RepresentationDimension <= 0 || RepresentationDimension > 1024)
            return false;

        if (FeedForwardDimension <= 0 || FeedForwardDimension > 4096)
            return false;

        if (AdaptationHiddenDimension <= 0 || AdaptationHiddenDimension > 2048)
            return false;

        // Check attention parameters
        if (UseAttention)
        {
            if (NumAttentionHeads <= 0 || NumAttentionHeads > 32)
                return false;

            if (AttentionDimension <= 0 || AttentionDimension > 256)
                return false;

            if (HiddenDimension % NumAttentionHeads != 0)
                return false;
        }

        // Check layer counts
        if (NumTransformerBlocks <= 0 || NumTransformerBlocks > 20)
            return false;

        if (NumDecoderLayers <= 0 || NumDecoderLayers > 10)
            return false;

        if (NumAdaptationLayers <= 0 || NumAdaptationLayers > 10)
            return false;

        // Check support set
        if (SupportSetSize <= 0 || SupportSetSize > ContextSize)
            return false;

        // Check dropout rate
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        // Check regularization weights
        if (FastWeightRegularization < 0.0 || FastWeightRegularization > 1.0)
            return false;

        if (AdaptationWeight < 0.0 || AdaptationWeight > 10.0)
            return false;

        if (UncertaintyWeight < 0.0 || UncertaintyWeight > 10.0)
            return false;

        // Check fast weight scale
        if (FastWeightScale <= 0.0 || FastWeightScale > 1.0)
            return false;

        // Check fast learning rate
        if (FastLearningRate <= 0.0 || FastLearningRate > 1.0)
            return false;

        // Check fast adaptation steps
        if (FastAdaptationSteps < 0 || FastAdaptationSteps > 100)
            return false;

        // Check max context points
        if (MaxContextPoints <= 0 || MaxContextPoints > 10000)
            return false;

        // Check hierarchical representation
        if (UseHierarchicalRepresentation)
        {
            if (NumHierarchyLevels <= 0 || NumHierarchyLevels > 10)
                return false;
        }

        // Check attention temperature
        if (AttentionTemperature <= 0.0 || AttentionTemperature > 10.0)
            return false;

        return true;
    }

    /// <summary>
    /// Gets the total number of encoder parameters.
    /// </summary>
    /// <returns>Total parameters in encoder network.</returns>
    public int GetEncoderParameterCount()
    {
        // Input embedding
        int paramsCount = InputDimension * HiddenDimension + HiddenDimension;

        // Attention layers
        if (UseAttention)
        {
            // Multi-head attention parameters
            paramsCount += NumTransformerBlocks * (
                3 * HiddenDimension * AttentionDimension + // Q, K, V projections
                AttentionDimension * AttentionDimension +  // Output projection
                HiddenDimension + 2 * AttentionDimension   // Bias terms
            );
        }

        // Transformer feed-forward layers
        paramsCount += NumTransformerBlocks * (
            HiddenDimension * FeedForwardDimension + FeedForwardDimension + // FFN
            FeedForwardDimension * HiddenDimension + HiddenDimension +       // Projection
            UseLayerNorm ? 2 * HiddenDimension * 2 : 0 // Layer norm parameters
        );

        // Output projection
        paramsCount += HiddenDimension * RepresentationDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of decoder parameters.
    /// </summary>
    /// <returns>Total parameters in decoder network.</returns>
    public int GetDecoderParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        int currentDim = InputDimension + RepresentationDimension;
        paramsCount += currentDim * HiddenDimension + HiddenDimension;
        currentDim = HiddenDimension;

        // Hidden layers
        for (int i = 0; i < NumDecoderLayers; i++)
        {
            paramsCount += currentDim * HiddenDimension + HiddenDimension;
        }

        // Output layer
        int outputDim = PredictUncertainty ? 2 * OutputDimension : OutputDimension;
        paramsCount += currentDim * outputDim;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of adaptation network parameters.
    /// </summary>
    /// <returns>Total parameters in adaptation network.</returns>
    public int GetAdaptationParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        int currentDim = RepresentationDimension;
        paramsCount += currentDim * AdaptationHiddenDimension + AdaptationHiddenDimension;
        currentDim = AdaptationHiddenDimension;

        // Hidden layers
        for (int i = 1; i < NumAdaptationLayers; i++)
        {
            paramsCount += currentDim * AdaptationHiddenDimension + AdaptationHiddenDimension;
        }

        // Output layer (fast weights)
        int numModelParams = GetEncoderParameterCount() + GetDecoderParameterCount();
        paramsCount += currentDim * numModelParams;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <returns>Total parameter count across all components.</returns>
    public int GetTotalParameterCount()
    {
        return GetEncoderParameterCount() + GetDecoderParameterCount() + GetAdaptationParameterCount();
    }

    /// <summary>
    /// Gets the effective context size after considering constraints.
    /// </summary>
    /// <returns>Effective context size for processing.</returns>
    public int GetEffectiveContextSize()
    {
        return Math.Min(ContextSize, MaxContextPoints);
    }
}