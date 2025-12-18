using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for Task-Adaptive Domain Adaptation Meta-learning (TADAM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TADAM extends MAML with domain adaptation capabilities through task-specific
/// feature transformation and adversarial domain learning. It learns to adapt
/// the feature space to be domain-invariant while preserving task-specific information.
/// </para>
/// <para><b>For Beginners:</b> This configuration controls how TADAM adapts across domains:
///
/// Key parameters:
/// - <b>NumDomains:</b> Number of different domains to handle
/// - <b>TaskEmbeddingDimension:</b> Size of task-specific representations
/// - <b>DomainAdversarialWeight:</b> Strength of domain adversarial training
/// - <b>UseFeatureTransformation:</b> Whether to adapt features per task
/// - <b>GradientReversalScale:</b> Scale factor for gradient reversal layer
/// </para>
/// <para>
/// <b>Advanced Features:</b>
/// - Adversarial domain adaptation with gradient reversal
/// - Task-specific feature transformation networks
/// - Residual blocks for stable training
/// - Multi-head attention for task encoding
/// - FiLM modulation for feature adaptation
/// - Hierarchical domain representations
/// </para>
/// </remarks>
public class TADAMAlgorithmOptions<T, TInput, TOutput> 
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
    /// Gets or sets the number of domains.
    /// </summary>
    /// <value>
    /// Number of different domains in the meta-dataset.
    /// TADAM will learn to adapt across these domains.
    /// Default is 5.
    /// </value>
    public int NumDomains { get; set; } = 5;

    /// <summary>
    /// Gets or sets the task embedding dimension.
    /// </summary>
    /// <value>
    /// Dimension of task-specific representation vectors.
    /// Captures task and domain information.
    /// Default is 64.
    /// </value>
    public int TaskEmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the hidden layer dimension.
    /// </summary>
    /// <value>
    /// Dimension of hidden layers in networks.
    /// Affects model capacity.
    /// Default is 256.
    /// </value>
    public int HiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the task encoder hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for task encoder network.
    /// Processes task-specific information.
    /// Default is 128.
    /// </value>
    public int TaskHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the domain classifier hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for domain classifier.
    /// Distinguishes between different domains.
    /// Default is 128.
    /// </value>
    public int DomainHiddenDimension { get; set; } = 128;

    /// <summary>
    /// Gets or sets the feature transformer hidden dimension.
    /// </summary>
    /// <value>
    /// Hidden dimension for feature transformation network.
    /// Adapts features for domain invariance.
    /// Default is 256.
    /// </value>
    public int FeatureHiddenDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the number of residual blocks.
    /// </summary>
    /// <value>
    /// Number of residual blocks in the base model.
    /// Helps with gradient flow in deep networks.
    /// Default is 4.
    /// </value>
    public int NumResidualBlocks { get; set; } = 4;

    /// <summary>
    /// Gets or sets the number of domain classifier layers.
    /// </summary>
    /// <value>
    /// Number of hidden layers in domain classifier.
    /// Default is 2.
    /// </value>
    public int NumDomainLayers { get; set; } = 2;

    /// <summary>
    /// Gets or sets the number of feature transformation layers.
    /// </summary>
    /// <value>
    /// Number of layers in feature transformer.
    /// Only used when UseFeatureTransformation is true.
    /// Default is 3.
    /// </value>
    public int NumFeatureLayers { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of task attention heads.
    /// </summary>
    /// <value>
    /// Number of attention heads for task encoding.
    /// Only used when UseTaskAttention is true.
    /// Default is 8.
    /// </value>
    public int NumTaskAttentionHeads { get; set; } = 8;

    /// <summary>
    /// Gets or sets the task attention dimension.
    /// </summary>
    /// <value>
    /// Dimension of each attention head.
    /// Only used when UseTaskAttention is true.
    /// Default is 32.
    /// </value>
    public int TaskAttentionDimension { get; set; } = 32;

    /// <summary>
    /// Gets or sets the support set size for adaptation.
    /// </summary>
    /// <value>
    /// Number of examples to use for task adaptation.
    /// Typical few-shot scenarios use 5 or 10.
    /// Default is 5.
    /// </value>
    public int SupportSetSize { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use task attention.
    /// </summary>
    /// <value>
    /// If true, uses multi-head attention for task encoding.
    /// Captures complex task relationships.
    /// Default is true.
    /// </value>
    public bool UseTaskAttention { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use feature transformation.
    /// </summary>
    /// <value>
    /// If true, applies task-specific feature transformation.
    /// Helps with domain adaptation.
    /// Default is true.
    /// </value>
    public bool UseFeatureTransformation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use layer normalization.
    /// </summary>
    /// <value>
    /// If true, applies layer normalization in residual blocks.
    /// Helps with training stability.
    /// Default is true.
    /// </value>
    public bool UseLayerNorm { get; set; } = true;

    /// <summary>
    /// Gets or sets the gradient reversal scale.
    /// </summary>
    /// <value>
    /// Scale factor for gradient reversal in domain classifier.
    /// Controls strength of adversarial training.
    /// Default is 1.0.
    /// </value>
    public double GradientReversalScale { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the domain adversarial weight.
    /// </summary>
    /// <value>
    /// Weight for domain adversarial loss in total objective.
    /// Balances task loss and domain adaptation.
    /// Default is 0.1.
    /// </value>
    public double DomainAdversarialWeight { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the feature regularization weight.
    /// </summary>
    /// <value>
    /// Weight for feature transformation regularization.
    /// Prevents over-adaptation of features.
    /// Default is 0.01.
    /// </value>
    public double FeatureRegularizationWeight { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the task parameter regularization weight.
    /// </summary>
    /// <value>
    /// L2 regularization for task-specific parameters.
    /// Prevents extreme parameter modifications.
    /// Default is 1e-4.
    /// </value>
    public double TaskParameterRegularization { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets the dropout rate.
    /// </summary>
    /// <value>
    /// Dropout rate for regularization (0.0 to 1.0).
    /// Applied to hidden layers.
    /// Default is 0.1.
    /// </value>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the adaptation temperature.
    /// </summary>
    /// <value>
    /// Temperature parameter for task-specific scaling.
    /// Controls adaptation magnitude.
    /// Default is 1.0.
    /// </value>
    public double AdaptationTemperature { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use hierarchical domains.
    /// </summary>
    /// <value>
    /// If true, models hierarchical domain relationships.
    /// Useful when domains have natural hierarchy.
    /// Default is false.
    /// </value>
    public bool UseHierarchicalDomains { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of domain hierarchy levels.
    /// </summary>
    /// <value>
    /// Number of levels in domain hierarchy.
    /// Only used when UseHierarchicalDomains is true.
    /// Default is 3.
    /// </value>
    public int NumDomainHierarchyLevels { get; set; } = 3;

    /// <summary>
    /// Gets or sets whether to use curriculum learning.
    /// </summary>
    /// <value>
    /// If true, gradually increases domain difficulty.
    /// Helps with stable training.
    /// Default is false.
    /// </value>
    public bool UseCurriculumLearning { get; set; } = false;

    /// <summary>
    /// Gets or sets the curriculum warmup episodes.
    /// </summary>
    /// <value>
    /// Number of episodes for curriculum warmup.
    /// Only used when UseCurriculumLearning is true.
    /// Default is 1000.
    /// </value>
    public int CurriculumWarmupEpisodes { get; set; } = 1000;

    /// <summary>
    /// Gets or sets whether to use domain-specific batch normalization.
    /// </summary>
    /// <value>
    /// If true, maintains separate BN statistics per domain.
    /// Helps with domain-specific feature distributions.
    /// Default is false.
    /// </value>
    public bool UseDomainSpecificBatchNorm { get; set; } = false;

    /// <summary>
    /// Gets or sets the momentum for domain statistics.
    /// </summary>
    /// <value>
    /// Momentum for updating domain-specific BN statistics.
    /// Only used when UseDomainSpecificBatchNorm is true.
    /// Default is 0.1.
    /// </value>
    public double DomainBatchNormMomentum { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use task-specific biases.
    /// </summary>
    /// <value>
    /// If true, learns task-specific bias terms.
    /// Allows for quick adaptation of outputs.
    /// Default is true.
    /// </value>
    public bool UseTaskSpecificBiases { get; set; } = true;

    /// <summary>
    /// Gets or sets the bias regularization weight.
    /// </summary>
    /// <value>
    /// L2 regularization for task-specific biases.
    /// Only used when UseTaskSpecificBiases is true.
    /// Default is 1e-5.
    /// </value>
    public double BiasRegularizationWeight { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use domain adversarial scheduler.
    /// </summary>
    /// <value>
    /// If true, schedules domain adversarial weight.
    /// Gradually increases domain adaptation difficulty.
    /// Default is false.
    /// </value>
    public bool UseDomainAdversarialScheduler { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum domain adversarial weight.
    /// </summary>
    /// <value>
    /// Maximum value for domain adversarial weight.
    /// Only used when UseDomainAdversarialScheduler is true.
    /// Default is 1.0.
    /// </value>
    public double MaxDomainAdversarialWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the domain adversarial warmup episodes.
    /// </summary>
    /// <value>
    /// Episodes over which to ramp up domain adversarial weight.
    /// Only used when UseDomainAdversarialScheduler is true.
    /// Default is 5000.
    /// </value>
    public int DomainAdversarialWarmupEpisodes { get; set; } = 5000;

    /// <summary>
    /// Creates a default TADAM configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default configuration based on TADAM paper:
    /// - Hidden dimension: 256
    /// - Task embedding dimension: 64
    /// - 4 residual blocks
    /// - Domain adversarial training
    /// - Task-specific feature transformation
    /// </remarks>
    public TADAMAlgorithmOptions()
    {
        // Set default values
        InnerLearningRate = NumOps.FromDouble(0.001);
        AdaptationSteps = 5;
    }

    /// <summary>
    /// Creates a TADAM configuration with custom values.
    /// </summary>
    /// <param name="inputDimension">Input data dimension.</param>
    /// <param name="outputDimension">Output data dimension.</param>
    /// <param name="numDomains">Number of domains.</param>
    /// <param name="taskEmbeddingDimension">Task embedding dimension.</param>
    /// <param name="hiddenDimension">Hidden layer dimension.</param>
    /// <param name="useFeatureTransformation">Whether to use feature transformation.</param>
    /// <param name="domainAdversarialWeight">Domain adversarial loss weight.</param>
    /// <param name="gradientReversalScale">Gradient reversal scale.</param>
    /// <param name="supportSetSize">Support set size.</param>
    /// <param name="innerLearningRate">Inner loop learning rate.</param>
    /// <param name="adaptationSteps">Number of adaptation steps.</param>
    /// <param name="numEpisodes">Number of training episodes.</param>
    public TADAMAlgorithmOptions(
        int inputDimension = 128,
        int outputDimension = 10,
        int numDomains = 5,
        int taskEmbeddingDimension = 64,
        int hiddenDimension = 256,
        bool useFeatureTransformation = true,
        double domainAdversarialWeight = 0.1,
        double gradientReversalScale = 1.0,
        int supportSetSize = 5,
        double innerLearningRate = 0.001,
        int adaptationSteps = 5,
        int numEpisodes = 10000)
    {
        InputDimension = inputDimension;
        OutputDimension = outputDimension;
        NumDomains = numDomains;
        TaskEmbeddingDimension = taskEmbeddingDimension;
        HiddenDimension = hiddenDimension;
        UseFeatureTransformation = useFeatureTransformation;
        DomainAdversarialWeight = domainAdversarialWeight;
        GradientReversalScale = gradientReversalScale;
        SupportSetSize = supportSetSize;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        AdaptationSteps = adaptationSteps;
        NumEpisodes = numEpisodes;
    }

    /// <summary>
    /// Validates the configuration parameters.
    /// </summary>
    /// <returns>True if all parameters are valid, false otherwise.</returns>
    public virtual bool IsValid()
    {
        // Check base class validation
            return false;

        // Check dimensions
        if (InputDimension <= 0 || InputDimension > 10000)
            return false;

        if (OutputDimension <= 0 || OutputDimension > 1000)
            return false;

        if (NumDomains <= 0 || NumDomains > 100)
            return false;

        if (TaskEmbeddingDimension <= 0 || TaskEmbeddingDimension > 512)
            return false;

        if (HiddenDimension <= 0 || HiddenDimension > 2048)
            return false;

        if (TaskHiddenDimension <= 0 || TaskHiddenDimension > 1024)
            return false;

        if (DomainHiddenDimension <= 0 || DomainHiddenDimension > 1024)
            return false;

        if (FeatureHiddenDimension <= 0 || FeatureHiddenDimension > 2048)
            return false;

        // Check layer counts
        if (NumResidualBlocks <= 0 || NumResidualBlocks > 20)
            return false;

        if (NumDomainLayers <= 0 || NumDomainLayers > 10)
            return false;

        if (NumFeatureLayers <= 0 || NumFeatureLayers > 10)
            return false;

        // Check attention parameters
        if (UseTaskAttention)
        {
            if (NumTaskAttentionHeads <= 0 || NumTaskAttentionHeads > 32)
                return false;

            if (TaskAttentionDimension <= 0 || TaskAttentionDimension > 256)
                return false;

            if (TaskHiddenDimension % NumTaskAttentionHeads != 0)
                return false;
        }

        // Check support set
        if (SupportSetSize <= 0 || SupportSetSize > 100)
            return false;

        // Check regularization weights
        if (DomainAdversarialWeight < 0.0 || DomainAdversarialWeight > 10.0)
            return false;

        if (FeatureRegularizationWeight < 0.0 || FeatureRegularizationWeight > 1.0)
            return false;

        if (TaskParameterRegularization < 0.0 || TaskParameterRegularization > 1.0)
            return false;

        if (BiasRegularizationWeight < 0.0 || BiasRegularizationWeight > 1.0)
            return false;

        // Check other hyperparameters
        if (DropoutRate < 0.0 || DropoutRate >= 1.0)
            return false;

        if (GradientReversalScale <= 0.0 || GradientReversalScale > 10.0)
            return false;

        if (AdaptationTemperature <= 0.0 || AdaptationTemperature > 10.0)
            return false;

        // Check hierarchical domains
        if (UseHierarchicalDomains)
        {
            if (NumDomainHierarchyLevels <= 0 || NumDomainHierarchyLevels > 10)
                return false;
        }

        // Check curriculum learning
        if (UseCurriculumLearning)
        {
            if (CurriculumWarmupEpisodes < 0)
                return false;
        }

        // Check domain-specific batch norm
        if (UseDomainSpecificBatchNorm)
        {
            if (DomainBatchNormMomentum < 0.0 || DomainBatchNormMomentum >= 1.0)
                return false;
        }

        // Check domain adversarial scheduler
        if (UseDomainAdversarialScheduler)
        {
            if (MaxDomainAdversarialWeight <= 0.0 || MaxDomainAdversarialWeight > 10.0)
                return false;

            if (DomainAdversarialWarmupEpisodes < 0)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Gets the current domain adversarial weight.
    /// </summary>
    /// <param name="episode">Current episode number.</param>
    /// <returns>Domain adversarial weight for this episode.</returns>
    public double GetCurrentDomainAdversarialWeight(int episode)
    {
        if (!UseDomainAdversarialScheduler)
        {
            return DomainAdversarialWeight;
        }

        if (episode >= DomainAdversarialWarmupEpisodes)
        {
            return MaxDomainAdversarialWeight;
        }

        // Linear ramp-up
        return MaxDomainAdversarialWeight * (episode / (double)DomainAdversarialWarmupEpisodes);
    }

    /// <summary>
    /// Gets the total number of base model parameters.
    /// </summary>
    /// <returns>Total parameters in base model.</returns>
    public int GetBaseModelParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        paramsCount += InputDimension * HiddenDimension + HiddenDimension;

        // Residual blocks
        for (int i = 0; i < NumResidualBlocks; i++)
        {
            // Each residual block typically has 2-3 layers
            paramsCount += 2 * (HiddenDimension * HiddenDimension + HiddenDimension);

            if (UseLayerNorm)
            {
                paramsCount += 2 * HiddenDimension * 2; // Layer norm parameters
            }
        }

        // Output layer
        paramsCount += HiddenDimension * OutputDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of task encoder parameters.
    /// </summary>
    /// <returns>Total parameters in task encoder.</returns>
    public int GetTaskEncoderParameterCount()
    {
        int paramsCount = 0;

        // Input layer
        paramsCount += 2 * InputDimension * TaskHiddenDimension + TaskHiddenDimension;

        // Attention layers
        if (UseTaskAttention)
        {
            paramsCount += NumTaskAttentionHeads * (
                TaskHiddenDimension * TaskAttentionDimension * 2 + // Q and K
                TaskHiddenDimension * TaskAttentionDimension +       // V
                TaskAttentionDimension * TaskAttentionDimension     // Output projection
            );
        }

        // Output layer
        paramsCount += TaskHiddenDimension * TaskEmbeddingDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of domain classifier parameters.
    /// </summary>
    /// <returns>Total parameters in domain classifier.</returns>
    public int GetDomainClassifierParameterCount()
    {
        int paramsCount = 0;

        // Input layer (after gradient reversal)
        paramsCount += TaskEmbeddingDimension * DomainHiddenDimension + DomainHiddenDimension;

        // Hidden layers
        for (int i = 1; i < NumDomainLayers; i++)
        {
            paramsCount += DomainHiddenDimension * DomainHiddenDimension + DomainHiddenDimension;
        }

        // Output layer
        paramsCount += DomainHiddenDimension * NumDomains;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of feature transformer parameters.
    /// </summary>
    /// <returns>Total parameters in feature transformer.</returns>
    public int GetFeatureTransformerParameterCount()
    {
        if (!UseFeatureTransformation)
        {
            return 0;
        }

        int paramsCount = 0;

        // Input layer
        paramsCount += InputDimension * FeatureHiddenDimension + FeatureHiddenDimension;

        // Hidden layers
        for (int i = 1; i < NumFeatureLayers; i++)
        {
            paramsCount += FeatureHiddenDimension * FeatureHiddenDimension + FeatureHiddenDimension;
        }

        // Output layer
        paramsCount += FeatureHiddenDimension * InputDimension;

        return paramsCount;
    }

    /// <summary>
    /// Gets the total number of parameters in the model.
    /// </summary>
    /// <returns>Total parameter count across all components.</returns>
    public int GetTotalParameterCount()
    {
        return GetBaseModelParameterCount() +
               GetTaskEncoderParameterCount() +
               GetDomainClassifierParameterCount() +
               GetFeatureTransformerParameterCount();
    }

    /// <summary>
    /// Gets the memory requirement estimate in MB.
    /// </summary>
    /// <returns>Estimated memory usage in megabytes.</returns>
    public double GetMemoryRequirementMB()
    {
        // Rough estimate: 4 bytes per parameter (float32) + overhead
        var totalParams = GetTotalParameterCount();
        var paramMemoryMB = (totalParams * 4.0) / (1024 * 1024);

        // Add overhead for activations, gradients, and task embeddings
        var overheadMB = (HiddenDimension + TaskEmbeddingDimension) * SupportSetSize * 4.0 / (1024 * 1024);

        return paramMemoryMB + overheadMB;
    }
}