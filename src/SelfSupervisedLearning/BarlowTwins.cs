using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Barlow Twins: Self-Supervised Learning via Redundancy Reduction.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Barlow Twins learns representations by making the
/// cross-correlation matrix between embeddings of two augmented views close to the
/// identity matrix. This achieves both invariance (diagonal = 1) and reduces
/// redundancy between features (off-diagonal = 0).</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>No negatives:</b> Unlike contrastive methods, doesn't need negative samples</item>
/// <item><b>No momentum encoder:</b> Uses a simpler symmetric architecture</item>
/// <item><b>Decorrelation:</b> Explicitly reduces redundancy between feature dimensions</item>
/// <item><b>No asymmetry:</b> Both branches are identical (no predictor needed)</item>
/// </list>
///
/// <para><b>Loss components:</b></para>
/// <list type="bullet">
/// <item><b>Invariance term:</b> Makes diagonal elements = 1 (same view features match)</item>
/// <item><b>Redundancy reduction:</b> Makes off-diagonal elements = 0 (decorrelation)</item>
/// </list>
///
/// <para><b>Cross-correlation matrix:</b></para>
/// <code>
/// C_ij = (1/N) * Σ_b z1[b,i] * z2[b,j]
/// Loss = Σ_i (1 - C_ii)² + λ * Σ_i Σ_{j≠i} C_ij²
/// </code>
///
/// <para><b>Reference:</b> Zbontar et al., "Barlow Twins: Self-Supervised Learning via
/// Redundancy Reduction" (ICML 2021)</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Barlow Twins: Self-Supervised Learning via Redundancy Reduction", "https://arxiv.org/abs/2103.03230", Year = 2021, Authors = "Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny")]
public class BarlowTwins<T> : SSLMethodBase<T>
{
    private readonly BarlowTwinsLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;

    /// <inheritdoc />
    public override string Name => "Barlow Twins";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.NonContrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => false;

    /// <summary>
    /// Gets the lambda (redundancy reduction weight) used in the loss.
    /// </summary>
    public double Lambda => _loss.Lambda;

    /// <summary>
    /// Initializes a new instance of the BarlowTwins class.
    /// </summary>
    /// <param name="encoder">The encoder network (shared between both branches).</param>
    /// <param name="projector">Projection head (typically 3-layer MLP with 8192 output dim).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public BarlowTwins(
        INeuralNetwork<T> encoder,
        IProjectorHead<T> projector,
        SSLConfig? config = null)
        : base(encoder, projector, config ?? new SSLConfig { Method = SSLMethodType.BarlowTwins })
    {
        var btConfig = _config.BarlowTwins ?? new BarlowTwinsConfig();
        var lambda = btConfig.Lambda ?? 0.0051; // Default from paper

        _loss = new BarlowTwinsLoss<T>(lambda);
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    private void UpdateParameters(T learningRate, Vector<T> accumulatedProjGrads)
    {
        // Update encoder
        var encoderGrads = new Vector<T>(_encoder.GetParameterGradients());
        var encoderParams = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(encoderParams, Engine.Multiply(encoderGrads, learningRate)));

        // Update projector with accumulated gradients from both views
        if (_projector is not null)
        {
            var projParams = _projector.GetParameters();
            _projector.SetParameters(Engine.Subtract(projParams, Engine.Multiply(accumulatedProjGrads, learningRate)));
        }
    }

    /// <summary>
    /// Creates a Barlow Twins instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 8192).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 8192).</param>
    /// <param name="lambda">Redundancy reduction weight (default: 0.0051).</param>
    /// <returns>A configured Barlow Twins instance.</returns>
    public static BarlowTwins<T> Create(
        INeuralNetwork<T> encoder,
        int encoderOutputDim,
        int projectionDim = 8192,
        int hiddenDim = 8192,
        double lambda = 0.0051)
    {
        // Barlow Twins uses a 3-layer projector with large dimensions
        var projector = new MLPProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, useBatchNormOnOutput: true);

        var config = new SSLConfig
        {
            Method = SSLMethodType.BarlowTwins,
            BarlowTwins = new BarlowTwinsConfig { Lambda = lambda }
        };

        return new BarlowTwins<T>(encoder, projector, config);
    

}
}
