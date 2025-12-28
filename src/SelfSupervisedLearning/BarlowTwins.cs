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

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (view1, view2) = _augmentation.ApplySimCLR(batch);

        // Forward pass for view 1
        var h1 = _encoder.ForwardWithMemory(view1);
        var z1 = _projector!.Project(h1);

        // Forward pass for view 2
        var h2 = _encoder.ForwardWithMemory(view2);
        var z2 = _projector.Project(h2);

        // Compute Barlow Twins loss with gradients
        var (loss, gradZ1, gradZ2) = _loss.ComputeLossWithGradients(z1, z2);

        // Backward pass for first view
        var gradH1 = _projector.Backward(gradZ1);
        _encoder.Backpropagate(gradH1);

        // Backward pass for second view (accumulates gradients)
        var gradH2 = _projector.Backward(gradZ2);
        _encoder.Backpropagate(gradH2);

        // Update parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateParameters(learningRate);

        // Compute cross-correlation for monitoring
        var crossCorr = _loss.ComputeCrossCorrelation(z1, z2, batchSize);

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize;
        result.NumNegativePairs = 0; // No negatives in Barlow Twins

        // Add Barlow Twins-specific metrics
        result.Metrics["off_diagonal_sum"] = _loss.OffDiagonalSum(crossCorr);
        result.Metrics["lambda"] = NumOps.FromDouble(Lambda);

        return result;
    }

    private void UpdateParameters(T learningRate)
    {
        // Update encoder
        var encoderGrads = _encoder.GetParameterGradients();
        var encoderParams = _encoder.GetParameters();
        var newEncoderParams = new T[encoderParams.Length];

        for (int i = 0; i < encoderParams.Length; i++)
        {
            newEncoderParams[i] = NumOps.Subtract(
                encoderParams[i],
                NumOps.Multiply(learningRate, encoderGrads[i]));
        }
        _encoder.UpdateParameters(new Vector<T>(newEncoderParams));

        // Update projector
        if (_projector is not null)
        {
            var projGrads = _projector.GetParameterGradients();
            var projParams = _projector.GetParameters();
            var newProjParams = new T[projParams.Length];

            for (int i = 0; i < projParams.Length; i++)
            {
                newProjParams[i] = NumOps.Subtract(
                    projParams[i],
                    NumOps.Multiply(learningRate, projGrads[i]));
            }
            _projector.SetParameters(new Vector<T>(newProjParams));
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
