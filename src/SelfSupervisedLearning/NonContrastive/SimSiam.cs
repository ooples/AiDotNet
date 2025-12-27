using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Infrastructure;
using AiDotNet.SelfSupervisedLearning.Infrastructure.ProjectorHeads;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning.NonContrastive;

/// <summary>
/// SimSiam: Exploring Simple Siamese Representation Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SimSiam shows that simple Siamese networks can learn
/// meaningful representations without negative pairs, momentum encoder, or large batches.
/// The key is the stop-gradient operation applied to one branch.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>No negatives:</b> Like BYOL, doesn't need negative samples</item>
/// <item><b>No momentum encoder:</b> Unlike BYOL, uses the same weights for both branches</item>
/// <item><b>No large batches:</b> Works with batch sizes as small as 256</item>
/// <item><b>Stop-gradient:</b> The key to preventing collapse</item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Branch 1: encoder → projector → predictor → p₁
/// Branch 2: encoder → projector → z₂ (stop-gradient)
/// Loss: D(p₁, stopgrad(z₂)) + D(p₂, stopgrad(z₁))
/// </code>
///
/// <para><b>Why it works:</b> The stop-gradient prevents both branches from collapsing
/// to the same constant output. The predictor makes one branch "predict" the other,
/// creating useful gradients for learning.</para>
///
/// <para><b>Reference:</b> Chen and He, "Exploring Simple Siamese Representation Learning"
/// (CVPR 2021)</para>
/// </remarks>
public class SimSiam<T> : SSLMethodBase<T>
{
    private readonly BYOLLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;

    /// <summary>
    /// Gets the typed symmetric projector.
    /// </summary>
    private SymmetricProjector<T> SymmetricProjector => (SymmetricProjector<T>)_projector!;

    /// <inheritdoc />
    public override string Name => "SimSiam";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.NonContrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => false; // Key difference from BYOL

    /// <summary>
    /// Initializes a new instance of the SimSiam class.
    /// </summary>
    /// <param name="encoder">The encoder network (shared between both branches).</param>
    /// <param name="projector">Symmetric projector with predictor.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public SimSiam(
        INeuralNetwork<T> encoder,
        SymmetricProjector<T> projector,
        SSLConfig? config = null)
        : base(encoder, projector, config ?? CreateSimSiamConfig())
    {
        if (projector is null)
            throw new ArgumentNullException(nameof(projector));

        if (!projector.HasPredictor)
            throw new ArgumentException("Projector must have a predictor head", nameof(projector));

        _loss = new BYOLLoss<T>();
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    private static SSLConfig CreateSimSiamConfig()
    {
        return new SSLConfig
        {
            Method = SSLMethodType.SimSiam,
            LearningRate = 0.05, // SGD with base LR
            UseCosineDecay = true,
            BatchSize = 512 // Works with smaller batches than SimCLR
        };
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (view1, view2) = _augmentation.ApplySimCLR(batch);

        // Forward pass for view 1
        var h1 = _encoder.ForwardWithMemory(view1);
        var z1 = SymmetricProjector.Project(h1);
        var p1 = SymmetricProjector.Predict(z1);

        // Forward pass for view 2
        var h2 = _encoder.ForwardWithMemory(view2);
        var z2 = SymmetricProjector.Project(h2);
        var p2 = SymmetricProjector.Predict(z2);

        // Apply stop-gradient to z1 and z2
        var z1Detached = StopGradient<T>.Detach(z1);
        var z2Detached = StopGradient<T>.Detach(z2);

        // Symmetric loss: D(p1, stopgrad(z2)) + D(p2, stopgrad(z1))
        var loss = _loss.ComputeSymmetricLoss(p1, z2Detached, p2, z1Detached);

        // Backward pass for both symmetric paths
        var (_, gradP1) = _loss.ComputeLossWithGradients(p1, z2Detached);
        var (_, gradP2) = _loss.ComputeLossWithGradients(p2, z1Detached);

        // Backpropagate gradients from first path (p1 -> z2)
        var gradZ1 = SymmetricProjector.Backward(gradP1);
        _encoder.Backpropagate(gradZ1);

        // Backpropagate gradients from second path (p2 -> z1)
        var gradZ2 = SymmetricProjector.Backward(gradP2);
        _encoder.Backpropagate(gradZ2);

        // Update parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateParameters(learningRate);

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize * 2; // Symmetric loss
        result.NumNegativePairs = 0; // No negatives in SimSiam

        // Add SimSiam-specific metrics
        result.Metrics["z1_std"] = ComputeStd(z1);
        result.Metrics["z2_std"] = ComputeStd(z2);

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
        var projGrads = SymmetricProjector.GetParameterGradients();
        var projParams = SymmetricProjector.GetParameters();
        var newProjParams = new T[projParams.Length];

        for (int i = 0; i < projParams.Length; i++)
        {
            newProjParams[i] = NumOps.Subtract(
                projParams[i],
                NumOps.Multiply(learningRate, projGrads[i]));
        }
        SymmetricProjector.SetParameters(new Vector<T>(newProjParams));
    }

    private T ComputeStd(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];

        // Compute mean
        T mean = NumOps.Zero;
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                mean = NumOps.Add(mean, tensor[b, d]);
            }
        }
        mean = NumOps.Divide(mean, NumOps.FromDouble(batchSize * dim));

        // Compute variance
        T variance = NumOps.Zero;
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                var diff = NumOps.Subtract(tensor[b, d], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
        }
        variance = NumOps.Divide(variance, NumOps.FromDouble(batchSize * dim));

        return NumOps.Sqrt(variance);
    }

    /// <summary>
    /// Creates a SimSiam instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 2048).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 2048).</param>
    /// <param name="predictorHiddenDim">Hidden dimension of the predictor (default: 512).</param>
    /// <returns>A configured SimSiam instance.</returns>
    public static SimSiam<T> Create(
        INeuralNetwork<T> encoder,
        int encoderOutputDim,
        int projectionDim = 2048,
        int hiddenDim = 2048,
        int predictorHiddenDim = 512)
    {
        var projector = new SymmetricProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, predictorHiddenDim);

        return new SimSiam<T>(encoder, projector);
    }
}
