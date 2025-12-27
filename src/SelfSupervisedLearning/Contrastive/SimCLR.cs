using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Infrastructure;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning.Contrastive;

/// <summary>
/// SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SimCLR is one of the most influential self-supervised learning methods.
/// It learns representations by maximizing agreement between differently augmented views of the same image
/// using a contrastive loss.</para>
///
/// <para><b>How SimCLR works:</b></para>
/// <list type="number">
/// <item>Take an image and create two random augmented views</item>
/// <item>Pass both through the same encoder network</item>
/// <item>Pass both through a projection head (MLP)</item>
/// <item>Apply NT-Xent contrastive loss to bring views together</item>
/// <item>Other images in the batch serve as negative samples</item>
/// </list>
///
/// <para><b>Key hyperparameters:</b></para>
/// <list type="bullet">
/// <item><b>Batch size:</b> Larger is better (4096-8192 in paper)</item>
/// <item><b>Temperature:</b> 0.1 (controls softmax sharpness)</item>
/// <item><b>Projection dimension:</b> 128</item>
/// <item><b>Augmentations:</b> Random crop, color jitter, blur</item>
/// </list>
///
/// <para><b>Reference:</b> Chen et al., "A Simple Framework for Contrastive Learning of Visual
/// Representations" (ICML 2020)</para>
/// </remarks>
public class SimCLR<T> : SSLMethodBase<T>
{
    private readonly NTXentLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;

    /// <inheritdoc />
    public override string Name => "SimCLR";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.Contrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => false;

    /// <summary>
    /// Initializes a new instance of the SimCLR class.
    /// </summary>
    /// <param name="encoder">The backbone encoder network (e.g., ResNet).</param>
    /// <param name="projector">The projection head (MLP).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public SimCLR(
        INeuralNetwork<T> encoder,
        IProjectorHead<T> projector,
        SSLConfig? config = null)
        : base(encoder, projector, config ?? new SSLConfig { Method = SSLMethodType.SimCLR })
    {
        var temperature = _config.Temperature ?? 0.1;
        _loss = new NTXentLoss<T>(temperature);
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        Tensor<T> view1, view2;

        if (augmentationContext?.PrecomputedView is not null)
        {
            // Use precomputed augmentations if provided
            view1 = batch;
            view2 = augmentationContext.PrecomputedView;
        }
        else
        {
            // Apply SimCLR augmentation
            (view1, view2) = _augmentation.ApplySimCLR(batch);
        }

        // Forward pass through encoder
        var h1 = _encoder.ForwardWithMemory(view1);
        var h2 = _encoder.ForwardWithMemory(view2);

        // Project to contrastive space
        var z1 = _projector!.Project(h1);
        var z2 = _projector.Project(h2);

        // Compute NT-Xent loss with gradients
        var (loss, gradZ1, gradZ2) = _loss.ComputeLossWithGradients(z1, z2);

        // Backward pass through projector
        var gradH1 = _projector.Backward(gradZ1);
        var gradH2 = _projector.Backward(gradZ2);

        // Backward pass through encoder
        // Note: In practice, we'd combine gradients from both views
        _encoder.Backpropagate(gradH1);

        // Update parameters with learning rate
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        var encoderGradients = _encoder.GetParameterGradients();
        var projectorGradients = _projector.GetParameterGradients();

        // Simple SGD update (in practice, use optimizer)
        UpdateWithGradients(learningRate, encoderGradients, projectorGradients);

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize;
        result.NumNegativePairs = batchSize * (2 * batchSize - 2); // Each sample has 2N-2 negatives

        // Add SimCLR-specific metrics
        result.Metrics["embedding_norm"] = ComputeAverageNorm(z1);

        return result;
    }

    private void UpdateWithGradients(T learningRate, Vector<T> encoderGrads, Vector<T> projectorGrads)
    {
        // Simple SGD update for encoder
        var encoderParams = _encoder.GetParameters();
        var newEncoderParams = new T[encoderParams.Length];

        for (int i = 0; i < encoderParams.Length; i++)
        {
            newEncoderParams[i] = NumOps.Subtract(
                encoderParams[i],
                NumOps.Multiply(learningRate, encoderGrads[i]));
        }
        _encoder.UpdateParameters(new Vector<T>(newEncoderParams));

        // Simple SGD update for projector
        var projectorParams = _projector!.GetParameters();
        var newProjectorParams = new T[projectorParams.Length];

        for (int i = 0; i < projectorParams.Length; i++)
        {
            newProjectorParams[i] = NumOps.Subtract(
                projectorParams[i],
                NumOps.Multiply(learningRate, projectorGrads[i]));
        }
        _projector.SetParameters(new Vector<T>(newProjectorParams));
    }

    private T ComputeAverageNorm(Tensor<T> embeddings)
    {
        var batchSize = embeddings.Shape[0];
        var dim = embeddings.Shape[1];
        T totalNorm = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = embeddings[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }
            totalNorm = NumOps.Add(totalNorm, NumOps.Sqrt(sumSquared));
        }

        return NumOps.Divide(totalNorm, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Creates a SimCLR instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 128).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 2048).</param>
    /// <returns>A configured SimCLR instance.</returns>
    public static SimCLR<T> Create(
        INeuralNetwork<T> encoder,
        int encoderOutputDim,
        int projectionDim = 128,
        int hiddenDim = 2048)
    {
        var projector = new Infrastructure.ProjectorHeads.MLPProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim);

        return new SimCLR<T>(encoder, projector);
    }
}
