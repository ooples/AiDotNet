using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Exploring Simple Siamese Representation Learning", "https://arxiv.org/abs/2011.10566", Year = 2021, Authors = "Xinlei Chen, Kaiming He")]
public class SimSiam<T> : SSLMethodBase<T>
{
    private readonly BYOLLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;

    /// <summary>
    /// Gets the typed symmetric projector.
    /// </summary>
    private SymmetricProjector<T> SymmetricProjector => (SymmetricProjector<T>)(_projector ?? throw new InvalidOperationException("Projector has not been initialized."));

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

    private void UpdateParameters(T learningRate)
    {
        // Update encoder
        var encoderGrads = new Vector<T>(_encoder.GetParameterGradients());
        var encoderParams = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(encoderParams, Engine.Multiply(encoderGrads, learningRate)));

        // Update projector
        var projGrads = SymmetricProjector.GetParameterGradients();
        var projParams = SymmetricProjector.GetParameters();
        SymmetricProjector.SetParameters(Engine.Subtract(projParams, Engine.Multiply(projGrads, learningRate)));
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
