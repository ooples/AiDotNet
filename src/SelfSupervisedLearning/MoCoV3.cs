using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Validation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// MoCo v3: An Empirical Study of Training Self-Supervised Vision Transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo v3 adapts momentum contrastive learning specifically for
/// Vision Transformers (ViT). It simplifies the framework by removing the memory queue and
/// using in-batch negatives with a symmetric loss.</para>
///
/// <para><b>Key changes from MoCo v1/v2:</b></para>
/// <list type="bullet">
/// <item><b>No memory queue:</b> Uses in-batch negatives like SimCLR</item>
/// <item><b>Symmetric loss:</b> Both views serve as queries and keys</item>
/// <item><b>Prediction head:</b> Adds a predictor MLP on one branch</item>
/// <item><b>ViT optimizations:</b> Random patch projection, no BN in MLP heads</item>
/// </list>
///
/// <para><b>Training stability for ViT:</b></para>
/// <list type="bullet">
/// <item>Uses lower learning rates and careful initialization</item>
/// <item>Gradient clipping and careful warmup</item>
/// <item>Momentum encoder still provides stable targets</item>
/// </list>
///
/// <para><b>Reference:</b> Chen et al., "An Empirical Study of Training Self-Supervised Vision
/// Transformers" (ICCV 2021)</para>
/// </remarks>
public class MoCoV3<T> : SSLMethodBase<T>
{
    private readonly IMomentumEncoder<T> _momentumEncoder;
    private readonly IProjectorHead<T>? _momentumProjector;
    private readonly IProjectorHead<T>? _predictor;
    private readonly InfoNCELoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;

    /// <inheritdoc />
    public override string Name => "MoCo v3";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.Contrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false; // MoCo v3 doesn't use queue

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Initializes a new instance of the MoCoV3 class.
    /// </summary>
    /// <param name="encoder">The online encoder network (ViT recommended).</param>
    /// <param name="momentumEncoder">The momentum encoder.</param>
    /// <param name="projector">Projection head for online encoder.</param>
    /// <param name="momentumProjector">Projection head for momentum encoder.</param>
    /// <param name="predictor">Predictor head (applied to online branch only).</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MoCoV3(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> momentumEncoder,
        IProjectorHead<T> projector,
        IProjectorHead<T> momentumProjector,
        IProjectorHead<T>? predictor = null,
        SSLConfig? config = null)
        : base(encoder, projector, config ?? CreateMoCoV3Config())
    {
        Guard.NotNull(momentumEncoder);
        _momentumEncoder = momentumEncoder;
        Guard.NotNull(momentumProjector);
        _momentumProjector = momentumProjector;
        _predictor = predictor;

        var temperature = _config.Temperature ?? 0.2; // MoCo v3 uses higher temperature
        _loss = new InfoNCELoss<T>(temperature);
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    private static SSLConfig CreateMoCoV3Config()
    {
        return new SSLConfig
        {
            Method = SSLMethodType.MoCoV3,
            Temperature = 0.2,
            LearningRate = 1.5e-4, // Lower LR for ViT
            WarmupEpochs = 40,
            UseCosineDecay = true,
            MoCo = new MoCoConfig
            {
                Momentum = 0.99,
                UseMLPProjector = true
            }
        };
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (view1, view2) = _augmentation.ApplySimCLR(batch);

        // Forward pass for view 1 through online encoder
        var h1 = _encoder.ForwardWithMemory(view1);
        var z1 = _projector!.Project(h1);
        var q1 = _predictor?.Project(z1) ?? z1;

        // Forward pass for view 2 through online encoder
        var h2 = _encoder.ForwardWithMemory(view2);
        var z2 = _projector.Project(h2);
        var q2 = _predictor?.Project(z2) ?? z2;

        // Forward pass through momentum encoder (no gradients)
        var hk1 = _momentumEncoder.Encode(view1);
        var k1 = _momentumProjector!.Project(hk1);

        var hk2 = _momentumEncoder.Encode(view2);
        var k2 = _momentumProjector.Project(hk2);

        // Detach keys (stop gradient)
        k1 = StopGradient<T>.Detach(k1);
        k2 = StopGradient<T>.Detach(k2);

        // Compute symmetric loss with gradients: L = loss(q1, k2) + loss(q2, k1)
        var (loss1, gradQ1, _) = _loss.ComputeLossInBatchWithGradients(q1, k2);
        var (loss2, gradQ2, _) = _loss.ComputeLossInBatchWithGradients(q2, k1);
        var loss = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Add(loss1, loss2));

        // Scale gradients by 0.5 (symmetric loss averaging)
        var half = NumOps.FromDouble(0.5);
        gradQ1 = ScaleGradient(gradQ1, half);
        gradQ2 = ScaleGradient(gradQ2, half);

        // Backward pass through predictor (if present) and projector for view 1
        var gradZ1 = _predictor?.Backward(gradQ1) ?? gradQ1;
        var gradH1 = _projector!.Backward(gradZ1);
        _encoder.Backpropagate(gradH1);

        // Backward pass through predictor (if present) and projector for view 2
        var gradZ2 = _predictor?.Backward(gradQ2) ?? gradQ2;
        var gradH2 = _projector.Backward(gradZ2);
        _encoder.Backpropagate(gradH2);

        // Update momentum encoder
        _momentumEncoder.UpdateFromMainEncoder(_encoder);

        // Update online encoder parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateParameters(learningRate);

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize * 2; // Symmetric loss
        result.NumNegativePairs = batchSize * (batchSize - 1) * 2;
        result.Metrics["momentum"] = NumOps.FromDouble(_momentumEncoder.Momentum);
        result.Metrics["loss_1_to_2"] = loss1;
        result.Metrics["loss_2_to_1"] = loss2;

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

        // Update predictor
        if (_predictor is not null)
        {
            var predGrads = _predictor.GetParameterGradients();
            var predParams = _predictor.GetParameters();
            var newPredParams = new T[predParams.Length];

            for (int i = 0; i < predParams.Length; i++)
            {
                newPredParams[i] = NumOps.Subtract(
                    predParams[i],
                    NumOps.Multiply(learningRate, predGrads[i]));
            }
            _predictor.SetParameters(new Vector<T>(newPredParams));
        }
    }

    private Tensor<T> ScaleGradient(Tensor<T> grad, T scale)
    {
        var batchSize = grad.Shape[0];
        var dim = grad.Shape[1];
        var scaled = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                scaled[i * dim + j] = NumOps.Multiply(grad[i, j], scale);
            }
        }

        return new Tensor<T>(scaled, [batchSize, dim]);
    }

    /// <inheritdoc />
    public override void OnEpochStart(int epochNumber)
    {
        base.OnEpochStart(epochNumber);

        // Update momentum schedule (cosine from base to 1.0)
        var mocoConfig = _config.MoCo ?? new MoCoConfig();
        var baseMomentum = mocoConfig.Momentum ?? 0.99;
        var totalEpochs = _config.PretrainingEpochs ?? 300;

        var newMomentum = MomentumEncoder<T>.ScheduleMomentum(
            baseMomentum, 1.0, epochNumber, totalEpochs);

        _momentumEncoder.SetMomentum(newMomentum);
    }
}
