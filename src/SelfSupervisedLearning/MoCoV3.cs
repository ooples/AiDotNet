using AiDotNet.Attributes;
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
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("An Empirical Study of Training Self-Supervised Vision Transformers", "https://arxiv.org/abs/2104.02057", Year = 2021, Authors = "Xinlei Chen, Saining Xie, Kaiming He")]
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

    private void UpdateParameters(T learningRate)
    {
        // Update encoder
        var encoderGrads = new Vector<T>(_encoder.GetParameterGradients());
        var encoderParams = _encoder.GetParameters();
        _encoder.UpdateParameters(Engine.Subtract(encoderParams, Engine.Multiply(encoderGrads, learningRate)));

        // Update projector
        if (_projector is not null)
        {
            var projGrads = new Vector<T>(_projector.GetParameterGradients());
            var projParams = _projector.GetParameters();
            _projector.SetParameters(Engine.Subtract(projParams, Engine.Multiply(projGrads, learningRate)));
        }

        // Update predictor
        if (_predictor is not null)
        {
            var predGrads = new Vector<T>(_predictor.GetParameterGradients());
            var predParams = _predictor.GetParameters();
            _predictor.SetParameters(Engine.Subtract(predParams, Engine.Multiply(predGrads, learningRate)));
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
