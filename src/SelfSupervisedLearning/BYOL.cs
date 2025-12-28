using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// BYOL: Bootstrap Your Own Latent - Self-supervised learning without negative samples.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> BYOL is a breakthrough method that learns representations
/// without requiring negative samples. It uses an online network that learns to predict
/// the output of a target network, which is updated as an exponential moving average (EMA)
/// of the online network.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>No negatives:</b> Unlike SimCLR or MoCo, BYOL doesn't need negative samples</item>
/// <item><b>Asymmetric architecture:</b> Online has a predictor, target doesn't</item>
/// <item><b>EMA target:</b> Target network is a slow-moving average of online network</item>
/// <item><b>Symmetric loss:</b> Both views serve as online and target</item>
/// </list>
///
/// <para><b>Architecture:</b></para>
/// <code>
/// Online: encoder → projector → predictor → p
/// Target: encoder → projector → z (stop-gradient)
/// Loss: MSE(normalize(p), normalize(z))
/// </code>
///
/// <para><b>Why it doesn't collapse:</b> The combination of the predictor (asymmetry),
/// EMA updates (target moves slowly), and batch normalization prevents trivial solutions.</para>
///
/// <para><b>Reference:</b> Grill et al., "Bootstrap Your Own Latent - A New Approach to
/// Self-Supervised Learning" (NeurIPS 2020)</para>
/// </remarks>
public class BYOL<T> : SSLMethodBase<T>
{
    private readonly IMomentumEncoder<T> _targetEncoder;
    private readonly SymmetricProjector<T> _onlineProjector;
    private readonly SymmetricProjector<T> _targetProjector;
    private readonly BYOLLoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;
    private readonly double _baseMomentum;

    /// <inheritdoc />
    public override string Name => "BYOL";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.NonContrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => false;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Initializes a new instance of the BYOL class.
    /// </summary>
    /// <param name="encoder">The online encoder network.</param>
    /// <param name="targetEncoder">The target encoder (momentum-updated copy).</param>
    /// <param name="onlineProjector">Symmetric projector with predictor for online network.</param>
    /// <param name="targetProjector">Symmetric projector (no predictor) for target network.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public BYOL(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> targetEncoder,
        SymmetricProjector<T> onlineProjector,
        SymmetricProjector<T> targetProjector,
        SSLConfig? config = null)
        : base(encoder, onlineProjector, config ?? new SSLConfig { Method = SSLMethodType.BYOL })
    {
        _targetEncoder = targetEncoder ?? throw new ArgumentNullException(nameof(targetEncoder));
        _onlineProjector = onlineProjector ?? throw new ArgumentNullException(nameof(onlineProjector));
        _targetProjector = targetProjector ?? throw new ArgumentNullException(nameof(targetProjector));

        if (!_onlineProjector.HasPredictor)
            throw new ArgumentException("Online projector must have a predictor head", nameof(onlineProjector));

        var byolConfig = _config.BYOL ?? new BYOLConfig();
        _baseMomentum = byolConfig.BaseMomentum ?? 0.996;

        _loss = new BYOLLoss<T>();
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (view1, view2) = _augmentation.ApplyBYOL(batch);

        // Online network forward pass for view 1
        var h1Online = _encoder.ForwardWithMemory(view1);
        var z1Online = _onlineProjector.Project(h1Online);
        var p1 = _onlineProjector.Predict(z1Online);

        // Online network forward pass for view 2
        var h2Online = _encoder.ForwardWithMemory(view2);
        var z2Online = _onlineProjector.Project(h2Online);
        var p2 = _onlineProjector.Predict(z2Online);

        // Target network forward pass (no gradients)
        var h1Target = _targetEncoder.Encode(view1);
        var z1Target = _targetProjector.Project(h1Target);
        z1Target = StopGradient<T>.Detach(z1Target);

        var h2Target = _targetEncoder.Encode(view2);
        var z2Target = _targetProjector.Project(h2Target);
        z2Target = StopGradient<T>.Detach(z2Target);

        // Symmetric loss: online(view1) predicts target(view2) and vice versa
        var loss = _loss.ComputeSymmetricLoss(p1, z2Target, p2, z1Target);

        // Backward pass through online network - first path (p1 → z2Target)
        var (_, gradP1) = _loss.ComputeLossWithGradients(p1, z2Target);
        var gradZ1 = _onlineProjector.Backward(gradP1);
        _encoder.Backpropagate(gradZ1);

        // Backward pass through online network - second path (p2 → z1Target)
        var (_, gradP2) = _loss.ComputeLossWithGradients(p2, z1Target);
        var gradZ2 = _onlineProjector.Backward(gradP2);
        _encoder.Backpropagate(gradZ2);

        // Update online network parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateOnlineParameters(learningRate);

        // Update target network with EMA
        _targetEncoder.UpdateFromMainEncoder(_encoder);
        UpdateTargetProjector();

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize * 2; // Symmetric loss
        result.NumNegativePairs = 0; // No negatives in BYOL
        result.Metrics["momentum"] = NumOps.FromDouble(_targetEncoder.Momentum);

        return result;
    }

    private void UpdateOnlineParameters(T learningRate)
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

        // Update online projector
        var projGrads = _onlineProjector.GetParameterGradients();
        var projParams = _onlineProjector.GetParameters();
        var newProjParams = new T[projParams.Length];

        for (int i = 0; i < projParams.Length; i++)
        {
            newProjParams[i] = NumOps.Subtract(
                projParams[i],
                NumOps.Multiply(learningRate, projGrads[i]));
        }
        _onlineProjector.SetParameters(new Vector<T>(newProjParams));
    }

    private void UpdateTargetProjector()
    {
        // EMA update for target projector
        var momentum = NumOps.FromDouble(_targetEncoder.Momentum);
        var oneMinusMomentum = NumOps.Subtract(NumOps.One, momentum);

        var onlineParams = _onlineProjector.GetParameters();
        var targetParams = _targetProjector.GetParameters();
        var newTargetParams = new T[targetParams.Length];

        for (int i = 0; i < targetParams.Length; i++)
        {
            newTargetParams[i] = NumOps.Add(
                NumOps.Multiply(momentum, targetParams[i]),
                NumOps.Multiply(oneMinusMomentum, onlineParams[i]));
        }
        _targetProjector.SetParameters(new Vector<T>(newTargetParams));
    }

    /// <inheritdoc />
    public override void OnEpochStart(int epochNumber)
    {
        base.OnEpochStart(epochNumber);

        // Update momentum schedule (cosine from base to 1.0)
        var byolConfig = _config.BYOL ?? new BYOLConfig();
        var totalEpochs = _config.PretrainingEpochs ?? 300;

        var newMomentum = MomentumEncoder<T>.ScheduleMomentum(
            _baseMomentum, 1.0, epochNumber, totalEpochs);

        _targetEncoder.SetMomentum(newMomentum);
    }

    /// <summary>
    /// Creates a BYOL instance with default configuration.
    /// </summary>
    /// <param name="encoder">The backbone encoder.</param>
    /// <param name="createEncoderCopy">Function to create a copy of the encoder for target.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    /// <param name="projectionDim">Dimension of the projection space (default: 256).</param>
    /// <param name="hiddenDim">Hidden dimension of the projector MLP (default: 4096).</param>
    /// <returns>A configured BYOL instance.</returns>
    public static BYOL<T> Create(
        INeuralNetwork<T> encoder,
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createEncoderCopy,
        int encoderOutputDim,
        int projectionDim = 256,
        int hiddenDim = 4096)
    {
        // Create projectors
        var onlineProjector = new SymmetricProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, predictorHiddenDim: hiddenDim);
        var targetProjector = new SymmetricProjector<T>(
            encoderOutputDim, hiddenDim, projectionDim, predictorHiddenDim: 0);

        // Copy projector parameters (without predictor)
        CopyProjectorToTarget(onlineProjector, targetProjector);

        // Create target encoder
        var encoderCopy = createEncoderCopy(encoder);
        var targetEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new BYOL<T>(encoder, targetEncoder, onlineProjector, targetProjector);
    }

    private static void CopyProjectorToTarget(
        SymmetricProjector<T> online, SymmetricProjector<T> target)
    {
        // Copy only the projector parameters (target doesn't have predictor)
        var onlineParams = online.GetParameters();
        var targetParamCount = target.ParameterCount;

        var targetParams = new T[targetParamCount];
        for (int i = 0; i < targetParamCount; i++)
        {
            targetParams[i] = onlineParams[i];
        }

        target.SetParameters(new Vector<T>(targetParams));
    }
}
