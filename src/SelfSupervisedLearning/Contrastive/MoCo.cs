using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;
using AiDotNet.SelfSupervisedLearning.Infrastructure;
using AiDotNet.SelfSupervisedLearning.Losses;

namespace AiDotNet.SelfSupervisedLearning.Contrastive;

/// <summary>
/// MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> MoCo is a contrastive learning method that uses a momentum encoder
/// and a memory queue to provide a large pool of consistent negative samples without requiring
/// huge batch sizes.</para>
///
/// <para><b>Key innovations:</b></para>
/// <list type="bullet">
/// <item><b>Momentum Encoder:</b> A slowly-updating copy of the encoder for consistent keys</item>
/// <item><b>Memory Queue:</b> Stores past embeddings as negative samples (65536 by default)</item>
/// <item><b>Asymmetric design:</b> Query from main encoder, keys from momentum encoder</item>
/// </list>
///
/// <para><b>How MoCo works:</b></para>
/// <list type="number">
/// <item>Pass query image through online encoder → query q</item>
/// <item>Pass key image through momentum encoder → positive key k+</item>
/// <item>Get negative keys k- from memory queue</item>
/// <item>Compute InfoNCE loss: pull q closer to k+, push away from k-</item>
/// <item>Update momentum encoder with EMA</item>
/// <item>Enqueue new keys, dequeue oldest</item>
/// </list>
///
/// <para><b>Reference:</b> He et al., "Momentum Contrast for Unsupervised Visual Representation
/// Learning" (CVPR 2020)</para>
/// </remarks>
public class MoCo<T> : SSLMethodBase<T>
{
    private readonly IMomentumEncoder<T> _momentumEncoder;
    private readonly IProjectorHead<T>? _momentumProjector;
    private readonly IMemoryBank<T> _memoryBank;
    private readonly InfoNCELoss<T> _loss;
    private readonly SSLAugmentationPolicies<T> _augmentation;
    private readonly double _baseMomentum;

    /// <inheritdoc />
    public override string Name => "MoCo";

    /// <inheritdoc />
    public override SSLMethodCategory Category => SSLMethodCategory.Contrastive;

    /// <inheritdoc />
    public override bool RequiresMemoryBank => true;

    /// <inheritdoc />
    public override bool UsesMomentumEncoder => true;

    /// <summary>
    /// Gets the memory bank used for negative samples.
    /// </summary>
    public IMemoryBank<T> MemoryBank => _memoryBank;

    /// <summary>
    /// Initializes a new instance of the MoCo class.
    /// </summary>
    /// <param name="encoder">The online encoder network.</param>
    /// <param name="momentumEncoder">The momentum encoder (copy of main encoder).</param>
    /// <param name="projector">Optional projection head for online encoder.</param>
    /// <param name="momentumProjector">Optional projection head for momentum encoder.</param>
    /// <param name="embeddingDim">Dimension of embeddings for memory bank.</param>
    /// <param name="config">Optional SSL configuration.</param>
    public MoCo(
        INeuralNetwork<T> encoder,
        IMomentumEncoder<T> momentumEncoder,
        IProjectorHead<T>? projector = null,
        IProjectorHead<T>? momentumProjector = null,
        int embeddingDim = 128,
        SSLConfig? config = null)
        : base(encoder, projector, config ?? new SSLConfig { Method = SSLMethodType.MoCoV2 })
    {
        _momentumEncoder = momentumEncoder ?? throw new ArgumentNullException(nameof(momentumEncoder));
        _momentumProjector = momentumProjector;

        var mocoConfig = _config.MoCo ?? new MoCoConfig();
        var queueSize = mocoConfig.QueueSize ?? 65536;
        _baseMomentum = mocoConfig.Momentum ?? 0.999;

        _memoryBank = new MemoryBank<T>(queueSize, embeddingDim, _config.Seed);

        var temperature = _config.Temperature ?? 0.07;
        _loss = new InfoNCELoss<T>(temperature);
        _augmentation = new SSLAugmentationPolicies<T>(_config.Seed);
    }

    /// <inheritdoc />
    protected override SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext)
    {
        var batchSize = batch.Shape[0];

        // Create two augmented views
        var (queryView, keyView) = _augmentation.ApplyMoCoV2(batch);

        // Forward pass: query through online encoder
        var hQuery = _encoder.ForwardWithMemory(queryView);
        var query = _projector?.Project(hQuery) ?? hQuery;

        // Forward pass: key through momentum encoder (no gradients)
        var hKey = _momentumEncoder.Encode(keyView);
        var key = _momentumProjector?.Project(hKey) ?? hKey;

        // Get negatives from memory bank
        var negatives = _memoryBank.GetAll();

        // Compute InfoNCE loss
        T loss;
        Tensor<T> gradQuery;

        if (_memoryBank.CurrentSize > 0)
        {
            (loss, gradQuery, _) = _loss.ComputeLossWithGradients(query, key, negatives);
        }
        else
        {
            // Fallback to in-batch negatives if queue is empty
            (loss, gradQuery, _) = _loss.ComputeLossInBatchWithGradients(query, key);
        }

        // Backward pass through online encoder
        if (_projector is not null)
        {
            var gradH = _projector.Backward(gradQuery);
            _encoder.Backpropagate(gradH);
        }
        else
        {
            _encoder.Backpropagate(gradQuery);
        }

        // Update online encoder parameters
        var learningRate = NumOps.FromDouble(GetEffectiveLearningRate());
        UpdateEncoderParameters(learningRate);

        // Update momentum encoder (EMA)
        _momentumEncoder.UpdateFromMainEncoder(_encoder);

        // Enqueue new keys to memory bank
        _memoryBank.Enqueue(key);

        // Create result
        var result = CreateStepResult(loss);
        result.NumPositivePairs = batchSize;
        result.NumNegativePairs = _memoryBank.CurrentSize;
        result.Metrics["queue_size"] = NumOps.FromDouble(_memoryBank.CurrentSize);
        result.Metrics["momentum"] = NumOps.FromDouble(_momentumEncoder.Momentum);

        return result;
    }

    private void UpdateEncoderParameters(T learningRate)
    {
        var encoderGrads = _encoder.GetParameterGradients();
        var encoderParams = _encoder.GetParameters();
        var newParams = new T[encoderParams.Length];

        for (int i = 0; i < encoderParams.Length; i++)
        {
            newParams[i] = NumOps.Subtract(
                encoderParams[i],
                NumOps.Multiply(learningRate, encoderGrads[i]));
        }

        _encoder.UpdateParameters(new Vector<T>(newParams));

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

    /// <inheritdoc />
    public override void Reset()
    {
        base.Reset();
        _memoryBank.Clear();
    }

    /// <inheritdoc />
    protected override int GetAdditionalParameterCount()
    {
        return _momentumEncoder.GetParameters().Length +
               (_momentumProjector?.ParameterCount ?? 0);
    }

    /// <inheritdoc />
    protected override Vector<T>? GetAdditionalParameters()
    {
        var momentumParams = _momentumEncoder.GetParameters();
        var projParams = _momentumProjector?.GetParameters();

        if (projParams is null)
            return momentumParams;

        var combined = new T[momentumParams.Length + projParams.Length];
        for (int i = 0; i < momentumParams.Length; i++)
            combined[i] = momentumParams[i];
        for (int i = 0; i < projParams.Length; i++)
            combined[momentumParams.Length + i] = projParams[i];

        return new Vector<T>(combined);
    }
}
