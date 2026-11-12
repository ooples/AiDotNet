using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Implements Ghost Batch Normalization, a regularization technique used in TabNet
/// that applies batch normalization to virtual mini-batches within each actual batch.
/// </summary>
/// <remarks>
/// <para>
/// Ghost Batch Normalization divides each training batch into smaller "virtual batches"
/// and computes separate normalization statistics for each. This provides a regularization
/// effect similar to using smaller batch sizes without the computational overhead.
/// </para>
/// <para>
/// <b>For Beginners:</b> Batch Normalization helps neural networks train faster by
/// normalizing the inputs to each layer. Ghost Batch Normalization takes this further
/// by adding controlled randomness through virtual batches.
///
/// Imagine you have a batch of 256 samples:
/// - Standard Batch Norm: Computes mean/variance over all 256 samples
/// - Ghost Batch Norm (virtual size 64): Computes 4 separate mean/variance calculations,
///   one for each group of 64 samples
///
/// This variation in statistics acts as regularization, helping prevent overfitting.
/// It's particularly effective for tabular data where overfitting is common.
/// </para>
/// <para>
/// The scale (gamma) and shift (beta) are trainable parameters registered with the layer machinery, so a
/// composite that registers this layer as a sub-layer exposes them to the parameter vector, checkpoints and
/// the autodiff tape. The running statistics are non-trainable buffers.
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Regularizer)]
[PipelineStage(PipelineStage.Training)]
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.ActivationNormalization)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, HasTrainingMode = true, TestInputShape = "4, 3", TestConstructorArgs = "3")]
[ElementWiseShape(Note = "Per-feature batch normalisation; every dimension is carried through.")]
[AutoParameters]
public partial class GhostBatchNormalization<T> : LayerBase<T>, IShapeContract
{
    private readonly int _numFeatures;
    private readonly int _virtualBatchSize;
    private readonly double _momentum;
    private readonly double _epsilon;

    // Learnable scale and shift
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _gamma;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _beta;

    private Tensor<T>? _gammaGradient;
    private Tensor<T>? _betaGradient;

    // Running statistics for inference
    [Buffer(Name = "running_mean", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _runningMean;
    [Buffer(Name = "running_variance", Role = PersistentTensorRole.Constant)]
    private Tensor<T> _runningVar;

    /// <summary>
    /// Gets the name of this layer.
    /// </summary>
    public string Name => "GhostBatchNormalization";

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the GhostBatchNormalization class.
    /// </summary>
    /// <param name="numFeatures">The number of features (channels) to normalize.</param>
    /// <param name="virtualBatchSize">The size of each virtual batch. Default is 128.</param>
    /// <param name="momentum">The momentum for running statistics. Default is 0.02.</param>
    /// <param name="epsilon">Small constant for numerical stability. Default is 1e-5.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating Ghost Batch Normalization:
    /// - numFeatures: Should match the number of features in your data
    /// - virtualBatchSize: Smaller = more regularization (try 32-128)
    /// - momentum: How quickly running stats adapt (smaller = slower adaptation)
    /// - epsilon: Prevents division by zero (rarely needs changing)
    /// </para>
    /// </remarks>
    public GhostBatchNormalization(
        int numFeatures,
        int virtualBatchSize = 128,
        double momentum = 0.02,
        double epsilon = 1e-5)
        : base(new[] { numFeatures }, new[] { numFeatures })
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "numFeatures must be positive.");

        _numFeatures = numFeatures;
        _virtualBatchSize = virtualBatchSize;
        _momentum = momentum;
        _epsilon = epsilon;

        // gamma = 1, beta = 0; running mean 0, running variance 1
        _gamma = Tensor<T>.CreateDefault([numFeatures], NumOps.One);
        _beta = Tensor<T>.CreateDefault([numFeatures], NumOps.Zero);
        _runningMean = Tensor<T>.CreateDefault([numFeatures], NumOps.Zero);
        _runningVar = Tensor<T>.CreateDefault([numFeatures], NumOps.One);

        RegisterTrainableParameter(_gamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_beta, PersistentTensorRole.NormalizationParams);
        RegisterBuffer(_runningMean, "running_mean");
        RegisterBuffer(_runningVar, "running_variance");
    }

    /// <summary>
    /// Performs the forward pass through the Ghost Batch Normalization layer.
    /// </summary>
    /// <param name="input">The input tensor of shape [batch_size, num_features].</param>
    /// <returns>The normalized output tensor.</returns>
    /// <remarks>
    /// <para>
    /// During training, the input is divided into virtual batches and each is normalized
    /// separately. During inference, the running statistics are used instead.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 2)
        {
            throw new ArgumentException($"Expected 2D input [batch_size, features], got {input.Rank}D", nameof(input));
        }

        int batchSize = input.Shape[0];
        int features = input.Shape[1];

        if (features != _numFeatures)
        {
            throw new ArgumentException($"Expected {_numFeatures} features, got {features}", nameof(input));
        }

        var engine = Engine;

        // Choose normalization statistics. Batch statistics need >= 2 samples to be
        // meaningful; with a single sample (common at inference and in the invariant
        // tests, which feed one example) the batch mean equals the sample, so the
        // centered value is identically 0 — the output would not depend on the input
        // and the gradient to upstream layers would vanish. In that case — and always
        // in inference mode — normalize with the running statistics, which are
        // input-INDEPENDENT constants, so the output (and gradient) still varies with x.
        // All Engine ops, so the autodiff tape records the computation (the previous
        // manual-loop implementation was a tape dead-end, so nothing upstream of a
        // GhostBatchNorm could train).
        bool useBatchStats = IsTrainingMode && batchSize >= 2;
        var minusOne = NumOps.FromDouble(-1.0);
        var eps = NumOps.FromDouble(_epsilon);
        Tensor<T> normalized;

        if (useBatchStats && _virtualBatchSize >= 2 && batchSize > _virtualBatchSize
            && batchSize % _virtualBatchSize == 0)
        {
            // Ghost Batch Normalization (Hoffer et al. 2017; the GBN TabNet uses for
            // regularization, Arik & Pfister 2019 §3.3): split the batch into virtual batches of
            // _virtualBatchSize samples and normalize EACH independently — the "ghost" semantics.
            // Reshape [B, F] -> [numGhosts, vbs, F], take per-ghost mean/var over the sample axis,
            // normalize within each ghost, reshape back. Tape-safe (Engine ops).
            int numGhosts = batchSize / _virtualBatchSize;
            var grouped = engine.Reshape(input, new[] { numGhosts, _virtualBatchSize, _numFeatures });
            var gMean = engine.ReduceMean(grouped, new[] { 1 }, keepDims: true);                 // [numGhosts,1,F]
            var gCentered = engine.TensorAdd(grouped, engine.TensorMultiplyScalar(gMean, minusOne));
            var gVar = engine.ReduceMean(engine.TensorMultiply(gCentered, gCentered), new[] { 1 }, keepDims: true);
            var gStd = engine.TensorSqrt(engine.TensorAddScalar(gVar, eps));
            var gNorm = engine.TensorDivide(gCentered, gStd);                            // [numGhosts,vbs,F]
            normalized = engine.Reshape(gNorm, new[] { batchSize, _numFeatures });

            // Running statistics track the FULL-batch mean/var (used at inference, which sees no
            // virtual-batch structure).
            var bMean = engine.ReduceMean(input, new[] { 0 }, keepDims: true);
            var bCentered = engine.TensorAdd(input, engine.TensorMultiplyScalar(bMean, minusOne));
            var bVar = engine.ReduceMean(engine.TensorMultiply(bCentered, bCentered), new[] { 0 }, keepDims: true);
            UpdateRunningStatistics(bMean, bVar);
        }
        else if (useBatchStats)
        {
            // Single virtual batch (batch <= virtualBatchSize, or not an exact multiple): normalize
            // over the whole batch — one ghost.
            var meanRow = engine.ReduceMean(input, new[] { 0 }, keepDims: true);
            var centered = engine.TensorAdd(input, engine.TensorMultiplyScalar(meanRow, minusOne));
            var varRow = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: true);
            UpdateRunningStatistics(meanRow, varRow);
            var std = engine.TensorSqrt(engine.TensorAddScalar(varRow, eps));
            normalized = engine.TensorDivide(centered, std);
        }
        else
        {
            // Inference / single-sample: input-independent running statistics.
            var meanRow = _runningMean.Reshape(new[] { 1, _numFeatures });
            var varRow = _runningVar.Reshape(new[] { 1, _numFeatures });
            var centered = engine.TensorAdd(input, engine.TensorMultiplyScalar(meanRow, minusOne));
            var std = engine.TensorSqrt(engine.TensorAddScalar(varRow, eps));
            normalized = engine.TensorDivide(centered, std);
        }

        // Scale/shift by the learnable gamma/beta (broadcast over the batch), on the tape
        var scaled = engine.TensorMultiply(normalized, _gamma.Reshape(new[] { 1, _numFeatures }));
        return engine.TensorAdd(scaled, _beta.Reshape(new[] { 1, _numFeatures }));
    }

    private void UpdateRunningStatistics(Tensor<T> meanRow, Tensor<T> varRow)
    {
        // Running stats are a non-tape side effect: running = (1 - momentum) running + momentum batch, written
        // in place so the registered buffer keeps its identity
        var engine = Engine;
        var keep = NumOps.FromDouble(1 - _momentum);
        var take = NumOps.FromDouble(_momentum);
        var mean = engine.TensorAdd(
            engine.TensorMultiplyScalar(_runningMean, keep),
            engine.TensorMultiplyScalar(meanRow.Contiguous().Reshape(_numFeatures), take));
        var variance = engine.TensorAdd(
            engine.TensorMultiplyScalar(_runningVar, keep),
            engine.TensorMultiplyScalar(varRow.Contiguous().Reshape(_numFeatures), take));
        mean.AsSpan().CopyTo(_runningMean.AsWritableSpan());
        variance.AsSpan().CopyTo(_runningVar.AsWritableSpan());
        engine.InvalidatePersistentTensor(_runningMean);
        engine.InvalidatePersistentTensor(_runningVar);
    }

    /// <summary>
    /// Performs the forward pass using running statistics (inference mode).
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The normalized output tensor.</returns>
    public Tensor<T> ForwardInference(Tensor<T> input)
    {
        var engine = Engine;
        int features = input.Shape[1];
        var std = engine.TensorSqrt(engine.TensorAddScalar(_runningVar, NumOps.FromDouble(_epsilon)));
        var normalized = engine.TensorDivide(
            engine.TensorSubtract(input, _runningMean.Reshape(1, features)), std.Reshape(1, features));
        return engine.TensorAdd(
            engine.TensorMultiply(normalized, _gamma.Reshape(1, features)), _beta.Reshape(1, features));
    }

    /// <summary>
    /// Plain gradient descent on gamma and beta when a manual backward has supplied their gradients; the tape
    /// training path updates the registered tensors directly instead.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient is null || _betaGradient is null) return;

        var gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        var beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
        gamma.AsSpan().CopyTo(_gamma.AsWritableSpan());
        beta.AsSpan().CopyTo(_beta.AsWritableSpan());
        Engine.InvalidatePersistentTensor(_gamma);
        Engine.InvalidatePersistentTensor(_beta);
    }

    /// <summary>
    /// Resets the gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        _gammaGradient = null;
        _betaGradient = null;
    }

    /// <inheritdoc/>
    public override void ClearGradients()
    {
        base.ClearGradients();
        ResetGradients();
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        ResetGradients();
    }

    /// <summary>
    /// Gets the output shape given an input shape.
    /// </summary>
    /// <param name="inputShape">The input shape.</param>
    /// <returns>The output shape (same as input for normalization layers).</returns>
    public int[] GetOutputShape(int[] inputShape)
    {
        return inputShape;
    }

    /// <summary>
    /// Gets a copy of the scale (gamma) parameters.
    /// </summary>
    public Vector<T> Gamma => _gamma.ToVector();

    /// <summary>
    /// Gets a copy of the shift (beta) parameters.
    /// </summary>
    public Vector<T> Beta => _beta.ToVector();

    /// <summary>
    /// Gets a copy of the running mean statistics.
    /// </summary>
    public Vector<T> RunningMean => _runningMean.ToVector();

    /// <summary>
    /// Gets a copy of the running variance statistics.
    /// </summary>
    public Vector<T> RunningVar => _runningVar.ToVector();
}
