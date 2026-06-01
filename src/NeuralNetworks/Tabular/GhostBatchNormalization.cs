using AiDotNet.Attributes;
using AiDotNet.Enums;
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
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.Regularizer)]
[PipelineStage(PipelineStage.Training)]
public class GhostBatchNormalization<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numFeatures;
    private readonly int _virtualBatchSize;
    private readonly double _momentum;
    private readonly double _epsilon;

    // Learnable parameters
    private Vector<T> _gamma; // Scale parameter
    private Vector<T> _beta;  // Shift parameter

    // Running statistics for inference
    private Vector<T> _runningMean;
    private Vector<T> _runningVar;

    // Gradients
    private Vector<T>? _gammaGrad;
    private Vector<T>? _betaGrad;

    // Cache for backward pass
    private Tensor<T>? _inputCache;
    private Tensor<T>? _normalizedCache;

    // Training vs inference mode. Propagated by the owning composite layer
    // (FeatureTransformerLayer / AttentiveTransformerLayer) so eval uses running stats.
    private bool _isTraining = true;

    /// <summary>
    /// Sets training vs inference mode. In inference (and for under-sized batches in training)
    /// the layer normalizes with the running statistics so the output stays input-dependent and
    /// deterministic.
    /// </summary>
    public void SetTrainingMode(bool isTraining) => _isTraining = isTraining;

    /// <summary>
    /// Gets the name of this layer.
    /// </summary>
    public string Name => "GhostBatchNormalization";

    /// <summary>
    /// Gets whether this layer supports training.
    /// </summary>
    public bool SupportsTraining => true;

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
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _numFeatures = numFeatures;
        _virtualBatchSize = virtualBatchSize;
        _momentum = momentum;
        _epsilon = epsilon;

        // Initialize learnable parameters
        _gamma = new Vector<T>(_numFeatures);
        _beta = new Vector<T>(_numFeatures);

        // Initialize running statistics
        _runningMean = new Vector<T>(_numFeatures);
        _runningVar = new Vector<T>(_numFeatures);

        // Initialize gamma to 1 and beta to 0
        for (int i = 0; i < _numFeatures; i++)
        {
            _gamma[i] = _numOps.One;
            _beta[i] = _numOps.Zero;
            _runningMean[i] = _numOps.Zero;
            _runningVar[i] = _numOps.One;
        }
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
    public Tensor<T> Forward(Tensor<T> input)
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

        var engine = AiDotNetEngine.Current;
        _inputCache = input;

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
        bool useBatchStats = _isTraining && batchSize >= 2;
        var minusOne = _numOps.FromDouble(-1.0);
        var eps = _numOps.FromDouble(_epsilon);
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
            var gCentered = engine.TensorBroadcastAdd(grouped, engine.TensorMultiplyScalar(gMean, minusOne));
            var gVar = engine.ReduceMean(engine.TensorMultiply(gCentered, gCentered), new[] { 1 }, keepDims: true);
            var gStd = engine.TensorSqrt(engine.TensorAddScalar(gVar, eps));
            var gNorm = engine.TensorBroadcastDivide(gCentered, gStd);                            // [numGhosts,vbs,F]
            normalized = engine.Reshape(gNorm, new[] { batchSize, _numFeatures });

            // Running statistics track the FULL-batch mean/var (used at inference, which sees no
            // virtual-batch structure).
            var bMean = engine.ReduceMean(input, new[] { 0 }, keepDims: true);
            var bCentered = engine.TensorBroadcastAdd(input, engine.TensorMultiplyScalar(bMean, minusOne));
            var bVar = engine.ReduceMean(engine.TensorMultiply(bCentered, bCentered), new[] { 0 }, keepDims: true);
            UpdateRunningStatistics(bMean, bVar);
        }
        else if (useBatchStats)
        {
            // Single virtual batch (batch <= virtualBatchSize, or not an exact multiple): normalize
            // over the whole batch — one ghost.
            var meanRow = engine.ReduceMean(input, new[] { 0 }, keepDims: true);
            var centered = engine.TensorBroadcastAdd(input, engine.TensorMultiplyScalar(meanRow, minusOne));
            var varRow = engine.ReduceMean(engine.TensorMultiply(centered, centered), new[] { 0 }, keepDims: true);
            UpdateRunningStatistics(meanRow, varRow);
            var std = engine.TensorSqrt(engine.TensorAddScalar(varRow, eps));
            normalized = engine.TensorBroadcastDivide(centered, std);
        }
        else
        {
            // Inference / single-sample: input-independent running statistics.
            var meanRow = Tensor<T>.FromVector(_runningMean).Reshape(new[] { 1, _numFeatures });
            var varRow = Tensor<T>.FromVector(_runningVar).Reshape(new[] { 1, _numFeatures });
            var centered = engine.TensorBroadcastAdd(input, engine.TensorMultiplyScalar(meanRow, minusOne));
            var std = engine.TensorSqrt(engine.TensorAddScalar(varRow, eps));
            normalized = engine.TensorBroadcastDivide(centered, std);
        }

        _normalizedCache = normalized;

        // Scale/shift by the learnable gamma/beta.
        var gammaRow = Tensor<T>.FromVector(_gamma).Reshape(new[] { 1, _numFeatures });
        var betaRow = Tensor<T>.FromVector(_beta).Reshape(new[] { 1, _numFeatures });
        var scaled = engine.TensorBroadcastMultiply(normalized, gammaRow);
        return engine.TensorBroadcastAdd(scaled, betaRow);
    }

    private void UpdateRunningStatistics(Tensor<T> meanRow, Tensor<T> varRow)
    {
        // Running stats are a non-tape side-effect; read the computed batch stats by value.
        var oneMinus = _numOps.FromDouble(1 - _momentum);
        var m = _numOps.FromDouble(_momentum);
        for (int f = 0; f < _numFeatures; f++)
        {
            _runningMean[f] = _numOps.Add(_numOps.Multiply(oneMinus, _runningMean[f]), _numOps.Multiply(m, meanRow[f]));
            _runningVar[f] = _numOps.Add(_numOps.Multiply(oneMinus, _runningVar[f]), _numOps.Multiply(m, varRow[f]));
        }
    }

    /// <summary>
    /// Performs the forward pass using running statistics (inference mode).
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The normalized output tensor.</returns>
    public Tensor<T> ForwardInference(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int features = input.Shape[1];
        var output = new Tensor<T>(input._shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                var normalized = _numOps.Divide(
                    _numOps.Subtract(input[b * features + f], _runningMean[f]),
                    _numOps.FromDouble(Math.Sqrt(_numOps.ToDouble(_runningVar[f]) + _epsilon)));

                output[b * features + f] = _numOps.Add(
                    _numOps.Multiply(_gamma[f], normalized),
                    _beta[f]);
            }
        }

        return output;
    }

    /// <summary>
    /// Gets the learnable parameters of this layer.
    /// </summary>
    /// <returns>A vector containing gamma and beta parameters.</returns>
    public Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(_numFeatures * 2);
        for (int i = 0; i < _numFeatures; i++)
        {
            parameters[i] = _gamma[i];
            parameters[_numFeatures + i] = _beta[i];
        }
        return parameters;
    }

    /// <summary>
    /// Sets the learnable parameters of this layer.
    /// </summary>
    /// <param name="parameters">A vector containing gamma and beta parameters.</param>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _numFeatures * 2)
        {
            throw new ArgumentException($"Expected {_numFeatures * 2} parameters, got {parameters.Length}");
        }

        for (int i = 0; i < _numFeatures; i++)
        {
            _gamma[i] = parameters[i];
            _beta[i] = parameters[_numFeatures + i];
        }
    }

    /// <summary>
    /// Gets the parameter gradients from the last backward pass.
    /// </summary>
    /// <returns>A vector containing gamma and beta gradients.</returns>
    public Vector<T> GetParameterGradients()
    {
        if (_gammaGrad == null || _betaGrad == null)
        {
            return new Vector<T>(_numFeatures * 2);
        }

        var gradients = new Vector<T>(_numFeatures * 2);
        for (int i = 0; i < _numFeatures; i++)
        {
            gradients[i] = _gammaGrad[i];
            gradients[_numFeatures + i] = _betaGrad[i];
        }
        return gradients;
    }

    /// <summary>
    /// Resets the gradients to zero.
    /// </summary>
    public void ResetGradients()
    {
        _gammaGrad = null;
        _betaGrad = null;
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
    /// Gets the number of trainable parameters in this layer.
    /// </summary>
    public long ParameterCount => _numFeatures * 2;

    /// <summary>
    /// Gets the scale (gamma) parameters.
    /// </summary>
    public Vector<T> Gamma => _gamma;

    /// <summary>
    /// Gets the shift (beta) parameters.
    /// </summary>
    public Vector<T> Beta => _beta;

    /// <summary>
    /// Gets the running mean statistics.
    /// </summary>
    public Vector<T> RunningMean => _runningMean;

    /// <summary>
    /// Gets the running variance statistics.
    /// </summary>
    public Vector<T> RunningVar => _runningVar;
}
