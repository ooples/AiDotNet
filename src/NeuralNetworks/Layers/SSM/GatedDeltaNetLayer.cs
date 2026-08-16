using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the GatedDeltaNet layer from NVIDIA, ICLR 2025.
/// </summary>
/// <remarks>
/// <para>
/// GatedDeltaNet combines the delta rule for fast weight updates with gated output, achieving
/// state-of-the-art performance among sub-quadratic architectures. It matches Transformer quality
/// on many benchmarks while maintaining linear O(n) complexity.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Short convolution on input (captures local context, like Mamba)
///   2. Compute Q, K, V projections from convolved input
///   3. Compute gates: beta (write strength) and alpha (forget gate)
///   4. Delta rule state update:
///      S_t = alpha_t * S_{t-1} + beta_t * (V_t - S_{t-1} * K_t) * K_t^T
///      The (V - S*K)*K^T term is the "delta rule": it only writes the DIFFERENCE
///      between the target V and what the state would currently retrieve for key K.
///   5. Output: O_t = S_t * Q_t
///   6. Gated output: y_t = gate_t * O_t
///   7. Output projection
/// </code>
/// </para>
/// <para>
/// The delta rule update is key: instead of blindly accumulating K*V outer products (like linear
/// attention), it computes the error (V - S*K) first and updates accordingly. This is exactly the
/// delta rule from neural network learning theory, applied to the fast weight matrix at each step.
/// </para>
/// <para><b>For Beginners:</b> GatedDeltaNet is one of the best sub-quadratic architectures as of 2025.
///
/// Think of the state matrix S as a "lookup table" that maps keys to values:
/// - Linear attention: "Just add key-value pairs to the table" -> entries pile up, old ones never corrected
/// - Delta rule: "Before adding, check if this key already has a value. Only write the correction."
///
/// This is like the difference between:
/// - Memorizing every flashcard answer independently (linear attention)
/// - Checking what you already know, then only memorizing what's new or different (delta rule)
///
/// The gating mechanism (alpha, beta) lets the model control:
/// - How much to forget old entries (alpha)
/// - How strongly to write new corrections (beta)
///
/// Combined with a short convolution for local context, this simple recipe matches Transformers
/// while being much more efficient for long sequences.
/// </para>
/// <para>
/// <b>Reference:</b> Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025.
/// https://arxiv.org/abs/2412.06464
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Gating)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving; relations DISCOVERED by probing, roles read from the forward. Like every layer in
// this folder it takes seqLen = Shape[rank-2] and modelDim = Shape[rank-1], so rank 2 is
// [Time, Features] with NO batch axis. OutputAxesFor is generated from these layouts.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GatedDeltaNetLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _convKernelSize;

    // Short convolution: [modelDim, convKernelSize]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _convWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _convBias;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Beta (write strength) projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _betaWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _betaBias;

    // Alpha (forget gate) projection: [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _alphaWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _alphaBias;

    // Output gate: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastConvOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastBeta;
    private Tensor<T>? _lastAlpha;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastSiluConv;
    private Tensor<T>? _lastDeltaRuleOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _convWeightsGradient;
    private Tensor<T>? _convBiasGradient;
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _betaWeightsGradient;
    private Tensor<T>? _betaBiasGradient;
    private Tensor<T>? _alphaWeightsGradient;
    private Tensor<T>? _alphaBiasGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the convolution kernel size.
    /// </summary>
    public int ConvKernelSize => _convKernelSize;

    /// <summary>
    /// Creates a new GatedDeltaNet layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own fast weight matrix.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="convKernelSize">
    /// Short convolution kernel size. Default: 4.
    /// <para><b>For Beginners:</b> Captures local context before the delta rule processes the sequence.
    /// Same role as the Conv1D in Mamba.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public GatedDeltaNetLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int convKernelSize = 4,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (convKernelSize <= 0)
            throw new ArgumentException($"Conv kernel size ({convKernelSize}) must be positive.", nameof(convKernelSize));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _convKernelSize = convKernelSize;

        _convWeights = new Tensor<T>([modelDimension, convKernelSize]);
        _convBias = new Tensor<T>([modelDimension]);
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _betaWeights = new Tensor<T>([modelDimension, numHeads]);
        _betaBias = new Tensor<T>([numHeads]);
        _alphaWeights = new Tensor<T>([modelDimension, numHeads]);
        _alphaBias = new Tensor<T>([numHeads]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_convWeights);
        _convBias.Fill(NumOps.Zero);
        InitializeTensor2D(_queryWeights);
        InitializeTensor2D(_keyWeights);
        InitializeTensor2D(_valueWeights);
        InitializeTensor2D(_betaWeights);
        _betaBias.Fill(NumOps.FromDouble(0.1));
        InitializeTensor2D(_alphaWeights);
        // Alpha bias ~ 2 so sigmoid(2) ≈ 0.88 -> strong initial memory retention
        for (int i = 0; i < _alphaBias.Length; i++)
            _alphaBias[i] = NumOps.FromDouble(2.0);
        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        _lastInput = input3D;

        // Step 1: Short convolution
        var convOutput = DepthwiseConv1DForward(input3D, batchSize, seqLen);
        var siluConv = Engine.Swish(convOutput);
        _lastConvOutput = convOutput;
        _lastSiluConv = siluConv;

        // Step 2: Q, K, V projections
        var siluFlat = Engine.Reshape(siluConv, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(siluFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(siluFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(siluFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 3: Gates
        var betaRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _betaWeights),
            Engine.Reshape(_betaBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        var alphaRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _alphaWeights),
            Engine.Reshape(_alphaBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastAlpha = alpha;

        var gateRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 4: Delta rule recurrence per head
        var output = Engine.GatedDeltaNetScanForward(q, k, v, alpha, beta, _numHeads);
        _lastDeltaRuleOutput = output;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(output, gate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    /// <summary>
    /// Depthwise causal Conv1D forward.
    /// </summary>
    private Tensor<T> DepthwiseConv1DForward(Tensor<T> input, int batchSize, int seqLen)
    {
        // IEngine.DepthwiseConv1D uses [B,C,T] and cross-correlation kernel order. Flip the
        // layer's lag-ordered weights, left-pad by K-1, and retain the first T outputs to obtain
        // exactly y[t,c] = bias[c] + sum_k weight[c,k] * input[t-k,c]. This records one
        // convolution node whose graph size is independent of sequence length.
        var channelsFirst = Engine.TensorPermute(input, new[] { 0, 2, 1 });
        var reversedKernel = Engine.Reshape(
            Engine.TensorFlip(_convWeights, new[] { 1 }),
            new[] { _modelDimension, 1, _convKernelSize });
        var padded = Engine.DepthwiseConv1D(
            channelsFirst, reversedKernel, stride: 1, padding: _convKernelSize - 1);
        var causal = Engine.TensorSlice(
            padded,
            new[] { 0, 0, 0 },
            new[] { batchSize, _modelDimension, seqLen });
        var timeMajor = Engine.TensorPermute(causal, new[] { 0, 2, 1 });
        return Engine.TensorBroadcastAdd(
            timeMajor,
            Engine.Reshape(_convBias, new[] { 1, 1, _modelDimension }));
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        // SiLU(x) = x * sigmoid(x)
        // SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template._shape);
        ones.Fill(NumOps.One);
        return ones;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_convWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _convWeights = Engine.TensorAdd(_convWeights, Engine.TensorMultiplyScalar(_convWeightsGradient, negLR));
        _convBias = Engine.TensorAdd(_convBias, Engine.TensorMultiplyScalar(_convBiasGradient!, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _betaWeights = Engine.TensorAdd(_betaWeights, Engine.TensorMultiplyScalar(_betaWeightsGradient!, negLR));
        _betaBias = Engine.TensorAdd(_betaBias, Engine.TensorMultiplyScalar(_betaBiasGradient!, negLR));
        _alphaWeights = Engine.TensorAdd(_alphaWeights, Engine.TensorMultiplyScalar(_alphaWeightsGradient!, negLR));
        _alphaBias = Engine.TensorAdd(_alphaBias, Engine.TensorMultiplyScalar(_alphaBiasGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_convWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_convBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_betaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_alphaWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_alphaBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _convWeights, _convBias,
        _queryWeights, _keyWeights, _valueWeights,
        _betaWeights, _betaBias,
        _alphaWeights, _alphaBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_convWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_convWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_convBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_betaBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_alphaWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_alphaBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _convWeightsGradient = null; _convBiasGradient = null; _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _betaWeightsGradient = null; _betaBiasGradient = null; _alphaWeightsGradient = null; _alphaBiasGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastConvOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastBeta = null;
        _lastAlpha = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastSiluConv = null;
        _lastDeltaRuleOutput = null;
        _originalInputShape = null;
        _convWeightsGradient = null;
        _convBiasGradient = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _betaWeightsGradient = null;
        _betaBiasGradient = null;
        _alphaWeightsGradient = null;
        _alphaBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["ConvKernelSize"] = _convKernelSize.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the query weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;
}
