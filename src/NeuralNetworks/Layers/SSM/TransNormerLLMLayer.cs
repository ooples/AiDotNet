using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the TransNormerLLM layer from "TransNormerLLM: A Faster and Better LLM" (Qin et al., 2023).
/// </summary>
/// <remarks>
/// <para>
/// TransNormerLLM uses "Lightning Attention" -- a linear attention variant with exponential decay and
/// efficient normalization. Unlike standard Transformers that use softmax attention (O(n^2)), Lightning
/// Attention achieves linear complexity O(n) by combining linear attention with a decay factor and
/// RMSNorm-based normalization.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from the input
///   2. Apply RMSNorm to Q and K (key innovation -- stabilizes linear attention)
///   3. Linear attention with exponential decay (recurrent form):
///      S_t = gamma * S_{t-1} + k_t * v_t^T    (running state matrix)
///      z_t = gamma * z_{t-1} + k_t             (running normalizer)
///      o_t = RMSNorm(S_t * q_t)                (normalized output)
///   4. Output gate: y = swish(X W_g + b_g) * o
///   5. Output projection: output = y W_o + b_o
/// </code>
/// </para>
/// <para>
/// The key innovations over standard linear attention:
/// - RMSNorm on Q and K prevents the magnitude explosion that plagues linear attention
/// - Exponential decay gamma provides a soft causal bias (like RetNet) without rotary PE
/// - Per-output RMSNorm stabilizes the attention output, preventing training instability
/// These simple modifications make linear attention competitive with softmax attention for LLMs.
/// </para>
/// <para><b>For Beginners:</b> TransNormerLLM makes "linear attention" actually work well for language models.
///
/// Standard linear attention has a known problem: it tends to become numerically unstable during training
/// because the accumulated state matrix can grow without bound. TransNormerLLM fixes this with two tricks:
///
/// 1. RMSNorm on Q and K: Before computing attention, the queries and keys are normalized. This is like
///    making sure all "questions" and "answers" have similar magnitude, preventing any single token from
///    dominating the accumulated state.
///
/// 2. Exponential decay: Old information naturally fades away (controlled by gamma), preventing the state
///    from accumulating indefinitely. This is similar to RetNet's approach but simpler.
///
/// Together, these allow TransNormerLLM to match or exceed Transformer quality while being much faster
/// for long sequences (linear vs quadratic complexity).
/// </para>
/// <para>
/// <b>Reference:</b> Qin et al., "TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer", 2023.
/// https://arxiv.org/abs/2307.14995
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Normalization)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features].
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class TransNormerLLMLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly double _decayRate;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // RMSNorm parameters for Q and K: [numHeads, headDim]
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _queryNormScale;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _keyNormScale;

    // Per-head decay parameters (gammas): [numHeads]
    [TrainableParameter(Role = PersistentTensorRole.ScaleParameters)]
    private Tensor<T> _gammas;

    // Output RMSNorm: [numHeads, headDim]
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _outputNormScale;

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

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastQueryNormed;
    private Tensor<T>? _lastKeyNormed;
    private Tensor<T>? _lastQueryRmsInv;
    private Tensor<T>? _lastKeyRmsInv;
    private Tensor<T>? _lastAttnRaw;
    private Tensor<T>? _lastAttnNormed;
    private Tensor<T>? _lastAttnRmsInv;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastGate;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _queryNormScaleGradient;
    private Tensor<T>? _keyNormScaleGradient;
    private Tensor<T>? _gammasGradient;
    private Tensor<T>? _outputNormScaleGradient;
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
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the decay rate.
    /// </summary>
    public double DecayRate => _decayRate;

    /// <summary>
    /// Creates a new TransNormerLLM layer with lightning attention.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of the vector representing each token.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head can focus on different patterns. More heads give
    /// more diversity but each head has a smaller dimension. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="decayRate">
    /// Exponential decay rate gamma. Default: 0.99.
    /// <para><b>For Beginners:</b> Controls how quickly old information fades. A value of 0.99 means
    /// about 1% of information is lost per step, so the effective context window is about 100 steps.
    /// Higher values (closer to 1.0) give longer memory but may be harder to train.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public TransNormerLLMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double decayRate = 0.99,
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
        if (decayRate <= 0.0 || decayRate >= 1.0)
            throw new ArgumentException($"Decay rate ({decayRate}) must be in (0, 1).", nameof(decayRate));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _decayRate = decayRate;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);

        _queryNormScale = new Tensor<T>([numHeads, _headDimension]);
        _keyNormScale = new Tensor<T>([numHeads, _headDimension]);

        _gammas = new Tensor<T>([numHeads]);

        _outputNormScale = new Tensor<T>([numHeads, _headDimension]);

        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);

        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        InitializeTensor2D(_keyWeights);
        InitializeTensor2D(_valueWeights);

        // RMSNorm scales initialized to 1.0
        for (int i = 0; i < _queryNormScale.Length; i++)
            _queryNormScale[i] = NumOps.One;
        for (int i = 0; i < _keyNormScale.Length; i++)
            _keyNormScale[i] = NumOps.One;
        for (int i = 0; i < _outputNormScale.Length; i++)
            _outputNormScale[i] = NumOps.One;

        // Initialize gammas with multi-scale decay rates centered around the specified decay rate
        // Heads get slightly different rates for multi-scale context capture
        for (int h = 0; h < _numHeads; h++)
        {
            double spread = 0.02;
            double offset = (h - (_numHeads - 1.0) / 2.0) / Math.Max(1, _numHeads - 1) * spread;
            double gamma = Math.Max(0.9, Math.Min(0.9999, _decayRate + offset));
            _gammas[h] = NumOps.FromDouble(gamma);
        }

        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <summary>
    /// Applies RMSNorm per head dimension and returns the inverse RMS for backward.
    /// </summary>
    private (Tensor<T> Output, Tensor<T> InverseRms) ApplyRMSNorm(
        Tensor<T> input, Tensor<T> scale, int batchSize, int seqLen)
    {
        var heads = Engine.Reshape(
            input,
            new[] { batchSize, seqLen, _numHeads, _headDimension });
        var meanSquare = Engine.TensorMultiplyScalar(
            Engine.ReduceSum(
                Engine.TensorSquare(heads), new[] { 3 }, keepDims: true),
            NumOps.FromDouble(1.0 / _headDimension));
        var inverseRms = Engine.TensorReciprocal(
            Engine.TensorSqrt(
                Engine.TensorAddScalar(meanSquare, NumOps.FromDouble(1e-6))));
        var normalized = Engine.TensorMultiply(heads, inverseRms);
        var scaled = Engine.TensorMultiply(
            normalized,
            Engine.Reshape(scale, new[] { 1, 1, _numHeads, _headDimension }));
        return (
            Engine.Reshape(scaled, new[] { batchSize, seqLen, _modelDimension }),
            Engine.Reshape(inverseRms, new[] { batchSize, seqLen, _numHeads }));
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

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: RMSNorm on Q and K
        var (qNormed, qRmsInv) = ApplyRMSNorm(q, _queryNormScale, batchSize, seqLen);
        var (kNormed, kRmsInv) = ApplyRMSNorm(k, _keyNormScale, batchSize, seqLen);
        _lastQueryNormed = qNormed;
        _lastKeyNormed = kNormed;
        _lastQueryRmsInv = qRmsInv;
        _lastKeyRmsInv = kRmsInv;

        // Step 3: Compute output gate
        var gateRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Swish(gateRaw);
        _lastGateRaw = gateRaw;
        _lastGate = gate;

        // Step 4: Lightning attention with decay (recurrent form)
        var attnRaw = LightningAttentionForward(qNormed, kNormed, v, batchSize, seqLen);
        _lastAttnRaw = attnRaw;

        // Step 5: RMSNorm on attention output
        var (attnNormed, attnRmsInv) = ApplyRMSNorm(
            attnRaw, _outputNormScale, batchSize, seqLen);
        _lastAttnNormed = attnNormed;
        _lastAttnRmsInv = attnRmsInv;

        // Step 6: Gated output
        var gatedOutput = Engine.TensorMultiply(gate, attnNormed);

        // Step 7: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension }));
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
    /// Lightning attention forward pass: linear attention with exponential decay.
    /// S_t = gamma * S_{t-1} + k_t * v_t^T
    /// o_t = S_t * q_t
    /// </summary>
    private Tensor<T> LightningAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var gate = Engine.TensorBroadcastTo(
            Engine.Reshape(_gammas, new[] { 1, 1, _numHeads }),
            new[] { batchSize, seqLen, _numHeads });
        // GLA stores S[value,key] and returns S*q. Swapping its key/value streams stores
        // S[key,value], so the fused read becomes the TransNormer recurrence's S*q orientation.
        return Engine.GlaScanForward(q, v, k, gate, _numHeads);
    }

    /// <summary>
    /// Backward pass through RMSNorm.
    /// </summary>
    private Tensor<T> RMSNormBackward(
        Tensor<T> dOutput, Tensor<T> input, Tensor<T> scale, Tensor<T> rmsInv,
        Tensor<T> scaleGradient, int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(input._shape);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int dimStart = h * _headDimension;
                    T invRms = rmsInv[new[] { bi, t, h }];

                    // Compute intermediate sums for the RMSNorm gradient
                    T sumDyXhat = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xNorm = NumOps.Multiply(input[new[] { bi, t, flatD }], invRms);
                        T dy = dOutput[new[] { bi, t, flatD }];
                        T s = scale[new[] { h, d }];

                        // Scale gradient
                        scaleGradient[new[] { h, d }] = NumOps.Add(
                            scaleGradient[new[] { h, d }],
                            NumOps.Multiply(dy, xNorm));

                        T dyScaled = NumOps.Multiply(dy, s);
                        sumDyXhat = NumOps.Add(sumDyXhat,
                            NumOps.Multiply(dyScaled, xNorm));
                    }

                    T invHD = NumOps.FromDouble(1.0 / _headDimension);
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xNorm = NumOps.Multiply(input[new[] { bi, t, flatD }], invRms);
                        T dy = dOutput[new[] { bi, t, flatD }];
                        T s = scale[new[] { h, d }];
                        T dyScaled = NumOps.Multiply(dy, s);

                        // dInput = invRms * (dyScaled - xNorm * invHD * sumDyXhat)
                        T correction = NumOps.Multiply(xNorm,
                            NumOps.Multiply(invHD, sumDyXhat));
                        dInput[new[] { bi, t, flatD }] = NumOps.Multiply(
                            invRms, NumOps.Subtract(dyScaled, correction));
                    }
                }
            }
        }

        return dInput;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var oneMinusSig = Engine.ScalarMinusTensor(NumOps.One, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAddScalar(xTimesOneMinusSig, NumOps.One);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_queryWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _queryNormScale = Engine.TensorAdd(_queryNormScale, Engine.TensorMultiplyScalar(_queryNormScaleGradient!, negLR));
        _keyNormScale = Engine.TensorAdd(_keyNormScale, Engine.TensorMultiplyScalar(_keyNormScaleGradient!, negLR));
        _gammas = Engine.TensorAdd(_gammas, Engine.TensorMultiplyScalar(_gammasGradient!, negLR));
        _outputNormScale = Engine.TensorAdd(_outputNormScale, Engine.TensorMultiplyScalar(_outputNormScaleGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Clamp gammas to valid range
        for (int h = 0; h < _numHeads; h++)
        {
            double gVal = NumOps.ToDouble(_gammas[h]);
            if (gVal <= 0.0) _gammas[h] = NumOps.FromDouble(1e-6);
            else if (gVal >= 1.0) _gammas[h] = NumOps.FromDouble(1.0 - 1e-6);
        }

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _keyWeights, _valueWeights,
        _queryNormScale, _keyNormScale,
        _gammas,
        _outputNormScale,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_queryNormScaleGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyNormScaleGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_gammasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputNormScaleGradient?.ToArray() ?? new T[_outputNormScale.Length]),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _queryNormScaleGradient = null; _keyNormScaleGradient = null; _gammasGradient = null;
        _outputNormScaleGradient = null; _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastQueryNormed = null;
        _lastKeyNormed = null;
        _lastQueryRmsInv = null;
        _lastKeyRmsInv = null;
        _lastAttnRaw = null;
        _lastAttnNormed = null;
        _lastAttnRmsInv = null;
        _lastGateRaw = null;
        _lastGate = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _queryNormScaleGradient = null;
        _keyNormScaleGradient = null;
        _gammasGradient = null;
        _outputNormScaleGradient = null;
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
        metadata["DecayRate"] = _decayRate.ToString("F4");

        var gammaValues = new string[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            gammaValues[h] = NumOps.ToDouble(_gammas[h]).ToString("F6");
        metadata["PerHeadDecayRates"] = string.Join(", ", gammaValues);

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

    /// <summary>
    /// Gets the per-head decay rates (gammas) for external inspection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each head has its own decay rate gamma in (0, 1). These are initialized near the specified
    /// decay rate but with slight offsets to create multi-scale context capture.
    /// </para>
    /// <para><b>For Beginners:</b> These control how quickly each head forgets old information.
    /// Different decay rates allow different heads to focus on different time scales.</para>
    /// </remarks>
    public Tensor<T> GetDecayRates() => _gammas;
}
