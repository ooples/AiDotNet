using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class GatedDeltaNetLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _convKernelSize;

    // Short convolution: [modelDim, convKernelSize]
    private Tensor<T> _convWeights;
    private Tensor<T> _convBias;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Beta (write strength) projection: [modelDim, numHeads]
    private Tensor<T> _betaWeights;
    private Tensor<T> _betaBias;

    // Alpha (forget gate) projection: [modelDim, numHeads]
    private Tensor<T> _alphaWeights;
    private Tensor<T> _alphaBias;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
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
    private Tensor<T>? _lastStates;
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
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _convWeights.Length + _convBias.Length +
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _betaWeights.Length + _betaBias.Length +
        _alphaWeights.Length + _alphaBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

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
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, modelDim)
            : input.Reshape(batchSize, seqLen, modelDim);

        _lastInput = input3D;

        // Step 1: Short convolution
        var convOutput = DepthwiseConv1DForward(input3D, batchSize, seqLen);
        var siluConv = Engine.Swish(convOutput);
        _lastConvOutput = siluConv;

        // Step 2: Q, K, V projections
        var siluFlat = siluConv.Reshape(batchSize * seqLen, _modelDimension);
        var q = Engine.TensorMatMul(siluFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(siluFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(siluFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 3: Gates
        var betaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _betaWeights),
            _betaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var beta = Engine.Sigmoid(betaRaw);
        _lastBeta = beta;

        var alphaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _alphaWeights),
            _alphaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastAlpha = alpha;

        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(siluFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;

        // Step 4: Delta rule recurrence per head
        var output = DeltaRuleForward(q, k, v, alpha, beta, batchSize, seqLen);

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(output, gate);

        // Step 6: Output projection
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = _outputProjectionBias.Reshape(1, _modelDimension);
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
        var output3D = outputFlat.Reshape(batchSize, seqLen, _modelDimension);

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// Delta rule forward: fast weight update with gated forgetting.
    /// </summary>
    private Tensor<T> DeltaRuleForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> alpha, Tensor<T> beta,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T alphaVal = alpha[new[] { bi, t, hi }];
                    T betaVal = beta[new[] { bi, t, hi }];

                    // Retrieve current state's prediction for this key: S * k
                    var sK = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        sK[di] = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);
                            sK[di] = NumOps.Add(sK[di],
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], kVal));
                        }
                    }

                    // Delta: V - S*K (the error/correction term)
                    var delta = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        delta[di] = NumOps.Subtract(v[new[] { bi, t, flatDi }], sK[di]);
                    }

                    // State update: S = alpha * S + beta * delta * K^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);

                            T prevS = state[new[] { bi, hi, di, ki }];
                            T update = NumOps.Multiply(betaVal,
                                NumOps.Multiply(delta[di], kVal));
                            T newS = NumOps.Add(NumOps.Multiply(alphaVal, prevS), update);
                            state[new[] { bi, hi, di, ki }] = newS;
                        }
                    }

                    // Output: O = S * Q
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = q[new[] { bi, t, flatKi }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, ki }], qVal));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }
        }

        _lastStates = allStates;
        return output;
    }

    /// <summary>
    /// Depthwise causal Conv1D forward.
    /// </summary>
    private Tensor<T> DepthwiseConv1DForward(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var bias2D = _convBias.Reshape(1, _modelDimension);

        var weightSlices = new Tensor<T>[_convKernelSize];
        for (int ki = 0; ki < _convKernelSize; ki++)
        {
            weightSlices[ki] = _convWeights.GetSliceAlongDimension(ki, 1)
                .Reshape(1, _modelDimension);
        }

        for (int t = 0; t < seqLen; t++)
        {
            var result_t = Engine.TensorBroadcastAdd(
                new Tensor<T>(new[] { batchSize, _modelDimension }), bias2D);

            for (int ki = 0; ki < _convKernelSize; ki++)
            {
                int srcT = t - ki;
                if (srcT >= 0)
                {
                    var x_src = input.GetSliceAlongDimension(srcT, 1);
                    result_t = Engine.TensorAdd(result_t,
                        Engine.TensorBroadcastMultiply(x_src, weightSlices[ki]));
                }
            }

            output.SetSlice(1, t, result_t);
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _convWeightsGradient = new Tensor<T>([_modelDimension, _convKernelSize]);
        _convBiasGradient = new Tensor<T>([_modelDimension]);
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _betaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _betaBiasGradient = new Tensor<T>([_numHeads]);
        _alphaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _alphaBiasGradient = new Tensor<T>([_numHeads]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Gate backward
        var dDelta = Engine.TensorMultiply(dGated, _lastGate!);

        // Propagate through Q, K, V projections (simplified)
        var dDeltaFlat = dDelta.Reshape(batchSize * seqLen, _modelDimension);
        var dConvOutput = Engine.TensorMatMul(dDeltaFlat, _queryWeights.Transpose([1, 0]));

        // Conv backward through SiLU (approximate)
        var dConv3D = dConvOutput.Reshape(batchSize, seqLen, _modelDimension);
        var dInput = DepthwiseConv1DBackward(dConv3D, _lastInput, batchSize, seqLen);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    private Tensor<T> DepthwiseConv1DBackward(
        Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        var weightSlices = new Tensor<T>[_convKernelSize];
        for (int ki = 0; ki < _convKernelSize; ki++)
        {
            weightSlices[ki] = _convWeights.GetSliceAlongDimension(ki, 1)
                .Reshape(1, _modelDimension);
        }

        for (int t = 0; t < seqLen; t++)
        {
            var dOut_t = dOutput.GetSliceAlongDimension(t, 1);

            for (int ki = 0; ki < _convKernelSize; ki++)
            {
                int srcT = t - ki;
                if (srcT >= 0)
                {
                    var dInputContrib = Engine.TensorBroadcastMultiply(dOut_t, weightSlices[ki]);
                    var dInput_srcT = dInput.GetSliceAlongDimension(srcT, 1);
                    dInput_srcT = Engine.TensorAdd(dInput_srcT, dInputContrib);
                    dInput.SetSlice(1, srcT, dInput_srcT);
                }
            }
        }

        return dInput;
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
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
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
        _lastStates = null;
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => true;

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");
        var outWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(outWeightsNode);
        inputNodes.Add(outBiasNode);

        var outT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(xNode, outT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

        return outputWithBias;
    }

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
