using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the HGRN2 layer from "HGRN2: Gated Linear RNNs with State Expansion" (Qin et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// HGRN2 extends HGRN (Hierarchical Gated Recurrent Network) with "state expansion", bridging the gap
/// between element-wise gated recurrences (vector state) and linear attention (matrix state). Instead of
/// maintaining a hidden vector h_t as in HGRN, HGRN2 maintains a hidden matrix S_t of shape
/// [head_dim x head_dim] per head, enabling richer state representations.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from the input
///   2. Compute forget gate g_t = sigmoid(W_g * x_t + b_g)   (per-head scalar)
///   3. State update (outer-product recurrence, per head):
///      S_t = g_t * S_{t-1} + k_t * v_t^T
///      This is like linear attention's state accumulation (k*v^T) but with a gated
///      forget factor g_t that controls how much of the previous state is retained.
///   4. Output readout: o_t = S_t * q_t   (matrix-vector product)
///   5. Gated output: y_t = gate_t * o_t
///   6. Output projection: final = W_out * y_t + b_out
/// </code>
/// </para>
/// <para>
/// The key insight of "state expansion" is that using an outer product k*v^T to build the state matrix
/// gives each head a rank-1 update per step. Over time the state accumulates a low-rank approximation
/// of the key-value associations, similar to how linear attention accumulates K^T V. The crucial
/// difference from linear attention is the forget gate g_t, which prevents unbounded state growth and
/// allows the model to selectively discard old information.
/// </para>
/// <para>
/// This bridges two extremes:
/// - HGRN (vector state): S_t is a vector, updated element-wise. Capacity limited by head_dim.
/// - Linear attention (matrix state): S_t = S_{t-1} + k_t * v_t^T, no forgetting. Unbounded growth.
/// - HGRN2 (gated matrix state): S_t = g_t * S_{t-1} + k_t * v_t^T. Best of both worlds.
/// </para>
/// <para><b>For Beginners:</b> HGRN2 is a sequence model that processes tokens one at a time while
/// maintaining a "memory matrix" for each attention head.
///
/// Think of each head's state matrix as a small notebook:
/// - At each step, the model writes a new "entry" (the outer product k*v^T) into the notebook
/// - The forget gate g_t controls how much the old notes fade: g=1 means perfect memory, g=0 means
///   forget everything from before
/// - To produce output, the model "looks up" information by multiplying the notebook by a query vector
///
/// Compared to a standard Transformer:
/// - Transformers re-read ALL previous tokens at every step (O(n^2) cost)
/// - HGRN2 compresses all history into a fixed-size matrix (O(n) cost, constant memory)
/// - The matrix state is much richer than a simple vector (like LSTM/GRU), letting HGRN2
///   remember more complex patterns
///
/// HGRN2 achieves competitive performance with Transformers on language modeling benchmarks while
/// being significantly more efficient for long sequences.
/// </para>
/// <para>
/// <b>Reference:</b> Qin et al., "HGRN2: Gated Linear RNNs with State Expansion", 2024.
/// https://arxiv.org/abs/2404.07904
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class HGRN2Layer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly double _forgetBias;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Forget gate projection: [modelDim, numHeads] + bias [numHeads]
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

    // Output gate: [modelDim, modelDim] + bias [modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim] + bias [modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <summary>
    /// Gets the model dimension (d_model).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head (d_model / numHeads).
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new HGRN2 layer with state expansion.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token's representation vector.
    /// Larger values let the model capture more features but use more memory.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own state matrix of shape
    /// [head_dim x head_dim]. More heads means more independent "memory notebooks",
    /// each tracking different aspects of the sequence. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="forgetBias">
    /// Initial bias for the forget gate. Default: 1.0.
    /// <para><b>For Beginners:</b> A positive bias makes the model start by remembering more
    /// (sigmoid(1.0) ~ 0.73). This helps with learning long-range dependencies early in training.
    /// Higher values mean stronger initial memory retention.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public HGRN2Layer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double forgetBias = 1.0,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _forgetBias = forgetBias;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _forgetGateBias = new Tensor<T>([numHeads]);
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
        InitializeTensor2D(_forgetGateWeights);

        // Initialize forget gate bias so sigmoid(bias) starts with reasonable retention
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(_forgetBias);

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

        // Step 1: Q, K, V projections
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(inputFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(inputFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Forget gate g_t = sigmoid(W_g * x_t + b_g), per-head scalar
        var forgetRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _forgetGateWeights),
            _forgetGateBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var forgetGate = Engine.Sigmoid(forgetRaw);
        _lastForgetGate = forgetGate;

        // Step 3: Output gate = swish(W_gate * x + b_gate)
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 4: Gated outer-product recurrence per head
        var recurrenceOutput = OuterProductRecurrenceForward(q, k, v, forgetGate, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(recurrenceOutput, gate);

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
    /// Outer-product gated recurrence forward pass.
    /// </summary>
    /// <remarks>
    /// For each head h and timestep t:
    ///   S_t[h] = g_t[h] * S_{t-1}[h] + k_t[h] * v_t[h]^T    (outer product update)
    ///   o_t[h] = S_t[h] * q_t[h]                                (readout via query)
    /// where S_t[h] is a [head_dim x head_dim] matrix.
    /// </remarks>
    private Tensor<T> OuterProductRecurrenceForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> forgetGate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        // Save all states for backward pass: [batch, seqLen+1, numHeads, headDim, headDim]
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T gVal = forgetGate[new[] { bi, t, hi }];

                    // State update: S_t = g_t * S_{t-1} + k_t * v_t^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T kVal = NumOps.Multiply(k[new[] { bi, t, flatDi }], scale);

                        for (int vi = 0; vi < _headDimension; vi++)
                        {
                            int flatVi = dimStart + vi;
                            T vVal = v[new[] { bi, t, flatVi }];

                            T prevS = state[new[] { bi, hi, di, vi }];
                            // outer product: k[di] * v[vi]
                            T outerProd = NumOps.Multiply(kVal, vVal);
                            T newS = NumOps.Add(NumOps.Multiply(gVal, prevS), outerProd);
                            state[new[] { bi, hi, di, vi }] = newS;
                        }
                    }

                    // Output readout: o_t = S_t * q_t
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int qi = 0; qi < _headDimension; qi++)
                        {
                            int flatQi = dimStart + qi;
                            T qVal = q[new[] { bi, t, flatQi }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(state[new[] { bi, hi, di, qi }], qVal));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }

            // Save state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi2 = 0; hi2 < _numHeads; hi2++)
                    for (int di = 0; di < _headDimension; di++)
                        for (int vi = 0; vi < _headDimension; vi++)
                            allStates[new[] { bi, t + 1, hi2, di, vi }] = state[new[] { bi, hi2, di, vi }];
        }

        _lastStates = allStates;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastForgetGate == null || _lastGate == null || _lastGateRaw == null ||
            _lastRecurrenceOutput == null || _lastStates == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _forgetGateWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _forgetGateBiasGradient = new Tensor<T>([_numHeads]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 6 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(_lastRecurrenceOutput, _lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating - gatedOutput = recurrenceOutput * gate
        var dRecurrenceOut = Engine.TensorMultiply(dGated, _lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastRecurrenceOutput);

        // Gate uses Swish; derivative: swish(x) + sigmoid(x) * (1 - swish(x))
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastGateRaw));

        // Gate weight gradients
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        // Accumulate input gradient from gate path
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: outer-product recurrence
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dForgetGate = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T gVal = _lastForgetGate[new[] { bi, t, hi }];

                    // Backward through output readout: o_t = S_t * q_t
                    // dS += dO * q^T, dQ += S^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dRecurrenceOut[new[] { bi, t, flatDi }];

                        for (int qi = 0; qi < _headDimension; qi++)
                        {
                            int flatQi = dimStart + qi;
                            T qVal = _lastQuery[new[] { bi, t, flatQi }];
                            T sVal = _lastStates[new[] { bi, t + 1, hi, di, qi }];

                            // dS[di,qi] += dO[di] * q[qi]
                            dState[new[] { bi, hi, di, qi }] = NumOps.Add(
                                dState[new[] { bi, hi, di, qi }],
                                NumOps.Multiply(dO, qVal));

                            // dQ[qi] += S[di,qi] * dO[di]
                            dQ[new[] { bi, t, flatQi }] = NumOps.Add(
                                dQ[new[] { bi, t, flatQi }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // Backward through state update: S_t = g_t * S_{t-1} + k_t * v_t^T
                    // dg += sum(dS .* S_{t-1})
                    // dS_{t-1} = g_t * dS
                    // dk += dS * v (scaled)
                    // dv += k^T * dS (scaled)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatDi }], scale);

                        for (int vi = 0; vi < _headDimension; vi++)
                        {
                            int flatVi = dimStart + vi;
                            T vVal = _lastValue[new[] { bi, t, flatVi }];
                            T dS = dState[new[] { bi, hi, di, vi }];
                            T sPrev = _lastStates[new[] { bi, t, hi, di, vi }];

                            // dg_t += dS[di,vi] * S_{t-1}[di,vi]
                            dForgetGate[new[] { bi, t, hi }] = NumOps.Add(
                                dForgetGate[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, sPrev));

                            // dk[di] += dS[di,vi] * v[vi] (with scale applied to k)
                            dK[new[] { bi, t, flatDi }] = NumOps.Add(
                                dK[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, vVal));

                            // dv[vi] += k[di] * dS[di,vi]
                            dV[new[] { bi, t, flatVi }] = NumOps.Add(
                                dV[new[] { bi, t, flatVi }],
                                NumOps.Multiply(kVal, dS));

                            // Propagate dState to previous timestep: dS_{t-1} = g_t * dS_t
                            dState[new[] { bi, hi, di, vi }] = NumOps.Multiply(gVal, dS);
                        }
                    }
                }
            }
        }

        // Forget gate through sigmoid derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        var forgetSigDeriv = Engine.TensorMultiply(_lastForgetGate,
            Engine.TensorSubtract(
                CreateOnesLike(_lastForgetGate), _lastForgetGate));
        var dForgetRaw = Engine.TensorMultiply(dForgetGate, forgetSigDeriv);

        // Forget gate weight gradients
        var dForgetFlat = dForgetRaw.Reshape(batchSize * seqLen, _numHeads);
        _forgetGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dForgetFlat);
        _forgetGateBiasGradient = Engine.ReduceSum(dForgetRaw, new int[] { 0, 1 });

        // Q, K, V weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]),
            Engine.TensorMultiplyScalar(dKFlat, scale));
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient from all projection paths
        var dInputTotal = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(Engine.TensorMultiplyScalar(dKFlat, scale), _keyWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dForgetFlat, _forgetGateWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromGate);

        var dInput = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        // SiLU(x) = x * sigmoid(x)
        // SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        var sig = Engine.Sigmoid(x);
        var ones = new Tensor<T>(x.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        var oneMinusSig = Engine.TensorSubtract(ones, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAdd(ones, xTimesOneMinusSig);
        return Engine.TensorMultiply(sig, onePlusXSig);
    }

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var ones = new Tensor<T>(template.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        return ones;
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
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
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
        _queryWeights, _keyWeights, _valueWeights,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastForgetGate = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastStates = null;
        _lastRecurrenceOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

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
        metadata["ForgetBias"] = _forgetBias.ToString("F2");
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
