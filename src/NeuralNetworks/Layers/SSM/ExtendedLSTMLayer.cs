using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// xLSTM modernizes the classic LSTM architecture with two key innovations:
/// 1. <b>sLSTM (scalar LSTM)</b>: Enhanced gating with exponential activation functions and
///    a new memory mixing mechanism. Uses scalar (diagonal) memory cells.
/// 2. <b>mLSTM (matrix LSTM)</b>: Replaces the scalar memory cell with a matrix-valued memory,
///    connecting LSTMs to modern linear attention/state space models.
/// </para>
/// <para>
/// This layer implements the mLSTM variant, which is the more impactful innovation:
/// <code>
///   // Gate computations
///   i_t = exp(W_i * x_t + b_i)    // Input gate (exponential, not sigmoid!)
///   f_t = sigmoid(W_f * x_t + b_f) OR exp(W_f * x_t + b_f)  // Forget gate
///   o_t = sigmoid(W_o * x_t + b_o)  // Output gate
///
///   // Key-Value projections (connecting to linear attention)
///   k_t = W_k * x_t / sqrt(d)
///   v_t = W_v * x_t
///   q_t = W_q * x_t
///
///   // Matrix memory cell update (covariance-based)
///   C_t = f_t * C_{t-1} + i_t * v_t * k_t^T    // Matrix cell = gated outer product
///   n_t = f_t * n_{t-1} + i_t * k_t              // Normalizer state
///
///   // Output
///   h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1)
/// </code>
/// </para>
/// <para>
/// The connection to linear attention: if f_t = 1 and i_t = 1, the matrix cell C_t accumulates
/// k*v outer products exactly like the state matrix in linear attention. The gates allow
/// selective forgetting and input scaling, which is what makes xLSTM competitive.
/// </para>
/// <para><b>For Beginners:</b> xLSTM is a modernized version of the classic LSTM (1997).
///
/// The original LSTM was the dominant sequence model for years, but was overtaken by Transformers.
/// xLSTM brings it back by fixing key limitations:
///
/// 1. <b>Exponential gating</b>: Instead of sigmoid (0 to 1), gates use exp() which can amplify
///    important signals, not just dampen them.
///
/// 2. <b>Matrix memory</b>: Instead of a vector cell, mLSTM uses a matrix. This is like having
///    a lookup table that maps keys to values, similar to attention but stored as a running sum.
///
/// The result: an LSTM that matches Transformer performance at scale while maintaining the
/// efficient O(1) per-step inference of RNNs.
/// </para>
/// <para>
/// <b>Reference:</b> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
/// https://arxiv.org/abs/2405.04517
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class ExtendedLSTMLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _headDimension;
    private readonly int _numHeads;

    // Input gate projection: [modelDim, modelDim]
    private Tensor<T> _inputGateWeights;
    private Tensor<T> _inputGateBias;

    // Forget gate projection: [modelDim, modelDim]
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

    // Output gate projection: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Query, Key, Value projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastCellStates;
    private Tensor<T>? _lastNormStates;
    private Tensor<T>? _lastInputGates;
    private Tensor<T>? _lastForgetGates;
    private Tensor<T>? _lastOutputGates;
    private Tensor<T>? _lastQ;
    private Tensor<T>? _lastK;
    private Tensor<T>? _lastV;
    private Tensor<T>? _lastHiddenPreProj;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads for the matrix memory.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _inputGateWeights.Length + _inputGateBias.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Extended LSTM (xLSTM) layer using the mLSTM (matrix memory) variant.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads for matrix memory. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own matrix memory cell.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public ExtendedLSTMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _inputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputGateBias = new Tensor<T>([modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateBias = new Tensor<T>([modelDimension]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_forgetGateWeights);
        // Forget gate bias initialized to positive values for long memory (LSTM best practice)
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(1.0);
        InitializeTensor(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_queryWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
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

        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T scaleK = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Matrix cell state per head: C[batch, head, headDim, headDim]
        var cellState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        // Normalizer state per head: n[batch, head, headDim]
        var normState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension });

        // Store gates and projections for backward
        var allInputGates = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allForgetGates = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allOutputGates = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allHiddenPreProj = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allCellStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        var allNormStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension });

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = input3D.GetSliceAlongDimension(t, 1);  // [batch, modelDim]

            // Gate computations
            var iGateRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _inputGateWeights),
                _inputGateBias.Reshape(1, _modelDimension));
            var fGateRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _forgetGateWeights),
                _forgetGateBias.Reshape(1, _modelDimension));
            var oGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _outputGateWeights),
                _outputGateBias.Reshape(1, _modelDimension)));

            // Exponential input gate (xLSTM innovation)
            var iGate = Engine.TensorExp(iGateRaw);
            // Sigmoid forget gate (stabilized)
            var fGate = Engine.Sigmoid(fGateRaw);

            allInputGates.SetSlice(1, t, iGate);
            allForgetGates.SetSlice(1, t, fGate);
            allOutputGates.SetSlice(1, t, oGate);

            // Q, K, V projections
            var q = Engine.TensorMatMul(x_t, _queryWeights);
            var k = Engine.TensorMultiplyScalar(
                Engine.TensorMatMul(x_t, _keyWeights), scaleK);
            var v = Engine.TensorMatMul(x_t, _valueWeights);

            allQ.SetSlice(1, t, q);
            allK.SetSlice(1, t, k);
            allV.SetSlice(1, t, v);

            // Update matrix cell state per head: C = f * C + i * (v * k^T)
            var h_t = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T iVal = iGate[new[] { bi, dimStart }];  // Use head-level gate
                    T fVal = fGate[new[] { bi, dimStart }];
                    T oVal = oGate[new[] { bi, dimStart }];

                    // Clamp input gate pre-activation before exp to prevent overflow
                    // exp(20) ≈ 4.85e8 which is safe; exp(>88) overflows float
                    double iRaw = NumOps.ToDouble(iVal);
                    double iClampedExp = Math.Exp(Math.Min(iRaw, 20.0));
                    iVal = NumOps.FromDouble(iClampedExp);

                    // Matrix cell update: C = f * C + i * (v outer k)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = v[new[] { bi, flatDi }];
                        T nPrev = normState[new[] { bi, hi, di }];

                        // Normalizer update: n = f * n + i * k
                        T kDi = k[new[] { bi, flatDi }];
                        T nNew = NumOps.Add(NumOps.Multiply(fVal, nPrev),
                            NumOps.Multiply(iVal, kDi));
                        normState[new[] { bi, hi, di }] = nNew;

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = k[new[] { bi, flatKi }];
                            T cPrev = cellState[new[] { bi, hi, di, ki }];

                            // C_new = f * C_prev + i * v * k
                            T cNew = NumOps.Add(
                                NumOps.Multiply(fVal, cPrev),
                                NumOps.Multiply(iVal, NumOps.Multiply(vVal, kVal)));
                            cellState[new[] { bi, hi, di, ki }] = cNew;

                            // Output: h = o * (C * q) / max(|n^T * q|, 1)
                            T qVal = q[new[] { bi, flatKi }];
                            h_t[new[] { bi, flatDi }] = NumOps.Add(
                                h_t[new[] { bi, flatDi }],
                                NumOps.Multiply(oVal, NumOps.Multiply(cNew, qVal)));
                        }

                        // Normalize
                        T nDotQ = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            nDotQ = NumOps.Add(nDotQ,
                                NumOps.Multiply(normState[new[] { bi, hi, ki }],
                                    q[new[] { bi, flatKi }]));
                        }
                        double nDotQAbs = Math.Abs(NumOps.ToDouble(nDotQ));
                        double normFactor = Math.Max(nDotQAbs, 1.0);

                        h_t[new[] { bi, flatDi }] = NumOps.Divide(
                            h_t[new[] { bi, flatDi }],
                            NumOps.FromDouble(normFactor));
                    }
                }
            }

            allHiddenPreProj.SetSlice(1, t, h_t);

            // Output projection
            var y_t = Engine.TensorMatMul(h_t, _outputProjectionWeights);
            var outBias = _outputProjectionBias.Reshape(1, _modelDimension);
            y_t = Engine.TensorBroadcastAdd(y_t, outBias);

            output.SetSlice(1, t, y_t);
        }

        _lastCellStates = allCellStates;
        _lastNormStates = allNormStates;
        _lastInputGates = allInputGates;
        _lastForgetGates = allForgetGates;
        _lastOutputGates = allOutputGates;
        _lastQ = allQ;
        _lastK = allK;
        _lastV = allV;
        _lastHiddenPreProj = allHiddenPreProj;

        var result = ApplyActivation(output);
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

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastCellStates == null || _lastNormStates == null ||
            _lastInputGates == null || _lastForgetGates == null ||
            _lastOutputGates == null || _lastQ == null || _lastK == null ||
            _lastV == null || _lastHiddenPreProj == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];
        T scaleK = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _inputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _inputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _forgetGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _forgetGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward: y = h * Wout + bout
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var hFlat = _lastHiddenPreProj.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            hFlat.Transpose([1, 0]), gradFlat);

        // dh = dOut * Wout^T  [batch*seqLen, modelDim]
        var dh = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Accumulate input gradient through all projection paths
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Running cell/norm state gradients for BPTT
        var dCellState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var dNormState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            var x_t = _lastInput.GetSliceAlongDimension(t, 1);
            var dh_t = dh.GetSliceAlongDimension(t, 1);  // [batch, modelDim]
            var q_t = _lastQ.GetSliceAlongDimension(t, 1);
            var k_t = _lastK.GetSliceAlongDimension(t, 1);
            var v_t = _lastV.GetSliceAlongDimension(t, 1);
            var iGate = _lastInputGates.GetSliceAlongDimension(t, 1);
            var fGate = _lastForgetGates.GetSliceAlongDimension(t, 1);
            var oGate = _lastOutputGates.GetSliceAlongDimension(t, 1);

            var dQ = new Tensor<T>(new[] { batchSize, _modelDimension });
            var dK = new Tensor<T>(new[] { batchSize, _modelDimension });
            var dV = new Tensor<T>(new[] { batchSize, _modelDimension });
            var dIGate = new Tensor<T>(new[] { batchSize, _modelDimension });
            var dFGate = new Tensor<T>(new[] { batchSize, _modelDimension });
            var dOGate = new Tensor<T>(new[] { batchSize, _modelDimension });

            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T iVal = iGate[new[] { bi, dimStart }];
                    T fVal = fGate[new[] { bi, dimStart }];
                    T oVal = oGate[new[] { bi, dimStart }];

                    // Recompute normalizer for this timestep
                    T nDotQ = NumOps.Zero;
                    for (int ki = 0; ki < _headDimension; ki++)
                    {
                        int flatKi = dimStart + ki;
                        nDotQ = NumOps.Add(nDotQ,
                            NumOps.Multiply(
                                _lastNormStates[new[] { bi, t + 1, hi, ki }],
                                q_t[new[] { bi, flatKi }]));
                    }
                    double normFactor = Math.Max(Math.Abs(NumOps.ToDouble(nDotQ)), 1.0);
                    T invNorm = NumOps.FromDouble(1.0 / normFactor);

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dhVal = NumOps.Multiply(dh_t[new[] { bi, flatDi }], invNorm);

                        // dOutput gate: dh/do = (C * q) / norm
                        T cq = NumOps.Zero;
                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            cq = NumOps.Add(cq,
                                NumOps.Multiply(
                                    _lastCellStates[new[] { bi, t + 1, hi, di, ki }],
                                    q_t[new[] { bi, flatKi }]));
                        }
                        dOGate[new[] { bi, flatDi }] = NumOps.Add(
                            dOGate[new[] { bi, flatDi }],
                            NumOps.Multiply(NumOps.Multiply(cq, invNorm), dh_t[new[] { bi, flatDi }]));

                        // Gradient through C*q: oVal * dhVal flows to both C and q
                        T oTimesdh = NumOps.Multiply(oVal, dhVal);

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T cellVal = _lastCellStates[new[] { bi, t + 1, hi, di, ki }];
                            T qVal = q_t[new[] { bi, flatKi }];

                            // dQ += o * dh * C[di,ki] / norm
                            dQ[new[] { bi, flatKi }] = NumOps.Add(
                                dQ[new[] { bi, flatKi }],
                                NumOps.Multiply(oTimesdh, cellVal));

                            // dCell += o * dh * q[ki] / norm
                            T dCell = NumOps.Multiply(oTimesdh, qVal);
                            dCellState[new[] { bi, hi, di, ki }] = NumOps.Add(
                                dCellState[new[] { bi, hi, di, ki }], dCell);
                        }
                    }

                    // Cell state backward: C = f * C_prev + i * v * k
                    // dC_prev = f * dC, dF += dC * C_prev, dI += dC * v * k, dV += dC * i * k, dK += dC * i * v
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = v_t[new[] { bi, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T dC = dCellState[new[] { bi, hi, di, ki }];
                            T kVal = k_t[new[] { bi, flatKi }];

                            // dForget gate from cell state
                            T cPrev = t > 0
                                ? _lastCellStates[new[] { bi, t, hi, di, ki }]
                                : NumOps.Zero;
                            dFGate[new[] { bi, dimStart }] = NumOps.Add(
                                dFGate[new[] { bi, dimStart }],
                                NumOps.Multiply(dC, cPrev));

                            // dInput gate from cell state
                            dIGate[new[] { bi, dimStart }] = NumOps.Add(
                                dIGate[new[] { bi, dimStart }],
                                NumOps.Multiply(dC, NumOps.Multiply(vVal, kVal)));

                            // dV from cell state
                            dV[new[] { bi, flatDi }] = NumOps.Add(
                                dV[new[] { bi, flatDi }],
                                NumOps.Multiply(dC, NumOps.Multiply(iVal, kVal)));

                            // dK from cell state
                            dK[new[] { bi, flatKi }] = NumOps.Add(
                                dK[new[] { bi, flatKi }],
                                NumOps.Multiply(dC, NumOps.Multiply(iVal, vVal)));

                            // Propagate to previous cell state
                            dCellState[new[] { bi, hi, di, ki }] = NumOps.Multiply(fVal, dC);
                        }
                    }
                }
            }

            // Gate derivative chains: input gate uses exp, forget/output use sigmoid
            // dIGate is w.r.t. exp output, chain rule: d/dx exp(x) = exp(x)
            var dIGateRaw = Engine.TensorMultiply(dIGate, iGate);

            // Sigmoid derivative: sig(x) * (1 - sig(x))
            var ones = new Tensor<T>(fGate.Shape);
            for (int idx = 0; idx < ones.Length; idx++) ones[idx] = NumOps.One;
            var fSigDeriv = Engine.TensorMultiply(fGate, Engine.TensorSubtract(ones, fGate));
            var dFGateRaw = Engine.TensorMultiply(dFGate, fSigDeriv);

            var oSigDeriv = Engine.TensorMultiply(oGate, Engine.TensorSubtract(ones, oGate));
            var dOGateRaw = Engine.TensorMultiply(dOGate, oSigDeriv);

            // Accumulate weight gradients: dW += x^T * dGateRaw
            var x_t_T = x_t.Transpose([1, 0]);
            _inputGateWeightsGradient = Engine.TensorAdd(_inputGateWeightsGradient,
                Engine.TensorMatMul(x_t_T, dIGateRaw));
            _inputGateBiasGradient = Engine.TensorAdd(_inputGateBiasGradient,
                Engine.ReduceSum(dIGateRaw, new int[] { 0 }));
            _forgetGateWeightsGradient = Engine.TensorAdd(_forgetGateWeightsGradient,
                Engine.TensorMatMul(x_t_T, dFGateRaw));
            _forgetGateBiasGradient = Engine.TensorAdd(_forgetGateBiasGradient,
                Engine.ReduceSum(dFGateRaw, new int[] { 0 }));
            _outputGateWeightsGradient = Engine.TensorAdd(_outputGateWeightsGradient,
                Engine.TensorMatMul(x_t_T, dOGateRaw));
            _outputGateBiasGradient = Engine.TensorAdd(_outputGateBiasGradient,
                Engine.ReduceSum(dOGateRaw, new int[] { 0 }));

            // Q, K, V weight gradients
            _queryWeightsGradient = Engine.TensorAdd(_queryWeightsGradient,
                Engine.TensorMatMul(x_t_T, dQ));
            _keyWeightsGradient = Engine.TensorAdd(_keyWeightsGradient,
                Engine.TensorMatMul(x_t_T, Engine.TensorMultiplyScalar(dK, scaleK)));
            _valueWeightsGradient = Engine.TensorAdd(_valueWeightsGradient,
                Engine.TensorMatMul(x_t_T, dV));

            // Input gradient: sum of all paths flowing through x_t
            var dX_t = Engine.TensorMatMul(dIGateRaw, _inputGateWeights.Transpose([1, 0]));
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorMatMul(dFGateRaw, _forgetGateWeights.Transpose([1, 0])));
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorMatMul(dOGateRaw, _outputGateWeights.Transpose([1, 0])));
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorMatMul(dQ, _queryWeights.Transpose([1, 0])));
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorMatMul(
                Engine.TensorMultiplyScalar(dK, scaleK), _keyWeights.Transpose([1, 0])));
            dX_t = Engine.TensorAdd(dX_t, Engine.TensorMatMul(dV, _valueWeights.Transpose([1, 0])));

            dInput.SetSlice(1, t, dX_t);
        }

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputGateWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
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
        _inputGateWeights, _inputGateBias,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _queryWeights, _keyWeights, _valueWeights,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastCellStates = null;
        _lastNormStates = null;
        _lastInputGates = null;
        _lastForgetGates = null;
        _lastOutputGates = null;
        _lastQ = null;
        _lastK = null;
        _lastV = null;
        _lastHiddenPreProj = null;
        _originalInputShape = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the forget gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetForgetGateWeights() => _forgetGateWeights;
}
