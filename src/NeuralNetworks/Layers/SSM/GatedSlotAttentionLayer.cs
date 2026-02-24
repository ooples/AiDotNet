using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Gated Slot Attention (GSA) layer from Li et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// Gated Slot Attention maintains a fixed-size set of "slots" that act as compressed memory,
/// combining ideas from slot attention (object-centric learning) with gated linear recurrences
/// for efficient linear-time sequence modeling.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute gates: forget_gate (controls memory retention) and input_gate (controls writing)
///   3. Slot update (gated write):
///      S_t = forget_gate_t * S_{t-1} + input_gate_t * (v_t outer k_t)
///      The forget gate controls how much old slot content is retained.
///      The input gate controls how strongly new key-value associations are written.
///   4. Slot read:
///      o_t = S_t * q_t
///      Queries read from the current slot state to produce output.
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key difference from standard linear attention: the fixed slot count (n_slots) bounds
/// memory usage regardless of sequence length, and the dual gating mechanism (forget + input)
/// provides fine-grained control over information flow. This is analogous to how LSTM gates
/// control cell state, but applied to a matrix-valued memory (the slots).
/// </para>
/// <para><b>For Beginners:</b> GSA is a memory-efficient attention alternative.
///
/// Imagine you have a whiteboard with a fixed number of "slots" (rows) for taking notes:
/// - Standard attention: You compare every word with every other word (expensive for long texts)
/// - GSA: You maintain a fixed-size "summary board" and update it as you read each word
///
/// At each step:
/// - The "forget gate" decides which old notes to erase (like erasing parts of the whiteboard)
/// - The "input gate" decides how strongly to write new notes
/// - The "key" determines WHERE on the board to write
/// - The "value" determines WHAT to write
/// - The "query" determines what to READ from the board
///
/// Because the board has a fixed number of slots, the memory cost stays constant
/// regardless of how long the text is, making GSA efficient for very long sequences.
///
/// The gating mechanism is crucial: without it, old information would pile up and the
/// slots would become increasingly noisy. The gates let the model learn to selectively
/// retain important information and overwrite stale content.
/// </para>
/// <para>
/// <b>Reference:</b> Li et al., "Gated Slot Attention for Efficient Linear-Time Sequence Modeling", 2024.
/// https://arxiv.org/abs/2409.07146
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GatedSlotAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numSlots;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Forget gate projection: [modelDim, numHeads]
    private Tensor<T> _forgetGateWeights;
    private Tensor<T> _forgetGateBias;

    // Input gate projection: [modelDim, numHeads]
    private Tensor<T> _inputGateWeights;
    private Tensor<T> _inputGateBias;

    // Initial slot embeddings: [numHeads, numSlots, headDim]
    private Tensor<T> _initialSlots;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastForgetGate;
    private Tensor<T>? _lastInputGate;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private Tensor<T>? _lastSlotStates;
    private Tensor<T>? _lastSlotReadOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _forgetGateWeightsGradient;
    private Tensor<T>? _forgetGateBiasGradient;
    private Tensor<T>? _inputGateWeightsGradient;
    private Tensor<T>? _inputGateBiasGradient;
    private Tensor<T>? _initialSlotsGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
    /// Gets the number of memory slots per head.
    /// </summary>
    public int NumSlots => _numSlots;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _forgetGateWeights.Length + _forgetGateBias.Length +
        _inputGateWeights.Length + _inputGateBias.Length +
        _initialSlots.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Gated Slot Attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of the data flowing through this layer.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own independent set of memory slots.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numSlots">
    /// Number of memory slots per head. Default: 64.
    /// <para><b>For Beginners:</b> This controls how much "memory" each head has. More slots can
    /// store more information but use more computation. The slot count bounds the memory
    /// regardless of sequence length, which is what makes GSA efficient.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public GatedSlotAttentionLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int numSlots = 64,
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
        if (numSlots <= 0)
            throw new ArgumentException($"Number of slots ({numSlots}) must be positive.", nameof(numSlots));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _numSlots = numSlots;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _forgetGateBias = new Tensor<T>([numHeads]);
        _inputGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _inputGateBias = new Tensor<T>([numHeads]);
        _initialSlots = new Tensor<T>([numHeads, numSlots, _headDimension]);
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
        InitializeTensor2D(_inputGateWeights);
        // Forget gate bias ~ 2 so sigmoid(2) ~ 0.88 -> strong initial memory retention
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(2.0);
        _inputGateBias.Fill(NumOps.FromDouble(0.1));
        InitializeSlots();
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

    private void InitializeSlots()
    {
        // Initialize slot embeddings with Xavier-like initialization
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (_numSlots + _headDimension)));
        for (int i = 0; i < _initialSlots.Length; i++)
            _initialSlots[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
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

        // Step 2: Gates
        var forgetRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _forgetGateWeights),
            _forgetGateBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var forgetGate = Engine.Sigmoid(forgetRaw);
        _lastForgetGate = forgetGate;

        var inputGateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _inputGateWeights),
            _inputGateBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var inputGate = Engine.Sigmoid(inputGateRaw);
        _lastInputGate = inputGate;

        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var outputGate = Engine.Swish(gateRaw);
        _lastOutputGate = outputGate;
        _lastOutputGateRaw = gateRaw;

        // Step 3: Slot update recurrence and slot read
        var slotOutput = SlotRecurrenceForward(q, k, v, forgetGate, inputGate, batchSize, seqLen);
        _lastSlotReadOutput = slotOutput;

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(slotOutput, outputGate);

        // Step 5: Output projection
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
    /// Slot recurrence forward: gated write to slots and query-based read from slots.
    /// </summary>
    /// <remarks>
    /// For each timestep t and each head h:
    ///   S_t[h] = forget_gate[t,h] * S_{t-1}[h] + input_gate[t,h] * v_t[h] outer k_t[h]
    ///   o_t[h] = S_t[h] * q_t[h]
    ///
    /// The slot matrix S has shape [numSlots, headDim] per head. The outer product v*k^T
    /// writes a rank-1 update into the slot space. Because numSlots is fixed, this bounds
    /// memory independent of sequence length.
    /// </remarks>
    private Tensor<T> SlotRecurrenceForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> forgetGate, Tensor<T> inputGate,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Slot state per head: [batch, numHeads, numSlots, headDim]
        var slotState = new Tensor<T>(new[] { batchSize, _numHeads, _numSlots, _headDimension });

        // Initialize slots from learned initial embeddings
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int si = 0; si < _numSlots; si++)
                    for (int di = 0; di < _headDimension; di++)
                        slotState[new[] { bi, hi, si, di }] = _initialSlots[new[] { hi, si, di }];

        // Save all slot states for backward pass: [batch, seqLen+1, numHeads, numSlots, headDim]
        var allStates = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _numSlots, _headDimension });

        // Save initial state at t=0
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int si = 0; si < _numSlots; si++)
                    for (int di = 0; di < _headDimension; di++)
                        allStates[new[] { bi, 0, hi, si, di }] = slotState[new[] { bi, hi, si, di }];

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T fGate = forgetGate[new[] { bi, t, hi }];
                    T iGate = inputGate[new[] { bi, t, hi }];

                    // Slot update: S_t = fGate * S_{t-1} + iGate * (v_t outer k_t)
                    // v_t is [headDim], k_t is [headDim] (mapped to slots via outer product)
                    // For GSA, the slot dimension maps keys to slots, so:
                    //   S[si, di] += iGate * v[di] * k_scaled[si % headDim]
                    // The numSlots can differ from headDim, so we use modular indexing
                    // to map the key space onto the slot space.
                    for (int si = 0; si < _numSlots; si++)
                    {
                        // Key index for this slot (maps slots to key dimensions)
                        int kIdx = si % _headDimension;
                        int flatKIdx = dimStart + kIdx;
                        T kVal = NumOps.Multiply(k[new[] { bi, t, flatKIdx }], keyScale);

                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T prevS = slotState[new[] { bi, hi, si, di }];
                            T vVal = v[new[] { bi, t, flatDi }];

                            // S_t[si,di] = fGate * S_{t-1}[si,di] + iGate * v[di] * k[kIdx]
                            T update = NumOps.Multiply(iGate, NumOps.Multiply(vVal, kVal));
                            T newS = NumOps.Add(NumOps.Multiply(fGate, prevS), update);
                            slotState[new[] { bi, hi, si, di }] = newS;
                        }
                    }

                    // Slot read: o_t[di] = sum_si S_t[si, di] * q_scaled[si % headDim]
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int si = 0; si < _numSlots; si++)
                        {
                            int qIdx = si % _headDimension;
                            int flatQIdx = dimStart + qIdx;
                            T qVal = q[new[] { bi, t, flatQIdx }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(slotState[new[] { bi, hi, si, di }], qVal));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }
                }
            }

            // Save slot state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi = 0; hi < _numHeads; hi++)
                    for (int si = 0; si < _numSlots; si++)
                        for (int di = 0; di < _headDimension; di++)
                            allStates[new[] { bi, t + 1, hi, si, di }] = slotState[new[] { bi, hi, si, di }];
        }

        _lastSlotStates = allStates;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastForgetGate == null || _lastInputGate == null ||
            _lastOutputGate == null || _lastOutputGateRaw == null ||
            _lastSlotReadOutput == null || _lastSlotStates == null)
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
        _inputGateWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _inputGateBiasGradient = new Tensor<T>([_numHeads]);
        _initialSlotsGradient = new Tensor<T>([_numHeads, _numSlots, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 5 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(_lastSlotReadOutput, _lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 4 backward: gating - gatedOutput = slotReadOutput * outputGate
        var dSlotRead = Engine.TensorMultiply(dGated, _lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastSlotReadOutput);

        // Output gate uses Swish, derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastOutputGateRaw));

        // Output gate weight gradients
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        // Accumulate input gradient from output gate path
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 3 backward: slot recurrence (backward through time)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dForgetGate = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });
        var dInputGate = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Gradient of slot state, propagated backward through time
        var dSlotState = new Tensor<T>(new[] { batchSize, _numHeads, _numSlots, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T fGate = _lastForgetGate[new[] { bi, t, hi }];
                    T iGate = _lastInputGate[new[] { bi, t, hi }];

                    // Backward through slot read: o_t[di] = sum_si S_t[si,di] * q[si%headDim]
                    // dS_t[si,di] += dO[di] * q[si%headDim]
                    // dQ[si%headDim] += sum_di dO[di] * S_t[si,di]  (accumulated over slots)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dSlotRead[new[] { bi, t, flatDi }];

                        for (int si = 0; si < _numSlots; si++)
                        {
                            int qIdx = si % _headDimension;
                            int flatQIdx = dimStart + qIdx;
                            T qVal = _lastQuery[new[] { bi, t, flatQIdx }];
                            T sVal = _lastSlotStates[new[] { bi, t + 1, hi, si, di }];

                            dSlotState[new[] { bi, hi, si, di }] = NumOps.Add(
                                dSlotState[new[] { bi, hi, si, di }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatQIdx }] = NumOps.Add(
                                dQ[new[] { bi, t, flatQIdx }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // Backward through slot update:
                    // S_t[si,di] = fGate * S_{t-1}[si,di] + iGate * v[di] * k_scaled[si%headDim]
                    //
                    // dS_{t-1}[si,di] = fGate * dS_t[si,di]
                    // dForgetGate += sum_{si,di} dS_t[si,di] * S_{t-1}[si,di]
                    // dInputGate += sum_{si,di} dS_t[si,di] * v[di] * k_scaled[si%headDim]
                    // dV[di] += sum_si dS_t[si,di] * iGate * k_scaled[si%headDim]
                    // dK[si%headDim] += sum_di dS_t[si,di] * iGate * v[di] * keyScale
                    for (int si = 0; si < _numSlots; si++)
                    {
                        int kIdx = si % _headDimension;
                        int flatKIdx = dimStart + kIdx;
                        T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKIdx }], keyScale);

                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T dS = dSlotState[new[] { bi, hi, si, di }];
                            T sPrev = _lastSlotStates[new[] { bi, t, hi, si, di }];
                            T vVal = _lastValue[new[] { bi, t, flatDi }];

                            // dForgetGate
                            dForgetGate[new[] { bi, t, hi }] = NumOps.Add(
                                dForgetGate[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, sPrev));

                            // dInputGate
                            dInputGate[new[] { bi, t, hi }] = NumOps.Add(
                                dInputGate[new[] { bi, t, hi }],
                                NumOps.Multiply(dS, NumOps.Multiply(vVal, kVal)));

                            // dV[di] += dS * iGate * k_scaled
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(NumOps.Multiply(dS, iGate), kVal));

                            // dK[kIdx] += dS * iGate * v[di] * keyScale
                            dK[new[] { bi, t, flatKIdx }] = NumOps.Add(
                                dK[new[] { bi, t, flatKIdx }],
                                NumOps.Multiply(NumOps.Multiply(dS, iGate),
                                    NumOps.Multiply(vVal, keyScale)));

                            // Propagate gradient to previous timestep
                            dSlotState[new[] { bi, hi, si, di }] = NumOps.Multiply(fGate, dS);
                        }
                    }
                }
            }
        }

        // Accumulate gradient into initial slots (from dSlotState at t=0)
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int si = 0; si < _numSlots; si++)
                    for (int di = 0; di < _headDimension; di++)
                        _initialSlotsGradient[new[] { hi, si, di }] = NumOps.Add(
                            _initialSlotsGradient[new[] { hi, si, di }],
                            dSlotState[new[] { bi, hi, si, di }]);

        // Forget gate and input gate through sigmoid derivative: sig'(x) = sig(x) * (1 - sig(x))
        var forgetSigDeriv = Engine.TensorMultiply(_lastForgetGate,
            Engine.TensorSubtract(CreateOnesLike(_lastForgetGate), _lastForgetGate));
        var dForgetGateRaw = Engine.TensorMultiply(dForgetGate, forgetSigDeriv);

        var inputSigDeriv = Engine.TensorMultiply(_lastInputGate,
            Engine.TensorSubtract(CreateOnesLike(_lastInputGate), _lastInputGate));
        var dInputGateRaw = Engine.TensorMultiply(dInputGate, inputSigDeriv);

        // Gate weight gradients
        var dForgetFlat = dForgetGateRaw.Reshape(batchSize * seqLen, _numHeads);
        _forgetGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dForgetFlat);
        _forgetGateBiasGradient = Engine.ReduceSum(dForgetGateRaw, new int[] { 0, 1 });

        var dInputGateFlat = dInputGateRaw.Reshape(batchSize * seqLen, _numHeads);
        _inputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dInputGateFlat);
        _inputGateBiasGradient = Engine.ReduceSum(dInputGateRaw, new int[] { 0, 1 });

        // Q, K, V weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient: accumulate from all projection paths
        var dInputTotal = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dForgetFlat, _forgetGateWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dInputGateFlat, _inputGateWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromGate);

        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
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
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient!, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _initialSlots = Engine.TensorAdd(_initialSlots, Engine.TensorMultiplyScalar(_initialSlotsGradient!, negLR));
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
        _inputGateWeights, _inputGateBias,
        _initialSlots,
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
        _lastInputGate = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _lastSlotStates = null;
        _lastSlotReadOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _initialSlotsGradient = null;
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
        metadata["NumSlots"] = _numSlots.ToString();
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
    /// Gets the initial slot embeddings for external inspection.
    /// </summary>
    public Tensor<T> GetInitialSlots() => _initialSlots;
}
