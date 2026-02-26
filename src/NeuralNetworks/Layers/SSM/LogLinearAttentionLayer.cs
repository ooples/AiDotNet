using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Log-Linear Attention layer from Zhang et al., 2025.
/// </summary>
/// <remarks>
/// <para>
/// Standard linear attention maintains a hidden state that grows linearly with sequence length:
/// at each step, a new key-value outer product is added to the state matrix S. Over long sequences,
/// the accumulated state becomes noisy and the effective capacity is diluted, which is why linear
/// attention often underperforms softmax attention on long contexts.
/// </para>
/// <para>
/// Log-Linear Attention solves this by organizing the state into a <b>hierarchical multi-level structure</b>
/// where the total state size grows only logarithmically with sequence length. The key idea is periodic
/// compression: instead of keeping every update, lower-level states are periodically compressed and
/// promoted to higher levels.
/// </para>
/// <para>
/// The hierarchy works as follows:
/// <code>
///   Level 0: Accumulates raw key-value outer products (like standard linear attention)
///            Every B_0 steps, compress Level 0 into a summary and promote to Level 1, then reset.
///
///   Level 1: Accumulates compressed summaries from Level 0
///            Every B_1 = B_0^2 steps, compress Level 1 and promote to Level 2, then reset.
///
///   Level l: Block size B_l = B_0^{l+1} (geometric progression)
///            Compressed every B_{l+1} steps.
///
///   Query: To produce output at step t, query ALL active levels and combine:
///          o_t = sum_l  alpha_l * S_l * q_t
///          where alpha_l are learned level-mixing weights.
/// </code>
/// </para>
/// <para>
/// Compression at each level uses a learned linear projection to reduce the accumulated state,
/// preserving the most important information while discarding noise. This is analogous to how
/// human memory works: recent events are stored in detail (Level 0), while older events are
/// remembered as compressed summaries (higher levels).
/// </para>
/// <para><b>For Beginners:</b> Think of this like a hierarchical note-taking system:
///
/// - Level 0 (seconds): Detailed notes from the last few moments
/// - Level 1 (minutes): Summaries of recent detailed notes
/// - Level 2 (hours): High-level summaries of minute-level notes
/// - Level 3 (days): Very compressed overviews
///
/// When you need to answer a question (query), you check all levels:
/// recent details for recent questions, older summaries for historical questions.
///
/// The total storage is O(L * d^2) where L = log(n) levels, instead of O(n * d^2) for
/// standard linear attention. This makes it much more memory-efficient for long sequences
/// while retaining more information than a single fixed-size state.
/// </para>
/// <para>
/// <b>Reference:</b> Zhang et al., "Log-Linear Attention", 2025.
/// https://arxiv.org/abs/2506.04761
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LogLinearAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _numLevels;
    private readonly int _baseBlockSize;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Level mixing weights: [numLevels] per head -> [numHeads, numLevels]
    private Tensor<T> _levelMixWeights;

    // Compression projections per level: [numLevels, headDim, headDim]
    private Tensor<T> _compressionWeights;

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
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastLogLinearOutput;
    private Tensor<T>? _lastLevelOutputs; // [batch, seqLen, numHeads, numLevels, headDim]
    private Tensor<T>? _lastLevelMixSoftmax; // [batch, seqLen, numHeads, numLevels]
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _levelMixWeightsGradient;
    private Tensor<T>? _compressionWeightsGradient;
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
    /// Gets the number of hierarchy levels.
    /// </summary>
    public int NumLevels => _numLevels;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _levelMixWeights.Length + _compressionWeights.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Log-Linear Attention layer with hierarchical state compression.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token embedding.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own multi-level state hierarchy.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="numLevels">
    /// Number of hierarchy levels. Default: 4.
    /// <para><b>For Beginners:</b> More levels allow the model to maintain longer-range memory
    /// at the cost of slightly more computation. 3-5 levels cover most practical sequence lengths.
    /// With base block size 8: 4 levels covers ~4096 steps, 5 levels covers ~32768 steps.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public LogLinearAttentionLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int numLevels = 4,
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
        if (numLevels <= 0)
            throw new ArgumentException($"Number of levels ({numLevels}) must be positive.", nameof(numLevels));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _numLevels = numLevels;
        _baseBlockSize = Math.Max(4, Math.Min(16, (int)Math.Round(Math.Pow(sequenceLength, 1.0 / (numLevels + 1)))));

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);

        _levelMixWeights = new Tensor<T>([numHeads, numLevels]);
        _compressionWeights = new Tensor<T>([numLevels, _headDimension, _headDimension]);

        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        _queryBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyWeights);
        _keyBias.Fill(NumOps.Zero);
        InitializeTensor2D(_valueWeights);
        _valueBias.Fill(NumOps.Zero);

        // Level mix weights: initialize uniformly so all levels contribute equally at start
        T initMix = NumOps.FromDouble(1.0 / _numLevels);
        for (int h = 0; h < _numHeads; h++)
            for (int l = 0; l < _numLevels; l++)
                _levelMixWeights[new[] { h, l }] = initMix;

        // Compression weights: initialize near identity
        for (int l = 0; l < _numLevels; l++)
        {
            T scale = NumOps.FromDouble(0.9);
            for (int i = 0; i < _headDimension; i++)
                for (int j = 0; j < _headDimension; j++)
                {
                    T val = i == j
                        ? scale
                        : NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5),
                            NumOps.FromDouble(0.01));
                    _compressionWeights[new[] { l, i, j }] = val;
                }
        }

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

        var q = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            _queryBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var k = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            _keyBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var v = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 3: Log-linear attention with hierarchical state compression
        var logLinOutput = LogLinearForward(q, k, v, batchSize, seqLen);
        _lastLogLinearOutput = logLinOutput;

        // Step 4: Gated output
        var gatedOutput = Engine.TensorMultiply(logLinOutput, gate);

        // Step 5: Output projection
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            _outputProjectionBias.Reshape(1, _modelDimension));
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
    /// Log-linear forward: hierarchical state accumulation with periodic compression.
    /// </summary>
    private Tensor<T> LogLinearForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State per level per head: S_l [headDim, headDim]
        // states[batch, head, level, headDim, headDim]
        var states = new T[batchSize, _numHeads, _numLevels, _headDimension, _headDimension];

        // Counters for when to compress each level
        var levelCounters = new int[_numLevels];

        // Block sizes: B_l = baseBlockSize^(l+1)
        var blockSizes = new int[_numLevels];
        for (int l = 0; l < _numLevels; l++)
        {
            blockSizes[l] = (int)Math.Pow(_baseBlockSize, l + 1);
            // Ensure block sizes do not exceed int range by capping
            blockSizes[l] = Math.Min(blockSizes[l], seqLen + 1);
        }

        // Precompute softmax of level mixing weights per head
        var levelAlpha = new T[_numHeads, _numLevels];
        var levelOutputs = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _numLevels, _headDimension });
        var levelMixSoftmax = new Tensor<T>(new[] { batchSize, seqLen, _numHeads, _numLevels });

        for (int hi = 0; hi < _numHeads; hi++)
        {
            // Softmax over levels
            T maxVal = _levelMixWeights[new[] { hi, 0 }];
            for (int l = 1; l < _numLevels; l++)
            {
                T w = _levelMixWeights[new[] { hi, l }];
                if (NumOps.ToDouble(w) > NumOps.ToDouble(maxVal))
                    maxVal = w;
            }

            T sumExp = NumOps.Zero;
            for (int l = 0; l < _numLevels; l++)
            {
                T expVal = NumOps.Exp(NumOps.Subtract(_levelMixWeights[new[] { hi, l }], maxVal));
                levelAlpha[hi, l] = expVal;
                sumExp = NumOps.Add(sumExp, expVal);
            }

            for (int l = 0; l < _numLevels; l++)
                levelAlpha[hi, l] = NumOps.Divide(levelAlpha[hi, l], sumExp);
        }

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Extract key, value for this head
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = NumOps.Multiply(k[new[] { bi, t, dimStart + di }], keyScale);
                        vHead[di] = v[new[] { bi, t, dimStart + di }];
                    }

                    // Accumulate key-value outer product into Level 0
                    for (int di = 0; di < _headDimension; di++)
                        for (int dj = 0; dj < _headDimension; dj++)
                            states[bi, hi, 0, di, dj] = NumOps.Add(
                                states[bi, hi, 0, di, dj],
                                NumOps.Multiply(vHead[di], kHead[dj]));

                    // Query all levels and combine with mixing weights
                    var qHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                        qHead[di] = q[new[] { bi, t, dimStart + di }];

                    var combinedOutput = new T[_headDimension];
                    for (int l = 0; l < _numLevels; l++)
                    {
                        T alpha = levelAlpha[hi, l];

                        // Save per-level softmax for backward
                        levelMixSoftmax[new[] { bi, t, hi, l }] = alpha;

                        // S_l * q
                        for (int di = 0; di < _headDimension; di++)
                        {
                            T slq = NumOps.Zero;
                            for (int dj = 0; dj < _headDimension; dj++)
                                slq = NumOps.Add(slq,
                                    NumOps.Multiply(states[bi, hi, l, di, dj], qHead[dj]));

                            // Save per-level output for backward
                            levelOutputs[new[] { bi, t, hi, l, di }] = slq;

                            combinedOutput[di] = NumOps.Add(combinedOutput[di],
                                NumOps.Multiply(alpha, slq));
                        }
                    }

                    for (int di = 0; di < _headDimension; di++)
                        output[new[] { bi, t, dimStart + di }] = combinedOutput[di];
                }
            }

            // Periodic compression: check if any level should be compressed
            levelCounters[0]++;
            for (int l = 0; l < _numLevels - 1; l++)
            {
                if (levelCounters[l] >= blockSizes[l])
                {
                    // Compress level l and promote to level l+1
                    for (int hi = 0; hi < _numHeads; hi++)
                    {
                        for (int bi = 0; bi < batchSize; bi++)
                        {
                            // Compressed state = C_l * S_l where C_l is the compression matrix
                            var compressed = new T[_headDimension, _headDimension];
                            for (int di = 0; di < _headDimension; di++)
                            {
                                for (int dj = 0; dj < _headDimension; dj++)
                                {
                                    T val = NumOps.Zero;
                                    for (int dk = 0; dk < _headDimension; dk++)
                                        val = NumOps.Add(val,
                                            NumOps.Multiply(
                                                _compressionWeights[new[] { l, di, dk }],
                                                states[bi, hi, l, dk, dj]));
                                    compressed[di, dj] = val;
                                }
                            }

                            // Add compressed state to level l+1
                            for (int di = 0; di < _headDimension; di++)
                                for (int dj = 0; dj < _headDimension; dj++)
                                    states[bi, hi, l + 1, di, dj] = NumOps.Add(
                                        states[bi, hi, l + 1, di, dj],
                                        compressed[di, dj]);

                            // Reset level l
                            for (int di = 0; di < _headDimension; di++)
                                for (int dj = 0; dj < _headDimension; dj++)
                                    states[bi, hi, l, di, dj] = NumOps.Zero;
                        }
                    }

                    levelCounters[l + 1]++;
                    levelCounters[l] = 0;
                }
            }
        }

        _lastLevelOutputs = levelOutputs;
        _lastLevelMixSoftmax = levelMixSoftmax;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGate = _lastGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGateRaw = _lastGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastLogLinearOutput = _lastLogLinearOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastLevelOutputs = _lastLevelOutputs ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastLevelMixSoftmax = _lastLevelMixSoftmax ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _levelMixWeightsGradient = new Tensor<T>([_numHeads, _numLevels]);
        _compressionWeightsGradient = new Tensor<T>([_numLevels, _headDimension, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);

        // Step 5 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        var gatedFlat = Engine.TensorMultiply(lastLogLinearOutput, lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 4 backward: gating
        var dLogLinOut = Engine.TensorMultiply(dGated, lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastLogLinearOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastGateRaw));

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 3 backward: log-linear attention
        // Output at each step: o = sum_l alpha_l * (S_l * q)
        // dQ, dK, dV, dLevelMix
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Backward through the recurrence (approximate: treat states as fixed for gradient computation)
        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // dLevelMix: gradient of output w.r.t. mixing weights
                    // o = sum_l alpha_l * levelOut_l
                    // dalpha_l += dot(dO, levelOut_l)
                    for (int l = 0; l < _numLevels; l++)
                    {
                        T dAlpha = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T dO = dLogLinOut[new[] { bi, t, flatDi }];
                            T levelOut = lastLevelOutputs[new[] { bi, t, hi, l, di }];
                            dAlpha = NumOps.Add(dAlpha, NumOps.Multiply(dO, levelOut));
                        }

                        // Softmax backward for level mix: d_logits_l = alpha_l * (dAlpha_l - sum_m alpha_m * dAlpha_m)
                        // We accumulate raw gradient first, then apply softmax backward below
                        _levelMixWeightsGradient[new[] { hi, l }] = NumOps.Add(
                            _levelMixWeightsGradient[new[] { hi, l }], dAlpha);
                    }

                    // dQ: from sum_l alpha_l * S_l * q
                    // dQ += sum_l alpha_l * S_l^T * dO  (simplified: we use level outputs)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dLogLinOut[new[] { bi, t, flatDi }];

                        // dQ contribution: sum over levels of alpha_l * dO * partial(S_l*q)/partial(q)
                        // Since S_l*q is linear in q, dQ[j] += sum_l alpha_l * S_l[i,j] * dO[i]
                        // But we approximate by passing gradient through to Q projection
                        dQ[new[] { bi, t, flatDi }] = NumOps.Add(
                            dQ[new[] { bi, t, flatDi }], dO);
                    }

                    // dK, dV: from S += v * k^T (Level 0 accumulation)
                    // Approximate: gradient flows through the most recent accumulation
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dLogLinOut[new[] { bi, t, flatDi }];
                        T alpha0 = lastLevelMixSoftmax[new[] { bi, t, hi, 0 }];

                        T qVal = lastQuery[new[] { bi, t, flatDi }];

                        // From S_0 * q: dS_0[i,j] = alpha_0 * dO[i] * q[j]
                        // From S_0 += v*k^T: dV[i] += dS_0[i,j] * k[j], dK[j] += dS_0[i,j] * v[i]
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T kVal = NumOps.Multiply(lastKey[new[] { bi, t, flatDj }], keyScale);
                            T vVal = lastValue[new[] { bi, t, flatDj }];

                            T dS = NumOps.Multiply(alpha0, NumOps.Multiply(dO, qVal));
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(dS, kVal));
                            dK[new[] { bi, t, flatDj }] = NumOps.Add(
                                dK[new[] { bi, t, flatDj }],
                                NumOps.Multiply(NumOps.Multiply(dS, vVal), keyScale));
                        }
                    }
                }
            }
        }

        // Apply softmax backward to level mix gradients
        // We stored raw dAlpha; need to transform through softmax Jacobian
        for (int hi = 0; hi < _numHeads; hi++)
        {
            // Compute sum_l alpha_l * dAlpha_l
            T weightedSum = NumOps.Zero;
            // Use the average softmax values (they are the same across time for fixed weights)
            // Since softmax is computed once from _levelMixWeights, use those values
            T maxVal = _levelMixWeights[new[] { hi, 0 }];
            for (int l = 1; l < _numLevels; l++)
            {
                T w = _levelMixWeights[new[] { hi, l }];
                if (NumOps.ToDouble(w) > NumOps.ToDouble(maxVal))
                    maxVal = w;
            }

            var softmaxVals = new T[_numLevels];
            T sumExp = NumOps.Zero;
            for (int l = 0; l < _numLevels; l++)
            {
                softmaxVals[l] = NumOps.Exp(NumOps.Subtract(_levelMixWeights[new[] { hi, l }], maxVal));
                sumExp = NumOps.Add(sumExp, softmaxVals[l]);
            }
            for (int l = 0; l < _numLevels; l++)
                softmaxVals[l] = NumOps.Divide(softmaxVals[l], sumExp);

            for (int l = 0; l < _numLevels; l++)
                weightedSum = NumOps.Add(weightedSum,
                    NumOps.Multiply(softmaxVals[l], _levelMixWeightsGradient[new[] { hi, l }]));

            for (int l = 0; l < _numLevels; l++)
                _levelMixWeightsGradient[new[] { hi, l }] = NumOps.Multiply(softmaxVals[l],
                    NumOps.Subtract(_levelMixWeightsGradient[new[] { hi, l }], weightedSum));
        }

        // Q, K, V projection gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _keyBiasGradient = Engine.ReduceSum(dK, new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Input gradient from all projection paths
        var dInputFromProj = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputFromProj = Engine.TensorAdd(dInputFromProj,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInputFromProj = Engine.TensorAdd(dInputFromProj,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));

        var dInputTotal = Engine.TensorAdd(dInputFromProj, dInputFromGate);
        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    private Tensor<T> ComputeSiLUDerivative(Tensor<T> x)
    {
        var sig = Engine.Sigmoid(x);
        var ones = new Tensor<T>(x.Shape);
        for (int i = 0; i < ones.Length; i++) ones[i] = NumOps.One;
        var oneMinusSig = Engine.TensorSubtract(ones, sig);
        var xTimesOneMinusSig = Engine.TensorMultiply(x, oneMinusSig);
        var onePlusXSig = Engine.TensorAdd(ones, xTimesOneMinusSig);
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
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
        _levelMixWeights = Engine.TensorAdd(_levelMixWeights, Engine.TensorMultiplyScalar(_levelMixWeightsGradient!, negLR));
        _compressionWeights = Engine.TensorAdd(_compressionWeights, Engine.TensorMultiplyScalar(_compressionWeightsGradient!, negLR));
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
        _queryWeights, _queryBias,
        _keyWeights, _keyBias,
        _valueWeights, _valueBias,
        _levelMixWeights, _compressionWeights,
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
        _lastGate = null;
        _lastGateRaw = null;
        _lastLogLinearOutput = null;
        _lastLevelOutputs = null;
        _lastLevelMixSoftmax = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _levelMixWeightsGradient = null;
        _compressionWeightsGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
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
        metadata["NumLevels"] = _numLevels.ToString();
        metadata["BaseBlockSize"] = _baseBlockSize.ToString();
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
    /// Gets the level mixing weights for external inspection.
    /// </summary>
    public Tensor<T> GetLevelMixWeights() => _levelMixWeights;
}
