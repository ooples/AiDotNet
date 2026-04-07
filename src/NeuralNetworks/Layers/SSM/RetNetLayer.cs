using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the RetNet (Retentive Network) layer from Sun et al., 2023.
/// </summary>
/// <remarks>
/// <para>
/// RetNet introduces a multi-scale retention mechanism that replaces softmax attention with an
/// exponential decay approach, supporting parallel, recurrent, and chunkwise computation modes.
/// This implementation uses the parallel formulation for training:
/// <code>
///   Retention(X) = (Q K^T odot D) V
/// </code>
/// where D is a causal decay mask with D_ij = gamma^(i-j) for i &gt;= j and 0 otherwise.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections: Q = X W_Q, K = X W_K, V = X W_V
///   2. Split into heads, each with its own decay rate gamma_h
///   3. Build causal decay mask D per head: D[i,j] = gamma_h^(i-j)
///   4. Compute retention: (Q K^T odot D) V  (parallel mode)
///      or equivalently in recurrent mode:
///        s_t = gamma * s_{t-1} + k_t^T v_t
///        o_t = q_t s_t
///   5. Apply group normalization (LayerNorm per head)
///   6. Output gate: y = swish(X W_g + b_g) odot retention_output
///   7. Output projection: output = y W_o + b_o
/// </code>
/// </para>
/// <para>
/// The multi-scale retention is the key innovation: each head operates at a different time scale
/// via its own decay rate gamma_h. Heads with gamma close to 1.0 retain long-range context, while
/// heads with smaller gamma focus on local context. This naturally creates a multi-scale representation
/// without requiring positional encodings.
/// </para>
/// <para>
/// The decay rates are initialized using the formula from the paper:
/// <code>
///   gamma_h = 1 - 2^(-5 - h * (log2(1) - log2(numHeads)) / numHeads)
/// </code>
/// which spaces the decay rates logarithmically between approximately 0.97 and 0.9999, ensuring
/// that heads cover a wide range of temporal scales.
/// </para>
/// <para><b>For Beginners:</b> RetNet is designed to be a successor to the Transformer architecture.
///
/// Think of a Transformer's attention as: "For every word, look at ALL other words and decide what's
/// important." This is powerful but expensive (quadratic cost).
///
/// RetNet replaces this with a "retention" mechanism that works like a fading memory:
/// - Recent words are remembered clearly (high weight)
/// - Older words gradually fade away (exponential decay)
/// - Different heads "forget" at different speeds:
///   * Some heads have long memory (gamma close to 1.0) - they capture long-range patterns
///   * Some heads have short memory (gamma around 0.97) - they capture local patterns
///
/// The big advantage: RetNet can be computed three ways:
/// 1. Parallel mode (for training): process the whole sequence at once, like a Transformer
/// 2. Recurrent mode (for inference): process one token at a time, like an RNN - O(1) per step
/// 3. Chunkwise mode: a hybrid that balances speed and parallelism
///
/// This means RetNet trains like a Transformer but generates text like an RNN, getting the best
/// of both worlds.
/// </para>
/// <para>
/// <b>Reference:</b> Sun et al., "Retentive Network: A Successor to Transformer for Large Language Models", 2023.
/// https://arxiv.org/abs/2307.08621
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class RetNetLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _queryBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _keyBias;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _valueBias;

    // Per-head decay parameters (gammas): [numHeads]
    // These are stored as raw values in (0, 1), initialized per the paper's formula.
    private Tensor<T> _gammas;

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

    // Group normalization parameters per head: scale [numHeads, headDim] and bias [numHeads, headDim]
    private Tensor<T> _groupNormScale;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _groupNormBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastRetentionOutput;
    private Tensor<T>? _lastNormedRetention;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastDecayMasks;
    private Tensor<T>? _lastRetentionScores;
    private Tensor<T>? _lastGroupNormMean;
    private Tensor<T>? _lastGroupNormVar;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _gammasGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;
    private Tensor<T>? _groupNormScaleGradient;
    private Tensor<T>? _groupNormBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension (d_model).
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of retention heads.
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
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _gammas.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length +
        _groupNormScale.Length + _groupNormBias.Length;

    /// <summary>
    /// Creates a new RetNet (Retentive Network) layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length for building the causal decay mask.
    /// <para><b>For Beginners:</b> The maximum number of tokens the layer can process at once.
    /// Longer sequences require more memory but capture longer context.</para>
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of the vector representing each token. Larger values
    /// allow the model to store more information per token but increase compute cost.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of retention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head operates at a different "forgetting speed."
    /// More heads means more diversity in how the model looks at temporal patterns.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public RetNetLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);
        _gammas = new Tensor<T>([numHeads]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);
        _groupNormScale = new Tensor<T>([numHeads, _headDimension]);
        _groupNormBias = new Tensor<T>([numHeads, _headDimension]);

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
        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Initialize gammas per the paper's formula:
        // gamma_h = 1 - 2^(-5 - h * range / numHeads)
        // This spaces decay rates logarithmically so heads cover different time scales.
        for (int h = 0; h < _numHeads; h++)
        {
            double exponent = -5.0 - (h * 8.0 / _numHeads);
            double gamma = 1.0 - Math.Pow(2.0, exponent);
            _gammas[h] = NumOps.FromDouble(gamma);
        }

        // Group normalization: scale = 1, bias = 0
        for (int i = 0; i < _groupNormScale.Length; i++)
            _groupNormScale[i] = NumOps.One;
        _groupNormBias.Fill(NumOps.Zero);

        RegisterTrainableParameter(_gammas, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_groupNormScale, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Xavier/Glorot initialization for 2D weight tensors.
    /// </summary>
    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape.ToArray();

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

        // Step 2: Compute gate (for gated multi-scale retention)
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastGateRaw = gateRaw;
        _lastGate = gate;

        // Step 3: Multi-scale retention with causal decay mask (parallel mode)
        var retentionOutput = MultiScaleRetentionForward(q, k, v, batchSize, seqLen);
        _lastRetentionOutput = retentionOutput;

        // Step 4: Group normalization (per-head LayerNorm)
        var normedRetention = GroupNormForward(retentionOutput, batchSize, seqLen);
        _lastNormedRetention = normedRetention;

        // Step 5: Gated output = gate * normed_retention
        var gatedOutput = Engine.TensorMultiply(gate, normedRetention);

        // Step 6: Output projection
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
    /// Computes multi-scale retention in parallel mode: Retention_h(X) = (Q_h K_h^T odot D_h) V_h
    /// where D_h[i,j] = gamma_h^(i-j) for i &gt;= j, 0 otherwise.
    /// </summary>
    private Tensor<T> MultiScaleRetentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

        // Pre-compute decay masks: D_h[i,j] = gamma_h^(i-j) for i >= j, 0 otherwise
        // Shape: [numHeads, seqLen, seqLen]
        var decayMasks = new Tensor<T>(new[] { _numHeads, seqLen, seqLen });
        for (int h = 0; h < _numHeads; h++)
        {
            T gamma = _gammas[h];
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    // D[i,j] = gamma^(i-j)
                    int diff = i - j;
                    T decayVal = NumOps.One;
                    for (int p = 0; p < diff; p++)
                        decayVal = NumOps.Multiply(decayVal, gamma);
                    decayMasks[new[] { h, i, j }] = decayVal;
                }
                // j > i entries are already zero from initialization
            }
        }
        _lastDecayMasks = decayMasks;

        // Store retention scores for backward pass: [batch, numHeads, seqLen, seqLen]
        var retentionScores = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, seqLen, seqLen });
        _lastRetentionScores = retentionScores;

        // For each head, compute retention
        T headScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int h = 0; h < _numHeads; h++)
            {
                int dimStart = h * _headDimension;

                // Compute Q_h K_h^T: [seqLen, seqLen]
                for (int i = 0; i < seqLen; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            T qVal = q[new[] { bi, i, flatD }];
                            T kVal = k[new[] { bi, j, flatD }];
                            dot = NumOps.Add(dot, NumOps.Multiply(qVal, kVal));
                        }
                        // Scale by 1/sqrt(d_k) and apply causal decay mask
                        T scaledDot = NumOps.Multiply(dot, headScale);
                        T decayVal = decayMasks[new[] { h, i, j }];
                        T score = NumOps.Multiply(scaledDot, decayVal);
                        retentionScores[new[] { bi, h, i, j }] = score;
                    }
                }

                // Multiply retention scores by V_h: output[i] = sum_j score[i,j] * V[j]
                for (int i = 0; i < seqLen; i++)
                {
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T sum = NumOps.Zero;
                        for (int j = 0; j <= i; j++)
                        {
                            T score = retentionScores[new[] { bi, h, i, j }];
                            T vVal = v[new[] { bi, j, flatD }];
                            sum = NumOps.Add(sum, NumOps.Multiply(score, vVal));
                        }
                        output[new[] { bi, i, flatD }] = sum;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Applies group normalization (per-head LayerNorm) to the retention output.
    /// Each head's output is independently normalized, then scaled and shifted.
    /// </summary>
    private Tensor<T> GroupNormForward(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(input.Shape.ToArray());
        T eps = NumOps.FromDouble(1e-6);

        // Store mean and variance for backward pass: [batchSize, seqLen, numHeads]
        var meanTensor = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads });
        var varTensor = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int dimStart = h * _headDimension;
                    T invHD = NumOps.FromDouble(1.0 / _headDimension);

                    // Compute mean
                    T mean = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        mean = NumOps.Add(mean, input[new[] { bi, t, flatD }]);
                    }
                    mean = NumOps.Multiply(mean, invHD);

                    // Compute variance
                    T variance = NumOps.Zero;
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T diff = NumOps.Subtract(input[new[] { bi, t, flatD }], mean);
                        variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                    }
                    variance = NumOps.Multiply(variance, invHD);

                    meanTensor[new[] { bi, t, h }] = mean;
                    varTensor[new[] { bi, t, h }] = variance;

                    // Normalize, scale, and shift
                    T invStd = NumOps.FromDouble(1.0 / Math.Sqrt(
                        NumOps.ToDouble(NumOps.Add(variance, eps))));
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T normalized = NumOps.Multiply(
                            NumOps.Subtract(input[new[] { bi, t, flatD }], mean),
                            invStd);
                        T scale = _groupNormScale[new[] { h, d }];
                        T bias = _groupNormBias[new[] { h, d }];
                        output[new[] { bi, t, flatD }] = NumOps.Add(
                            NumOps.Multiply(normalized, scale), bias);
                    }
                }
            }
        }

        _lastGroupNormMean = meanTensor;
        _lastGroupNormVar = varTensor;
        return output;
    }

    /// <summary>
    /// Backward pass through group normalization.
    /// </summary>
    private Tensor<T> GroupNormBackward(
        Tensor<T> dNormed, Tensor<T> retentionInput,
        int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(retentionInput.Shape.ToArray());
        T eps = NumOps.FromDouble(1e-6);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    int dimStart = h * _headDimension;
                    T mean = (_lastGroupNormMean ?? throw new InvalidOperationException("_lastGroupNormMean has not been initialized."))[new[] { bi, t, h }];
                    T variance = (_lastGroupNormVar ?? throw new InvalidOperationException("_lastGroupNormVar has not been initialized."))[new[] { bi, t, h }];
                    T invStd = NumOps.FromDouble(1.0 / Math.Sqrt(
                        NumOps.ToDouble(NumOps.Add(variance, eps))));
                    T invHD = NumOps.FromDouble(1.0 / _headDimension);

                    // Accumulate group norm parameter gradients and compute intermediate sums
                    T sumDy = NumOps.Zero;
                    T sumDyXhat = NumOps.Zero;

                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xHat = NumOps.Multiply(
                            NumOps.Subtract(retentionInput[new[] { bi, t, flatD }], mean),
                            invStd);
                        T dy = dNormed[new[] { bi, t, flatD }];
                        T scale = _groupNormScale[new[] { h, d }];

                        // dScale += dy * xHat, dBias += dy
                        (_groupNormScaleGradient ?? throw new InvalidOperationException("_groupNormScaleGradient has not been initialized."))[new[] { h, d }] = NumOps.Add(
                            _groupNormScaleGradient[new[] { h, d }],
                            NumOps.Multiply(dy, xHat));
                        (_groupNormBiasGradient ?? throw new InvalidOperationException("_groupNormBiasGradient has not been initialized."))[new[] { h, d }] = NumOps.Add(
                            _groupNormBiasGradient[new[] { h, d }], dy);

                        // For input gradient, dy is scaled by the norm scale
                        T dyScaled = NumOps.Multiply(dy, scale);
                        sumDy = NumOps.Add(sumDy, dyScaled);
                        sumDyXhat = NumOps.Add(sumDyXhat, NumOps.Multiply(dyScaled, xHat));
                    }

                    // Compute input gradient for each dimension in this head
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T xHat = NumOps.Multiply(
                            NumOps.Subtract(retentionInput[new[] { bi, t, flatD }], mean),
                            invStd);
                        T dy = dNormed[new[] { bi, t, flatD }];
                        T scale = _groupNormScale[new[] { h, d }];
                        T dyScaled = NumOps.Multiply(dy, scale);

                        // dInput = invStd * (dyScaled - invHD * (sumDy + xHat * sumDyXhat))
                        T correction = NumOps.Multiply(invHD,
                            NumOps.Add(sumDy, NumOps.Multiply(xHat, sumDyXhat)));
                        dInput[new[] { bi, t, flatD }] = NumOps.Multiply(
                            invStd, NumOps.Subtract(dyScaled, correction));
                    }
                }
            }
        }

        return dInput;
    }

    /// <summary>
    /// Computes the derivative of the SiLU (Swish) activation function.
    /// SiLU(x) = x * sigmoid(x), SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    /// </summary>
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
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
        _gammas = Engine.TensorAdd(_gammas, Engine.TensorMultiplyScalar(_gammasGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
        _groupNormScale = Engine.TensorAdd(_groupNormScale, Engine.TensorMultiplyScalar(_groupNormScaleGradient!, negLR));
        _groupNormBias = Engine.TensorAdd(_groupNormBias, Engine.TensorMultiplyScalar(_groupNormBiasGradient!, negLR));

        // Clamp gammas to (0, 1) after update to maintain valid decay rates
        for (int h = 0; h < _numHeads; h++)
        {
            double gVal = NumOps.ToDouble(_gammas[h]);
            if (gVal <= 0.0) _gammas[h] = NumOps.FromDouble(1e-6);
            else if (gVal >= 1.0) _gammas[h] = NumOps.FromDouble(1.0 - 1e-6);
        }

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
        _gammas,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias,
        _groupNormScale, _groupNormBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_queryBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_gammasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_groupNormScaleGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_groupNormBiasGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _queryBiasGradient = null; _keyWeightsGradient = null; _keyBiasGradient = null; _valueWeightsGradient = null; _valueBiasGradient = null; _gammasGradient = null; _groupNormScaleGradient = null; _groupNormBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastRetentionOutput = null;
        _lastNormedRetention = null;
        _lastGateRaw = null;
        _lastGate = null;
        _lastDecayMasks = null;
        _lastRetentionScores = null;
        _lastGroupNormMean = null;
        _lastGroupNormVar = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _gammasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
        _groupNormScaleGradient = null;
        _groupNormBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();

        // Include the decay rates for inspection
        var gammaValues = new string[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            gammaValues[h] = NumOps.ToDouble(_gammas[h]).ToString("F6");
        metadata["DecayRates"] = string.Join(", ", gammaValues);

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
    /// Each head has its own decay rate gamma_h in (0, 1). Higher values mean slower decay
    /// (longer memory), while lower values mean faster decay (shorter memory).
    /// </para>
    /// <para><b>For Beginners:</b> These numbers control how quickly each attention head "forgets."
    /// A value of 0.99 means the head remembers information for a long time, while 0.90 means
    /// it focuses mainly on recent tokens.</para>
    /// </remarks>
    public Tensor<T> GetDecayRates() => _gammas;
}
