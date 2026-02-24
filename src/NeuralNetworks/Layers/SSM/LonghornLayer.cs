using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Longhorn layer from "Longhorn: State Space Models are Amortized Online Learners" (Liu et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// Longhorn reinterprets state space model state updates as amortized online learning. Rather than treating
/// the recurrent state as an opaque hidden state, Longhorn views it as a weight matrix of an online learner
/// that is continuously updated to predict values from keys. The decay/forget gate acts as a data-dependent
/// learning rate controlling how quickly the "learner" adapts to new observations.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. Compute alpha (learning rate / update gate) per head via learned projection + sigmoid
///   3. State update as exponential moving average of outer products:
///      S_t = (1 - alpha_t) * S_{t-1} + alpha_t * v_t * k_t^T
///      This is the key equation: alpha_t controls the interpolation between
///      remembering the old state and writing the new key-value association.
///   4. Output: o_t = S_t * q_t (query the online learner's weight matrix)
///   5. Output projection with optional group normalization
/// </code>
/// </para>
/// <para>
/// The critical insight is that this state update rule is equivalent to an online learning rule:
/// - The state matrix S acts as the "model weights" of a linear predictor
/// - Each (k_t, v_t) pair is a new "training example"
/// - alpha_t is the "learning rate" that controls how much to update
/// - (1 - alpha_t) is the "retention rate" that controls how much to remember
/// - Querying S with q_t is equivalent to running inference on the online learner
/// </para>
/// <para>
/// Unlike GatedDeltaNet which uses a delta rule (error-corrective) update, Longhorn uses a simpler
/// exponential moving average update. This means it does not check what the state already knows before
/// writing -- it simply blends the old state with the new outer product. Despite this simplicity, the
/// online learning perspective enables principled initialization and understanding of the model's behavior.
/// </para>
/// <para><b>For Beginners:</b> Longhorn is a sequence model that maintains an internal "memory" which
/// works like a student learning from a stream of examples.
///
/// Imagine a student studying flashcards one at a time:
/// - Each flashcard has a "key" (the question) and a "value" (the answer)
/// - The student maintains a mental model (the state matrix S) that maps keys to values
/// - For each new flashcard, the student decides how much to "learn" from it (alpha)
///   - High alpha: "This is important, update my understanding significantly"
///   - Low alpha: "I already know this well, barely change my understanding"
/// - To answer a question (query), the student applies their mental model
///
/// The key difference from standard attention:
/// - Standard attention: Re-reads ALL flashcards every time (O(n^2) cost)
/// - Longhorn: Maintains a compressed summary that gets updated incrementally (O(n) cost)
///
/// This makes Longhorn much more efficient for long sequences while still capturing important patterns.
/// </para>
/// <para>
/// <b>Reference:</b> Liu et al., "Longhorn: State Space Models are Amortized Online Learners", 2024.
/// https://arxiv.org/abs/2407.14207
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class LonghornLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Alpha (learning rate / update gate) projection: [modelDim, numHeads]
    private Tensor<T> _alphaWeights;
    private Tensor<T> _alphaBias;

    // Group normalization parameters per head: [numHeads, headDim]
    private Tensor<T> _groupNormGamma;
    private Tensor<T> _groupNormBeta;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastAlpha;
    private Tensor<T>? _lastStates;
    private Tensor<T>? _lastRecurrenceOutput;
    private Tensor<T>? _lastNormedOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _alphaWeightsGradient;
    private Tensor<T>? _alphaBiasGradient;
    private Tensor<T>? _groupNormGammaGradient;
    private Tensor<T>? _groupNormBetaGradient;
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
    /// Gets the dimension per head (modelDimension / numHeads).
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the total number of trainable parameters in this layer.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _alphaWeights.Length + _alphaBias.Length +
        _groupNormGamma.Length + _groupNormBeta.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Longhorn layer.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length this layer will process.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of each token's representation vector.
    /// Larger values capture more information but require more computation.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own "online learner" state matrix.
    /// Must evenly divide modelDimension. More heads let the model learn different types of
    /// key-value associations simultaneously.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public LonghornLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
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

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);
        _alphaWeights = new Tensor<T>([modelDimension, numHeads]);
        _alphaBias = new Tensor<T>([numHeads]);
        _groupNormGamma = new Tensor<T>([numHeads, _headDimension]);
        _groupNormBeta = new Tensor<T>([numHeads, _headDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    /// <summary>
    /// Initializes all trainable parameters using Xavier/Glorot initialization for weight matrices
    /// and appropriate constants for biases.
    /// </summary>
    private void InitializeParameters()
    {
        InitializeTensor2D(_queryWeights);
        _queryBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyWeights);
        _keyBias.Fill(NumOps.Zero);
        InitializeTensor2D(_valueWeights);
        _valueBias.Fill(NumOps.Zero);
        InitializeTensor2D(_alphaWeights);
        // Alpha bias ~ -2 so sigmoid(-2) ~ 0.12 -> slow initial learning rate (strong retention)
        // This is the "online learning rate" -- starting conservative is better for stability
        for (int i = 0; i < _alphaBias.Length; i++)
            _alphaBias[i] = NumOps.FromDouble(-2.0);
        // Group norm: gamma = 1, beta = 0 (identity initialization)
        for (int i = 0; i < _groupNormGamma.Length; i++)
            _groupNormGamma[i] = NumOps.One;
        _groupNormBeta.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    /// <summary>
    /// Applies Xavier/Glorot uniform initialization to a 2D weight tensor.
    /// </summary>
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

        var qFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _queryWeights),
            _queryBias.Reshape(1, _modelDimension));
        var q = qFlat.Reshape(batchSize, seqLen, _modelDimension);

        var kFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _keyWeights),
            _keyBias.Reshape(1, _modelDimension));
        var k = kFlat.Reshape(batchSize, seqLen, _modelDimension);

        var vFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension));
        var v = vFlat.Reshape(batchSize, seqLen, _modelDimension);

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Alpha (learning rate / update gate) via sigmoid
        var alphaRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _alphaWeights),
            _alphaBias.Reshape(1, _numHeads)).Reshape(batchSize, seqLen, _numHeads);
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastAlpha = alpha;

        // Step 3: Online learning recurrence per head
        var recurrenceOutput = OnlineLearnerForward(q, k, v, alpha, batchSize, seqLen);
        _lastRecurrenceOutput = recurrenceOutput;

        // Step 4: Group normalization per head
        var normedOutput = ApplyGroupNorm(recurrenceOutput, batchSize, seqLen);
        _lastNormedOutput = normedOutput;

        // Step 5: Output projection
        var normedFlat = normedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorMatMul(normedFlat, _outputProjectionWeights);
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
    /// Longhorn online learner forward: exponential moving average of outer products.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For each timestep t and head h, the recurrence is:
    /// <code>
    ///   S_t = (1 - alpha_t) * S_{t-1} + alpha_t * v_t * k_t^T
    ///   o_t = S_t * q_t
    /// </code>
    /// The alpha_t gate acts as the "learning rate" of the online learner. When alpha is high,
    /// the state aggressively adopts the new observation. When alpha is low, the state retains
    /// its existing knowledge and barely incorporates the new data.
    /// </para>
    /// <para>
    /// This is a convex combination: at each step, the state is an interpolation between
    /// the previous state and the new rank-1 outer product v*k^T. This ensures the state
    /// always has bounded spectral norm (assuming normalized inputs), preventing the
    /// unbounded growth issue of pure additive linear attention.
    /// </para>
    /// </remarks>
    private Tensor<T> OnlineLearnerForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> alpha,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State matrix per head: [batch, numHeads, headDim, headDim]
        var state = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        // Save all states for backward pass: [batch, seqLen+1, numHeads, headDim, headDim]
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
                    T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaVal);

                    // State update: S_t = (1 - alpha_t) * S_{t-1} + alpha_t * v_t * k_t^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = v[new[] { bi, t, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(k[new[] { bi, t, flatKi }], keyScale);

                            T prevS = state[new[] { bi, hi, di, ki }];
                            T outerProduct = NumOps.Multiply(vVal, kVal);
                            T newS = NumOps.Add(
                                NumOps.Multiply(oneMinusAlpha, prevS),
                                NumOps.Multiply(alphaVal, outerProduct));
                            state[new[] { bi, hi, di, ki }] = newS;
                        }
                    }

                    // Output: o_t = S_t * q_t
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

            // Save state snapshot for backward pass
            for (int bi = 0; bi < batchSize; bi++)
                for (int hi2 = 0; hi2 < _numHeads; hi2++)
                    for (int di = 0; di < _headDimension; di++)
                        for (int ki = 0; ki < _headDimension; ki++)
                            allStates[new[] { bi, t + 1, hi2, di, ki }] = state[new[] { bi, hi2, di, ki }];
        }

        _lastStates = allStates;
        return output;
    }

    /// <summary>
    /// Applies group normalization to the recurrence output, normalizing within each head independently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Group normalization is applied per-head to stabilize the output of the online learning recurrence.
    /// Each head's output is normalized to have zero mean and unit variance, then scaled and shifted
    /// by learnable gamma and beta parameters.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyGroupNorm(Tensor<T> input, int batchSize, int seqLen)
    {
        var output = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Compute mean for this head
                    T mean = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        mean = NumOps.Add(mean, input[new[] { bi, t, flatDi }]);
                    }
                    mean = NumOps.Divide(mean, NumOps.FromDouble(_headDimension));

                    // Compute variance for this head
                    T variance = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T diff = NumOps.Subtract(input[new[] { bi, t, flatDi }], mean);
                        variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                    }
                    variance = NumOps.Divide(variance, NumOps.FromDouble(_headDimension));

                    // Normalize and apply scale/shift
                    T invStd = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, epsilon)));
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T normalized = NumOps.Multiply(
                            NumOps.Subtract(input[new[] { bi, t, flatDi }], mean),
                            invStd);
                        T gamma = _groupNormGamma[new[] { hi, di }];
                        T beta = _groupNormBeta[new[] { hi, di }];
                        output[new[] { bi, t, flatDi }] = NumOps.Add(
                            NumOps.Multiply(gamma, normalized), beta);
                    }
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastAlpha == null || _lastRecurrenceOutput == null ||
            _lastNormedOutput == null || _lastStates == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize all gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _alphaWeightsGradient = new Tensor<T>([_modelDimension, _numHeads]);
        _alphaBiasGradient = new Tensor<T>([_numHeads]);
        _groupNormGammaGradient = new Tensor<T>([_numHeads, _headDimension]);
        _groupNormBetaGradient = new Tensor<T>([_numHeads, _headDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 5 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var normedFlat = _lastNormedOutput.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(
            normedFlat.Transpose([1, 0]), gradFlat);

        var dNormed = Engine.TensorMatMul(
            gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 4 backward: group normalization
        var dRecurrence = GroupNormBackward(dNormed, _lastRecurrenceOutput, batchSize, seqLen);

        // Step 3 backward: online learning recurrence (reverse time)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dAlpha = new Tensor<T>(new[] { batchSize, seqLen, _numHeads });

        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        var dState = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    T alphaVal = _lastAlpha[new[] { bi, t, hi }];
                    T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaVal);

                    // --- Backward through: o_t = S_t * q_t ---
                    // dS_t += dO outer q_t, dQ += S_t^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dRecurrence[new[] { bi, t, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T qVal = _lastQuery[new[] { bi, t, flatKi }];
                            T sVal = _lastStates[new[] { bi, t + 1, hi, di, ki }];

                            dState[new[] { bi, hi, di, ki }] = NumOps.Add(
                                dState[new[] { bi, hi, di, ki }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatKi }] = NumOps.Add(
                                dQ[new[] { bi, t, flatKi }],
                                NumOps.Multiply(dO, sVal));
                        }
                    }

                    // --- Backward through: S_t = (1 - alpha_t) * S_{t-1} + alpha_t * v_t * k_t^T ---
                    // dAlpha_t += sum_{di,ki} dS_t[di,ki] * (-S_{t-1}[di,ki] + v[di]*k[ki])
                    // dV[di] += alpha_t * sum_ki(dS_t[di,ki] * k[ki])
                    // dK[ki] += alpha_t * sum_di(dS_t[di,ki] * v[di])
                    // dS_{t-1}[di,ki] = (1 - alpha_t) * dS_t[di,ki]

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T vVal = _lastValue[new[] { bi, t, flatDi }];

                        for (int ki = 0; ki < _headDimension; ki++)
                        {
                            int flatKi = dimStart + ki;
                            T kVal = NumOps.Multiply(_lastKey[new[] { bi, t, flatKi }], keyScale);
                            T dS = dState[new[] { bi, hi, di, ki }];
                            T sPrev = _lastStates[new[] { bi, t, hi, di, ki }];

                            // dAlpha: derivative of interpolation w.r.t. alpha
                            // d/d(alpha) [(1-alpha)*S_prev + alpha*v*k^T] = -S_prev + v*k^T
                            T outerProduct = NumOps.Multiply(vVal, kVal);
                            T alphaGrad = NumOps.Multiply(dS,
                                NumOps.Subtract(outerProduct, sPrev));
                            dAlpha[new[] { bi, t, hi }] = NumOps.Add(
                                dAlpha[new[] { bi, t, hi }], alphaGrad);

                            // dV[di] += alpha_t * dS[di,ki] * k[ki]
                            dV[new[] { bi, t, flatDi }] = NumOps.Add(
                                dV[new[] { bi, t, flatDi }],
                                NumOps.Multiply(NumOps.Multiply(alphaVal, dS), kVal));

                            // dK[ki] += alpha_t * dS[di,ki] * v[di]
                            dK[new[] { bi, t, flatKi }] = NumOps.Add(
                                dK[new[] { bi, t, flatKi }],
                                NumOps.Multiply(NumOps.Multiply(alphaVal, dS), vVal));

                            // Propagate dState to previous timestep: dS_{t-1} = (1-alpha_t) * dS_t
                            dState[new[] { bi, hi, di, ki }] = NumOps.Multiply(oneMinusAlpha, dS);
                        }
                    }
                }
            }
        }

        // Step 2 backward: alpha through sigmoid derivative
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        var alphaSigDeriv = Engine.TensorMultiply(_lastAlpha,
            Engine.TensorSubtract(CreateOnesLike(_lastAlpha), _lastAlpha));
        var dAlphaRaw = Engine.TensorMultiply(dAlpha, alphaSigDeriv);

        // Step 1 backward: projection weight gradients
        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);

        var dAlphaFlat = dAlphaRaw.Reshape(batchSize * seqLen, _numHeads);
        _alphaWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dAlphaFlat);
        _alphaBiasGradient = Engine.ReduceSum(dAlphaRaw, new int[] { 0, 1 });

        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]),
            Engine.TensorMultiplyScalar(dKFlat, keyScale));
        _keyBiasGradient = Engine.ReduceSum(
            Engine.TensorMultiplyScalar(dK, keyScale), new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Input gradient: sum contributions from all projection paths
        var dInput = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(Engine.TensorMultiplyScalar(dKFlat, keyScale),
                _keyWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dAlphaFlat, _alphaWeights.Transpose([1, 0])));

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Backward pass through group normalization, computing gradients for gamma, beta,
    /// and propagating gradients to the input.
    /// </summary>
    private Tensor<T> GroupNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);
        T headDimT = NumOps.FromDouble(_headDimension);
        var gammaGrad = _groupNormGammaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");
        var betaGrad = _groupNormBetaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Recompute forward statistics
                    T mean = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        mean = NumOps.Add(mean, input[new[] { bi, t, flatDi }]);
                    }
                    mean = NumOps.Divide(mean, headDimT);

                    T variance = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T diff = NumOps.Subtract(input[new[] { bi, t, flatDi }], mean);
                        variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                    }
                    variance = NumOps.Divide(variance, headDimT);
                    T invStd = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, epsilon)));

                    // Compute normalized values and accumulate gamma/beta gradients
                    var xHat = new T[_headDimension];
                    T dGammaSum = NumOps.Zero;
                    T dBetaSum = NumOps.Zero;
                    T dxHatSum = NumOps.Zero;
                    T dxHatXHatSum = NumOps.Zero;

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        xHat[di] = NumOps.Multiply(
                            NumOps.Subtract(input[new[] { bi, t, flatDi }], mean),
                            invStd);

                        T gamma = _groupNormGamma[new[] { hi, di }];
                        T dOut = dOutput[new[] { bi, t, flatDi }];

                        // Accumulate gamma and beta gradients
                        gammaGrad[new[] { hi, di }] = NumOps.Add(
                            gammaGrad[new[] { hi, di }],
                            NumOps.Multiply(dOut, xHat[di]));
                        betaGrad[new[] { hi, di }] = NumOps.Add(
                            betaGrad[new[] { hi, di }], dOut);

                        // dxHat = dOutput * gamma
                        T dxHat = NumOps.Multiply(dOut, gamma);
                        dxHatSum = NumOps.Add(dxHatSum, dxHat);
                        dxHatXHatSum = NumOps.Add(dxHatXHatSum,
                            NumOps.Multiply(dxHat, xHat[di]));
                    }

                    // Compute input gradient using the group norm backward formula
                    // dInput = invStd * (dxHat - mean(dxHat) - xHat * mean(dxHat * xHat)) / 1
                    T meanDxHat = NumOps.Divide(dxHatSum, headDimT);
                    T meanDxHatXHat = NumOps.Divide(dxHatXHatSum, headDimT);

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T gamma = _groupNormGamma[new[] { hi, di }];
                        T dOut = dOutput[new[] { bi, t, flatDi }];
                        T dxHat = NumOps.Multiply(dOut, gamma);

                        T gradVal = NumOps.Multiply(invStd,
                            NumOps.Subtract(dxHat,
                                NumOps.Add(meanDxHat,
                                    NumOps.Multiply(xHat[di], meanDxHatXHat))));
                        dInput[new[] { bi, t, flatDi }] = gradVal;
                    }
                }
            }
        }

        return dInput;
    }

    /// <summary>
    /// Creates a tensor of ones with the same shape as the template tensor.
    /// </summary>
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
        _queryBias = Engine.TensorAdd(_queryBias, Engine.TensorMultiplyScalar(_queryBiasGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _keyBias = Engine.TensorAdd(_keyBias, Engine.TensorMultiplyScalar(_keyBiasGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _valueBias = Engine.TensorAdd(_valueBias, Engine.TensorMultiplyScalar(_valueBiasGradient!, negLR));
        _alphaWeights = Engine.TensorAdd(_alphaWeights, Engine.TensorMultiplyScalar(_alphaWeightsGradient!, negLR));
        _alphaBias = Engine.TensorAdd(_alphaBias, Engine.TensorMultiplyScalar(_alphaBiasGradient!, negLR));
        _groupNormGamma = Engine.TensorAdd(_groupNormGamma, Engine.TensorMultiplyScalar(_groupNormGammaGradient!, negLR));
        _groupNormBeta = Engine.TensorAdd(_groupNormBeta, Engine.TensorMultiplyScalar(_groupNormBetaGradient!, negLR));
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

    /// <summary>
    /// Returns all trainable parameter tensors in a consistent order for serialization.
    /// </summary>
    private Tensor<T>[] GetAllTensors() =>
    [
        _queryWeights, _queryBias,
        _keyWeights, _keyBias,
        _valueWeights, _valueBias,
        _alphaWeights, _alphaBias,
        _groupNormGamma, _groupNormBeta,
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
        _lastAlpha = null;
        _lastStates = null;
        _lastRecurrenceOutput = null;
        _lastNormedOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _alphaWeightsGradient = null;
        _alphaBiasGradient = null;
        _groupNormGammaGradient = null;
        _groupNormBetaGradient = null;
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
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection or analysis.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the query weights for external inspection or analysis.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the alpha (learning rate gate) weights for external inspection or analysis.
    /// </summary>
    public Tensor<T> GetAlphaWeights() => _alphaWeights;
}
