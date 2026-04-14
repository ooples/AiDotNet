using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the TTT (Test-Time Training) layer from Sun et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// TTT replaces the fixed-size hidden state of traditional RNNs with a more expressive hidden state:
/// the WEIGHTS of a small inner model. At each timestep the layer performs a gradient-based learning
/// step on those weights, effectively training the inner model on the fly while processing the sequence.
/// </para>
/// <para>
/// This file implements the <b>TTT-Linear</b> variant where the inner model is a single linear map W.
/// The recurrence at each timestep t is:
/// <code>
///   1. Project input x_t into key (x_k), value (x_v), and query (q_t)
///   2. Compute self-supervised loss: L(W, x_t) = ||W * x_k - x_v||^2
///   3. Gradient of loss w.r.t. W: dL/dW = 2 * (W * x_k - x_v) * x_k^T
///   4. Update inner weights: W_t = W_{t-1} - eta * dL/dW
///      i.e. W_t = W_{t-1} - eta * 2 * (W_{t-1} * x_k - x_v) * x_k^T
///   5. Produce output: o_t = W_t * q_t
///   6. Apply output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is that the hidden state (W) grows in expressivity with the size of the inner model.
/// Unlike a fixed-size RNN state vector, W can represent arbitrary linear relationships and is updated
/// using a principled gradient descent rule (the delta rule). This makes TTT a bridge between
/// recurrent models and in-context learning: the model literally "learns at test time."
/// </para>
/// <para>
/// Multi-head operation: each head maintains its own inner weight matrix W_h of size [headDim, headDim].
/// This allows different heads to learn different relationships from the sequence, similar to multi-head
/// attention. The heads operate independently and their outputs are concatenated.
/// </para>
/// <para><b>For Beginners:</b> Traditional RNNs store their memory as a fixed-size vector (like a notepad
/// with limited space). TTT stores memory as a small neural network's weights (like having a student
/// who learns from each example).
///
/// At each step in the sequence:
/// - The "student" (inner model W) tries to predict the value from the key
/// - It computes how wrong it was (the loss)
/// - It takes a learning step to improve (gradient descent)
/// - Then it answers a query using its updated knowledge
///
/// This means TTT can adapt to the specific patterns in each sequence, much like how you get better
/// at a task the more examples you see. The inner learning rate (eta) controls how quickly the
/// student adapts -- too fast and it forgets old information, too slow and it cannot keep up.
///
/// TTT-Linear is competitive with Transformers and Mamba on language modeling benchmarks while
/// maintaining linear O(n) complexity in sequence length.
/// </para>
/// <para>
/// <b>Reference:</b> Sun et al., "Learning to (Learn at Test Time): RNNs with Expressive Hidden States", 2024.
/// https://arxiv.org/abs/2407.04620
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class TTTLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly T _innerLearningRate;

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

    // Inner model initial weights per head: [numHeads, headDim, headDim]
    // This is W_0, the starting point for inner model weights at the beginning of each sequence
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _innerWeightsInit;

    // Learnable inner learning rate scale per head: [numHeads]
    private Tensor<T> _etaScale;

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

    // Layer normalization parameters for the inner model input (pre-norm)
    private Tensor<T> _lnGamma; // [modelDim]
    private Tensor<T> _lnBeta;  // [modelDim]

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastNormalized;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastTTTOutput;
    private Tensor<T>? _lastInnerWeights; // All W_t snapshots: [batch, seqLen+1, numHeads, headDim, headDim]
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _innerWeightsInitGradient;
    private Tensor<T>? _etaScaleGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;
    private Tensor<T>? _lnGammaGradient;
    private Tensor<T>? _lnBetaGradient;

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
    /// Gets the inner learning rate used for test-time training updates.
    /// </summary>
    public T InnerLearningRate => _innerLearningRate;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _innerWeightsInit.Length + _etaScale.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length +
        _lnGamma.Length + _lnBeta.Length;

    /// <summary>
    /// Creates a new TTT (Test-Time Training) layer implementing the TTT-Linear variant.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each input/output token embedding.
    /// Larger values can capture more information but use more memory.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own inner weight matrix W.
    /// Multiple heads let the model learn different types of relationships simultaneously.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="innerLearningRate">
    /// Learning rate for the inner model gradient updates (eta). Default: 0.01.
    /// <para><b>For Beginners:</b> Controls how fast the inner "student" model adapts at each step.
    /// Too high: the model overwrites old knowledge too quickly (catastrophic forgetting).
    /// Too low: the model cannot adapt to new patterns in the sequence.
    /// The layer also learns a per-head scale factor for this rate.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public TTTLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double innerLearningRate = 0.01,
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
        if (innerLearningRate <= 0)
            throw new ArgumentException($"Inner learning rate ({innerLearningRate}) must be positive.", nameof(innerLearningRate));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _innerLearningRate = NumOps.FromDouble(innerLearningRate);

        // Q, K, V projections
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);

        // Inner model initial weights: one W_0 per head, each [headDim, headDim]
        _innerWeightsInit = new Tensor<T>([numHeads, _headDimension, _headDimension]);

        // Per-head learnable learning rate scale (multiplied by base eta)
        _etaScale = new Tensor<T>([numHeads]);

        // Output gate
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);

        // Output projection
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        // Layer norm
        _lnGamma = new Tensor<T>([modelDimension]);
        _lnBeta = new Tensor<T>([modelDimension]);

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

        // Initialize inner weights W_0 close to zero (small identity-like initialization)
        // so the inner model starts near zero and learns from the sequence
        T initScale = NumOps.FromDouble(0.01);
        for (int h = 0; h < _numHeads; h++)
        {
            for (int i = 0; i < _headDimension; i++)
            {
                for (int j = 0; j < _headDimension; j++)
                {
                    T val = i == j
                        ? initScale
                        : NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5),
                            NumOps.FromDouble(0.001));
                    _innerWeightsInit[new[] { h, i, j }] = val;
                }
            }
        }

        // Eta scale initialized to 1.0 so effective lr = innerLearningRate * 1.0
        for (int h = 0; h < _numHeads; h++)
            _etaScale[h] = NumOps.One;

        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

        // Layer norm: gamma=1, beta=0
        for (int i = 0; i < _modelDimension; i++)
        {
            _lnGamma[i] = NumOps.One;
            _lnBeta[i] = NumOps.Zero;
        }
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
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

        // Step 1: Layer normalization
        var normalized = LayerNormForward(input3D, batchSize, seqLen);
        _lastNormalized = normalized;

        // Step 2: Q, K, V projections
        var normFlat = Engine.Reshape(normalized, new[] { batchSize * seqLen, _modelDimension });

        var q = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _queryWeights),
            Engine.Reshape(_queryBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        var k = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _keyWeights),
            Engine.Reshape(_keyBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        var v = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _valueWeights),
            Engine.Reshape(_valueBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 3: Output gate (SiLU activation)
        var gateRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Swish(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 4: TTT-Linear recurrence per head
        // At each step t: W_t = W_{t-1} - eta * 2 * (W_{t-1} * k_t - v_t) * k_t^T
        // Output: o_t = W_t * q_t
        var tttOutput = TTTLinearForward(q, k, v, batchSize, seqLen);
        _lastTTTOutput = tttOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(tttOutput, gate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension }));
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        // Step 7: Residual connection
        var result = Engine.TensorAdd(output3D, input3D);
        result = ApplyActivation(result);
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
    /// TTT-Linear forward: inner model weight update via gradient descent on self-supervised loss.
    /// </summary>
    /// <remarks>
    /// For each timestep t and each head h:
    ///   1. Compute prediction error: e_t = W_{t-1} * k_t - v_t
    ///   2. Compute gradient: grad = 2 * e_t * k_t^T (outer product, scaled)
    ///   3. Update: W_t = W_{t-1} - eta_h * grad
    ///   4. Output: o_t = W_t * q_t
    ///
    /// This is mathematically equivalent to the delta rule applied to a linear associative memory,
    /// where keys map to values. The inner model literally learns a key-value lookup at test time.
    /// </remarks>
    private Tensor<T> TTTLinearForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

        // Inner weights state: starts from W_0 (learnable initialization) for each batch element
        var innerW = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allInnerW = TensorAllocator.Rent<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

        // Initialize inner weights to W_0
        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                for (int di = 0; di < _headDimension; di++)
                {
                    for (int dj = 0; dj < _headDimension; dj++)
                    {
                        T w0 = _innerWeightsInit[new[] { hi, di, dj }];
                        innerW[new[] { bi, hi, di, dj }] = w0;
                        allInnerW[new[] { bi, 0, hi, di, dj }] = w0;
                    }
                }
            }
        }

        T two = NumOps.FromDouble(2.0);

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                // Effective learning rate for this head: eta * etaScale[h]
                T etaH = NumOps.Multiply(_innerLearningRate, _etaScale[hi]);

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Extract key and value vectors for this head
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = k[new[] { bi, t, dimStart + di }];
                        vHead[di] = v[new[] { bi, t, dimStart + di }];
                    }

                    // Step 1: Compute prediction error e = W * k - v
                    var error = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T wk = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            wk = NumOps.Add(wk,
                                NumOps.Multiply(innerW[new[] { bi, hi, di, dj }], kHead[dj]));
                        }
                        error[di] = NumOps.Subtract(wk, vHead[di]);
                    }

                    // Step 2-3: Update W_t = W_{t-1} - eta * 2 * error * k^T
                    // Gradient: dL/dW = 2 * error * k^T (outer product)
                    // Update combines steps: W -= eta * 2 * error * k^T
                    T etaTwo = NumOps.Multiply(etaH, two);
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T scaledError = NumOps.Multiply(etaTwo, error[di]);
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T grad = NumOps.Multiply(scaledError, kHead[dj]);
                            innerW[new[] { bi, hi, di, dj }] = NumOps.Subtract(
                                innerW[new[] { bi, hi, di, dj }], grad);
                        }
                    }

                    // Step 4: Output o = W_t * q
                    var qHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                        qHead[di] = q[new[] { bi, t, dimStart + di }];

                    for (int di = 0; di < _headDimension; di++)
                    {
                        T oVal = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(innerW[new[] { bi, hi, di, dj }], qHead[dj]));
                        }
                        output[new[] { bi, t, dimStart + di }] = oVal;
                    }

                    // Save W_t snapshot for backward pass
                    for (int di = 0; di < _headDimension; di++)
                        for (int dj = 0; dj < _headDimension; dj++)
                            allInnerW[new[] { bi, t + 1, hi, di, dj }] = innerW[new[] { bi, hi, di, dj }];
                }
            }
        }

        _lastInnerWeights = allInnerW;
        return output;
    }

    /// <summary>
    /// Simple layer normalization across the last dimension.
    /// </summary>
    private Tensor<T> LayerNormForward(Tensor<T> input, int batchSize, int seqLen)
    {
        // Standard LayerNorm over the last axis (modelDimension). Replaces 4
        // nested scalar passes (mean + variance + normalize + affine) with a
        // single fused Engine.LayerNorm call. Backward still has the manual
        // recompute path below — Engine.LayerNormBackward could fold that in
        // as a follow-up but leaves a larger refactor behind.
        return Engine.LayerNorm(input, _lnGamma, _lnBeta, 1e-5, out _, out _);
    }

    /// <summary>
    /// Backward pass for layer normalization.
    /// </summary>
    private Tensor<T> LayerNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T eps = NumOps.FromDouble(1e-5);
        T invDim = NumOps.Divide(NumOps.One, NumOps.FromDouble(_modelDimension));
        var lnGammaGrad = _lnGammaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");
        var lnBetaGrad = _lnBetaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Recompute mean, variance, invStd
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, t, d }]);
                mean = NumOps.Multiply(mean, invDim);

                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Multiply(variance, invDim);
                T invStd = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));

                // Compute normalized values and gradient accumulations
                T dMean = NumOps.Zero;
                T dVar = NumOps.Zero;

                for (int d = 0; d < _modelDimension; d++)
                {
                    T xCentered = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    T xNorm = NumOps.Multiply(xCentered, invStd);
                    T dNorm = NumOps.Multiply(dOutput[new[] { bi, t, d }], _lnGamma[d]);

                    // Accumulate gamma and beta gradients
                    lnGammaGrad[d] = NumOps.Add(lnGammaGrad[d],
                        NumOps.Multiply(dOutput[new[] { bi, t, d }], xNorm));
                    lnBetaGrad[d] = NumOps.Add(lnBetaGrad[d],
                        dOutput[new[] { bi, t, d }]);

                    dMean = NumOps.Add(dMean, dNorm);
                    dVar = NumOps.Subtract(dVar,
                        NumOps.Multiply(NumOps.FromDouble(0.5),
                            NumOps.Multiply(dNorm,
                                NumOps.Multiply(xCentered,
                                    NumOps.Multiply(invStd, NumOps.Multiply(invStd, invStd))))));
                }

                dMean = NumOps.Multiply(NumOps.Negate(invStd), dMean);

                for (int d = 0; d < _modelDimension; d++)
                {
                    T xCentered = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    T dNorm = NumOps.Multiply(dOutput[new[] { bi, t, d }], _lnGamma[d]);

                    T dX = NumOps.Multiply(dNorm, invStd);
                    dX = NumOps.Add(dX, NumOps.Multiply(NumOps.Multiply(two(dVar), xCentered), invDim));
                    dX = NumOps.Add(dX, NumOps.Multiply(dMean, invDim));

                    dInput[new[] { bi, t, d }] = dX;
                }
            }
        }

        return dInput;
    }

    /// <summary>
    /// Returns the constant 2.0 as type T.
    /// </summary>
    private static T two(T _) => MathHelper.GetNumericOperations<T>().FromDouble(2.0);

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
        _innerWeightsInit = Engine.TensorAdd(_innerWeightsInit, Engine.TensorMultiplyScalar(_innerWeightsInitGradient!, negLR));
        _etaScale = Engine.TensorAdd(_etaScale, Engine.TensorMultiplyScalar(_etaScaleGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
        _lnGamma = Engine.TensorAdd(_lnGamma, Engine.TensorMultiplyScalar(_lnGammaGradient!, negLR));
        _lnBeta = Engine.TensorAdd(_lnBeta, Engine.TensorMultiplyScalar(_lnBetaGradient!, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_queryBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_innerWeightsInit, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

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
        _innerWeightsInit, _etaScale,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias,
        _lnGamma, _lnBeta
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
            new Vector<T>(_innerWeightsInitGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_etaScaleGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_lnGammaGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_lnBetaGradient?.ToArray() ?? Array.Empty<T>()));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _queryBiasGradient = null; _keyWeightsGradient = null; _keyBiasGradient = null; _valueWeightsGradient = null; _valueBiasGradient = null; _innerWeightsInitGradient = null; _etaScaleGradient = null; _lnGammaGradient = null; _lnBetaGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastNormalized = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastTTTOutput = null;
        _lastInnerWeights = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _innerWeightsInitGradient = null;
        _etaScaleGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
        _lnGammaGradient = null;
        _lnBetaGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["InnerLearningRate"] = NumOps.ToDouble(_innerLearningRate).ToString("G");
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
    /// Gets the inner model initial weights for external inspection.
    /// </summary>
    public Tensor<T> GetInnerWeightsInit() => _innerWeightsInit;
}
