using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class TTTLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly T _innerLearningRate;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Inner model initial weights per head: [numHeads, headDim, headDim]
    // This is W_0, the starting point for inner model weights at the beginning of each sequence
    private Tensor<T> _innerWeightsInit;

    // Learnable inner learning rate scale per head: [numHeads]
    private Tensor<T> _etaScale;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
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

        // Step 1: Layer normalization
        var normalized = LayerNormForward(input3D, batchSize, seqLen);
        _lastNormalized = normalized;

        // Step 2: Q, K, V projections
        var normFlat = normalized.Reshape(batchSize * seqLen, _modelDimension);

        var q = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _queryWeights),
            _queryBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var k = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _keyWeights),
            _keyBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        var v = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _valueWeights),
            _valueBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);

        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 3: Output gate (SiLU activation)
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(normFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
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
        var gatedFlat = gatedOutput.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(gatedFlat, _outputProjectionWeights),
            _outputProjectionBias.Reshape(1, _modelDimension));
        var output3D = outputFlat.Reshape(batchSize, seqLen, _modelDimension);

        // Step 7: Residual connection
        var result = Engine.TensorAdd(output3D, input3D);
        result = ApplyActivation(result);
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
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Inner weights state: starts from W_0 (learnable initialization) for each batch element
        var innerW = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allInnerW = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

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
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T eps = NumOps.FromDouble(1e-5);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Compute mean
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_modelDimension));

                // Compute variance
                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_modelDimension));

                // Normalize and apply affine
                T invStd = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));
                for (int d = 0; d < _modelDimension; d++)
                {
                    T normalized = NumOps.Multiply(
                        NumOps.Subtract(input[new[] { bi, t, d }], mean), invStd);
                    output[new[] { bi, t, d }] = NumOps.Add(
                        NumOps.Multiply(_lnGamma[d], normalized), _lnBeta[d]);
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastNormalized == null ||
            _lastQuery == null || _lastKey == null || _lastValue == null ||
            _lastGate == null || _lastGateRaw == null || _lastTTTOutput == null ||
            _lastInnerWeights == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Residual: gradient flows through both the projection and the skip connection
        var dResidual = activationGrad; // gradient to skip connection (input)
        var dProjected = activationGrad; // gradient through the TTT path

        // Initialize all gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _innerWeightsInitGradient = new Tensor<T>([_numHeads, _headDimension, _headDimension]);
        _etaScaleGradient = new Tensor<T>([_numHeads]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);
        _lnGammaGradient = new Tensor<T>([_modelDimension]);
        _lnBetaGradient = new Tensor<T>([_modelDimension]);

        // Step 6 backward: output projection
        var gradFlat = dProjected.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var gatedFlat = Engine.TensorMultiply(_lastTTTOutput, _lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating - gatedOutput = tttOutput * gate
        var dTTTOut = Engine.TensorMultiply(dGated, _lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, _lastTTTOutput);

        // Gate uses SiLU/Swish: derivative = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(_lastGateRaw));

        // Gate weight gradients
        var normFlat = _lastNormalized.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        // Gradient flowing back to normalized input from gate path
        var dNormFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: TTT-Linear recurrence (most complex part)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        T two = NumOps.FromDouble(2.0);

        // Backward through time for each head
        var dInnerW = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;
                T etaH = NumOps.Multiply(_innerLearningRate, _etaScale[hi]);
                T etaTwo = NumOps.Multiply(etaH, two);

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // ----- Output: o = W_t * q -----
                    // dW_t += dO * q^T, dQ += W_t^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dTTTOut[new[] { bi, t, flatDi }];

                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T qVal = _lastQuery[new[] { bi, t, flatDj }];
                            T wVal = _lastInnerWeights[new[] { bi, t + 1, hi, di, dj }];

                            // dW_t[di,dj] += dO[di] * q[dj]
                            dInnerW[new[] { bi, hi, di, dj }] = NumOps.Add(
                                dInnerW[new[] { bi, hi, di, dj }],
                                NumOps.Multiply(dO, qVal));

                            // dQ[dj] += W_t[di,dj] * dO[di]
                            dQ[new[] { bi, t, flatDj }] = NumOps.Add(
                                dQ[new[] { bi, t, flatDj }],
                                NumOps.Multiply(wVal, dO));
                        }
                    }

                    // ----- Update: W_t = W_{t-1} - eta*2 * error * k^T -----
                    // where error = W_{t-1}*k - v
                    // Recompute error using W_{t-1}
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = _lastKey[new[] { bi, t, dimStart + di }];
                        vHead[di] = _lastValue[new[] { bi, t, dimStart + di }];
                    }

                    var error = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T wk = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            wk = NumOps.Add(wk,
                                NumOps.Multiply(_lastInnerWeights[new[] { bi, t, hi, di, dj }], kHead[dj]));
                        }
                        error[di] = NumOps.Subtract(wk, vHead[di]);
                    }

                    // dW_{t-1} from update rule:
                    // W_t = W_{t-1} - eta*2*error*k^T
                    // dW_{t-1} = dW_t (identity contribution)
                    //           - eta*2 * (dW_t contribution through error)
                    // Since error = W_{t-1}*k - v, d_error/dW_{t-1} = (...)*k
                    // Full: dW_{t-1}[i,j] += dW_t[i,j]  (already accumulated)
                    //   plus from the gradient through the update:
                    //   dW_{t-1}[i,j] -= eta*2 * sum_m(dW_t[i,m] * k[m]) * k[j]
                    // But this is implicit since dW_t already flows through W_t = f(W_{t-1})

                    // For dK: dK += contribution from update
                    // dL/dk from the update = -eta*2 * (error^T * dW + dW * k * error^T ... )
                    // Simplified: propagate dW through the update step
                    for (int di = 0; di < _headDimension; di++)
                    {
                        // dK from: grad[di,dj] = eta*2 * error[di] * k[dj]
                        // dK[dj] += -eta*2 * error[di] * dW[di,dj]  (negated because W -= grad)
                        // dError[di] += -eta*2 * sum_j(dW[di,dj] * k[dj])
                        T dErrorAccum = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T dW_ij = dInnerW[new[] { bi, hi, di, dj }];

                            // dK contribution
                            dK[new[] { bi, t, dimStart + dj }] = NumOps.Subtract(
                                dK[new[] { bi, t, dimStart + dj }],
                                NumOps.Multiply(etaTwo, NumOps.Multiply(error[di], dW_ij)));

                            // Accumulate for dError
                            dErrorAccum = NumOps.Add(dErrorAccum,
                                NumOps.Multiply(dW_ij, kHead[dj]));
                        }

                        // dError[di] from update
                        T dError_di = NumOps.Multiply(NumOps.Negate(etaTwo), dErrorAccum);

                        // error = W*k - v, so:
                        // dV[di] -= dError[di] (error = Wk - v, so dv = -dError)
                        dV[new[] { bi, t, dimStart + di }] = NumOps.Subtract(
                            dV[new[] { bi, t, dimStart + di }], dError_di);

                        // dW_{t-1}[di,dj] += dError[di] * k[dj] (from error = W*k - v)
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            dInnerW[new[] { bi, hi, di, dj }] = NumOps.Add(
                                dInnerW[new[] { bi, hi, di, dj }],
                                NumOps.Multiply(dError_di, kHead[dj]));
                        }

                        // dK[dj] += dError[di] * W_{t-1}[di,dj] (from error = W*k - v)
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T wPrev = _lastInnerWeights[new[] { bi, t, hi, di, dj }];
                            dK[new[] { bi, t, dimStart + dj }] = NumOps.Add(
                                dK[new[] { bi, t, dimStart + dj }],
                                NumOps.Multiply(dError_di, wPrev));
                        }
                    }

                    // dEtaScale[hi] from the update: sum over all (i,j) of -2 * eta_base * error[i]*k[j] * dW_t[i,j]
                    // (accumulated across all timesteps)
                    T dEtaContrib = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T gradVal = NumOps.Multiply(two, NumOps.Multiply(error[di], kHead[dj]));
                            T dWVal = dInnerW[new[] { bi, hi, di, dj }];
                            dEtaContrib = NumOps.Add(dEtaContrib,
                                NumOps.Multiply(gradVal, dWVal));
                        }
                    }
                    _etaScaleGradient[hi] = NumOps.Subtract(_etaScaleGradient[hi],
                        NumOps.Multiply(_innerLearningRate, dEtaContrib));
                }
            }
        }

        // Accumulate dInnerW into dInnerWeightsInit (W_0 gradient)
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int di = 0; di < _headDimension; di++)
                    for (int dj = 0; dj < _headDimension; dj++)
                        _innerWeightsInitGradient[new[] { hi, di, dj }] = NumOps.Add(
                            _innerWeightsInitGradient[new[] { hi, di, dj }],
                            dInnerW[new[] { bi, hi, di, dj }]);

        // Q, K, V projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dKFlat);
        _keyBiasGradient = Engine.ReduceSum(dK, new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Gradient flowing back to normalized input from Q, K, V projections
        var dNormFromQKV = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dNormFromQKV = Engine.TensorAdd(dNormFromQKV,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dNormFromQKV = Engine.TensorAdd(dNormFromQKV,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));

        // Combine all gradients flowing to normalized input
        var dNormTotal = Engine.TensorAdd(dNormFromQKV, dNormFromGate);
        var dNorm3D = dNormTotal.Reshape(batchSize, seqLen, _modelDimension);

        // Layer norm backward
        var dInput = LayerNormBackward(dNorm3D, _lastInput, batchSize, seqLen);

        // Add residual gradient
        dInput = Engine.TensorAdd(dInput, dResidual);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    /// <summary>
    /// Backward pass for layer normalization.
    /// </summary>
    private Tensor<T> LayerNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
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
        _innerWeightsInit = Engine.TensorAdd(_innerWeightsInit, Engine.TensorMultiplyScalar(_innerWeightsInitGradient!, negLR));
        _etaScale = Engine.TensorAdd(_etaScale, Engine.TensorMultiplyScalar(_etaScaleGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
        _lnGamma = Engine.TensorAdd(_lnGamma, Engine.TensorMultiplyScalar(_lnGammaGradient!, negLR));
        _lnBeta = Engine.TensorAdd(_lnBeta, Engine.TensorMultiplyScalar(_lnBetaGradient!, negLR));
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
