using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the MesaNet layer from Grazzi et al., 2025.
/// </summary>
/// <remarks>
/// <para>
/// MesaNet ("Locally Optimal Test-Time Training") improves upon TTT by replacing gradient descent
/// with a closed-form ridge regression update for the inner model weights. Instead of taking a noisy
/// gradient step at each timestep, MesaNet computes the <b>locally optimal</b> weight matrix W_t that
/// minimizes the reconstruction error plus a regularization term toward the initial weights W_0.
/// </para>
/// <para>
/// The optimization at each timestep t:
/// <code>
///   W_t = argmin_W  ||W * K_t - V_t||^2  +  lambda * ||W - W_0||^2
///
///   Closed-form solution (batch form):
///     W_t = (K_t * K_t^T + lambda * I)^{-1} * (K_t * V_t^T + lambda * W_0)
///
///   Incremental Woodbury update (avoids recomputing from scratch):
///     When a new key-value pair (k_t, v_t) arrives:
///     P_t = P_{t-1} - P_{t-1}*k_t * (1 + k_t^T*P_{t-1}*k_t)^{-1} * k_t^T*P_{t-1}
///     W_t = W_{t-1} + (v_t - W_{t-1}*k_t) * k_t^T * P_t
///
///   where P_t = (sum_{i=1}^{t} k_i*k_i^T + lambda*I)^{-1} is the inverse covariance.
///
///   Output: o_t = W_t * q_t
/// </code>
/// </para>
/// <para>
/// The Woodbury identity allows rank-1 updates to the inverse covariance matrix P, making each step
/// O(d^2) instead of O(d^3) for a full matrix inversion. This keeps the overall complexity linear
/// in sequence length while achieving a strictly better update than gradient descent.
/// </para>
/// <para>
/// Why this is better than TTT's gradient descent:
/// - TTT: W_t = W_{t-1} - eta * grad  (approximate, depends on learning rate choice)
/// - MesaNet: W_t = optimal solution of ridge regression (exact, only depends on lambda)
/// - No learning rate sensitivity -- lambda is more stable and interpretable
/// - Converges in one step per observation rather than requiring multiple gradient steps
/// </para>
/// <para><b>For Beginners:</b> Imagine you have a student (the inner model W) who learns from examples.
///
/// TTT approach: After each example, the student takes a small step toward the right answer
/// (gradient descent). The step size (learning rate) is tricky to set -- too big and the student
/// overshoots, too small and learning is slow.
///
/// MesaNet approach: After each example, the student computes the BEST POSSIBLE answer given all
/// examples seen so far, with a gentle preference toward their initial knowledge (regularization).
/// There is no step size to tune -- the answer is mathematically optimal.
///
/// The Woodbury trick makes this efficient: instead of re-solving the entire problem from scratch
/// each time, MesaNet incrementally updates the solution as new examples arrive. This is like
/// updating a running average instead of recomputing from all data points.
/// </para>
/// <para>
/// <b>Reference:</b> Grazzi et al., "MesaNet: Locally Optimal Test-Time Training", 2025.
/// https://arxiv.org/abs/2506.05233
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MesaNetLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly T _regularization;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _queryBias;
    private Tensor<T> _keyWeights;
    private Tensor<T> _keyBias;
    private Tensor<T> _valueWeights;
    private Tensor<T> _valueBias;

    // Inner model initial weights per head: [numHeads, headDim, headDim]
    private Tensor<T> _innerWeightsInit;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Layer normalization parameters
    private Tensor<T> _lnGamma;
    private Tensor<T> _lnBeta;

    // Cached values for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastNormalized;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastMesaOutput;
    private Tensor<T>? _lastInnerWeights; // [batch, seqLen+1, numHeads, headDim, headDim]
    private Tensor<T>? _lastPMatrices;    // [batch, seqLen+1, numHeads, headDim, headDim]
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _queryBiasGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _keyBiasGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _valueBiasGradient;
    private Tensor<T>? _innerWeightsInitGradient;
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
    /// Gets the regularization strength (lambda).
    /// </summary>
    public T Regularization => _regularization;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _queryWeights.Length + _queryBias.Length +
        _keyWeights.Length + _keyBias.Length +
        _valueWeights.Length + _valueBias.Length +
        _innerWeightsInit.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length +
        _lnGamma.Length + _lnBeta.Length;

    /// <summary>
    /// Creates a new MesaNet layer implementing locally optimal test-time training.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each input/output token embedding.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own inner weight matrix and inverse covariance.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="regularization">
    /// Ridge regression regularization strength (lambda). Default: 0.01.
    /// <para><b>For Beginners:</b> Controls how strongly the inner model prefers staying close to its
    /// initial weights W_0. Higher values mean more conservative updates; lower values mean the model
    /// adapts more aggressively to new observations. Unlike a learning rate, lambda has a clear
    /// statistical interpretation as a prior strength.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MesaNetLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        double regularization = 0.01,
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
        if (regularization <= 0)
            throw new ArgumentException($"Regularization ({regularization}) must be positive.", nameof(regularization));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _regularization = NumOps.FromDouble(regularization);

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _queryBias = new Tensor<T>([modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyBias = new Tensor<T>([modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueBias = new Tensor<T>([modelDimension]);

        _innerWeightsInit = new Tensor<T>([numHeads, _headDimension, _headDimension]);

        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

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

        T initScale = NumOps.FromDouble(0.01);
        for (int h = 0; h < _numHeads; h++)
            for (int i = 0; i < _headDimension; i++)
                for (int j = 0; j < _headDimension; j++)
                {
                    T val = i == j
                        ? initScale
                        : NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5),
                            NumOps.FromDouble(0.001));
                    _innerWeightsInit[new[] { h, i, j }] = val;
                }

        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);

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

        // Step 4: Mesa (ridge regression) recurrence per head using Woodbury updates
        var mesaOutput = MesaForward(q, k, v, batchSize, seqLen);
        _lastMesaOutput = mesaOutput;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(mesaOutput, gate);

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
    /// Mesa forward: ridge regression with incremental Woodbury updates.
    /// </summary>
    /// <remarks>
    /// For each timestep t and each head h:
    ///   1. Rank-1 update to inverse covariance:  P_t = P_{t-1} - P_{t-1}*k*(1+k^T*P_{t-1}*k)^{-1}*k^T*P_{t-1}
    ///   2. Update inner weights:  W_t = W_{t-1} + (v - W_{t-1}*k) * k^T * P_t
    ///   3. Output:  o_t = W_t * q_t
    /// </remarks>
    private Tensor<T> MesaForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // State: inner weights W and inverse covariance P per head
        // W: [batch, numHeads, headDim, headDim]
        // P: [batch, numHeads, headDim, headDim] — initialized to (1/lambda) * I
        var innerW = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var pMatrix = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });
        var allInnerW = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });
        var allP = new Tensor<T>(new[] { batchSize, seqLen + 1, _numHeads, _headDimension, _headDimension });

        T invLambda = NumOps.Divide(NumOps.One, _regularization);

        // Initialize W to W_0 and P to (1/lambda)*I
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

                        T pVal = di == dj ? invLambda : NumOps.Zero;
                        pMatrix[new[] { bi, hi, di, dj }] = pVal;
                        allP[new[] { bi, 0, hi, di, dj }] = pVal;
                    }
                }
            }
        }

        for (int t = 0; t < seqLen; t++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Extract key and value for this head
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = k[new[] { bi, t, dimStart + di }];
                        vHead[di] = v[new[] { bi, t, dimStart + di }];
                    }

                    // Step 1: Compute P*k
                    var pk = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        pk[di] = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                            pk[di] = NumOps.Add(pk[di],
                                NumOps.Multiply(pMatrix[new[] { bi, hi, di, dj }], kHead[dj]));
                    }

                    // Compute scalar: 1 + k^T * P * k
                    T kPk = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                        kPk = NumOps.Add(kPk, NumOps.Multiply(kHead[di], pk[di]));
                    T denom = NumOps.Add(NumOps.One, kPk);
                    T invDenom = NumOps.Divide(NumOps.One, denom);

                    // Woodbury update: P_t = P_{t-1} - (P*k)*(P*k)^T / (1 + k^T*P*k)
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T correction = NumOps.Multiply(invDenom,
                                NumOps.Multiply(pk[di], pk[dj]));
                            pMatrix[new[] { bi, hi, di, dj }] = NumOps.Subtract(
                                pMatrix[new[] { bi, hi, di, dj }], correction);
                        }
                    }

                    // Step 2: Compute prediction error: e = W*k - v
                    var error = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T wk = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                            wk = NumOps.Add(wk,
                                NumOps.Multiply(innerW[new[] { bi, hi, di, dj }], kHead[dj]));
                        error[di] = NumOps.Subtract(wk, vHead[di]);
                    }

                    // Compute k^T * P_t (using updated P)
                    var kP = new T[_headDimension];
                    for (int dj = 0; dj < _headDimension; dj++)
                    {
                        kP[dj] = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                            kP[dj] = NumOps.Add(kP[dj],
                                NumOps.Multiply(kHead[di], pMatrix[new[] { bi, hi, di, dj }]));
                    }

                    // Update W: W_t = W_{t-1} - error * (k^T * P_t) = W_{t-1} + (v - W*k) * k^T * P_t
                    for (int di = 0; di < _headDimension; di++)
                    {
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T update = NumOps.Multiply(error[di], kP[dj]);
                            innerW[new[] { bi, hi, di, dj }] = NumOps.Subtract(
                                innerW[new[] { bi, hi, di, dj }], update);
                        }
                    }

                    // Step 3: Output o = W_t * q
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(innerW[new[] { bi, hi, di, dj }],
                                    q[new[] { bi, t, flatDj }]));
                        }
                        output[new[] { bi, t, flatDi }] = oVal;
                    }

                    // Save snapshots
                    for (int di = 0; di < _headDimension; di++)
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            allInnerW[new[] { bi, t + 1, hi, di, dj }] = innerW[new[] { bi, hi, di, dj }];
                            allP[new[] { bi, t + 1, hi, di, dj }] = pMatrix[new[] { bi, hi, di, dj }];
                        }
                }
            }
        }

        _lastInnerWeights = allInnerW;
        _lastPMatrices = allP;
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
                T mean = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                    mean = NumOps.Add(mean, input[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_modelDimension));

                T variance = NumOps.Zero;
                for (int d = 0; d < _modelDimension; d++)
                {
                    T diff = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_modelDimension));

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
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastNormalized = _lastNormalized ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGate = _lastGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGateRaw = _lastGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastMesaOutput = _lastMesaOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastInnerWeights = _lastInnerWeights ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastPMatrices = _lastPMatrices ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        var dResidual = activationGrad;
        var dProjected = activationGrad;

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _queryBiasGradient = new Tensor<T>([_modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyBiasGradient = new Tensor<T>([_modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueBiasGradient = new Tensor<T>([_modelDimension]);
        _innerWeightsInitGradient = new Tensor<T>([_numHeads, _headDimension, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);
        _lnGammaGradient = new Tensor<T>([_modelDimension]);
        _lnBetaGradient = new Tensor<T>([_modelDimension]);

        // Step 6 backward: output projection
        var gradFlat = dProjected.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(dProjected, new int[] { 0, 1 });

        var gatedFlat = Engine.TensorMultiply(lastMesaOutput, lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating
        var dMesaOut = Engine.TensorMultiply(dGated, lastGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastMesaOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastGateRaw));

        var normFlat = lastNormalized.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dNormFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 4 backward: Mesa recurrence (approximate via BPTT through W updates)
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dInnerW = new Tensor<T>(new[] { batchSize, _numHeads, _headDimension, _headDimension });

        for (int t = seqLen - 1; t >= 0; t--)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int bi = 0; bi < batchSize; bi++)
                {
                    // Output: o = W_t * q -> dW_t += dO * q^T, dQ += W_t^T * dO
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T dO = dMesaOut[new[] { bi, t, flatDi }];

                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            int flatDj = dimStart + dj;
                            T qVal = lastQuery[new[] { bi, t, flatDj }];
                            T wVal = lastInnerWeights[new[] { bi, t + 1, hi, di, dj }];

                            dInnerW[new[] { bi, hi, di, dj }] = NumOps.Add(
                                dInnerW[new[] { bi, hi, di, dj }],
                                NumOps.Multiply(dO, qVal));

                            dQ[new[] { bi, t, flatDj }] = NumOps.Add(
                                dQ[new[] { bi, t, flatDj }],
                                NumOps.Multiply(wVal, dO));
                        }
                    }

                    // Backward through W update: W_t = W_{t-1} - error * kP
                    // where error = W_{t-1}*k - v and kP = k^T * P_t
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        kHead[di] = lastKey[new[] { bi, t, dimStart + di }];
                        vHead[di] = lastValue[new[] { bi, t, dimStart + di }];
                    }

                    // Recompute error using W_{t-1}
                    var error = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T wk = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                            wk = NumOps.Add(wk,
                                NumOps.Multiply(lastInnerWeights[new[] { bi, t, hi, di, dj }], kHead[dj]));
                        error[di] = NumOps.Subtract(wk, vHead[di]);
                    }

                    // Recompute kP using P_t
                    var kP = new T[_headDimension];
                    for (int dj = 0; dj < _headDimension; dj++)
                    {
                        kP[dj] = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                            kP[dj] = NumOps.Add(kP[dj],
                                NumOps.Multiply(kHead[di], lastPMatrices[new[] { bi, t + 1, hi, di, dj }]));
                    }

                    // Propagate through W_t = W_{t-1} - error * kP^T
                    for (int di = 0; di < _headDimension; di++)
                    {
                        T dErrorAccum = NumOps.Zero;
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T dW_ij = dInnerW[new[] { bi, hi, di, dj }];

                            // dK contribution from error*kP term
                            dK[new[] { bi, t, dimStart + dj }] = NumOps.Subtract(
                                dK[new[] { bi, t, dimStart + dj }],
                                NumOps.Multiply(error[di], dW_ij));

                            dErrorAccum = NumOps.Add(dErrorAccum,
                                NumOps.Multiply(dW_ij, kP[dj]));
                        }

                        T dError_di = NumOps.Negate(dErrorAccum);

                        // dV from error = Wk - v -> dV = -dError
                        dV[new[] { bi, t, dimStart + di }] = NumOps.Subtract(
                            dV[new[] { bi, t, dimStart + di }], dError_di);

                        // dW_{t-1} from error = W*k - v
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            dInnerW[new[] { bi, hi, di, dj }] = NumOps.Add(
                                dInnerW[new[] { bi, hi, di, dj }],
                                NumOps.Multiply(dError_di, kHead[dj]));
                        }

                        // dK from error = W*k - v
                        for (int dj = 0; dj < _headDimension; dj++)
                        {
                            T wPrev = lastInnerWeights[new[] { bi, t, hi, di, dj }];
                            dK[new[] { bi, t, dimStart + dj }] = NumOps.Add(
                                dK[new[] { bi, t, dimStart + dj }],
                                NumOps.Multiply(dError_di, wPrev));
                        }
                    }
                }
            }
        }

        // Accumulate W_0 gradient
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
                for (int di = 0; di < _headDimension; di++)
                    for (int dj = 0; dj < _headDimension; dj++)
                        _innerWeightsInitGradient[new[] { hi, di, dj }] = NumOps.Add(
                            _innerWeightsInitGradient[new[] { hi, di, dj }],
                            dInnerW[new[] { bi, hi, di, dj }]);

        // Q, K, V projection gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dQFlat);
        _queryBiasGradient = Engine.ReduceSum(dQ, new int[] { 0, 1 });
        _keyWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dKFlat);
        _keyBiasGradient = Engine.ReduceSum(dK, new int[] { 0, 1 });
        _valueWeightsGradient = Engine.TensorMatMul(normFlat.Transpose([1, 0]), dVFlat);
        _valueBiasGradient = Engine.ReduceSum(dV, new int[] { 0, 1 });

        // Gradient to normalized input
        var dNormFromQKV = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dNormFromQKV = Engine.TensorAdd(dNormFromQKV,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dNormFromQKV = Engine.TensorAdd(dNormFromQKV,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));

        var dNormTotal = Engine.TensorAdd(dNormFromQKV, dNormFromGate);
        var dNorm3D = dNormTotal.Reshape(batchSize, seqLen, _modelDimension);

        // Layer norm backward
        var dInput = LayerNormBackward(dNorm3D, lastInput, batchSize, seqLen);

        // Add residual gradient
        dInput = Engine.TensorAdd(dInput, dResidual);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput.Reshape(_originalInputShape);

        return dInput;
    }

    private Tensor<T> LayerNormBackward(Tensor<T> dOutput, Tensor<T> input, int batchSize, int seqLen)
    {
        var dInput = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T eps = NumOps.FromDouble(1e-5);
        T invDim = NumOps.Divide(NumOps.One, NumOps.FromDouble(_modelDimension));
        T twoVal = NumOps.FromDouble(2.0);
        var lnGammaGrad = _lnGammaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");
        var lnBetaGrad = _lnBetaGradient
            ?? throw new InvalidOperationException("Gradients must be initialized before backward pass.");

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
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

                T dMean = NumOps.Zero;
                T dVar = NumOps.Zero;

                for (int d = 0; d < _modelDimension; d++)
                {
                    T xCentered = NumOps.Subtract(input[new[] { bi, t, d }], mean);
                    T xNorm = NumOps.Multiply(xCentered, invStd);
                    T dNorm = NumOps.Multiply(dOutput[new[] { bi, t, d }], _lnGamma[d]);

                    lnGammaGrad[d] = NumOps.Add(lnGammaGrad[d],
                        NumOps.Multiply(dOutput[new[] { bi, t, d }], xNorm));
                    lnBetaGrad[d] = NumOps.Add(lnBetaGrad[d], dOutput[new[] { bi, t, d }]);

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
                    dX = NumOps.Add(dX, NumOps.Multiply(NumOps.Multiply(twoVal, NumOps.Multiply(dVar, xCentered)), invDim));
                    dX = NumOps.Add(dX, NumOps.Multiply(dMean, invDim));
                    dInput[new[] { bi, t, d }] = dX;
                }
            }
        }

        return dInput;
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
        _innerWeightsInit = Engine.TensorAdd(_innerWeightsInit, Engine.TensorMultiplyScalar(_innerWeightsInitGradient!, negLR));
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
        _innerWeightsInit,
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
        _lastMesaOutput = null;
        _lastInnerWeights = null;
        _lastPMatrices = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _queryBiasGradient = null;
        _keyWeightsGradient = null;
        _keyBiasGradient = null;
        _valueWeightsGradient = null;
        _valueBiasGradient = null;
        _innerWeightsInitGradient = null;
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
        metadata["Regularization"] = NumOps.ToDouble(_regularization).ToString("G");
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
