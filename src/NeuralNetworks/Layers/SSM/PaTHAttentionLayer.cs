using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the PaTH Attention (Positional-aware Transformer via Householder) layer
/// from Mao et al., 2025 (arXiv:2505.16381).
/// </summary>
/// <remarks>
/// <para>
/// PaTH Attention replaces traditional positional encodings (sinusoidal, rotary, etc.) with
/// Householder reflections applied to queries and keys. Each position in the sequence has a
/// learned reflection vector p, and the corresponding Householder transform H = I - 2*p*p^T/||p||^2
/// is applied to Q and K before computing attention. This embeds positional information directly
/// into the geometry of the attention space rather than adding it as a bias.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compute Q, K, V projections from input
///   2. For each position i, compute Householder vector p_i (learned)
///   3. Householder transform: H_i = I - 2 * p_i * p_i^T / ||p_i||^2
///   4. Apply H_i to Q at position i, H_j to K at position j
///   5. Standard attention: softmax(H_i*Q_i * (H_j*K_j)^T / sqrt(d)) * V
///   6. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is that Householder reflections are orthogonal transformations. Unlike additive
/// positional encodings that can distort the magnitude of embeddings, Householder reflections
/// preserve norms while encoding position. The attention score between two positions depends on
/// BOTH the content (Q, K) and the relative position (H_i vs H_j), but in a multiplicative way
/// that is more expressive than simple additive bias.
/// </para>
/// <para><b>For Beginners:</b> PaTH Attention is a new way to tell the model WHERE each token
/// is in the sequence, using geometry instead of adding numbers.
///
/// Traditional approaches add position information:
/// - "I am token 5" gets added as a number pattern to the token's embedding
/// - This can interfere with the token's meaning
///
/// PaTH instead REFLECTS the token's query/key vectors using a mirror unique to each position:
/// - Position 1 has mirror A, position 2 has mirror B, etc.
/// - Each mirror "rotates" the query/key differently based on position
/// - The attention score naturally captures both WHAT the token means AND WHERE it is
///
/// Think of it like a hall of mirrors:
/// - Each position has its own unique mirror (Householder reflection)
/// - When you look at a token through its position's mirror, you see a unique view
/// - Two tokens at different positions produce different reflections
/// - The similarity between reflections captures both content and position
///
/// This is mathematically cleaner because reflections preserve vector lengths (they just
/// change direction), whereas adding position numbers changes lengths and can distort meaning.
/// </para>
/// <para>
/// <b>Reference:</b> Mao et al., "PaTH Attention: Positional-aware Transformer via Householder", 2025.
/// https://arxiv.org/abs/2505.16381
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class PaTHAttentionLayer<T> : LayerBase<T>
{
    private readonly int _sequenceLength;
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Householder vectors: [sequenceLength, numHeads, headDim]
    // Each position has a per-head Householder reflection vector
    private Tensor<T> _householderVectors;

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
    private Tensor<T>? _lastReflectedQ;
    private Tensor<T>? _lastReflectedK;
    private Tensor<T>? _lastAttentionWeights;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastAttentionOutput;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _householderVectorsGradient;
    private Tensor<T>? _outputGateWeightsGradient;
    private Tensor<T>? _outputGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <summary>Gets the sequence length.</summary>
    public int SequenceLength => _sequenceLength;

    /// <summary>Gets the model dimension.</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of attention heads.</summary>
    public int NumHeads => _numHeads;

    /// <summary>Gets the dimension per head.</summary>
    public int HeadDimension => _headDimension;

    /// <inheritdoc />
    public override int ParameterCount =>
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _householderVectors.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new PaTH Attention layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length. Determines the number of learned Householder vectors.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head has its own set of Householder reflection vectors,
    /// allowing different heads to encode position differently.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public PaTHAttentionLayer(
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

        _sequenceLength = sequenceLength;
        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _householderVectors = new Tensor<T>([sequenceLength, numHeads, _headDimension]);
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
        InitializeHouseholderVectors();
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

    private void InitializeHouseholderVectors()
    {
        // Initialize Householder vectors with small random values, then normalize
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));
        for (int pos = 0; pos < _sequenceLength; pos++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                // Generate random direction and normalize
                T normSq = NumOps.Zero;
                for (int di = 0; di < _headDimension; di++)
                {
                    T val = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
                    _householderVectors[new[] { pos, hi, di }] = val;
                    normSq = NumOps.Add(normSq, NumOps.Multiply(val, val));
                }

                // Normalize to unit length for stable initialization
                T norm = NumOps.Sqrt(NumOps.Add(normSq, NumOps.FromDouble(1e-8)));
                for (int di = 0; di < _headDimension; di++)
                {
                    T val = _householderVectors[new[] { pos, hi, di }];
                    _householderVectors[new[] { pos, hi, di }] = NumOps.Divide(val, norm);
                }
            }
        }
    }

    /// <summary>
    /// Applies Householder reflection H = I - 2*p*p^T/||p||^2 to a vector x.
    /// Result: x - 2 * p * (p^T * x) / ||p||^2.
    /// </summary>
    private void ApplyHouseholderReflection(
        T[] x, int pos, int headIndex, T[] result)
    {
        // Compute p^T * x and ||p||^2
        T dotPX = NumOps.Zero;
        T normPSq = NumOps.Zero;

        for (int di = 0; di < _headDimension; di++)
        {
            T pVal = _householderVectors[new[] { pos, headIndex, di }];
            dotPX = NumOps.Add(dotPX, NumOps.Multiply(pVal, x[di]));
            normPSq = NumOps.Add(normPSq, NumOps.Multiply(pVal, pVal));
        }

        // H*x = x - 2 * (p^T * x) / ||p||^2 * p
        T normPSqSafe = NumOps.Add(normPSq, NumOps.FromDouble(1e-10));
        T coeff = NumOps.Multiply(NumOps.FromDouble(2.0), NumOps.Divide(dotPX, normPSqSafe));

        for (int di = 0; di < _headDimension; di++)
        {
            T pVal = _householderVectors[new[] { pos, headIndex, di }];
            result[di] = NumOps.Subtract(x[di], NumOps.Multiply(coeff, pVal));
        }
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

        // Step 2: Apply Householder reflections to Q and K
        var reflectedQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var reflectedK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int posIndex = Math.Min(t, _sequenceLength - 1);
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Extract head-specific Q and K vectors
                    var qHead = new T[_headDimension];
                    var kHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        qHead[di] = q[new[] { bi, t, flatDi }];
                        kHead[di] = k[new[] { bi, t, flatDi }];
                    }

                    // Apply Householder reflection
                    var rQ = new T[_headDimension];
                    var rK = new T[_headDimension];
                    ApplyHouseholderReflection(qHead, posIndex, hi, rQ);
                    ApplyHouseholderReflection(kHead, posIndex, hi, rK);

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        reflectedQ[new[] { bi, t, flatDi }] = rQ[di];
                        reflectedK[new[] { bi, t, flatDi }] = rK[di];
                    }
                }
            }
        }
        _lastReflectedQ = reflectedQ;
        _lastReflectedK = reflectedK;

        // Step 3: Compute attention with reflected Q and K
        var attnOutput = ComputeAttention(reflectedQ, reflectedK, v, batchSize, seqLen);
        _lastAttentionOutput = attnOutput;

        // Step 4: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Sigmoid(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(attnOutput, gate);

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
    /// Standard scaled dot-product attention using reflected Q and K.
    /// </summary>
    private Tensor<T> ComputeAttention(
        Tensor<T> reflectedQ, Tensor<T> reflectedK, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        _lastAttentionWeights = new Tensor<T>(new[] { batchSize, _numHeads, seqLen, seqLen });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int ti = 0; ti < seqLen; ti++)
                {
                    // Compute scores: reflectedQ_i * reflectedK_j^T / sqrt(d)
                    T maxScore = NumOps.FromDouble(-1e9);
                    var scores = new T[seqLen];

                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T dot = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(
                                    reflectedQ[new[] { bi, ti, flatDi }],
                                    reflectedK[new[] { bi, tj, flatDi }]));
                        }
                        scores[tj] = NumOps.Multiply(dot, scale);
                        double scoreVal = NumOps.ToDouble(scores[tj]);
                        double maxVal = NumOps.ToDouble(maxScore);
                        if (scoreVal > maxVal)
                            maxScore = scores[tj];
                    }

                    // Softmax
                    T sumExp = NumOps.Zero;
                    var expScores = new T[seqLen];
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        expScores[tj] = NumOps.Exp(NumOps.Subtract(scores[tj], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[tj]);
                    }

                    T sumExpSafe = NumOps.Add(sumExp, NumOps.FromDouble(1e-10));
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T weight = NumOps.Divide(expScores[tj], sumExpSafe);
                        _lastAttentionWeights[new[] { bi, hi, ti, tj }] = weight;
                    }

                    // Weighted sum of values
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        T oVal = NumOps.Zero;
                        for (int tj = 0; tj < seqLen; tj++)
                        {
                            T weight = _lastAttentionWeights[new[] { bi, hi, ti, tj }];
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(weight, v[new[] { bi, tj, flatDi }]));
                        }
                        output[new[] { bi, ti, flatDi }] = oVal;
                    }
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
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastReflectedQ = _lastReflectedQ ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastReflectedK = _lastReflectedK ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastAttentionWeights = _lastAttentionWeights ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGate = _lastGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastGateRaw = _lastGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastAttentionOutput = _lastAttentionOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize gradients
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _householderVectorsGradient = new Tensor<T>([_sequenceLength, _numHeads, _headDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Output projection backward
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastAttentionOutput, lastGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Gate backward
        var dAttnOut = Engine.TensorMultiply(dGated, lastGate);
        var dGateSig = Engine.TensorMultiply(dGated, lastAttentionOutput);
        var sigDeriv = Engine.TensorMultiply(lastGate,
            Engine.TensorSubtract(CreateOnesLike(lastGate), lastGate));
        var dGateRaw = Engine.TensorMultiply(dGateSig, sigDeriv);

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });
        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Attention backward -> get dReflectedQ, dReflectedK, dV
        var dReflectedQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dReflectedK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T attnScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int ti = 0; ti < seqLen; ti++)
                {
                    // dV: dV[tj] += attn_weight[ti, tj] * dO[ti]
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T weight = lastAttentionWeights[new[] { bi, hi, ti, tj }];
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            T dO = dAttnOut[new[] { bi, ti, flatDi }];
                            dV[new[] { bi, tj, flatDi }] = NumOps.Add(
                                dV[new[] { bi, tj, flatDi }],
                                NumOps.Multiply(weight, dO));
                        }
                    }

                    // Softmax backward
                    var dAttnWeights = new T[seqLen];
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T dAW = NumOps.Zero;
                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dAW = NumOps.Add(dAW,
                                NumOps.Multiply(dAttnOut[new[] { bi, ti, flatDi }],
                                    lastValue[new[] { bi, tj, flatDi }]));
                        }
                        dAttnWeights[tj] = dAW;
                    }

                    T sumAttnDAttn = NumOps.Zero;
                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T w = lastAttentionWeights[new[] { bi, hi, ti, tj }];
                        sumAttnDAttn = NumOps.Add(sumAttnDAttn, NumOps.Multiply(w, dAttnWeights[tj]));
                    }

                    for (int tj = 0; tj < seqLen; tj++)
                    {
                        T w = lastAttentionWeights[new[] { bi, hi, ti, tj }];
                        T dScore = NumOps.Multiply(w, NumOps.Subtract(dAttnWeights[tj], sumAttnDAttn));
                        dScore = NumOps.Multiply(dScore, attnScale);

                        for (int di = 0; di < _headDimension; di++)
                        {
                            int flatDi = dimStart + di;
                            dReflectedQ[new[] { bi, ti, flatDi }] = NumOps.Add(
                                dReflectedQ[new[] { bi, ti, flatDi }],
                                NumOps.Multiply(dScore, lastReflectedK[new[] { bi, tj, flatDi }]));
                            dReflectedK[new[] { bi, tj, flatDi }] = NumOps.Add(
                                dReflectedK[new[] { bi, tj, flatDi }],
                                NumOps.Multiply(dScore, lastReflectedQ[new[] { bi, ti, flatDi }]));
                        }
                    }
                }
            }
        }

        // Householder backward: H*x = x - 2*(p^T*x)/(||p||^2) * p
        // dQ = H * dReflectedQ (Householder is self-inverse, so backward = forward transform)
        // dP accumulates gradients from both Q and K paths
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int posIndex = Math.Min(t, _sequenceLength - 1);
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Apply Householder backward to dReflectedQ -> dQ
                    var dRQ = new T[_headDimension];
                    var dRK = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        dRQ[di] = dReflectedQ[new[] { bi, t, flatDi }];
                        dRK[di] = dReflectedK[new[] { bi, t, flatDi }];
                    }

                    // H is its own inverse/transpose, so dQ = H * dReflectedQ
                    var dQHead = new T[_headDimension];
                    var dKHead = new T[_headDimension];
                    ApplyHouseholderReflection(dRQ, posIndex, hi, dQHead);
                    ApplyHouseholderReflection(dRK, posIndex, hi, dKHead);

                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        dQ[new[] { bi, t, flatDi }] = dQHead[di];
                        dK[new[] { bi, t, flatDi }] = dKHead[di];
                    }

                    // Householder vector gradient: d/dp [x - 2*(p^T*x)/||p||^2 * p]
                    // Accumulated from both Q and K paths
                    var qHead = new T[_headDimension];
                    var kHead = new T[_headDimension];
                    for (int di = 0; di < _headDimension; di++)
                    {
                        int flatDi = dimStart + di;
                        qHead[di] = lastQuery[new[] { bi, t, flatDi }];
                        kHead[di] = lastKey[new[] { bi, t, flatDi }];
                    }

                    // Compute gradient for p from Q path
                    AccumulateHouseholderGradient(qHead, dRQ, posIndex, hi);
                    // Compute gradient for p from K path
                    AccumulateHouseholderGradient(kHead, dRK, posIndex, hi);
                }
            }
        }

        // Projection weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        // Input gradient
        var dInputTotal = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal,
            Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0])));
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromGate);

        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Accumulates gradient of the Householder vector from one input vector path.
    /// For H*x = x - 2*(p^T*x)/(||p||^2) * p, the gradient w.r.t. p involves:
    /// dp += -2/(||p||^2) * [(dH*x)*x^T + x*(dH*x)^T - 2*(p^T*x)/(||p||^2) * (dH*x)*p^T] * p
    /// Simplified: dp += -2/(||p||^2) * [dOut * (p^T*x) + p * (dOut^T*x)] + correction
    /// </summary>
    private void AccumulateHouseholderGradient(T[] x, T[] dOut, int pos, int headIndex)
    {
        var gradTensor = _householderVectorsGradient;
        if (gradTensor == null) return;

        // Compute p^T * x and ||p||^2
        T dotPX = NumOps.Zero;
        T normPSq = NumOps.Zero;
        for (int di = 0; di < _headDimension; di++)
        {
            T pVal = _householderVectors[new[] { pos, headIndex, di }];
            dotPX = NumOps.Add(dotPX, NumOps.Multiply(pVal, x[di]));
            normPSq = NumOps.Add(normPSq, NumOps.Multiply(pVal, pVal));
        }

        T normPSqSafe = NumOps.Add(normPSq, NumOps.FromDouble(1e-10));
        T twoOverNormSq = NumOps.Divide(NumOps.FromDouble(2.0), normPSqSafe);

        // Compute dOut^T * x
        T dotDOutX = NumOps.Zero;
        for (int di = 0; di < _headDimension; di++)
            dotDOutX = NumOps.Add(dotDOutX, NumOps.Multiply(dOut[di], x[di]));

        // Compute dOut^T * p
        T dotDOutP = NumOps.Zero;
        for (int di = 0; di < _headDimension; di++)
        {
            T pVal = _householderVectors[new[] { pos, headIndex, di }];
            dotDOutP = NumOps.Add(dotDOutP, NumOps.Multiply(dOut[di], pVal));
        }

        // dp_i = -2/||p||^2 * [dOut_i * dotPX + p_i * dotDOutX
        //         - 2 * dotPX * dotDOutP / ||p||^2 * p_i]
        T correctionCoeff = NumOps.Multiply(NumOps.FromDouble(2.0),
            NumOps.Divide(NumOps.Multiply(dotPX, dotDOutP), normPSqSafe));

        for (int di = 0; di < _headDimension; di++)
        {
            T pVal = _householderVectors[new[] { pos, headIndex, di }];
            T grad = NumOps.Multiply(NumOps.Negate(twoOverNormSq),
                NumOps.Subtract(
                    NumOps.Add(
                        NumOps.Multiply(dOut[di], dotPX),
                        NumOps.Multiply(pVal, dotDOutX)),
                    NumOps.Multiply(correctionCoeff, pVal)));

            gradTensor[new[] { pos, headIndex, di }] = NumOps.Add(
                gradTensor[new[] { pos, headIndex, di }], grad);
        }
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
        _householderVectors = Engine.TensorAdd(_householderVectors, Engine.TensorMultiplyScalar(_householderVectorsGradient!, negLR));
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
        _householderVectors,
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
        _lastReflectedQ = null;
        _lastReflectedK = null;
        _lastAttentionWeights = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastAttentionOutput = null;
        _originalInputShape = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _householderVectorsGradient = null;
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
        metadata["SequenceLength"] = _sequenceLength.ToString();
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
    /// Gets the query weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the Householder vectors for external inspection.
    /// </summary>
    public Tensor<T> GetHouseholderVectors() => _householderVectors;
}
