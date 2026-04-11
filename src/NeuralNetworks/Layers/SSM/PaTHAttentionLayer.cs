using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

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
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.AttentionComputation)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class PaTHAttentionLayer<T> : LayerBase<T>
{
    private readonly int _sequenceLength;
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Householder vectors: [sequenceLength, numHeads, headDim]
    // Each position has a per-head Householder reflection vector
    private Tensor<T> _householderVectors;

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
        RegisterTrainableParameter(_householderVectors, PersistentTensorRole.Weights);
        InitializeTensor2D(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
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

        // Step 1: Q, K, V projections
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 2: Apply Householder reflections to Q and K
        var reflectedQ = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        var reflectedK = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

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
        var gateRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Sigmoid(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // Step 5: Gated output
        var gatedOutput = Engine.TensorMultiply(attnOutput, gate);

        // Step 6: Output projection
        var gatedFlat = Engine.Reshape(gatedOutput, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(gatedFlat, _outputProjectionWeights);
        var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, outBias);
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);
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
    /// Standard scaled dot-product attention using reflected Q and K.
    /// </summary>
    private Tensor<T> ComputeAttention(
        Tensor<T> reflectedQ, Tensor<T> reflectedK, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        _lastAttentionWeights = TensorAllocator.Rent<T>(new[] { batchSize, _numHeads, seqLen, seqLen });

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
        var ones = new Tensor<T>(template._shape);
        ones.Fill(NumOps.One);
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

    public override Vector<T> GetParameterGradients()
    {
        if (_queryWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_queryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_keyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_valueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_householderVectorsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputGateWeightsGradient?.ToArray() ?? new T[_outputGateWeights.Length]),
            new Vector<T>(_outputGateBiasGradient?.ToArray() ?? new T[_outputGateBias.Length]),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _householderVectorsGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null; _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

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
