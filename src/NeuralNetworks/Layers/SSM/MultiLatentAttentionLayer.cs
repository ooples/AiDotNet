using AiDotNet.Autodiff;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Multi-Latent Attention (MLA) layer from DeepSeek-V2 (Aixin Liu et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// Multi-Latent Attention compresses the KV cache via low-rank factorization, dramatically reducing
/// memory usage during inference. Instead of caching full-dimensional K and V for every token, MLA
/// caches a much smaller latent vector c_t, from which K and V are reconstructed on the fly.
/// </para>
/// <para>
/// The architecture:
/// <code>
///   1. Compress input into low-rank latent: c_t = W_c * x_t  (latentDim &lt;&lt; modelDim)
///   2. Reconstruct K and V from the latent:
///      K_t = W_k * c_t, V_t = W_v * c_t
///   3. Compute Q from input (full rank): Q_t = W_q * x_t
///   4. Multi-head attention with reconstructed K, V:
///      score[t,s] = Q_t^T * K_s / sqrt(headDim)
///      attn[t,s] = softmax(score) over s &lt;= t (causal)
///      O_t = sum_s attn[t,s] * V_s
///   5. Output gate and projection
/// </code>
/// </para>
/// <para>
/// The key insight is KV cache compression: during inference, you only need to store c_t (latentDim per token)
/// instead of K and V (2 * modelDim per token). When latentDim = modelDim/4, this yields an 8x reduction
/// in KV cache memory, which is the primary bottleneck for long-context LLM serving.
/// </para>
/// <para><b>For Beginners:</b> MLA is a memory-efficient attention mechanism used in DeepSeek-V2.
///
/// Standard attention is like keeping a complete notebook for every word you've read:
/// - You write the full Key (what this word is about) and Value (what it says) for every word
/// - When answering a question (Query), you look up all Key-Value pairs
/// - For a 100K word document, that's a LOT of notebook pages
///
/// MLA is like keeping compressed sticky notes instead:
/// - For each word, you write just a short summary (the latent c_t)
/// - When you need the full Key or Value, you expand the summary on the fly
/// - You still get nearly the same answer quality, but use far less paper
///
/// The "Multi-Latent" name comes from having multiple attention heads, each with its own
/// latent compression. This is what enables DeepSeek-V2 to handle very long contexts efficiently.
/// </para>
/// <para>
/// <b>Reference:</b> Aixin Liu et al., "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
/// Language Model", 2024. https://arxiv.org/abs/2405.04434
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class MultiLatentAttentionLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _latentDimension;

    // Latent compression: [modelDim, latentDim]
    private Tensor<T> _compressWeights;
    private Tensor<T> _compressBias;

    // K, V reconstruction from latent: [latentDim, modelDim]
    private Tensor<T> _keyUpWeights;
    private Tensor<T> _valueUpWeights;

    // Q projection (full rank): [modelDim, modelDim]
    private Tensor<T> _queryWeights;

    // Output gate: [modelDim, modelDim]
    private Tensor<T> _outputGateWeights;
    private Tensor<T> _outputGateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastLatent;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastAttnWeights;
    private Tensor<T>? _lastAttnOutput;
    private Tensor<T>? _lastOutputGate;
    private Tensor<T>? _lastOutputGateRaw;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _compressWeightsGradient;
    private Tensor<T>? _compressBiasGradient;
    private Tensor<T>? _keyUpWeightsGradient;
    private Tensor<T>? _valueUpWeightsGradient;
    private Tensor<T>? _queryWeightsGradient;
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
    /// Gets the latent dimension for KV cache compression.
    /// </summary>
    public int LatentDimension => _latentDimension;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _compressWeights.Length + _compressBias.Length +
        _keyUpWeights.Length + _valueUpWeights.Length +
        _queryWeights.Length +
        _outputGateWeights.Length + _outputGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new Multi-Latent Attention (MLA) layer.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token's representation vector.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head attends to different aspects of the input.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="latentDimension">
    /// Dimension of the compressed latent for KV cache. Default: 64.
    /// <para><b>For Beginners:</b> Smaller values mean more compression (less memory) but
    /// potentially less accurate attention. DeepSeek-V2 uses latentDim = modelDim/4 or similar.
    /// This is the key parameter that controls the memory-quality tradeoff.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public MultiLatentAttentionLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int latentDimension = 64,
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
        if (latentDimension <= 0)
            throw new ArgumentException($"Latent dimension ({latentDimension}) must be positive.", nameof(latentDimension));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _latentDimension = latentDimension;

        _compressWeights = new Tensor<T>([modelDimension, latentDimension]);
        _compressBias = new Tensor<T>([latentDimension]);
        _keyUpWeights = new Tensor<T>([latentDimension, modelDimension]);
        _valueUpWeights = new Tensor<T>([latentDimension, modelDimension]);
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_compressWeights);
        _compressBias.Fill(NumOps.Zero);
        InitializeTensor2D(_keyUpWeights);
        InitializeTensor2D(_valueUpWeights);
        InitializeTensor2D(_queryWeights);
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

        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);

        // Step 1: Compress input to latent c_t = W_c * x_t + b_c
        var latentFlat = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _compressWeights),
            _compressBias.Reshape(1, _latentDimension));
        var latent = latentFlat.Reshape(batchSize, seqLen, _latentDimension);
        _lastLatent = latent;

        // Step 2: Reconstruct K and V from latent
        var kFlat = Engine.TensorMatMul(latentFlat, _keyUpWeights);
        var vFlat = Engine.TensorMatMul(latentFlat, _valueUpWeights);
        var k = kFlat.Reshape(batchSize, seqLen, _modelDimension);
        var v = vFlat.Reshape(batchSize, seqLen, _modelDimension);
        _lastKey = k;
        _lastValue = v;

        // Step 3: Q projection (full rank)
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;

        // Step 4: Output gate
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _outputGateWeights),
            _outputGateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Swish(gateRaw);
        _lastOutputGate = gate;
        _lastOutputGateRaw = gateRaw;

        // Step 5: Causal multi-head attention
        var attnOutput = CausalMultiHeadAttention(q, k, v, batchSize, seqLen);
        _lastAttnOutput = attnOutput;

        // Step 6: Gated output
        var gatedOutput = Engine.TensorMultiply(attnOutput, gate);

        // Step 7: Output projection
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
    /// Causal multi-head attention: softmax(Q*K^T / sqrt(d_k)) * V with causal mask.
    /// </summary>
    private Tensor<T> CausalMultiHeadAttention(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var allAttnWeights = new Tensor<T>(new[] { batchSize, _numHeads, seqLen, seqLen });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int tq = 0; tq < seqLen; tq++)
                {
                    // Compute attention scores for this query position (causal: s <= tq)
                    var scores = new T[tq + 1];
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);

                    for (int tk = 0; tk <= tq; tk++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, tq, flatD }],
                                    k[new[] { bi, tk, flatD }]));
                        }
                        scores[tk] = NumOps.Multiply(dot, scale);
                        if (NumOps.ToDouble(scores[tk]) > NumOps.ToDouble(maxScore))
                            maxScore = scores[tk];
                    }

                    // Softmax
                    T sumExp = NumOps.Zero;
                    var expScores = new T[tq + 1];
                    for (int tk = 0; tk <= tq; tk++)
                    {
                        expScores[tk] = NumOps.Exp(NumOps.Subtract(scores[tk], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[tk]);
                    }
                    T sumExpInv = NumOps.Divide(NumOps.One,
                        NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));

                    // Weighted sum of values
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T oVal = NumOps.Zero;
                        for (int tk = 0; tk <= tq; tk++)
                        {
                            T attnW = NumOps.Multiply(expScores[tk], sumExpInv);
                            allAttnWeights[new[] { bi, hi, tq, tk }] = attnW;
                            oVal = NumOps.Add(oVal,
                                NumOps.Multiply(attnW, v[new[] { bi, tk, flatD }]));
                        }
                        output[new[] { bi, tq, flatD }] = oVal;
                    }
                }
            }
        }

        _lastAttnWeights = allAttnWeights;
        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var lastInput = _lastInput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutput = _lastOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastLatent = _lastLatent ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastQuery = _lastQuery ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastKey = _lastKey ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastValue = _lastValue ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastAttnWeights = _lastAttnWeights ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastAttnOutput = _lastAttnOutput ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGate = _lastOutputGate ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");
        var lastOutputGateRaw = _lastOutputGateRaw ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = lastInput.Shape[0];
        int seqLen = lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(lastOutput, grad3D);

        // Initialize gradients
        _compressWeightsGradient = new Tensor<T>([_modelDimension, _latentDimension]);
        _compressBiasGradient = new Tensor<T>([_latentDimension]);
        _keyUpWeightsGradient = new Tensor<T>([_latentDimension, _modelDimension]);
        _valueUpWeightsGradient = new Tensor<T>([_latentDimension, _modelDimension]);
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputGateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Step 7 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        var gatedFlat = Engine.TensorMultiply(lastAttnOutput, lastOutputGate)
            .Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionWeightsGradient = Engine.TensorMatMul(gatedFlat.Transpose([1, 0]), gradFlat);

        var dGated = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 6 backward: gating
        var dAttnOutput = Engine.TensorMultiply(dGated, lastOutputGate);
        var dGateSwish = Engine.TensorMultiply(dGated, lastAttnOutput);
        var dGateRaw = Engine.TensorMultiply(dGateSwish, ComputeSiLUDerivative(lastOutputGateRaw));

        var inputFlat = lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _outputGateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _outputGateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _outputGateWeights.Transpose([1, 0]));

        // Step 5 backward: causal multi-head attention
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int tq = 0; tq < seqLen; tq++)
                {
                    // dAttnWeights from dOutput * V^T
                    var dAttnW = new T[tq + 1];
                    for (int tk = 0; tk <= tq; tk++)
                    {
                        T dAW = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            T dO = dAttnOutput[new[] { bi, tq, flatD }];
                            T vVal = lastValue[new[] { bi, tk, flatD }];
                            dAW = NumOps.Add(dAW, NumOps.Multiply(dO, vVal));

                            // dV += attnWeight * dOutput
                            T aw = lastAttnWeights[new[] { bi, hi, tq, tk }];
                            dV[new[] { bi, tk, flatD }] = NumOps.Add(
                                dV[new[] { bi, tk, flatD }],
                                NumOps.Multiply(aw, dO));
                        }
                        dAttnW[tk] = dAW;
                    }

                    // Backward through softmax: dScore = attn * (dAttnW - dot(attn, dAttnW))
                    T dotAW = NumOps.Zero;
                    for (int tk = 0; tk <= tq; tk++)
                    {
                        T aw = lastAttnWeights[new[] { bi, hi, tq, tk }];
                        dotAW = NumOps.Add(dotAW, NumOps.Multiply(aw, dAttnW[tk]));
                    }

                    for (int tk = 0; tk <= tq; tk++)
                    {
                        T aw = lastAttnWeights[new[] { bi, hi, tq, tk }];
                        T dScore = NumOps.Multiply(aw, NumOps.Subtract(dAttnW[tk], dotAW));
                        T dScoreScaled = NumOps.Multiply(dScore, scale);

                        // dQ += dScore * K, dK += dScore * Q
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dQ[new[] { bi, tq, flatD }] = NumOps.Add(
                                dQ[new[] { bi, tq, flatD }],
                                NumOps.Multiply(dScoreScaled, lastKey[new[] { bi, tk, flatD }]));
                            dK[new[] { bi, tk, flatD }] = NumOps.Add(
                                dK[new[] { bi, tk, flatD }],
                                NumOps.Multiply(dScoreScaled, lastQuery[new[] { bi, tq, flatD }]));
                        }
                    }
                }
            }
        }

        // Q weight gradients
        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);

        // K and V gradients go back through up-projection then compression
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        var latentFlat = lastLatent.Reshape(batchSize * seqLen, _latentDimension);
        _keyUpWeightsGradient = Engine.TensorMatMul(latentFlat.Transpose([1, 0]), dKFlat);
        _valueUpWeightsGradient = Engine.TensorMatMul(latentFlat.Transpose([1, 0]), dVFlat);

        // dLatent from K and V paths
        var dLatentFromK = Engine.TensorMatMul(dKFlat, _keyUpWeights.Transpose([1, 0]));
        var dLatentFromV = Engine.TensorMatMul(dVFlat, _valueUpWeights.Transpose([1, 0]));
        var dLatentFlat = Engine.TensorAdd(dLatentFromK, dLatentFromV);

        // Compression weight gradients
        _compressWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dLatentFlat);
        _compressBiasGradient = Engine.ReduceSum(
            dLatentFlat.Reshape(batchSize, seqLen, _latentDimension), new int[] { 0, 1 });

        // Input gradient from all paths
        var dInput = Engine.TensorAdd(dInputFromGate,
            Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0])));
        dInput = Engine.TensorAdd(dInput,
            Engine.TensorMatMul(dLatentFlat, _compressWeights.Transpose([1, 0])));

        var dInput3D = dInput.Reshape(batchSize, seqLen, _modelDimension);

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
        if (_compressWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        var compressBiasGradient = _compressBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var keyUpWeightsGradient = _keyUpWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var valueUpWeightsGradient = _valueUpWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var queryWeightsGradient = _queryWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateWeightsGradient = _outputGateWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputGateBiasGradient = _outputGateBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionWeightsGradient = _outputProjectionWeightsGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");
        var outputProjectionBiasGradient = _outputProjectionBiasGradient ?? throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _compressWeights = Engine.TensorAdd(_compressWeights, Engine.TensorMultiplyScalar(_compressWeightsGradient, negLR));
        _compressBias = Engine.TensorAdd(_compressBias, Engine.TensorMultiplyScalar(compressBiasGradient, negLR));
        _keyUpWeights = Engine.TensorAdd(_keyUpWeights, Engine.TensorMultiplyScalar(keyUpWeightsGradient, negLR));
        _valueUpWeights = Engine.TensorAdd(_valueUpWeights, Engine.TensorMultiplyScalar(valueUpWeightsGradient, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(queryWeightsGradient, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(outputGateWeightsGradient, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(outputGateBiasGradient, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(outputProjectionWeightsGradient, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(outputProjectionBiasGradient, negLR));
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
        _compressWeights, _compressBias,
        _keyUpWeights, _valueUpWeights,
        _queryWeights,
        _outputGateWeights, _outputGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastLatent = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastAttnWeights = null;
        _lastAttnOutput = null;
        _lastOutputGate = null;
        _lastOutputGateRaw = null;
        _originalInputShape = null;
        _compressWeightsGradient = null;
        _compressBiasGradient = null;
        _keyUpWeightsGradient = null;
        _valueUpWeightsGradient = null;
        _queryWeightsGradient = null;
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
        metadata["LatentDimension"] = _latentDimension.ToString();
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
    /// Gets the compression weights for external inspection.
    /// </summary>
    public Tensor<T> GetCompressionWeights() => _compressWeights;
}
