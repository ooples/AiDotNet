using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the BASED (Bidirectional Attention with Sliding-window and Expanded features) layer from
/// "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff" (Arora et al., 2024).
/// </summary>
/// <remarks>
/// <para>
/// BASED is a hybrid architecture that combines two attention mechanisms to get the best of both worlds:
/// <code>
///   1. Linear Attention (global, O(n) complexity):
///      - Uses Taylor expansion feature map: phi(x) = [1, x, x (tensor) x / sqrt(2)]
///      - Maintains a running state matrix S and normalizer z
///      - For each step: S += phi(k) * v^T, z += phi(k), output = (S * phi(q)) / (z * phi(q))
///      - Provides global context across the entire sequence
///
///   2. Sliding Window Attention (local, O(n*w) complexity):
///      - Standard causal softmax attention restricted to a window of size w
///      - For each position t, attends only to positions [t-w+1, t]
///      - Provides precise local recall for nearby tokens
///
///   3. Combination:
///      - output = alpha * linear_attention_output + (1 - alpha) * window_attention_output
///      - A learned mixing gate alpha controls the blend per position
///      - Final output projection maps back to model dimension
/// </code>
/// </para>
/// <para>
/// The key insight is that linear attention is fast (O(n)) but struggles with recall-intensive tasks
/// because the feature map approximation loses precision. Adding a small sliding window (w=64 or 128)
/// cheaply fixes this: the window handles precise local recall while linear attention covers global context.
/// The total complexity is O(n * (d + w)) which is still linear in sequence length for fixed window size.
/// </para>
/// <para><b>For Beginners:</b> BASED combines two ways of paying attention to solve a key tradeoff:
///
/// Imagine you're reading a long book and need to answer questions about it:
/// - Linear attention is like having a summary of everything you've read so far. It's fast to update
///   and gives you the gist, but you might miss exact details (like a specific name on page 42).
/// - Sliding window attention is like being able to flip back a few pages to check exact wording.
///   It's precise but only covers recent pages.
///
/// BASED combines both: the summary (linear attention) gives global context, and the "flip back"
/// (sliding window) handles precise recall of recent information. Together, they match the quality
/// of full Transformer attention while being much more efficient for long sequences.
///
/// The Taylor expansion feature map phi(x) = [1, x, x*x/sqrt(2)] is a mathematical trick that
/// approximates the softmax kernel exp(q*k) using polynomial terms. The constant 1 captures baseline
/// attention, x captures first-order similarity, and x*x/sqrt(2) captures quadratic interactions.
/// </para>
/// <para>
/// <b>Reference:</b> Arora et al., "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff", 2024.
/// https://arxiv.org/abs/2402.18668
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
public partial class BASEDLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _windowSize;
    private readonly int _featureExpansion;
    private readonly int _expandedDimension;

    // Linear attention Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _linearQueryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _linearKeyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _linearValueWeights;

    // Sliding window attention Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _windowQueryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _windowKeyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _windowValueWeights;

    // Feature map scale parameter for Taylor expansion: [numHeads, headDim]
    private Tensor<T> _featureMapScale;

    // Mixing gate: learned alpha per head [modelDim, numHeads]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _mixingGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _mixingGateBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastLinearQuery;
    private Tensor<T>? _lastLinearKey;
    private Tensor<T>? _lastLinearValue;
    private Tensor<T>? _lastWindowQuery;
    private Tensor<T>? _lastWindowKey;
    private Tensor<T>? _lastWindowValue;
    private Tensor<T>? _lastLinearFeatureQ;
    private Tensor<T>? _lastLinearFeatureK;
    private Tensor<T>? _lastLinearOutput;
    private Tensor<T>? _lastWindowOutput;
    private Tensor<T>? _lastMixingAlpha;
    private Tensor<T>? _lastMixingAlphaRaw;
    private Tensor<T>? _lastCombinedOutput;
    private Tensor<T>? _lastWindowScores;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _linearQueryWeightsGradient;
    private Tensor<T>? _linearKeyWeightsGradient;
    private Tensor<T>? _linearValueWeightsGradient;
    private Tensor<T>? _windowQueryWeightsGradient;
    private Tensor<T>? _windowKeyWeightsGradient;
    private Tensor<T>? _windowValueWeightsGradient;
    private Tensor<T>? _featureMapScaleGradient;
    private Tensor<T>? _mixingGateWeightsGradient;
    private Tensor<T>? _mixingGateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

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
    /// Gets the sliding window size for local attention.
    /// </summary>
    public int WindowSize => _windowSize;

    /// <summary>
    /// Gets the feature expansion factor for the Taylor feature map.
    /// </summary>
    public int FeatureExpansion => _featureExpansion;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _linearQueryWeights.Length + _linearKeyWeights.Length + _linearValueWeights.Length +
        _windowQueryWeights.Length + _windowKeyWeights.Length + _windowValueWeights.Length +
        _featureMapScale.Length +
        _mixingGateWeights.Length + _mixingGateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

    /// <summary>
    /// Creates a new BASED layer that combines linear attention with sliding window attention.
    /// </summary>
    /// <param name="sequenceLength">
    /// Maximum sequence length.
    /// </param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The size of each token's representation vector. Larger values
    /// can capture more information but use more memory and compute.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head can focus on different aspects of the input simultaneously.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="windowSize">
    /// Sliding window size for local attention. Default: 64.
    /// <para><b>For Beginners:</b> How many recent tokens each position can attend to with full precision.
    /// Larger windows improve recall but increase compute. The paper finds 64-128 works well.</para>
    /// </param>
    /// <param name="featureExpansion">
    /// Feature expansion factor for Taylor map. Default: 2.
    /// <para><b>For Beginners:</b> Controls the size of the polynomial feature map. Higher values
    /// give better approximation of softmax but use more memory. The default of 2 uses the
    /// second-order Taylor expansion phi(x) = [1, x, x*x/sqrt(2)].</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public BASEDLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int windowSize = 64,
        int featureExpansion = 2,
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
        if (windowSize <= 0)
            throw new ArgumentException($"Window size ({windowSize}) must be positive.", nameof(windowSize));
        if (featureExpansion <= 0)
            throw new ArgumentException($"Feature expansion ({featureExpansion}) must be positive.", nameof(featureExpansion));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _windowSize = windowSize;
        _featureExpansion = featureExpansion;
        // Expanded dimension: 1 (constant) + headDim (linear) + headDim * (headDim + 1) / 2 (quadratic, upper triangle)
        // For simplicity with the featureExpansion parameter, we use headDim * featureExpansion
        _expandedDimension = _headDimension * featureExpansion;

        // Linear attention projections
        _linearQueryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _linearKeyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _linearValueWeights = new Tensor<T>([modelDimension, modelDimension]);

        // Window attention projections
        _windowQueryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _windowKeyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _windowValueWeights = new Tensor<T>([modelDimension, modelDimension]);

        // Feature map learnable scale: per-head scaling for Taylor expansion
        _featureMapScale = new Tensor<T>([numHeads, _headDimension]);

        // Mixing gate: projects input to per-head mixing coefficient
        _mixingGateWeights = new Tensor<T>([modelDimension, numHeads]);
        _mixingGateBias = new Tensor<T>([numHeads]);

        // Output projection
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor2D(_linearQueryWeights);
        InitializeTensor2D(_linearKeyWeights);
        InitializeTensor2D(_linearValueWeights);
        InitializeTensor2D(_windowQueryWeights);
        InitializeTensor2D(_windowKeyWeights);
        InitializeTensor2D(_windowValueWeights);

        // Feature map scale initialized to 1.0 (identity scaling)
        for (int i = 0; i < _featureMapScale.Length; i++)
            _featureMapScale[i] = NumOps.One;

        InitializeTensor2D(_mixingGateWeights);
        // Bias initialized to 0.5 so sigmoid(0.5) ~ 0.62, slightly favoring linear attention initially
        for (int i = 0; i < _mixingGateBias.Length; i++)
            _mixingGateBias[i] = NumOps.FromDouble(0.5);

        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
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

        // Step 1: Compute projections for both attention mechanisms
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });

        // Linear attention Q, K, V
        var linQ = Engine.Reshape(Engine.TensorMatMul(inputFlat, _linearQueryWeights), new[] { batchSize, seqLen, _modelDimension });
        var linK = Engine.Reshape(Engine.TensorMatMul(inputFlat, _linearKeyWeights), new[] { batchSize, seqLen, _modelDimension });
        var linV = Engine.Reshape(Engine.TensorMatMul(inputFlat, _linearValueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastLinearQuery = linQ;
        _lastLinearKey = linK;
        _lastLinearValue = linV;

        // Window attention Q, K, V
        var winQ = Engine.Reshape(Engine.TensorMatMul(inputFlat, _windowQueryWeights), new[] { batchSize, seqLen, _modelDimension });
        var winK = Engine.Reshape(Engine.TensorMatMul(inputFlat, _windowKeyWeights), new[] { batchSize, seqLen, _modelDimension });
        var winV = Engine.Reshape(Engine.TensorMatMul(inputFlat, _windowValueWeights), new[] { batchSize, seqLen, _modelDimension });
        _lastWindowQuery = winQ;
        _lastWindowKey = winK;
        _lastWindowValue = winV;

        // Step 2: Compute mixing gate alpha
        var alphaRaw = Engine.Reshape(Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _mixingGateWeights),
            Engine.Reshape(_mixingGateBias, new[] { 1, _numHeads })), new[] { batchSize, seqLen, _numHeads });
        var alpha = Engine.Sigmoid(alphaRaw);
        _lastMixingAlphaRaw = alphaRaw;
        _lastMixingAlpha = alpha;

        // Step 3: Linear attention with Taylor feature map
        var linearOutput = LinearAttentionForward(linQ, linK, linV, batchSize, seqLen);
        _lastLinearOutput = linearOutput;

        // Step 4: Sliding window causal attention
        var windowOutput = SlidingWindowAttentionForward(winQ, winK, winV, batchSize, seqLen);
        _lastWindowOutput = windowOutput;

        // Step 5: Combine linear and window attention using learned alpha
        var combined = CombineAttentionOutputs(linearOutput, windowOutput, alpha, batchSize, seqLen);
        _lastCombinedOutput = combined;

        // Step 6: Output projection
        var combinedFlat = Engine.Reshape(combined, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(combinedFlat, _outputProjectionWeights);
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
    /// Applies the Taylor expansion feature map: phi(x) = [x, x * x / sqrt(2)] (scaled by learned parameters).
    /// The constant term is handled implicitly in the state accumulation.
    /// </summary>
    private void ApplyFeatureMap(
        T[] headVector, int headIndex, T[] featureOutput)
    {
        // The Taylor expansion of exp(q*k) around 0 gives: 1 + q*k + (q*k)^2/2 + ...
        // For the feature map: phi(x) = [1, x, x (tensor) x / sqrt(2)]
        // We use a simplified version: phi(x) = [x * scale, x * x * scale / sqrt(2)]
        // where scale is learned per-head

        T sqrtTwo = NumOps.Sqrt(NumOps.FromDouble(2.0));
        int halfExpanded = _headDimension;

        for (int d = 0; d < _headDimension; d++)
        {
            T scaleVal = _featureMapScale[new[] { headIndex, d }];
            T scaled = NumOps.Multiply(headVector[d], scaleVal);

            // First-order term: x * scale
            featureOutput[d] = scaled;

            // Second-order term: (x * scale)^2 / sqrt(2)
            if (_featureExpansion >= 2 && d < halfExpanded)
            {
                featureOutput[_headDimension + d] = NumOps.Divide(
                    NumOps.Multiply(scaled, scaled), sqrtTwo);
            }
        }
    }

    /// <summary>
    /// Linear attention forward pass using the causal linear attention recurrence with Taylor feature map.
    /// For each position t: S_t = S_{t-1} + phi(k_t) * v_t^T, z_t = z_{t-1} + phi(k_t)
    /// output_t = (S_t * phi(q_t)) / (z_t^T * phi(q_t) + epsilon)
    /// </summary>
    private Tensor<T> LinearAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T epsilon = NumOps.FromDouble(1e-6);
        T keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Per-head feature dimension
        int featureDim = _expandedDimension;

        // Allocate feature map buffers
        var phiQ = new T[featureDim];
        var phiK = new T[featureDim];

        // State and normalizer per batch and head
        // S: [featureDim, headDim] - the running key-value state
        // z: [featureDim] - the running normalizer
        var stateS = new T[batchSize, _numHeads, featureDim, _headDimension];
        var stateZ = new T[batchSize, _numHeads, featureDim];

        // Initialize states to zero
        for (int bi = 0; bi < batchSize; bi++)
            for (int hi = 0; hi < _numHeads; hi++)
            {
                for (int fi = 0; fi < featureDim; fi++)
                {
                    stateZ[bi, hi, fi] = NumOps.Zero;
                    for (int di = 0; di < _headDimension; di++)
                        stateS[bi, hi, fi, di] = NumOps.Zero;
                }
            }

        // Cache feature maps for backward pass
        var featureQCache = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads, featureDim });
        var featureKCache = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads, featureDim });
        _lastLinearFeatureQ = featureQCache;
        _lastLinearFeatureK = featureKCache;

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Extract head vectors and apply key scaling
                    var qHead = new T[_headDimension];
                    var kHead = new T[_headDimension];
                    var vHead = new T[_headDimension];

                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        qHead[d] = NumOps.Multiply(q[new[] { bi, t, flatD }], keyScale);
                        kHead[d] = NumOps.Multiply(k[new[] { bi, t, flatD }], keyScale);
                        vHead[d] = v[new[] { bi, t, flatD }];
                    }

                    // Apply Taylor feature map
                    for (int i = 0; i < featureDim; i++)
                    {
                        phiQ[i] = NumOps.Zero;
                        phiK[i] = NumOps.Zero;
                    }
                    ApplyFeatureMap(qHead, hi, phiQ);
                    ApplyFeatureMap(kHead, hi, phiK);

                    // Cache features
                    for (int fi = 0; fi < featureDim; fi++)
                    {
                        featureQCache[new[] { bi, t, hi, fi }] = phiQ[fi];
                        featureKCache[new[] { bi, t, hi, fi }] = phiK[fi];
                    }

                    // Update state: S += phi(k) * v^T
                    for (int fi = 0; fi < featureDim; fi++)
                    {
                        for (int di = 0; di < _headDimension; di++)
                        {
                            stateS[bi, hi, fi, di] = NumOps.Add(
                                stateS[bi, hi, fi, di],
                                NumOps.Multiply(phiK[fi], vHead[di]));
                        }
                        // Update normalizer: z += phi(k)
                        stateZ[bi, hi, fi] = NumOps.Add(stateZ[bi, hi, fi], phiK[fi]);
                    }

                    // Compute output: o = S^T * phi(q), norm = z^T * phi(q)
                    T norm = epsilon;
                    for (int fi = 0; fi < featureDim; fi++)
                        norm = NumOps.Add(norm, NumOps.Multiply(stateZ[bi, hi, fi], phiQ[fi]));

                    for (int di = 0; di < _headDimension; di++)
                    {
                        T oVal = NumOps.Zero;
                        for (int fi = 0; fi < featureDim; fi++)
                            oVal = NumOps.Add(oVal, NumOps.Multiply(stateS[bi, hi, fi, di], phiQ[fi]));

                        int flatD = dimStart + di;
                        output[new[] { bi, t, flatD }] = NumOps.Divide(oVal, norm);
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Sliding window causal attention: standard softmax attention restricted to a local window.
    /// For each position t, attends only to positions max(0, t-w+1) through t.
    /// </summary>
    private Tensor<T> SlidingWindowAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        // Store attention scores for backward pass: [batch, seqLen, numHeads, windowSize]
        var scores = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _numHeads, _windowSize });
        _lastWindowScores = scores;

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int windowStart = Math.Max(0, t - _windowSize + 1);
                int windowLen = t - windowStart + 1;

                for (int hi = 0; hi < _numHeads; hi++)
                {
                    int dimStart = hi * _headDimension;

                    // Compute attention scores within the window
                    var rawScores = new T[windowLen];
                    for (int wi = 0; wi < windowLen; wi++)
                    {
                        int srcT = windowStart + wi;
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, t, flatD }], k[new[] { bi, srcT, flatD }]));
                        }
                        rawScores[wi] = NumOps.Multiply(dot, scale);
                    }

                    // Softmax over the window
                    T maxScore = rawScores[0];
                    for (int wi = 1; wi < windowLen; wi++)
                    {
                        if (NumOps.GreaterThan(rawScores[wi], maxScore))
                            maxScore = rawScores[wi];
                    }

                    var expScores = new T[windowLen];
                    T sumExp = NumOps.Zero;
                    for (int wi = 0; wi < windowLen; wi++)
                    {
                        expScores[wi] = NumOps.Exp(NumOps.Subtract(rawScores[wi], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[wi]);
                    }

                    T sumExpInv = NumOps.Divide(NumOps.One, NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));
                    var attnWeights = new T[windowLen];
                    for (int wi = 0; wi < windowLen; wi++)
                    {
                        attnWeights[wi] = NumOps.Multiply(expScores[wi], sumExpInv);
                        // Store for backward pass
                        if (wi < _windowSize)
                            scores[new[] { bi, t, hi, wi }] = attnWeights[wi];
                    }

                    // Weighted sum of values
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T val = NumOps.Zero;
                        for (int wi = 0; wi < windowLen; wi++)
                        {
                            int srcT = windowStart + wi;
                            val = NumOps.Add(val,
                                NumOps.Multiply(attnWeights[wi], v[new[] { bi, srcT, flatD }]));
                        }
                        output[new[] { bi, t, flatD }] = val;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Combines linear attention and window attention outputs using learned per-head mixing gate.
    /// output = alpha * linear_output + (1 - alpha) * window_output
    /// Alpha is expanded from [batch, seq, numHeads] to [batch, seq, modelDim] by repeating per head.
    /// </summary>
    private Tensor<T> CombineAttentionOutputs(
        Tensor<T> linearOutput, Tensor<T> windowOutput,
        Tensor<T> alpha, int batchSize, int seqLen)
    {
        var combined = TensorAllocator.Rent<T>(new[] { batchSize, seqLen, _modelDimension });

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int hi = 0; hi < _numHeads; hi++)
                {
                    T alphaVal = alpha[new[] { bi, t, hi }];
                    T oneMinusAlpha = NumOps.Subtract(NumOps.One, alphaVal);
                    int dimStart = hi * _headDimension;

                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T linVal = linearOutput[new[] { bi, t, flatD }];
                        T winVal = windowOutput[new[] { bi, t, flatD }];
                        combined[new[] { bi, t, flatD }] = NumOps.Add(
                            NumOps.Multiply(alphaVal, linVal),
                            NumOps.Multiply(oneMinusAlpha, winVal));
                    }
                }
            }
        }

        return combined;
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
        if (_linearQueryWeightsGradient == null || _linearKeyWeightsGradient == null || _linearValueWeightsGradient == null ||
            _windowQueryWeightsGradient == null || _windowKeyWeightsGradient == null || _windowValueWeightsGradient == null ||
            _featureMapScaleGradient == null || _mixingGateWeightsGradient == null || _mixingGateBiasGradient == null ||
            _outputProjectionWeightsGradient == null || _outputProjectionBiasGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _linearQueryWeights = Engine.TensorAdd(_linearQueryWeights, Engine.TensorMultiplyScalar(_linearQueryWeightsGradient, negLR));
        _linearKeyWeights = Engine.TensorAdd(_linearKeyWeights, Engine.TensorMultiplyScalar(_linearKeyWeightsGradient, negLR));
        _linearValueWeights = Engine.TensorAdd(_linearValueWeights, Engine.TensorMultiplyScalar(_linearValueWeightsGradient, negLR));
        _windowQueryWeights = Engine.TensorAdd(_windowQueryWeights, Engine.TensorMultiplyScalar(_windowQueryWeightsGradient, negLR));
        _windowKeyWeights = Engine.TensorAdd(_windowKeyWeights, Engine.TensorMultiplyScalar(_windowKeyWeightsGradient, negLR));
        _windowValueWeights = Engine.TensorAdd(_windowValueWeights, Engine.TensorMultiplyScalar(_windowValueWeightsGradient, negLR));
        _featureMapScale = Engine.TensorAdd(_featureMapScale, Engine.TensorMultiplyScalar(_featureMapScaleGradient, negLR));
        _mixingGateWeights = Engine.TensorAdd(_mixingGateWeights, Engine.TensorMultiplyScalar(_mixingGateWeightsGradient, negLR));
        _mixingGateBias = Engine.TensorAdd(_mixingGateBias, Engine.TensorMultiplyScalar(_mixingGateBiasGradient, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_linearQueryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linearKeyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_linearValueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_windowQueryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_windowKeyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_windowValueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_mixingGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_mixingGateBias, PersistentTensorRole.Biases);
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
        _linearQueryWeights, _linearKeyWeights, _linearValueWeights,
        _windowQueryWeights, _windowKeyWeights, _windowValueWeights,
        _featureMapScale,
        _mixingGateWeights, _mixingGateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_linearQueryWeightsGradient == null) return new Vector<T>(ParameterCount);
        return Vector<T>.Concatenate(
            new Vector<T>(_linearQueryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_linearKeyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_linearValueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_windowQueryWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_windowKeyWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_windowValueWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_featureMapScaleGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_mixingGateWeightsGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_mixingGateBiasGradient?.ToArray() ?? Array.Empty<T>()),
            new Vector<T>(_outputProjectionWeightsGradient?.ToArray() ?? new T[_outputProjectionWeights.Length]),
            new Vector<T>(_outputProjectionBiasGradient?.ToArray() ?? new T[_outputProjectionBias.Length]));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _linearQueryWeightsGradient = null; _linearKeyWeightsGradient = null; _linearValueWeightsGradient = null; _windowQueryWeightsGradient = null; _windowKeyWeightsGradient = null; _windowValueWeightsGradient = null; _featureMapScaleGradient = null; _mixingGateWeightsGradient = null; _mixingGateBiasGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastLinearQuery = null;
        _lastLinearKey = null;
        _lastLinearValue = null;
        _lastWindowQuery = null;
        _lastWindowKey = null;
        _lastWindowValue = null;
        _lastLinearFeatureQ = null;
        _lastLinearFeatureK = null;
        _lastLinearOutput = null;
        _lastWindowOutput = null;
        _lastMixingAlpha = null;
        _lastMixingAlphaRaw = null;
        _lastCombinedOutput = null;
        _lastWindowScores = null;
        _originalInputShape = null;
        _linearQueryWeightsGradient = null;
        _linearKeyWeightsGradient = null;
        _linearValueWeightsGradient = null;
        _windowQueryWeightsGradient = null;
        _windowKeyWeightsGradient = null;
        _windowValueWeightsGradient = null;
        _featureMapScaleGradient = null;
        _mixingGateWeightsGradient = null;
        _mixingGateBiasGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        metadata["WindowSize"] = _windowSize.ToString();
        metadata["FeatureExpansion"] = _featureExpansion.ToString();
        metadata["ExpandedDimension"] = _expandedDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the linear attention query weights for external inspection.
    /// </summary>
    public Tensor<T> GetLinearQueryWeights() => _linearQueryWeights;

    /// <summary>
    /// Gets the window attention query weights for external inspection.
    /// </summary>
    public Tensor<T> GetWindowQueryWeights() => _windowQueryWeights;

    /// <summary>
    /// Gets the feature map scale parameters for external inspection.
    /// </summary>
    public Tensor<T> GetFeatureMapScale() => _featureMapScale;
}
