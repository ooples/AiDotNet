using AiDotNet.Tensors.Engines;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Implements Anomaly Transformer for time series anomaly detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Anomaly Transformer uses the attention mechanism to detect anomalies.
/// Normal points tend to have similar attention patterns to their neighbors, while anomalies
/// show distinct attention patterns. It uses "Association Discrepancy" as the anomaly score.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Encode time series using positional encoding
/// 2. Apply self-attention to learn temporal relationships
/// 3. Compute association discrepancy between prior and series associations
/// 4. High discrepancy indicates anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Long time series with complex patterns
/// - When you need to capture long-range dependencies
/// - Multivariate time series anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Model dimensions: 64
/// - Number of heads: 4
/// - Sequence length: 100
/// - Epochs: 10
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Xu, J., et al. (2022).
/// "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.AnomalyDetection)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy",
    "https://openreview.net/forum?id=LzQQ89U1qm_",
    Year = 2022,
    Authors = "Jiehui Xu, Haixu Wu, Jianmin Wang, Mingsheng Long")]
public partial class AnomalyTransformerDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _modelDim;
    private readonly int _numHeads;
    private readonly int _seqLength;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Attention weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _Wq; // Query projection
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _Wk; // Key projection
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _Wv; // Value projection
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _Wo; // Output projection

    // Feed-forward weights
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _W1;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _b1;
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _W2;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T>? _b2;

    // Input projection
    [AiDotNet.Attributes.TrainableParameter]
    private Matrix<T>? _inputProj;

    // Prior association (learnable Gaussian kernel)
    private T _priorSigma;

    private int _inputDim;

    // Normalization parameters
    [Buffer]
    private Vector<T>? _dataMeans;
    [Buffer]
    private Vector<T>? _dataStds;

    /// <summary>
    /// Gets the model dimensions.
    /// </summary>
    public int ModelDim => _modelDim;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the sequence length.
    /// </summary>
    public int SeqLength => _seqLength;

    /// <summary>
    /// Creates a new Anomaly Transformer detector.
    /// </summary>
    /// <param name="modelDim">Dimensions of model. Default is 64.</param>
    /// <param name="numHeads">Number of attention heads. Default is 4.</param>
    /// <param name="seqLength">Length of input sequences. Default is 100.</param>
    /// <param name="epochs">Number of training epochs. Default is 10.</param>
    /// <param name="learningRate">Learning rate. Default is 0.0001.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public AnomalyTransformerDetector(int modelDim = 64, int numHeads = 4, int seqLength = 100,
        int epochs = 10, double learningRate = 0.0001,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (modelDim < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim),
                "Model dimensions must be at least 1. Recommended is 64.");
        }

        if (numHeads < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(numHeads),
                "Number of heads must be at least 1. Recommended is 4.");
        }

        if (modelDim % numHeads != 0)
        {
            throw new ArgumentException(
                $"Model dimensions ({modelDim}) must be divisible by number of heads ({numHeads}).",
                nameof(modelDim));
        }

        if (seqLength < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(seqLength),
                "Sequence length must be at least 1. Recommended is 100.");
        }

        if (epochs < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(epochs),
                "Epochs must be at least 1. Recommended is 10.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate),
                "Learning rate must be positive. Recommended is 0.0001.");
        }

        if (seqLength < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(seqLength),
                "Sequence length must be at least 2 for sliding window scoring. Recommended is 100.");
        }

        _modelDim = modelDim;
        _numHeads = numHeads;
        _seqLength = seqLength;
        _epochs = epochs;
        _learningRate = learningRate;
        _priorSigma = NumOps.One;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _inputDim = X.Columns;

        if (n < _seqLength)
        {
            throw new ArgumentException(
                $"Not enough samples for sequence length {_seqLength}. Need at least {_seqLength} samples.",
                nameof(X));
        }

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(X);
        _dataMeans = means;
        _dataStds = stds;

        // Initialize weights
        InitializeWeights();

        // Train
        Train(normalizedData);

        // Calculate scores for training data
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (Matrix<T> normalized, Vector<T> means, Vector<T> stds) NormalizeData(Matrix<T> data)
    {
        // Per-feature mean and population std (std below 1e-10 replaced by 1), then (x - mean) / std
        var engine = Engine;
        var values = AnomalyTensorOps.ToTensor(data);
        var (std, mean) = engine.TensorStdMean(values, new[] { 0 }, 0, false);
        var tiny = engine.TensorLessThan(std, NumOps.FromDouble(1e-10));
        var stds = engine.TensorWhere(tiny, Tensor<T>.CreateOnes(new[] { data.Columns }), std);
        var normalized = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(values, mean), stds);
        return (normalized.ToMatrix(), mean.ToVector(), stds.ToVector());
    }

    private void InitializeWeights()
    {
        double scale = Math.Sqrt(2.0 / _modelDim);
        double scaleInput = Math.Sqrt(2.0 / (_inputDim + _modelDim));

        // Input projection
        _inputProj = InitializeMatrix(_inputDim, _modelDim, scaleInput);

        // Attention weights
        _Wq = InitializeMatrix(_modelDim, _modelDim, scale);
        _Wk = InitializeMatrix(_modelDim, _modelDim, scale);
        _Wv = InitializeMatrix(_modelDim, _modelDim, scale);
        _Wo = InitializeMatrix(_modelDim, _modelDim, scale);

        // Feed-forward
        int ffDim = _modelDim * 4;
        double scaleFF1 = Math.Sqrt(2.0 / (_modelDim + ffDim));
        double scaleFF2 = Math.Sqrt(2.0 / (ffDim + _modelDim));

        _W1 = InitializeMatrix(_modelDim, ffDim, scaleFF1);
        _b1 = InitializeVector(ffDim);
        _W2 = InitializeMatrix(ffDim, _modelDim, scaleFF2);
        _b2 = InitializeVector(_modelDim);

        // Prior sigma (learnable)
        _priorSigma = NumOps.One;
    }

    private Matrix<T> InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double val = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
                matrix[i, j] = NumOps.FromDouble(val);
            }
        }
        return matrix;
    }

    private Vector<T> InitializeVector(int size) => new Vector<T>(size);

    private void Train(Matrix<T> data)
    {
        var engine = Engine;
        int n = data.Rows;
        int numSeqs = n - _seqLength + 1;
        var values = AnomalyTensorOps.ToTensor(data);
        T lr = NumOps.FromDouble(_learningRate);
        T clipValue = NumOps.FromDouble(5.0);
        T one = NumOps.One;
        T minSigma = NumOps.FromDouble(0.1);
        T lossScale = NumOps.FromDouble(2.0 / (_seqLength * _modelDim));

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, numSeqs).OrderBy(_ => _random.NextDouble()).ToArray();

            foreach (var idx in indices)
            {
                // Sequence (seqLength x inputDim) and the forward pass with intermediate caching
                var seq = engine.TensorSlice(values, new[] { idx, 0 }, new[] { _seqLength, _inputDim });
                var pass = ForwardPass(seq);

                // Reconstruction loss gradient (MSE) against the projected input:
                // dL/dOutput = 2 * (output - projected) / N
                var dOutput = engine.TensorMultiplyScalar(engine.TensorSubtract(pass.Output, pass.Projected), lossScale);

                // Backprop through feed-forward (output -> attnOutput), attention (attnOutput -> projected)
                // and the input projection (projected -> seq)
                var dAttnOutput = BackpropFeedForward(pass.AttnOutput, pass.FFHidden, dOutput, lr, clipValue);
                var dProjected = BackpropAttention(pass.Projected, dAttnOutput, lr, clipValue);
                BackpropInputProjection(seq, dProjected, lr, clipValue);

                // Update prior sigma based on association discrepancy (at least 0.1)
                T discDiff = NumOps.Subtract(engine.TensorMean(pass.AssocDisc), one);
                _priorSigma = NumOps.Subtract(_priorSigma, NumOps.Multiply(lr, discDiff));
                if (NumOps.LessThan(_priorSigma, minSigma))
                {
                    _priorSigma = minSigma;
                }
            }
        }
    }

    // weights - lr * clamp(gradient, -clip, clip)
    private Tensor<T> Step(Tensor<T> weights, Tensor<T> gradient, T lr, T clip)
    {
        var engine = Engine;
        return engine.TensorSubtract(weights,
            engine.TensorMultiplyScalar(engine.TensorClamp(gradient, NumOps.Negate(clip), clip), lr));
    }

    private sealed class ForwardResult
    {
        public required Tensor<T> Projected { get; init; }
        public required Tensor<T> AttnOutput { get; init; }
        public required Tensor<T> Attention { get; init; }
        public required Tensor<T> FFHidden { get; init; }
        public required Tensor<T> Output { get; init; }
        public required Tensor<T> AssocDisc { get; init; }
    }

    // Input projection + positional encoding, self-attention, feed-forward (with residuals) and the
    // association discrepancy of one sequence ([seqLen, inputDim]).
    private ForwardResult ForwardPass(Tensor<T> sequence)
    {
        var inputProj = _inputProj;
        var W1 = _W1;
        var b1 = _b1;
        var W2 = _W2;
        var b2 = _b2;
        if (inputProj == null || W1 == null || b1 == null || W2 == null || b2 == null)
        {
            throw new InvalidOperationException("Model weights not initialized.");
        }

        var engine = Engine;
        int seqLen = sequence.Shape[0];

        // Input projection [seqLen, inputDim] @ [inputDim, modelDim], plus positional encoding
        var projected = engine.TensorAdd(engine.TensorMatMul(sequence, Tensor<T>.FromMatrix(inputProj)),
            PositionalEncoding(seqLen));

        // Self-attention
        var (attnOutput, attention) = SelfAttention(projected);

        // Feed-forward: hidden = ReLU(attnOutput @ W1 + b1), output = hidden @ W2 + b2 + attnOutput
        var ffHidden = engine.TensorMax(engine.TensorAdd(engine.TensorMatMul(attnOutput, Tensor<T>.FromMatrix(W1)),
            AnomalyTensorOps.BroadcastRow(b1.AsTensor(), seqLen)), NumOps.Zero);
        var output = engine.TensorAdd(engine.TensorAdd(engine.TensorMatMul(ffHidden, Tensor<T>.FromMatrix(W2)),
            AnomalyTensorOps.BroadcastRow(b2.AsTensor(), seqLen)), attnOutput);

        // Association discrepancy |KL(series || prior)| per position, both shifted by 1e-10
        T epsilon = NumOps.FromDouble(1e-10);
        var p = engine.TensorAddScalar(attention, epsilon);
        var q = engine.TensorAddScalar(ComputePriorAssociation(seqLen), epsilon);
        var assocDisc = engine.TensorAbs(engine.ReduceSum(
            engine.TensorMultiply(p, engine.TensorLog(engine.TensorDivide(p, q))), new[] { 1 }, false));

        return new ForwardResult
        {
            Projected = projected,
            AttnOutput = attnOutput,
            Attention = attention,
            FFHidden = ffHidden,
            Output = output,
            AssocDisc = assocDisc
        };
    }

    private Tensor<T> BackpropFeedForward(Tensor<T> attnOutput, Tensor<T> ffHidden, Tensor<T> dOutput, T lr, T clipValue)
    {
        var W1 = _W1;
        var b1 = _b1;
        var W2 = _W2;
        var b2 = _b2;
        if (W1 == null || b1 == null || W2 == null || b2 == null)
        {
            throw new InvalidOperationException("Feed-forward weights not initialized.");
        }

        var engine = Engine;
        var w1 = Tensor<T>.FromMatrix(W1);
        var w2 = Tensor<T>.FromMatrix(W2);

        // Second layer (residual connection: output = W2*h + attnOutput): dW2 = h^T dOutput, dB2 = sum dOutput,
        // dh = dOutput W2^T through the ReLU
        var dW2 = engine.TensorMatMul(engine.TensorTranspose(ffHidden), dOutput);
        var dB2 = engine.ReduceSum(dOutput, new[] { 0 }, false);
        var dH = engine.TensorMultiply(engine.TensorMatMul(dOutput, engine.TensorTranspose(w2)),
            engine.TensorGreaterThan(ffHidden, NumOps.Zero));

        // First layer: dW1 = attnOutput^T dh, dB1 = sum dh; dAttnOutput = dOutput (residual) + dh W1^T
        var dW1 = engine.TensorMatMul(engine.TensorTranspose(attnOutput), dH);
        var dB1 = engine.ReduceSum(dH, new[] { 0 }, false);
        var dAttnOutput = engine.TensorAdd(dOutput, engine.TensorMatMul(dH, engine.TensorTranspose(w1)));

        // Apply weight updates with gradient clipping
        _W2 = Step(w2, dW2, lr, clipValue).ToMatrix();
        _b2 = Step(b2.AsTensor(), dB2, lr, clipValue).ToVector();
        _W1 = Step(w1, dW1, lr, clipValue).ToMatrix();
        _b1 = Step(b1.AsTensor(), dB1, lr, clipValue).ToVector();

        return dAttnOutput;
    }

    // Exact backward pass of SelfAttention for every head: output = concat_h(A_h V_h) Wo + projected with
    // A_h = softmax(Q_h K_h^T / sqrt(headDim)). Returns dL/dprojected through the residual and the Q, K and V
    // paths, and updates Wq, Wk, Wv and Wo.
    private Tensor<T> BackpropAttention(Tensor<T> projected, Tensor<T> dAttnOutput, T lr, T clipValue)
    {
        var Wq = _Wq;
        var Wk = _Wk;
        var Wv = _Wv;
        var Wo = _Wo;
        if (Wq == null || Wk == null || Wv == null || Wo == null)
        {
            throw new InvalidOperationException("Attention weights not initialized.");
        }

        var engine = Engine;
        int seqLen = projected.Shape[0];
        int heads = _numHeads;
        int headDim = _modelDim / heads;
        T scale = NumOps.Sqrt(NumOps.FromDouble(headDim));
        var wq = Tensor<T>.FromMatrix(Wq);
        var wk = Tensor<T>.FromMatrix(Wk);
        var wv = Tensor<T>.FromMatrix(Wv);
        var wo = Tensor<T>.FromMatrix(Wo);
        var projectedT = engine.TensorTranspose(projected);

        // Recompute the per-head forward pass (the attention weights are not updated before this point)
        var Q = SplitHeads(engine.TensorMatMul(projected, wq), seqLen, heads, headDim);
        var K = SplitHeads(engine.TensorMatMul(projected, wk), seqLen, heads, headDim);
        var V = SplitHeads(engine.TensorMatMul(projected, wv), seqLen, heads, headDim);
        var attention = engine.TensorSoftmax(
            engine.TensorDivideScalar(engine.TensorBatchMatMul(Q, engine.TensorSwapAxes(K, 1, 2)), scale), 2);
        var concatenated = MergeHeads(engine.TensorBatchMatMul(attention, V), seqLen);

        // Output projection: dWo = concat^T dOut, dConcat = dOut Wo^T (split back into heads)
        var dWo = engine.TensorMatMul(engine.TensorTranspose(concatenated), dAttnOutput);
        var dHeads = SplitHeads(engine.TensorMatMul(dAttnOutput, engine.TensorTranspose(wo)), seqLen, heads, headDim);

        // headOutput_h = A_h V_h: dV_h = A_h^T dO_h, dA_h = dO_h V_h^T
        var dV = engine.TensorBatchMatMul(engine.TensorSwapAxes(attention, 1, 2), dHeads);
        var dAttn = engine.TensorBatchMatMul(dHeads, engine.TensorSwapAxes(V, 1, 2));

        // Softmax Jacobian per row: dScore = A * (dA - sum_l A[., l] dA[., l]), then the 1 / scale
        var rowDot = engine.ReduceSum(engine.TensorMultiply(attention, dAttn), new[] { 2 }, true);
        var dScore = engine.TensorDivideScalar(engine.TensorMultiply(attention,
            engine.TensorSubtract(dAttn, engine.TensorBroadcastTo(rowDot, new[] { heads, seqLen, seqLen }))), scale);

        // score_h = Q_h K_h^T (scaled above): dQ_h = dScore_h K_h, dK_h = dScore_h^T Q_h
        var dQ = MergeHeads(engine.TensorBatchMatMul(dScore, K), seqLen);
        var dK = MergeHeads(engine.TensorBatchMatMul(engine.TensorSwapAxes(dScore, 1, 2), Q), seqLen);
        var dVMerged = MergeHeads(dV, seqLen);
        var dWq = engine.TensorMatMul(projectedT, dQ);
        var dWk = engine.TensorMatMul(projectedT, dK);
        var dWv = engine.TensorMatMul(projectedT, dVMerged);

        // dL/dprojected: the residual connection plus the Q, K and V projection paths
        var dProjected = engine.TensorAdd(
            engine.TensorAdd(dAttnOutput, engine.TensorMatMul(dQ, engine.TensorTranspose(wq))),
            engine.TensorAdd(engine.TensorMatMul(dK, engine.TensorTranspose(wk)),
                engine.TensorMatMul(dVMerged, engine.TensorTranspose(wv))));

        // Apply weight updates
        _Wo = Step(wo, dWo, lr, clipValue).ToMatrix();
        _Wq = Step(wq, dWq, lr, clipValue).ToMatrix();
        _Wk = Step(wk, dWk, lr, clipValue).ToMatrix();
        _Wv = Step(wv, dWv, lr, clipValue).ToMatrix();

        return dProjected;
    }

    private void BackpropInputProjection(Tensor<T> sequence, Tensor<T> dProjected, T lr, T clipValue)
    {
        var inputProj = _inputProj;
        if (inputProj == null)
        {
            throw new InvalidOperationException("Input projection not initialized.");
        }

        var engine = Engine;
        var dInputProj = engine.TensorMatMul(engine.TensorTranspose(sequence), dProjected);
        _inputProj = Step(Tensor<T>.FromMatrix(inputProj), dInputProj, lr, clipValue).ToMatrix();
    }

    // Sinusoidal positional encoding [seqLen, modelDim]: angle = pos / 10000^(2 * (i / 2) / modelDim),
    // sin for even i and cos for odd i.
    private Tensor<T> PositionalEncoding(int seqLen)
    {
        var engine = Engine;
        int modelDim = _modelDim;
        var pairIndex = new int[modelDim];
        var even = new int[modelDim];
        for (int i = 0; i < modelDim; i++)
        {
            pairIndex[i] = 2 * (i / 2);
            even[i] = i % 2 == 0 ? 1 : 0;
        }

        var exponents = engine.TensorDivideScalar(
            Tensor<int>.FromVector(Vector<int>.Wrap(pairIndex), new[] { modelDim }).Cast<T>(), NumOps.FromDouble(modelDim));
        var divisor = engine.TensorPower(engine.Fill(modelDim, NumOps.FromDouble(10000)).AsTensor(), exponents);
        var angle = engine.TensorDivide(
            AnomalyTensorOps.BroadcastColumn(engine.TensorArange<T>(0, seqLen, 1), modelDim),
            AnomalyTensorOps.BroadcastRow(divisor, seqLen));
        var evenMask = AnomalyTensorOps.BroadcastRow(
            Tensor<int>.FromVector(Vector<int>.Wrap(even), new[] { modelDim }).Cast<T>(), seqLen);
        return engine.TensorWhere(evenMask, engine.TensorSin(angle), engine.TensorCos(angle));
    }

    // Multi-head self-attention with an output projection and residual connection; also returns the
    // attention averaged across heads ([seqLen, seqLen]).
    private (Tensor<T> output, Tensor<T> attention) SelfAttention(Tensor<T> x)
    {
        var Wq = _Wq;
        var Wk = _Wk;
        var Wv = _Wv;
        var Wo = _Wo;
        if (Wq == null || Wk == null || Wv == null || Wo == null)
        {
            throw new InvalidOperationException("Attention weights not initialized.");
        }

        var engine = Engine;
        int seqLen = x.Shape[0];
        int heads = _numHeads;
        int headDim = _modelDim / heads;
        T scale = NumOps.Sqrt(NumOps.FromDouble(headDim));

        // Q, K, V projections (full model dimension), split into heads [heads, seqLen, headDim]
        var Q = SplitHeads(engine.TensorMatMul(x, Tensor<T>.FromMatrix(Wq)), seqLen, heads, headDim);
        var K = SplitHeads(engine.TensorMatMul(x, Tensor<T>.FromMatrix(Wk)), seqLen, heads, headDim);
        var V = SplitHeads(engine.TensorMatMul(x, Tensor<T>.FromMatrix(Wv)), seqLen, heads, headDim);

        // Per-head scaled dot-product attention (softmax over each row) applied to V
        var scores = engine.TensorDivideScalar(engine.TensorBatchMatMul(Q, engine.TensorSwapAxes(K, 1, 2)), scale);
        var headAttention = engine.TensorSoftmax(scores, 2);
        var avgAttention = engine.ReduceSum(engine.TensorDivideScalar(headAttention, NumOps.FromDouble(heads)), new[] { 0 }, false);
        var headOutputs = engine.TensorBatchMatMul(headAttention, V);

        // Concatenate head outputs into attnOutput [seqLen, modelDim], then the output projection with
        // residual connection
        var attnOutput = engine.Reshape(engine.TensorPermute(headOutputs, new[] { 1, 0, 2 }), new[] { seqLen, _modelDim });
        var output = engine.TensorAdd(engine.TensorMatMul(attnOutput, Tensor<T>.FromMatrix(Wo)), x);
        return (output, avgAttention);
    }

    // [heads, seqLen, headDim] as [seqLen, heads * headDim] (the inverse of SplitHeads).
    private Tensor<T> MergeHeads(Tensor<T> x, int seqLen)
        => Engine.Reshape(Engine.TensorPermute(x, new[] { 1, 0, 2 }), new[] { seqLen, _modelDim });

    // [seqLen, heads * headDim] as [heads, seqLen, headDim].
    private Tensor<T> SplitHeads(Tensor<T> x, int seqLen, int heads, int headDim)
        => Engine.TensorPermute(Engine.Reshape(x, new[] { seqLen, heads, headDim }), new[] { 1, 0, 2 });

    // Row-normalized Gaussian kernel exp(-(i - j)^2 / (2 sigma^2)) over positions.
    private Tensor<T> ComputePriorAssociation(int seqLen)
    {
        var engine = Engine;
        T twoSigmaSq = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(_priorSigma, _priorSigma));
        var positions = engine.TensorArange<T>(0, seqLen, 1);
        var offset = engine.TensorSubtract(AnomalyTensorOps.BroadcastColumn(positions, seqLen),
            AnomalyTensorOps.BroadcastRow(positions, seqLen));
        var prior = engine.TensorExp(engine.TensorNegate(engine.TensorDivideScalar(engine.TensorSquare(offset), twoSigmaSq)));
        return AnomalyTensorOps.DivideColumn(prior, engine.ReduceSum(prior, new[] { 1 }, false));
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var dataMeans = _dataMeans;
        var dataStds = _dataStds;
        if (dataMeans == null || dataStds == null)
        {
            throw new InvalidOperationException("Model not properly fitted. Normalization parameters missing.");
        }

        var engine = Engine;
        int n = X.Rows;

        // Normalized data
        var data = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), dataMeans.AsTensor()), dataStds.AsTensor());

        // Score using sliding windows (step at least 1), taking the max of overlapping scores. Per the
        // Anomaly Transformer paper, the anomaly score combines association discrepancy and the per-timestep
        // reconstruction error RMSE(output, projected input): disc * error + error.
        // Each window keeps only its own [seqLength] scores; one scatter-max then reduces the overlaps.
        int stepSize = Math.Max(1, _seqLength / 2);
        T modelDim = NumOps.FromDouble(_modelDim);
        var windowScores = new List<Tensor<T>>();
        var positions = new List<int>();
        for (int windowStart = 0; windowStart + _seqLength <= n; windowStart += stepSize)
        {
            var pass = ForwardPass(engine.TensorSlice(data, new[] { windowStart, 0 }, new[] { _seqLength, _inputDim }));
            var reconError = engine.TensorSqrt(engine.TensorDivideScalar(engine.ReduceSum(
                engine.TensorSquare(engine.TensorSubtract(pass.Output, pass.Projected)), new[] { 1 }, false), modelDim));
            windowScores.Add(engine.TensorAdd(engine.TensorMultiply(pass.AssocDisc, reconError), reconError));
            positions.AddRange(Enumerable.Range(windowStart, _seqLength));
        }

        // Points without window coverage use the simple distance ||x||; then multiply by the value deviation
        // signal (1 + ||x||). The association discrepancy and reconstruction error from the transformer
        // capture temporal pattern anomalies, while value deviation captures point-level magnitude
        // anomalies. Using multiplicative combination ensures extreme values get proportionally amplified
        // scores, which is critical when the model hasn't fully converged (few training epochs).
        var norm = engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(data), new[] { 1 }, false));
        var scores = windowScores.Count == 0
            ? norm
            : engine.TensorScatterReduce(norm, 0,
                Tensor<int>.FromVector(Vector<int>.Wrap(positions.ToArray()), new[] { positions.Count }),
                engine.TensorConcatenate(windowScores.ToArray(), 0), ScatterReduceMode.AMax, false);
        return engine.TensorMultiply(scores, engine.TensorAddScalar(norm, NumOps.One)).ToVector();
    }
}
