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
public class AnomalyTransformerDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _modelDim;
    private readonly int _numHeads;
    private readonly int _seqLength;
    private readonly int _epochs;
    private readonly double _learningRate;

    // Attention weights
    private Matrix<T>? _Wq; // Query projection
    private Matrix<T>? _Wk; // Key projection
    private Matrix<T>? _Wv; // Value projection
    private Matrix<T>? _Wo; // Output projection

    // Feed-forward weights
    private Matrix<T>? _W1;
    private Vector<T>? _b1;
    private Matrix<T>? _W2;
    private Vector<T>? _b2;

    // Input projection
    private Matrix<T>? _inputProj;

    // Prior association (learnable Gaussian kernel)
    private T _priorSigma;

    private int _inputDim;

    // Normalization parameters
    private Vector<T>? _dataMeans;
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
        int n = data.Rows;
        int d = data.Columns;

        var means = new Vector<T>(d);
        var stds = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            means[j] = NumOps.Divide(sum, NumOps.FromDouble(n));

            T variance = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            double stdVal = Math.Sqrt(NumOps.ToDouble(variance) / n);
            if (stdVal < 1e-10) stdVal = 1;
            stds[j] = NumOps.FromDouble(stdVal);
        }

        var normalized = new Matrix<T>(n, d);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T diff = NumOps.Subtract(data[i, j], means[j]);
                normalized[i, j] = NumOps.Divide(diff, stds[j]);
            }
        }

        return (normalized, means, stds);
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

    private Vector<T> InitializeVector(int size)
    {
        var vector = new Vector<T>(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = NumOps.Zero;
        }
        return vector;
    }

    private void Train(Matrix<T> data)
    {
        int n = data.Rows;
        int numSeqs = n - _seqLength + 1;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, numSeqs).OrderBy(_ => _random.NextDouble()).ToArray();

            foreach (var idx in indices)
            {
                // Extract sequence as a Matrix<T> (seqLength x inputDim)
                var seq = new Matrix<T>(_seqLength, _inputDim);
                for (int t = 0; t < _seqLength; t++)
                {
                    for (int j = 0; j < _inputDim; j++)
                    {
                        seq[t, j] = data[idx + t, j];
                    }
                }

                // Forward pass and compute loss
                var (_, _, assocDisc) = Forward(seq);

                // Compute average discrepancy
                T avgDisc = NumOps.Zero;
                for (int i = 0; i < assocDisc.Length; i++)
                {
                    avgDisc = NumOps.Add(avgDisc, assocDisc[i]);
                }
                avgDisc = NumOps.Divide(avgDisc, NumOps.FromDouble(assocDisc.Length));

                // Update prior sigma based on association discrepancy
                // Simplified: adjust sigma to minimize reconstruction + maximize discrepancy for anomalies
                T discDiff = NumOps.Subtract(avgDisc, NumOps.One);
                T update = NumOps.Multiply(NumOps.FromDouble(_learningRate), discDiff);
                _priorSigma = NumOps.Subtract(_priorSigma, update);

                // Ensure sigma stays positive
                double sigmaVal = NumOps.ToDouble(_priorSigma);
                if (sigmaVal < 0.1)
                {
                    _priorSigma = NumOps.FromDouble(0.1);
                }
            }
        }
    }

    private (Matrix<T> output, Matrix<T> attention, Vector<T> assocDisc) Forward(Matrix<T> sequence)
    {
        // Capture nullable fields for proper null checking
        var inputProj = _inputProj;
        if (inputProj == null)
        {
            throw new InvalidOperationException("Weights not initialized.");
        }

        int seqLen = sequence.Rows;

        // Project input to model dimension: projected[t, j] = sum(sequence[t, i] * inputProj[i, j])
        var projected = new Matrix<T>(seqLen, _modelDim);
        for (int t = 0; t < seqLen; t++)
        {
            for (int j = 0; j < _modelDim; j++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < _inputDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(sequence[t, i], inputProj[i, j]));
                }
                projected[t, j] = sum;
            }
        }

        // Add positional encoding
        AddPositionalEncoding(projected);

        // Self-attention
        var (attnOutput, attention) = SelfAttention(projected);

        // Compute prior association (Gaussian kernel based on position distance)
        var priorAssoc = ComputePriorAssociation(seqLen);

        // Association discrepancy: KL divergence between series and prior associations
        var assocDisc = new Vector<T>(seqLen);
        T epsilon = NumOps.FromDouble(1e-10);
        for (int i = 0; i < seqLen; i++)
        {
            T kl = NumOps.Zero;
            for (int j = 0; j < seqLen; j++)
            {
                T p = NumOps.Add(attention[i, j], epsilon);
                T q = NumOps.Add(priorAssoc[i, j], epsilon);
                // KL = p * log(p/q)
                double pVal = NumOps.ToDouble(p);
                double qVal = NumOps.ToDouble(q);
                double klTerm = pVal * Math.Log(pVal / qVal);
                kl = NumOps.Add(kl, NumOps.FromDouble(klTerm));
            }
            // Take absolute value
            double klVal = Math.Abs(NumOps.ToDouble(kl));
            assocDisc[i] = NumOps.FromDouble(klVal);
        }

        // Feed-forward
        var output = FeedForward(attnOutput);

        return (output, attention, assocDisc);
    }

    private void AddPositionalEncoding(Matrix<T> x)
    {
        int seqLen = x.Rows;

        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < _modelDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / _modelDim);
                double pe = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                x[pos, i] = NumOps.Add(x[pos, i], NumOps.FromDouble(pe));
            }
        }
    }

    private (Matrix<T> output, Matrix<T> attention) SelfAttention(Matrix<T> x)
    {
        // Capture nullable fields for proper null checking
        var Wq = _Wq;
        var Wk = _Wk;
        var Wv = _Wv;
        var Wo = _Wo;

        if (Wq == null || Wk == null || Wv == null || Wo == null)
        {
            throw new InvalidOperationException("Attention weights not initialized.");
        }

        int seqLen = x.Rows;
        int headDim = _modelDim / _numHeads;

        // Compute Q, K, V
        var Q = new Matrix<T>(seqLen, _modelDim);
        var K = new Matrix<T>(seqLen, _modelDim);
        var V = new Matrix<T>(seqLen, _modelDim);

        for (int t = 0; t < seqLen; t++)
        {
            for (int j = 0; j < _modelDim; j++)
            {
                T qSum = NumOps.Zero;
                T kSum = NumOps.Zero;
                T vSum = NumOps.Zero;
                for (int i = 0; i < _modelDim; i++)
                {
                    qSum = NumOps.Add(qSum, NumOps.Multiply(x[t, i], Wq[i, j]));
                    kSum = NumOps.Add(kSum, NumOps.Multiply(x[t, i], Wk[i, j]));
                    vSum = NumOps.Add(vSum, NumOps.Multiply(x[t, i], Wv[i, j]));
                }
                Q[t, j] = qSum;
                K[t, j] = kSum;
                V[t, j] = vSum;
            }
        }

        // Compute attention scores
        T scale = NumOps.FromDouble(Math.Sqrt(headDim));
        var attention = new Matrix<T>(seqLen, seqLen);

        for (int i = 0; i < seqLen; i++)
        {
            T maxScore = NumOps.FromDouble(double.MinValue);

            for (int j = 0; j < seqLen; j++)
            {
                T score = NumOps.Zero;
                for (int k = 0; k < _modelDim; k++)
                {
                    score = NumOps.Add(score, NumOps.Multiply(Q[i, k], K[j, k]));
                }
                score = NumOps.Divide(score, scale);
                attention[i, j] = score;
                if (NumOps.ToDouble(score) > NumOps.ToDouble(maxScore))
                {
                    maxScore = score;
                }
            }

            // Softmax
            T sum = NumOps.Zero;
            for (int j = 0; j < seqLen; j++)
            {
                double expVal = Math.Exp(NumOps.ToDouble(attention[i, j]) - NumOps.ToDouble(maxScore));
                attention[i, j] = NumOps.FromDouble(expVal);
                sum = NumOps.Add(sum, attention[i, j]);
            }
            for (int j = 0; j < seqLen; j++)
            {
                attention[i, j] = NumOps.Divide(attention[i, j], sum);
            }
        }

        // Apply attention to values
        var attnOutput = new Matrix<T>(seqLen, _modelDim);
        for (int i = 0; i < seqLen; i++)
        {
            for (int k = 0; k < _modelDim; k++)
            {
                T sum = NumOps.Zero;
                for (int j = 0; j < seqLen; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(attention[i, j], V[j, k]));
                }
                attnOutput[i, k] = sum;
            }
        }

        // Output projection with residual connection
        var output = new Matrix<T>(seqLen, _modelDim);
        for (int i = 0; i < seqLen; i++)
        {
            for (int k = 0; k < _modelDim; k++)
            {
                T proj = NumOps.Zero;
                for (int m = 0; m < _modelDim; m++)
                {
                    proj = NumOps.Add(proj, NumOps.Multiply(attnOutput[i, m], Wo[m, k]));
                }
                // Residual connection
                output[i, k] = NumOps.Add(proj, x[i, k]);
            }
        }

        return (output, attention);
    }

    private Matrix<T> ComputePriorAssociation(int seqLen)
    {
        var prior = new Matrix<T>(seqLen, seqLen);
        T twoSigmaSq = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(_priorSigma, _priorSigma));

        for (int i = 0; i < seqLen; i++)
        {
            T sum = NumOps.Zero;

            for (int j = 0; j < seqLen; j++)
            {
                // Gaussian kernel based on position distance
                double dist = Math.Abs(i - j);
                double distSq = dist * dist;
                double gaussVal = Math.Exp(-distSq / NumOps.ToDouble(twoSigmaSq));
                prior[i, j] = NumOps.FromDouble(gaussVal);
                sum = NumOps.Add(sum, prior[i, j]);
            }

            // Normalize
            for (int j = 0; j < seqLen; j++)
            {
                prior[i, j] = NumOps.Divide(prior[i, j], sum);
            }
        }

        return prior;
    }

    private Matrix<T> FeedForward(Matrix<T> x)
    {
        // Capture nullable fields for proper null checking
        var W1 = _W1;
        var b1 = _b1;
        var W2 = _W2;
        var b2 = _b2;

        if (W1 == null || b1 == null || W2 == null || b2 == null)
        {
            throw new InvalidOperationException("Feed-forward weights not initialized.");
        }

        int seqLen = x.Rows;
        int ffDim = W1.Columns;

        var output = new Matrix<T>(seqLen, _modelDim);

        for (int t = 0; t < seqLen; t++)
        {
            // First layer with ReLU
            var h = new Vector<T>(ffDim);
            for (int j = 0; j < ffDim; j++)
            {
                T sum = b1[j];
                for (int i = 0; i < _modelDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(x[t, i], W1[i, j]));
                }
                // ReLU
                double val = NumOps.ToDouble(sum);
                h[j] = NumOps.FromDouble(Math.Max(0, val));
            }

            // Second layer with residual connection
            for (int j = 0; j < _modelDim; j++)
            {
                T sum = b2[j];
                for (int i = 0; i < ffDim; i++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(h[i], W2[i, j]));
                }
                // Residual connection
                output[t, j] = NumOps.Add(sum, x[t, j]);
            }
        }

        return output;
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

        int n = X.Rows;
        var scores = new Vector<T>(n);

        // Convert to normalized Matrix<T>
        var data = new Matrix<T>(n, _inputDim);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _inputDim; j++)
            {
                T diff = NumOps.Subtract(X[i, j], dataMeans[j]);
                data[i, j] = NumOps.Divide(diff, dataStds[j]);
            }
        }

        // Track which points have been scored
        var hasScore = new bool[n];

        // Score using sliding window
        int windowStart = 0;
        int stepSize = Math.Max(1, _seqLength / 2); // Ensure step size is at least 1

        while (windowStart + _seqLength <= n)
        {
            var seq = new Matrix<T>(_seqLength, _inputDim);
            for (int t = 0; t < _seqLength; t++)
            {
                for (int j = 0; j < _inputDim; j++)
                {
                    seq[t, j] = data[windowStart + t, j];
                }
            }

            var (_, _, assocDisc) = Forward(seq);

            // Assign scores to each point in the window
            for (int t = 0; t < _seqLength; t++)
            {
                int idx = windowStart + t;
                if (hasScore[idx])
                {
                    // Take max of overlapping scores
                    if (NumOps.ToDouble(assocDisc[t]) > NumOps.ToDouble(scores[idx]))
                    {
                        scores[idx] = assocDisc[t];
                    }
                }
                else
                {
                    scores[idx] = assocDisc[t];
                    hasScore[idx] = true;
                }
            }

            windowStart += stepSize;
        }

        // Handle remaining points that weren't scored
        for (int i = 0; i < n; i++)
        {
            if (!hasScore[i])
            {
                // Use simple distance for points without window coverage
                T dist = NumOps.Zero;
                for (int j = 0; j < _inputDim; j++)
                {
                    dist = NumOps.Add(dist, NumOps.Multiply(data[i, j], data[i, j]));
                }
                scores[i] = dist;
            }
        }

        return scores;
    }
}
