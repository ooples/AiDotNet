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
    private double[,]? _Wq; // Query projection
    private double[,]? _Wk; // Key projection
    private double[,]? _Wv; // Value projection
    private double[,]? _Wo; // Output projection

    // Feed-forward weights
    private double[,]? _W1;
    private double[]? _b1;
    private double[,]? _W2;
    private double[]? _b2;

    // Input projection
    private double[,]? _inputProj;

    // Prior association (learnable Gaussian kernel)
    private double _priorSigma;

    private int _inputDim;

    // Normalization parameters
    private double[]? _dataMeans;
    private double[]? _dataStds;

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

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Normalize data
        var (normalizedData, means, stds) = NormalizeData(data);
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

    private (double[][] normalized, double[] means, double[] stds) NormalizeData(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        var means = new double[d];
        var stds = new double[d];

        for (int j = 0; j < d; j++)
        {
            for (int i = 0; i < n; i++)
            {
                means[j] += data[i][j];
            }
            means[j] /= n;

            for (int i = 0; i < n; i++)
            {
                stds[j] += Math.Pow(data[i][j] - means[j], 2);
            }
            stds[j] = Math.Sqrt(stds[j] / n);
            if (stds[j] < 1e-10) stds[j] = 1;
        }

        var normalized = new double[n][];
        for (int i = 0; i < n; i++)
        {
            normalized[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                normalized[i][j] = (data[i][j] - means[j]) / stds[j];
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
        _b1 = new double[ffDim];
        _W2 = InitializeMatrix(ffDim, _modelDim, scaleFF2);
        _b2 = new double[_modelDim];

        // Prior sigma (learnable)
        _priorSigma = 1.0;
    }

    private double[,] InitializeMatrix(int rows, int cols, double scale)
    {
        var matrix = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                matrix[i, j] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * scale;
            }
        }
        return matrix;
    }

    private void Train(double[][] data)
    {
        int n = data.Length;
        int numSeqs = n - _seqLength + 1;

        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            var indices = Enumerable.Range(0, numSeqs).OrderBy(_ => _random.NextDouble()).ToArray();

            foreach (var idx in indices)
            {
                // Extract sequence
                var seq = new double[_seqLength][];
                for (int t = 0; t < _seqLength; t++)
                {
                    seq[t] = data[idx + t];
                }

                // Forward pass and compute loss
                var (_, _, assocDisc) = Forward(seq);

                // Update prior sigma based on association discrepancy
                // Simplified: adjust sigma to minimize reconstruction + maximize discrepancy for anomalies
                double avgDisc = assocDisc.Average();
                _priorSigma = Math.Max(0.1, _priorSigma - _learningRate * (avgDisc - 1.0));
            }
        }
    }

    private (double[][] output, double[][] attention, double[] assocDisc) Forward(double[][] sequence)
    {
        int seqLen = sequence.Length;

        // Project input to model dimension
        var projected = new double[seqLen][];
        for (int t = 0; t < seqLen; t++)
        {
            projected[t] = new double[_modelDim];
            for (int j = 0; j < _modelDim; j++)
            {
                for (int i = 0; i < _inputDim; i++)
                {
                    projected[t][j] += sequence[t][i] * _inputProj![i, j];
                }
            }
        }

        // Add positional encoding
        AddPositionalEncoding(projected);

        // Self-attention
        var (attnOutput, attention) = SelfAttention(projected);

        // Compute prior association (Gaussian kernel based on position distance)
        var priorAssoc = ComputePriorAssociation(seqLen);

        // Association discrepancy: KL divergence between series and prior associations
        var assocDisc = new double[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            double kl = 0;
            for (int j = 0; j < seqLen; j++)
            {
                double p = attention[i][j] + 1e-10;
                double q = priorAssoc[i][j] + 1e-10;
                kl += p * Math.Log(p / q);
            }
            assocDisc[i] = Math.Abs(kl);
        }

        // Feed-forward
        var output = FeedForward(attnOutput);

        return (output, attention, assocDisc);
    }

    private void AddPositionalEncoding(double[][] x)
    {
        int seqLen = x.Length;

        for (int pos = 0; pos < seqLen; pos++)
        {
            for (int i = 0; i < _modelDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / _modelDim);
                if (i % 2 == 0)
                {
                    x[pos][i] += Math.Sin(angle);
                }
                else
                {
                    x[pos][i] += Math.Cos(angle);
                }
            }
        }
    }

    private (double[][] output, double[][] attention) SelfAttention(double[][] x)
    {
        int seqLen = x.Length;
        int headDim = _modelDim / _numHeads;

        // Compute Q, K, V
        var Q = new double[seqLen][];
        var K = new double[seqLen][];
        var V = new double[seqLen][];

        for (int t = 0; t < seqLen; t++)
        {
            Q[t] = new double[_modelDim];
            K[t] = new double[_modelDim];
            V[t] = new double[_modelDim];

            for (int j = 0; j < _modelDim; j++)
            {
                for (int i = 0; i < _modelDim; i++)
                {
                    Q[t][j] += x[t][i] * _Wq![i, j];
                    K[t][j] += x[t][i] * _Wk![i, j];
                    V[t][j] += x[t][i] * _Wv![i, j];
                }
            }
        }

        // Compute attention scores
        double scale = Math.Sqrt(headDim);
        var attention = new double[seqLen][];

        for (int i = 0; i < seqLen; i++)
        {
            attention[i] = new double[seqLen];
            double maxScore = double.MinValue;

            for (int j = 0; j < seqLen; j++)
            {
                double score = 0;
                for (int k = 0; k < _modelDim; k++)
                {
                    score += Q[i][k] * K[j][k];
                }
                score /= scale;
                attention[i][j] = score;
                if (score > maxScore) maxScore = score;
            }

            // Softmax
            double sum = 0;
            for (int j = 0; j < seqLen; j++)
            {
                attention[i][j] = Math.Exp(attention[i][j] - maxScore);
                sum += attention[i][j];
            }
            for (int j = 0; j < seqLen; j++)
            {
                attention[i][j] /= sum;
            }
        }

        // Apply attention to values
        var attnOutput = new double[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            attnOutput[i] = new double[_modelDim];
            for (int j = 0; j < seqLen; j++)
            {
                for (int k = 0; k < _modelDim; k++)
                {
                    attnOutput[i][k] += attention[i][j] * V[j][k];
                }
            }

            // Output projection
            var projected = new double[_modelDim];
            for (int k = 0; k < _modelDim; k++)
            {
                for (int m = 0; m < _modelDim; m++)
                {
                    projected[k] += attnOutput[i][m] * _Wo![m, k];
                }
            }
            attnOutput[i] = projected;

            // Residual connection
            for (int k = 0; k < _modelDim; k++)
            {
                attnOutput[i][k] += x[i][k];
            }
        }

        return (attnOutput, attention);
    }

    private double[][] ComputePriorAssociation(int seqLen)
    {
        var prior = new double[seqLen][];

        for (int i = 0; i < seqLen; i++)
        {
            prior[i] = new double[seqLen];
            double sum = 0;

            for (int j = 0; j < seqLen; j++)
            {
                // Gaussian kernel based on position distance
                double dist = Math.Abs(i - j);
                prior[i][j] = Math.Exp(-dist * dist / (2 * _priorSigma * _priorSigma));
                sum += prior[i][j];
            }

            // Normalize
            for (int j = 0; j < seqLen; j++)
            {
                prior[i][j] /= sum;
            }
        }

        return prior;
    }

    private double[][] FeedForward(double[][] x)
    {
        int seqLen = x.Length;
        int ffDim = _W1!.GetLength(1);

        var output = new double[seqLen][];

        for (int t = 0; t < seqLen; t++)
        {
            // First layer with ReLU
            var h = new double[ffDim];
            for (int j = 0; j < ffDim; j++)
            {
                h[j] = _b1![j];
                for (int i = 0; i < _modelDim; i++)
                {
                    h[j] += x[t][i] * _W1[i, j];
                }
                h[j] = Math.Max(0, h[j]); // ReLU
            }

            // Second layer
            output[t] = new double[_modelDim];
            for (int j = 0; j < _modelDim; j++)
            {
                output[t][j] = _b2![j];
                for (int i = 0; i < ffDim; i++)
                {
                    output[t][j] += h[i] * _W2![i, j];
                }
            }

            // Residual connection
            for (int j = 0; j < _modelDim; j++)
            {
                output[t][j] += x[t][j];
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

        // Convert to normalized double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_inputDim];
            for (int j = 0; j < _inputDim; j++)
            {
                data[i][j] = (NumOps.ToDouble(X[i, j]) - dataMeans[j]) / dataStds[j];
            }
        }

        // Track which points have been scored
        var hasScore = new bool[n];

        // Score using sliding window
        int windowStart = 0;
        int stepSize = Math.Max(1, _seqLength / 2); // Ensure step size is at least 1

        while (windowStart + _seqLength <= n)
        {
            var seq = new double[_seqLength][];
            for (int t = 0; t < _seqLength; t++)
            {
                seq[t] = data[windowStart + t];
            }

            var (_, _, assocDisc) = Forward(seq);

            // Assign scores to each point in the window
            for (int t = 0; t < _seqLength; t++)
            {
                int idx = windowStart + t;
                if (hasScore[idx])
                {
                    // Take max of overlapping scores
                    double currentScore = NumOps.ToDouble(scores[idx]);
                    scores[idx] = NumOps.FromDouble(Math.Max(currentScore, assocDisc[t]));
                }
                else
                {
                    scores[idx] = NumOps.FromDouble(assocDisc[t]);
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
                double dist = 0;
                for (int j = 0; j < _inputDim; j++)
                {
                    dist += data[i][j] * data[i][j];
                }
                scores[i] = NumOps.FromDouble(dist);
            }
        }

        return scores;
    }
}
