using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using SUOD (Scalable Unsupervised Outlier Detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SUOD is an acceleration framework that combines multiple
/// anomaly detection algorithms efficiently. It uses approximation techniques to
/// speed up detection while maintaining accuracy.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple diverse base detectors
/// 2. Use random projection for dimensionality reduction
/// 3. Combine scores using robust averaging
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Large datasets where speed matters
/// - When you want ensemble benefits with good performance
/// - As a general-purpose robust detector
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Base detectors: LOF, k-NN, Isolation Forest
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., et al. (2021). "SUOD: Accelerating Large-Scale Unsupervised
/// Heterogeneous Outlier Detection." MLSys.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("SUOD: Accelerating Large-Scale Unsupervised Heterogeneous Outlier Detection", "https://doi.org/10.48550/arXiv.2003.05731", Year = 2021, Authors = "Yue Zhao, Xiyang Hu, Cheng Cheng, Cong Wang, Changlin Wan, Wen Wang, Jianing Yang, Haoping Bai, Zheng Li, Cao Xiao, Yunlong Wang, Zhi Qiao, Jiashu Sun, Leman Akoglu")]
public class SUODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly bool _useRandomProjection;
    private readonly int _nProjectedFeatures;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private Matrix<T>? _projectionMatrix;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets whether random projection is used.
    /// </summary>
    public bool UseRandomProjection => _useRandomProjection;

    /// <summary>
    /// Creates a new SUOD anomaly detector.
    /// </summary>
    /// <param name="useRandomProjection">
    /// Whether to use random projection for dimensionality reduction. Default is true.
    /// </param>
    /// <param name="nProjectedFeatures">
    /// Number of features after projection. Default is 10.
    /// Only used if useRandomProjection is true.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SUODDetector(bool useRandomProjection = true, int nProjectedFeatures = 10,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nProjectedFeatures < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nProjectedFeatures),
                "NProjectedFeatures must be at least 1. Recommended is 10.");
        }

        _useRandomProjection = useRandomProjection;
        _nProjectedFeatures = nProjectedFeatures;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;
        Matrix<T> processedData = X;

        // Apply random projection if enabled and beneficial
        if (_useRandomProjection && X.Columns > _nProjectedFeatures)
        {
            _projectionMatrix = CreateRandomProjectionMatrix(X.Columns);
            processedData = ApplyProjection(X);
        }

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();

        int k = Math.Min(10, processedData.Rows - 1);

        // Detector 1: LOF
        var lof = new DistanceBased.LocalOutlierFactor<T>(
            numNeighbors: k,
            contamination: _contamination,
            randomSeed: _randomSeed);
        lof.Fit(processedData);
        _baseDetectors.Add(lof);

        // Detector 2: k-NN
        var knn = new DistanceBased.KNNDetector<T>(
            k: k,
            contamination: _contamination,
            randomSeed: _randomSeed + 1);
        knn.Fit(processedData);
        _baseDetectors.Add(knn);

        // Detector 3: Isolation Forest
        var iforest = new TreeBased.IsolationForest<T>(
            numTrees: 50,
            maxSamples: Math.Min(256, processedData.Rows),
            contamination: _contamination,
            randomSeed: _randomSeed + 2);
        iforest.Fit(processedData);
        _baseDetectors.Add(iforest);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        Matrix<T> processedData = X;
        if (_projectionMatrix != null)
        {
            processedData = ApplyProjection(X);
        }

        // Collect scores from all detectors
        var allScores = new List<Vector<T>>();

        foreach (var detector in _baseDetectors ?? throw new InvalidOperationException("Model not properly fitted."))
        {
            var scores = detector.ScoreAnomalies(processedData);
            var normalizedScores = NormalizeScores(scores);
            allScores.Add(normalizedScores);
        }

        // Combine using robust average (trimmed mean)
        var combinedScores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var pointScores = allScores.Select(s => s[i]).OrderBy(v => NumOps.ToDouble(v)).ToArray();

            // Trimmed mean (exclude highest and lowest if we have enough detectors)
            T combined;
            if (pointScores.Length > 2)
            {
                combined = NumOps.Zero;
                int count = pointScores.Length - 2;
                for (int j = 1; j <= count; j++)
                {
                    combined = NumOps.Add(combined, pointScores[j]);
                }
                combined = NumOps.Divide(combined, NumOps.FromDouble(count));
            }
            else
            {
                combined = NumOps.Zero;
                for (int j = 0; j < pointScores.Length; j++)
                {
                    combined = NumOps.Add(combined, pointScores[j]);
                }
                combined = NumOps.Divide(combined, NumOps.FromDouble(pointScores.Length));
            }

            combinedScores[i] = combined;
        }

        return combinedScores;
    }

    private Vector<T> NormalizeScores(Vector<T> scores)
    {
        var result = new Vector<T>(scores.Length);
        T min = NumOps.MaxValue;
        T max = NumOps.MinValue;

        for (int i = 0; i < scores.Length; i++)
        {
            if (NumOps.LessThan(scores[i], min)) min = scores[i];
            if (NumOps.GreaterThan(scores[i], max)) max = scores[i];
        }

        // Min-max normalization
        T range = NumOps.Subtract(max, min);
        T eps = NumOps.FromDouble(1e-10);

        if (NumOps.GreaterThan(range, eps))
        {
            for (int i = 0; i < scores.Length; i++)
            {
                result[i] = NumOps.Divide(NumOps.Subtract(scores[i], min), range);
            }
        }

        return result;
    }

    private Matrix<T> CreateRandomProjectionMatrix(int originalDimensions)
    {
        // Gaussian random projection
        var matrix = new Matrix<T>(originalDimensions, _nProjectedFeatures);
        double scale = 1.0 / Math.Sqrt(_nProjectedFeatures);

        for (int i = 0; i < originalDimensions; i++)
        {
            for (int j = 0; j < _nProjectedFeatures; j++)
            {
                // Box-Muller transform for Gaussian random numbers
                double u1 = 1.0 - _random.NextDouble();
                double u2 = 1.0 - _random.NextDouble();
                double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                matrix[i, j] = NumOps.FromDouble(z * scale);
            }
        }

        return matrix;
    }

    private Matrix<T> ApplyProjection(Matrix<T> X)
    {
        int n = X.Rows;
        int d = _nProjectedFeatures;
        var projected = new Matrix<T>(n, d);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < X.Columns; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(X[i, k], (_projectionMatrix ?? throw new InvalidOperationException("_projectionMatrix has not been initialized."))[k, j]));
                }
                projected[i, j] = sum;
            }
        }

        return projected;
    }
}
