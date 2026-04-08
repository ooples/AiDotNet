using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using XGBOD (Extreme Gradient Boosting Outlier Detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> XGBOD is a semi-supervised method that creates new features from
/// unsupervised outlier detection scores, then trains a supervised classifier on these
/// enhanced features. It combines the best of both worlds.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple unsupervised outlier detectors
/// 2. Generate outlier scores as new features (TOS: Transformed Outlier Scores)
/// 3. Combine TOS with original features
/// 4. Train gradient boosting classifier on enhanced features
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have some labeled anomaly examples
/// - When feature engineering with outlier scores helps
/// - As a powerful ensemble method
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10 unsupervised detectors
/// - Boosting rounds: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., Hryniewicki, M.K. (2018). "XGBOD: Improving Supervised Outlier
/// Detection with Unsupervised Representation Learning." IJCNN.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning", "https://doi.org/10.1109/IJCNN.2018.8489605", Year = 2018, Authors = "Yue Zhao, Maciej K. Hryniewicki")]
public class XGBODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _boostingRounds;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private Vector<T>? _weights; // Simplified boosting weights
    private Vector<T>? _featureMean;
    private Vector<T>? _featureStd;
    private int _nOriginalFeatures;

    /// <summary>
    /// Gets the number of base estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the number of boosting rounds.
    /// </summary>
    public int BoostingRounds => _boostingRounds;

    /// <summary>
    /// Creates a new XGBOD anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of unsupervised base detectors. Default is 10.</param>
    /// <param name="boostingRounds">Number of boosting iterations. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public XGBODDetector(int nEstimators = 10, int boostingRounds = 100,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 1. Recommended is 10.");
        }

        if (boostingRounds < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(boostingRounds),
                "BoostingRounds must be at least 1. Recommended is 100.");
        }

        _nEstimators = nEstimators;
        _boostingRounds = boostingRounds;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nOriginalFeatures = X.Columns;

        // Guard against tiny datasets - need at least 3 samples for meaningful detection
        if (n < 3)
        {
            throw new ArgumentException(
                $"XGBOD requires at least 3 samples, but got {n}.",
                nameof(X));
        }

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        var tosScores = new List<Vector<T>>(); // Transformed Outlier Scores

        // Base k for neighbor-based detectors, capped at n - 1
        int baseK = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            // Cap neighbor count at n - 1 to avoid exceeding sample count
            int neighborCount = Math.Min(Math.Max(1, baseK + e), n - 1);

            switch (e % 5)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: neighborCount,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: neighborCount,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 2:
                    detector = new TreeBased.IsolationForest<T>(
                        numTrees: 50 + e * 10,
                        maxSamples: Math.Min(256, n),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 3:
                    detector = new Statistical.ZScoreDetector<T>(
                        zThreshold: 2.5 + e * 0.1,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                default:
                    detector = new Statistical.IQRDetector<T>(
                        multiplier: 1.5 + e * 0.1,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
            }

            detector.Fit(X);
            _baseDetectors.Add(detector);

            // Get TOS (transformed outlier scores)
            tosScores.Add(detector.ScoreAnomalies(X));
        }

        // Create enhanced feature matrix [original features | TOS]
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;
        var enhancedData = new Matrix<T>(n, nEnhancedFeatures);

        for (int i = 0; i < n; i++)
        {
            // Original features
            for (int j = 0; j < _nOriginalFeatures; j++)
            {
                enhancedData[i, j] = X[i, j];
            }

            // TOS features
            for (int e = 0; e < _nEstimators; e++)
            {
                enhancedData[i, _nOriginalFeatures + e] = tosScores[e][i];
            }
        }

        // Normalize enhanced features
        _featureMean = new Vector<T>(nEnhancedFeatures);
        _featureStd = new Vector<T>(nEnhancedFeatures);
        T nT = NumOps.FromDouble(n);
        T eps = NumOps.FromDouble(1e-10);

        for (int j = 0; j < nEnhancedFeatures; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, enhancedData[i, j]);
            }
            _featureMean[j] = NumOps.Divide(sum, nT);

            T varSum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                T diff = NumOps.Subtract(enhancedData[i, j], _featureMean[j]);
                varSum = NumOps.Add(varSum, NumOps.Multiply(diff, diff));
            }
            _featureStd[j] = NumOps.Sqrt(NumOps.Divide(varSum, nT));
            if (NumOps.LessThan(_featureStd[j], eps)) _featureStd[j] = NumOps.One;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < nEnhancedFeatures; j++)
            {
                enhancedData[i, j] = NumOps.Divide(NumOps.Subtract(enhancedData[i, j], _featureMean[j]), _featureStd[j]);
            }
        }

        // Train simple gradient boosting on enhanced features
        _weights = TrainSimplifiedBoosting(enhancedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Vector<T> TrainSimplifiedBoosting(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        // Initialize weights (uniform)
        var weights = new Vector<T>(d);
        T initW = NumOps.Divide(NumOps.One, NumOps.FromDouble(d));
        for (int j = 0; j < d; j++)
        {
            weights[j] = initW;
        }

        // Sample weights (start uniform)
        var sampleWeights = new Vector<T>(n);
        T initSW = NumOps.Divide(NumOps.One, NumOps.FromDouble(n));
        for (int i = 0; i < n; i++)
        {
            sampleWeights[i] = initSW;
        }

        // Create pseudo-labels based on TOS majority vote
        var pseudoLabels = new int[n];
        int nAnomaly = (int)(n * _contamination);
        T nEstT = NumOps.FromDouble(_nEstimators);

        var avgTos = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            avgTos[i] = NumOps.Zero;
            for (int e = 0; e < _nEstimators; e++)
            {
                avgTos[i] = NumOps.Add(avgTos[i], data[i, _nOriginalFeatures + e]);
            }
            avgTos[i] = NumOps.Divide(avgTos[i], nEstT);
        }

        var sortedIndices = Enumerable.Range(0, n)
            .OrderByDescending(i => NumOps.ToDouble(avgTos[i]))
            .ToArray();

        for (int i = 0; i < nAnomaly; i++)
        {
            pseudoLabels[sortedIndices[i]] = 1; // Anomaly
        }

        T half = NumOps.FromDouble(0.5);
        T eps = NumOps.FromDouble(1e-10);

        // Boosting iterations
        for (int round = 0; round < _boostingRounds; round++)
        {
            // Find best feature to split on
            T bestGain = NumOps.MinValue;
            int bestFeature = 0;
            T bestThreshold = NumOps.Zero;

            for (int j = 0; j < d; j++)
            {
                // Try median as threshold
                var colValues = new T[n];
                for (int i = 0; i < n; i++) colValues[i] = data[i, j];
                var sorted = colValues.OrderBy(v => NumOps.ToDouble(v)).ToArray();
                T threshold = sorted[n / 2];

                // Compute weighted error
                T error = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    int predicted = NumOps.GreaterThan(data[i, j], threshold) ? 1 : 0;
                    if (predicted != pseudoLabels[i])
                    {
                        error = NumOps.Add(error, sampleWeights[i]);
                    }
                }

                // Information gain proxy
                T gain = NumOps.Subtract(half, error);
                if (NumOps.GreaterThan(gain, bestGain))
                {
                    bestGain = gain;
                    bestFeature = j;
                    bestThreshold = threshold;
                }
            }

            // Update weights
            T numerator = NumOps.Add(NumOps.Subtract(NumOps.One, bestGain), half);
            T denominator = NumOps.Add(NumOps.Add(bestGain, half), eps);
            T alpha = NumOps.Multiply(half, NumOps.Log(NumOps.Divide(numerator, denominator)));
            weights[bestFeature] = NumOps.Add(weights[bestFeature], alpha);

            // Update sample weights
            T absAlpha = NumOps.Abs(alpha);
            T expAlpha = NumOps.Exp(absAlpha);
            for (int i = 0; i < n; i++)
            {
                int predicted = NumOps.GreaterThan(data[i, bestFeature], bestThreshold) ? 1 : 0;
                if (predicted != pseudoLabels[i])
                {
                    sampleWeights[i] = NumOps.Multiply(sampleWeights[i], expAlpha);
                }
            }

            // Normalize sample weights
            T sumWeights = NumOps.Zero;
            for (int i = 0; i < n; i++) sumWeights = NumOps.Add(sumWeights, sampleWeights[i]);
            for (int i = 0; i < n; i++) sampleWeights[i] = NumOps.Divide(sampleWeights[i], sumWeights);
        }

        // Normalize feature weights
        T sumW = NumOps.Zero;
        for (int j = 0; j < d; j++) sumW = NumOps.Add(sumW, weights[j]);
        for (int j = 0; j < d; j++) weights[j] = NumOps.Divide(weights[j], sumW);

        return weights;
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

        if (X.Columns != _nOriginalFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nOriginalFeatures} features.",
                nameof(X));
        }

        int n = X.Rows;
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;

        var featureMean = _featureMean ?? throw new InvalidOperationException("Feature mean not computed.");
        var featureStd = _featureStd ?? throw new InvalidOperationException("Feature std not computed.");
        var weights = _weights ?? throw new InvalidOperationException("Weights not computed.");

        // Get TOS for test data
        var tosScores = new List<Vector<T>>();
        foreach (var detector in _baseDetectors ?? throw new InvalidOperationException("Model not properly fitted."))
        {
            tosScores.Add(detector.ScoreAnomalies(X));
        }

        var resultScores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Build enhanced feature vector
            var enhanced = new Vector<T>(nEnhancedFeatures);

            // Original features (normalized)
            for (int j = 0; j < _nOriginalFeatures; j++)
            {
                enhanced[j] = NumOps.Divide(NumOps.Subtract(X[i, j], featureMean[j]), featureStd[j]);
            }

            // TOS features (normalized)
            for (int e = 0; e < _nEstimators; e++)
            {
                int j = _nOriginalFeatures + e;
                enhanced[j] = NumOps.Divide(NumOps.Subtract(tosScores[e][i], featureMean[j]), featureStd[j]);
            }

            // Weighted sum as final score using Engine.DotProduct
            T score = Engine.DotProduct(weights, enhanced);

            resultScores[i] = score;
        }

        return resultScores;
    }
}
