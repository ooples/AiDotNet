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
public class XGBODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _boostingRounds;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private double[]? _weights; // Simplified boosting weights
    private double[]? _featureMean;
    private double[]? _featureStd;
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

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        var tosScores = new List<double[]>(); // Transformed Outlier Scores

        int k = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            switch (e % 5)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: Math.Max(1, k + e),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: Math.Max(1, k + e),
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
            var scores = detector.ScoreAnomalies(X);
            var doubleScores = new double[n];
            for (int i = 0; i < n; i++)
            {
                doubleScores[i] = NumOps.ToDouble(scores[i]);
            }
            tosScores.Add(doubleScores);
        }

        // Create enhanced feature matrix [original features | TOS]
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;
        var enhancedData = new double[n][];

        for (int i = 0; i < n; i++)
        {
            enhancedData[i] = new double[nEnhancedFeatures];

            // Original features
            for (int j = 0; j < _nOriginalFeatures; j++)
            {
                enhancedData[i][j] = NumOps.ToDouble(X[i, j]);
            }

            // TOS features
            for (int e = 0; e < _nEstimators; e++)
            {
                enhancedData[i][_nOriginalFeatures + e] = tosScores[e][i];
            }
        }

        // Normalize enhanced features
        _featureMean = new double[nEnhancedFeatures];
        _featureStd = new double[nEnhancedFeatures];

        for (int j = 0; j < nEnhancedFeatures; j++)
        {
            _featureMean[j] = enhancedData.Average(row => row[j]);
            _featureStd[j] = Math.Sqrt(enhancedData.Average(row => Math.Pow(row[j] - _featureMean[j], 2)));
            if (_featureStd[j] < 1e-10) _featureStd[j] = 1;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < nEnhancedFeatures; j++)
            {
                enhancedData[i][j] = (enhancedData[i][j] - _featureMean[j]) / _featureStd[j];
            }
        }

        // Train simple gradient boosting on enhanced features
        // Using a simplified version with weighted feature importance
        _weights = TrainSimplifiedBoosting(enhancedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[] TrainSimplifiedBoosting(double[][] data)
    {
        int n = data.Length;
        int d = data[0].Length;

        // Initialize weights (uniform)
        var weights = new double[d];
        for (int j = 0; j < d; j++)
        {
            weights[j] = 1.0 / d;
        }

        // Sample weights (start uniform)
        var sampleWeights = new double[n];
        for (int i = 0; i < n; i++)
        {
            sampleWeights[i] = 1.0 / n;
        }

        // Create pseudo-labels based on TOS majority vote
        var pseudoLabels = new int[n];
        int nAnomaly = (int)(n * _contamination);

        var avgTos = new double[n];
        for (int i = 0; i < n; i++)
        {
            avgTos[i] = 0;
            for (int e = 0; e < _nEstimators; e++)
            {
                avgTos[i] += data[i][_nOriginalFeatures + e];
            }
            avgTos[i] /= _nEstimators;
        }

        var sortedIndices = avgTos.Select((v, i) => (v, i))
            .OrderByDescending(x => x.v)
            .Select(x => x.i)
            .ToArray();

        for (int i = 0; i < nAnomaly; i++)
        {
            pseudoLabels[sortedIndices[i]] = 1; // Anomaly
        }

        // Boosting iterations
        for (int round = 0; round < _boostingRounds; round++)
        {
            // Find best feature to split on
            double bestGain = double.MinValue;
            int bestFeature = 0;
            double bestThreshold = 0;

            for (int j = 0; j < d; j++)
            {
                // Try median as threshold
                var values = data.Select((row, i) => (row[j], pseudoLabels[i], sampleWeights[i])).ToArray();
                double threshold = values.OrderBy(v => v.Item1).Skip(n / 2).First().Item1;

                // Compute weighted error
                double error = 0;
                for (int i = 0; i < n; i++)
                {
                    int predicted = data[i][j] > threshold ? 1 : 0;
                    if (predicted != pseudoLabels[i])
                    {
                        error += sampleWeights[i];
                    }
                }

                // Information gain proxy
                double gain = 0.5 - error;
                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = j;
                    bestThreshold = threshold;
                }
            }

            // Update weights
            double alpha = 0.5 * Math.Log((1 - bestGain + 0.5) / (bestGain + 0.5 + 1e-10));
            weights[bestFeature] += alpha;

            // Update sample weights
            for (int i = 0; i < n; i++)
            {
                int predicted = data[i][bestFeature] > bestThreshold ? 1 : 0;
                if (predicted != pseudoLabels[i])
                {
                    sampleWeights[i] *= Math.Exp(Math.Abs(alpha));
                }
            }

            // Normalize sample weights
            double sumWeights = sampleWeights.Sum();
            for (int i = 0; i < n; i++)
            {
                sampleWeights[i] /= sumWeights;
            }
        }

        // Normalize feature weights
        double sumW = weights.Sum();
        for (int j = 0; j < d; j++)
        {
            weights[j] /= sumW;
        }

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

        int n = X.Rows;
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;

        // Get TOS for test data
        var tosScores = new List<double[]>();
        foreach (var detector in _baseDetectors!)
        {
            var scores = detector.ScoreAnomalies(X);
            var doubleScores = new double[n];
            for (int i = 0; i < n; i++)
            {
                doubleScores[i] = NumOps.ToDouble(scores[i]);
            }
            tosScores.Add(doubleScores);
        }

        var resultScores = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            // Build enhanced feature vector
            var enhanced = new double[nEnhancedFeatures];

            // Original features (normalized)
            for (int j = 0; j < _nOriginalFeatures; j++)
            {
                enhanced[j] = (NumOps.ToDouble(X[i, j]) - _featureMean![j]) / _featureStd![j];
            }

            // TOS features (normalized)
            for (int e = 0; e < _nEstimators; e++)
            {
                int j = _nOriginalFeatures + e;
                enhanced[j] = (tosScores[e][i] - _featureMean![j]) / _featureStd![j];
            }

            // Weighted sum as final score
            double score = 0;
            for (int j = 0; j < nEnhancedFeatures; j++)
            {
                score += _weights![j] * enhanced[j];
            }

            resultScores[i] = NumOps.FromDouble(score);
        }

        return resultScores;
    }
}
