using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using Feature Bagging ensemble method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Feature Bagging creates multiple anomaly detectors, each trained on
/// a random subset of features. The final score is the combination of all detector scores.
/// This helps handle high-dimensional data where different feature subsets may reveal different anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Create n_estimators base detectors
/// 2. Each detector uses a random subset of features
/// 3. Train each detector on its feature subset
/// 4. Combine scores using averaging or maximum
/// </para>
/// <para>
/// <b>When to use:</b>
/// - High-dimensional data
/// - When different feature combinations may reveal different anomalies
/// - When you want more robust detection than single-detector methods
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10
/// - Max features: 0.5 (50% of features per detector)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Lazarevic, A., Kumar, V. (2005). "Feature Bagging for Outlier Detection." KDD.
/// </para>
/// </remarks>
public class FeatureBaggingDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly double _maxFeatures;
    private readonly CombinationMethod _combination;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private List<int[]>? _featureSubsets;
    private Matrix<T>? _trainingData;

    /// <summary>
    /// Gets the number of estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the maximum features fraction.
    /// </summary>
    public double MaxFeatures => _maxFeatures;

    /// <summary>
    /// Method for combining detector scores.
    /// </summary>
    public enum CombinationMethod
    {
        /// <summary>Average of all detector scores.</summary>
        Average,
        /// <summary>Maximum of all detector scores.</summary>
        Maximum
    }

    /// <summary>
    /// Creates a new Feature Bagging anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of base detectors. Default is 10.</param>
    /// <param name="maxFeatures">
    /// Fraction of features to use per detector. Default is 0.5 (50%).
    /// </param>
    /// <param name="combination">Method for combining scores. Default is Average.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public FeatureBaggingDetector(int nEstimators = 10, double maxFeatures = 0.5,
        CombinationMethod combination = CombinationMethod.Average,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 1. Recommended is 10.");
        }

        if (maxFeatures <= 0 || maxFeatures > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxFeatures),
                "MaxFeatures must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.5.");
        }

        _nEstimators = nEstimators;
        _maxFeatures = maxFeatures;
        _combination = combination;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _trainingData = X;
        int nFeatures = X.Columns;
        int subsetSize = Math.Max(1, (int)(nFeatures * _maxFeatures));

        // Create base detectors and feature subsets
        _baseDetectors = new List<IAnomalyDetector<T>>();
        _featureSubsets = new List<int[]>();

        for (int e = 0; e < _nEstimators; e++)
        {
            // Generate random feature subset
            var featureSubset = GenerateFeatureSubset(nFeatures, subsetSize, e);
            _featureSubsets.Add(featureSubset);

            // Create subset data
            var subsetData = ExtractFeatureSubset(X, featureSubset);

            // Create and train base detector (using LOF as default)
            var detector = new DistanceBased.LocalOutlierFactor<T>(
                numNeighbors: Math.Min(10, X.Rows - 1),
                contamination: _contamination,
                randomSeed: _randomSeed + e);

            detector.Fit(subsetData);
            _baseDetectors.Add(detector);
        }

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

        var allScores = new List<Vector<T>>();

        // Get scores from each base detector
        for (int e = 0; e < _nEstimators; e++)
        {
            var subsetData = ExtractFeatureSubset(X, _featureSubsets![e]);
            var detectorScores = _baseDetectors![e].ScoreAnomalies(subsetData);
            allScores.Add(detectorScores);
        }

        // Combine scores
        var combinedScores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            double combined = 0;

            if (_combination == CombinationMethod.Average)
            {
                for (int e = 0; e < _nEstimators; e++)
                {
                    combined += NumOps.ToDouble(allScores[e][i]);
                }
                combined /= _nEstimators;
            }
            else // Maximum
            {
                combined = double.MinValue;
                for (int e = 0; e < _nEstimators; e++)
                {
                    double score = NumOps.ToDouble(allScores[e][i]);
                    if (score > combined) combined = score;
                }
            }

            combinedScores[i] = NumOps.FromDouble(combined);
        }

        return combinedScores;
    }

    private int[] GenerateFeatureSubset(int totalFeatures, int subsetSize, int seed)
    {
        var random = new Random(_randomSeed + seed);
        var allFeatures = Enumerable.Range(0, totalFeatures).ToList();

        // Shuffle
        for (int i = allFeatures.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            int temp = allFeatures[i];
            allFeatures[i] = allFeatures[j];
            allFeatures[j] = temp;
        }

        return allFeatures.Take(subsetSize).OrderBy(x => x).ToArray();
    }

    private Matrix<T> ExtractFeatureSubset(Matrix<T> X, int[] featureIndices)
    {
        var subset = new Matrix<T>(X.Rows, featureIndices.Length);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < featureIndices.Length; j++)
            {
                subset[i, j] = X[i, featureIndices[j]];
            }
        }

        return subset;
    }
}
