using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using Random Subspace ensemble method.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Random Subspace creates multiple detectors, each trained on a randomly
/// selected subset of features. This helps detect anomalies that may only be visible in certain
/// feature subspaces, which is common in high-dimensional data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Generate n_estimators random feature subsets
/// 2. Train a base detector on each subspace
/// 3. Combine scores using averaging or voting
/// </para>
/// <para>
/// <b>When to use:</b>
/// - High-dimensional data
/// - When anomalies hide in subspaces
/// - For robust detection across feature combinations
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 20
/// - Max features: sqrt(n_features)
/// - Combination: Average
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Keller, F., Muller, E., Bohm, K. (2012). "HiCS: High Contrast Subspaces for
/// Density-Based Outlier Ranking." ICDE.
/// </para>
/// </remarks>
public class RandomSubspaceDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _maxFeatures;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private List<int[]>? _featureSubsets;
    private int _nFeatures;

    /// <summary>
    /// Gets the number of estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the maximum features per subspace.
    /// </summary>
    public int MaxFeatures => _maxFeatures;

    /// <summary>
    /// Creates a new Random Subspace anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of base detectors. Default is 20.</param>
    /// <param name="maxFeatures">
    /// Max features per subspace. -1 means sqrt(n_features). Default is -1.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public RandomSubspaceDetector(int nEstimators = 20, int maxFeatures = -1,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 1. Recommended is 20.");
        }

        if (maxFeatures < -1 || maxFeatures == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxFeatures),
                "MaxFeatures must be -1 (auto-detect) or >= 1. Use -1 for sqrt(n_features).");
        }

        _nEstimators = nEstimators;
        _maxFeatures = maxFeatures;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;
        int d = _nFeatures;

        // Determine subspace size
        int subspaceSize = _maxFeatures > 0
            ? Math.Min(_maxFeatures, d)
            : Math.Max(1, (int)Math.Sqrt(d));

        // Create base detectors with random subspaces
        _baseDetectors = new List<IAnomalyDetector<T>>();
        _featureSubsets = new List<int[]>();

        for (int e = 0; e < _nEstimators; e++)
        {
            // Generate random feature subset
            var featureSubset = Enumerable.Range(0, d)
                .OrderBy(_ => _random.NextDouble())
                .Take(subspaceSize)
                .OrderBy(f => f)
                .ToArray();

            _featureSubsets.Add(featureSubset);

            // Extract subspace data
            var subspaceData = ExtractSubspace(X, featureSubset);

            // Train detector (using Isolation Forest as base)
            var detector = new TreeBased.IsolationForest<T>(
                numTrees: 50,
                maxSamples: Math.Min(256, n),
                contamination: _contamination,
                randomSeed: _randomSeed + e);

            detector.Fit(subspaceData);
            _baseDetectors.Add(detector);
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Matrix<T> ExtractSubspace(Matrix<T> X, int[] featureIndices)
    {
        var subspace = new Matrix<T>(X.Rows, featureIndices.Length);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < featureIndices.Length; j++)
            {
                subspace[i, j] = X[i, featureIndices[j]];
            }
        }

        return subspace;
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var allScores = new double[_nEstimators][];

        // Get scores from each detector
        for (int e = 0; e < _nEstimators; e++)
        {
            var subspaceData = ExtractSubspace(X, _featureSubsets![e]);
            var scores = _baseDetectors![e].ScoreAnomalies(subspaceData);

            allScores[e] = new double[X.Rows];
            for (int i = 0; i < X.Rows; i++)
            {
                allScores[e][i] = NumOps.ToDouble(scores[i]);
            }
        }

        // Combine scores (average)
        var combinedScores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            double avg = 0;
            for (int e = 0; e < _nEstimators; e++)
            {
                avg += allScores[e][i];
            }
            avg /= _nEstimators;

            combinedScores[i] = NumOps.FromDouble(avg);
        }

        return combinedScores;
    }
}
