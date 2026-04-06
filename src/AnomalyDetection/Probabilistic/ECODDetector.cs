using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using ECOD (Empirical Cumulative Distribution Functions for Outlier Detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ECOD uses the cumulative distribution function (CDF) to identify
/// outliers. Points with extreme values in any dimension have low probability under the
/// empirical distribution and are flagged as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute empirical CDF for each feature
/// 2. Calculate tail probabilities for each point
/// 3. Combine probabilities across features
/// 4. Points with very low probabilities are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Large datasets (linear complexity O(n))
/// - High-dimensional data
/// - When you need a fast, parameter-free method
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - No parameters to tune (parameter-free)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Li, Z., et al. (2022). "ECOD: Unsupervised Outlier Detection Using
/// Empirical Cumulative Distribution Functions." IEEE TKDE.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ModelPaper("ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions", "https://doi.org/10.1109/TKDE.2022.3159580", Year = 2022, Authors = "Zheng Li, Yue Zhao, Xiyang Hu, Nicola Botta, Cezar Ionescu, George H. Chen")]
public class ECODDetector<T> : AnomalyDetectorBase<T>
{
    private Vector<T>[]? _sortedFeatureValues;
    private int _nFeatures;
    private int _nTrainingSamples;

    /// <summary>
    /// Creates a new ECOD anomaly detector.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public ECODDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _nTrainingSamples = X.Rows;

        // Store sorted values for each feature to compute empirical CDF
        _sortedFeatureValues = new Vector<T>[_nFeatures];

        for (int j = 0; j < _nFeatures; j++)
        {
            var featureValues = new T[X.Rows];
            for (int i = 0; i < X.Rows; i++)
            {
                featureValues[i] = X[i, j];
            }
            Array.Sort(featureValues, (a, b) => NumOps.Compare(a, b));
            _sortedFeatureValues[j] = new Vector<T>(featureValues);
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

        var scores = new Vector<T>(X.Rows);

        T eps = NumOps.FromDouble(1e-10);
        var sortedVals = _sortedFeatureValues ?? throw new InvalidOperationException("_sortedFeatureValues has not been initialized.");

        for (int i = 0; i < X.Rows; i++)
        {
            T combinedScore = NumOps.Zero;

            for (int j = 0; j < _nFeatures; j++)
            {
                T value = X[i, j];

                // Compute empirical CDF: F(x) = proportion of values <= x
                T leftCdf = ComputeECDF(value, sortedVals[j]);
                T rightCdf = NumOps.Subtract(NumOps.One, leftCdf);

                // Use the minimum of left and right tail probabilities
                T tailProb = NumOps.LessThan(leftCdf, rightCdf) ? leftCdf : rightCdf;

                // Avoid log(0) by clamping
                if (NumOps.LessThan(tailProb, eps)) tailProb = eps;

                // Negative log probability (higher = more anomalous)
                combinedScore = NumOps.Subtract(combinedScore, NumOps.Log(tailProb));
            }

            scores[i] = combinedScore;
        }

        return scores;
    }

    private T ComputeECDF(T value, Vector<T> sortedValues)
    {
        // Linear search for count of values <= value (sorted array)
        int left = 0;
        int right = sortedValues.Length - 1;

        while (left <= right)
        {
            int mid = (left + right) / 2;
            if (!NumOps.GreaterThan(sortedValues[mid], value))
                left = mid + 1;
            else
                right = mid - 1;
        }

        // CDF = (number of values <= x) / n
        return NumOps.FromDouble((double)left / sortedValues.Length);
    }
}
