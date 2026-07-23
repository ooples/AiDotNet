using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Probabilistic;

/// <summary>
/// Detects anomalies using Copula-Based Outlier Detection (COPOD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> COPOD uses copulas (statistical functions that describe dependencies
/// between variables) to model the joint probability of data points. Points with very low
/// joint probability are anomalies - they are statistically unlikely given the data distribution.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Transform each feature to empirical probability using ECDF
/// 2. Model the joint distribution using empirical copula
/// 3. Compute the negative log probability as the anomaly score
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When features have different marginal distributions
/// - When you want a parameter-free method
/// - High-dimensional data with complex dependencies
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Contamination: 0.1 (10%)
/// - No hyperparameters to tune (parameter-free method)
/// </para>
/// <para>
/// Reference: Li, Z., et al. (2020). "COPOD: Copula-Based Outlier Detection." ICDM.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("COPOD: Copula-Based Outlier Detection", "https://doi.org/10.1109/ICDM50108.2020.00135", Year = 2020, Authors = "Zheng Li, Yue Zhao, Nicola Botta, Cezar Ionescu, Xiyang Hu")]
public class COPODDetector<T> : AnomalyDetectorBase<T>
{
    private Vector<T>[]? _sortedFeatureValues;
    private int _nFeatures;
    private int _nSamples;

    /// <summary>
    /// Creates a new COPOD anomaly detector.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public COPODDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        _nFeatures = X.Columns;
        _nSamples = X.Rows;

        // Pre-sort feature values for O(log n) ECDF lookups
        _sortedFeatureValues = new Vector<T>[_nFeatures];
        for (int j = 0; j < _nFeatures; j++)
        {
            var values = new T[_nSamples];
            for (int i = 0; i < _nSamples; i++)
            {
                values[i] = X[i, j];
            }
            Array.Sort(values, (a, b) => NumOps.Compare(a, b));
            _sortedFeatureValues[j] = new Vector<T>(values);
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

        for (int i = 0; i < X.Rows; i++)
        {
            T score = NumOps.Zero;

            // For each feature, compute the empirical tail probability
            for (int j = 0; j < _nFeatures; j++)
            {
                T value = X[i, j];

                // Compute empirical CDF (left tail) and survival function (right tail)
                T leftTailProb = ComputeECDF(j, value);
                T rightTailProb = NumOps.Subtract(NumOps.One, leftTailProb);

                // Add negative log of minimum tail probability (skewness-corrected)
                T minTailProb = NumOps.LessThan(leftTailProb, rightTailProb) ? leftTailProb : rightTailProb;

                // Avoid log(0)
                if (NumOps.LessThan(minTailProb, eps)) minTailProb = eps;

                score = NumOps.Subtract(score, NumOps.Log(minTailProb));
            }

            scores[i] = score;
        }

        return scores;
    }

    private T ComputeECDF(int featureIndex, T value)
    {
        var sortedValues = _sortedFeatureValues;
        if (sortedValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Linear search for count of values <= value (binary search needs IComparable<T>)
        var featureValues = sortedValues[featureIndex];
        int count = 0;
        for (int i = 0; i < featureValues.Length; i++)
        {
            if (!NumOps.GreaterThan(featureValues[i], value))
            {
                count++;
            }
            else
            {
                break; // Values are sorted, so we can stop early
            }
        }

        // Return proportion, adjusted to avoid 0 and 1
        return NumOps.FromDouble((count + 0.5) / (_nSamples + 1));
    }
}
