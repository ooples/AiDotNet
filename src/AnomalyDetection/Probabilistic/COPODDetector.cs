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
public partial class COPODDetector<T> : AnomalyDetectorBase<T>
{
    [Buffer]
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
        var (sortedColumns, _) = Engine.TensorSort(AnomalyTensorOps.ToTensor(X), 0, false);
        for (int j = 0; j < _nFeatures; j++)
        {
            _sortedFeatureValues[j] = Engine.TensorSliceAxis(sortedColumns, 1, j).ToVector();
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

        T eps = NumOps.FromDouble(1e-10);
        var sortedValues = _sortedFeatureValues;
        if (sortedValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        var data = AnomalyTensorOps.ToTensor(X);
        var scores = Tensor<T>.CreateZeros(new[] { X.Rows });
        T half = NumOps.FromDouble(0.5);
        T denominator = NumOps.FromDouble(_nSamples + 1);

        // For each feature, compute the empirical tail probability
        for (int j = 0; j < _nFeatures; j++)
        {
            // Empirical CDF (left tail), adjusted to avoid 0 and 1: (count(values <= x) + 0.5) / (n + 1)
            var atOrBelow = engine.TensorSearchSorted(sortedValues[j].AsTensor(), engine.TensorSliceAxis(data, 1, j), true);
            var leftTailProb = engine.TensorDivideScalar(engine.TensorAddScalar(atOrBelow.Cast<T>(), half), denominator);
            var rightTailProb = engine.ScalarMinusTensor(NumOps.One, leftTailProb);

            // Add negative log of minimum tail probability, clamped away from 0 to avoid log(0)
            var minTailProb = engine.TensorMax(engine.TensorMin(leftTailProb, rightTailProb), eps);
            scores = engine.TensorSubtract(scores, engine.TensorLog(minTailProb));
        }

        return scores.ToVector();
    }
}
