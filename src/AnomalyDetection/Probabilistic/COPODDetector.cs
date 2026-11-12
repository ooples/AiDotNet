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
        // PyOD scores the training data against the training sample alone
        var trainingScores = ScoreAnomaliesInternal(X, false);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        // PyOD decision_function: the empirical distributions are taken over X_train and X together
        return ScoreAnomaliesInternal(X, true);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, bool includeQuery)
    {
        ValidateInput(X);

        var sortedValues = _sortedFeatureValues ?? throw new InvalidOperationException("Model not properly fitted.");
        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        // Tail terms U_left = -log F_left, U_right = -log F_right, and the skewness-selected tail
        // U_skew = U_left where the feature is left-skewed, U_right where right-skewed (both when unskewed)
        var engine = Engine;
        var (left, right, skewLeft, skewRight) = AnomalyTensorOps.EcdfTails(sortedValues, AnomalyTensorOps.ToTensor(X), includeQuery);
        var skew = engine.TensorAdd(AnomalyTensorOps.MultiplyRow(left, skewLeft), AnomalyTensorOps.MultiplyRow(right, skewRight));
        // O = max(U_skew, (U_left + U_right) / 2) per feature, summed over the features (PyOD COPOD)
        var outlyingness = engine.TensorMax(skew,
            engine.TensorMultiplyScalar(engine.TensorAdd(left, right), NumOps.FromDouble(0.5)));
        return engine.ReduceSum(outlyingness, new[] { 1 }, false).ToVector();
    }
}
