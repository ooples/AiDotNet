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
[ResearchPaper("ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions", "https://doi.org/10.1109/TKDE.2022.3159580", Year = 2022, Authors = "Zheng Li, Yue Zhao, Xiyang Hu, Nicola Botta, Cezar Ionescu, George H. Chen")]
public partial class ECODDetector<T> : AnomalyDetectorBase<T>
{
    [Buffer]
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
        // O = max(U_left, U_right, U_skew) per feature, summed over the features (PyOD ECOD)
        var outlyingness = engine.TensorMax(engine.TensorMax(left, right), skew);
        return engine.ReduceSum(outlyingness, new[] { 1 }, false).ToVector();
    }
}
