using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Recursive Feature Elimination for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// RFE performs feature selection by recursively removing features and building
/// a model on the remaining features. It ranks features by importance and removes
/// the least important ones until the desired number of features is reached.
/// </para>
/// <para>
/// The algorithm:
/// 1. Train a model on all features and compute feature importances
/// 2. Remove the least important feature(s)
/// 3. Repeat until desired number of features is reached
/// </para>
/// <para><b>For Beginners:</b> RFE is like an elimination tournament for features:
/// - Start with all features
/// - Remove the weakest performer each round
/// - Keep going until you have the desired number of features
/// - The surviving features are the most important ones
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RFE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _step;
    private readonly RFEImportanceMethod _importanceMethod;

    // Fitted parameters
    private int[]? _ranking; // Ranking for each feature (1 = selected, 2+ = eliminated)
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private Vector<T>? _featureImportances;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the step size (features removed per iteration).
    /// </summary>
    public int Step => _step;

    /// <summary>
    /// Gets the feature ranking (1 = selected, 2+ = elimination order).
    /// </summary>
    public int[]? Ranking => _ranking;

    /// <summary>
    /// Gets the feature importances from the final model.
    /// </summary>
    public Vector<T>? FeatureImportances => _featureImportances;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="RFE{T}"/>.
    /// </summary>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 5.</param>
    /// <param name="step">Number of features to remove at each iteration. Defaults to 1.</param>
    /// <param name="importanceMethod">Method for computing feature importance. Defaults to Correlation.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public RFE(
        int nFeaturesToSelect = 5,
        int step = 1,
        RFEImportanceMethod importanceMethod = RFEImportanceMethod.Correlation,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (step < 1)
        {
            throw new ArgumentException("Step must be at least 1.", nameof(step));
        }

        _nFeaturesToSelect = nFeaturesToSelect;
        _step = step;
        _importanceMethod = importanceMethod;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RFE requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits RFE by recursively eliminating features.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Track remaining features (ascending) and the elimination round of each eliminated feature
        var remaining = new SortedSet<int>(Enumerable.Range(0, p));
        var eliminatedInRound = new int[p];
        int rounds = 0;

        // Recursive elimination
        while (remaining.Count > nToSelect)
        {
            rounds++;
            var features = remaining.ToArray();

            // Compute importances for remaining features
            var importances = ComputeImportances(x, y, features);

            // Determine how many to remove this iteration
            int nToRemove = Math.Min(_step, remaining.Count - nToSelect);

            // Remove the least important features (lowest index first among ties)
            foreach (int position in PreprocessingTensorOps.StableOrder(importances, descending: false).Take(nToRemove))
            {
                eliminatedInRound[features[position]] = rounds;
                remaining.Remove(features[position]);
            }
        }

        // Rank like scikit-learn's RFE.ranking_: 1 for the selected features; a feature eliminated in round r of
        // R rounds gets R - r + 2, so the features removed together share a rank and the last ones removed rank 2
        _ranking = eliminatedInRound.Select(round => round == 0 ? 1 : rounds - round + 2).ToArray();

        // Create support mask and selected indices
        _supportMask = _ranking.Select(rank => rank == 1).ToArray();
        _selectedIndices = remaining.ToArray();

        // Compute final feature importances on selected features (0 for the eliminated ones)
        var finalImportances = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (_selectedIndices.Length > 0)
        {
            finalImportances = Engine.TensorIndexCopy(finalImportances, 0,
                new Tensor<int>((int[])_selectedIndices.Clone(), new[] { _selectedIndices.Length }),
                ComputeImportances(x, y, _selectedIndices));
        }

        _featureImportances = finalImportances.ToVector();

        IsFitted = true;
    }

    /// <summary>
    /// Importances ([k]) of the given columns of <paramref name="x"/> for <paramref name="y"/>: |Pearson r| (0 for a
    /// constant column or target), |coefficient| of the least-squares fit with intercept on all of them (as
    /// scikit-learn's RFE with LinearRegression), or the univariate regression F statistic
    /// <c>r^2 / ((1 - r^2) / (n - 2))</c> (scikit-learn's f_regression; 0 for a constant column or target).
    /// </summary>
    private Tensor<T> ComputeImportances(Tensor<T> x, Tensor<T> y, int[] featureIndices)
    {
        int n = x.Shape[0];
        int k = featureIndices.Length;
        var columns = Engine.TensorIndexSelect(x, new Tensor<int>((int[])featureIndices.Clone(), new[] { k }), 1);

        switch (_importanceMethod)
        {
            case RFEImportanceMethod.Correlation:
                // Use absolute correlation as importance
                return Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(columns, y));

            case RFEImportanceMethod.LinearCoefficients:
                // Use absolute value of OLS coefficients
                return Engine.TensorAbs(WrapperScoringOps.LeastSquaresSlopes(columns, y, out _, out _));

            case RFEImportanceMethod.FStatistic:
                return FStatistics(columns, y, n);

            default:
                return PreprocessingTensorOps.Filled(new[] { k }, NumOps.Zero);
        }
    }

    private Tensor<T> FStatistics(Tensor<T> columns, Tensor<T> y, int n)
    {
        int k = columns.Shape[1];
        var zeros = PreprocessingTensorOps.Filled(new[] { k }, NumOps.Zero);
        int df2 = n - 2;
        T epsilon = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var centered = FilterStatisticTensorOps.CenterColumns(columns);
        var yc = Engine.Reshape(Engine.TensorSubtractScalar(y, Engine.TensorMean(y)), new[] { n, 1 });
        T ssYY = Engine.TensorSumOfSquares(yc);
        if (df2 <= 0 || NumOps.LessThan(ssYY, epsilon))
            return zeros;

        var ssXX = SelectorTensorOps.ColumnSumOfSquares(centered);
        var varying = PreprocessingTensorOps.AtLeast(ssXX, epsilon);
        // r^2 = ssXY^2 / (ssXX ssYY) from r of the max-magnitude-scaled columns: ssXY^2 and ssXX ssYY overflow a
        // narrow T such as Half (Half's 65504 is passed by ssXY = 256)
        var r = PreprocessingTensorOps.CenteredCorrelations(centered, yc);
        var r2 = Engine.TensorMultiply(r, r);
        var f = Engine.TensorDivide(r2,
            Engine.TensorDivideScalar(PreprocessingTensorOps.OneMinus(r2), NumOps.FromDouble(df2)));
        return Engine.TensorWhere(varying, f, zeros);
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting the most important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("RFE has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RFE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("RFE has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}

/// <summary>
/// Specifies the method for computing feature importance in RFE.
/// </summary>
public enum RFEImportanceMethod
{
    /// <summary>
    /// Use absolute correlation with target.
    /// </summary>
    Correlation,

    /// <summary>
    /// Use absolute value of linear regression coefficients.
    /// </summary>
    LinearCoefficients,

    /// <summary>
    /// Use F-statistic from univariate regression.
    /// </summary>
    FStatistic
}
