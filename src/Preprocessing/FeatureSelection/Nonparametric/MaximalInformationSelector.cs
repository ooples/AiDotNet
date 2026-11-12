using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonparametric;

/// <summary>
/// Maximal Information Coefficient (MIC) feature selection (Reshef et al. 2011, "Detecting Novel Associations in
/// Large Data Sets", Science 334).
/// </summary>
/// <remarks>
/// <para>
/// Scores every feature by its MIC with the target and keeps the highest-scoring ones. MIC is the largest
/// normalized mutual information <c>I(X; Y) / log(min(x, y))</c> over the x-by-y grids with <c>x y &lt;= B(n)</c>,
/// where <c>B(n) = n^alpha</c>. It is computed with the paper's ApproxMaxMI heuristic exactly as minepy does
/// (equipartitioned rows, at most <c>c x</c> superclumps per column, the OptimizeXAxis dynamic program, both axis
/// orders).
/// </para>
/// <para><b>For Beginners:</b> MIC is like correlation but can detect any type
/// of relationship, not just linear ones. It scores relationships on a scale
/// of 0 to 1, where 1 means a perfect (possibly complex) relationship exists.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaximalInformationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly double _c;

    private Vector<T>? _micScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>The exponent of the grid budget <c>B(n) = n^alpha</c> (or B itself when it is at least 4).</summary>
    public double Alpha => _alpha;

    /// <summary>The clump factor: the x axis is split into at most <c>c</c> times as many superclumps as columns.</summary>
    public double C => _c;

    /// <summary>The MIC of every feature with the target.</summary>
    public Vector<T>? MICScores => _micScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a MIC feature selector.
    /// </summary>
    /// <param name="nFeaturesToSelect">The number of features to keep.</param>
    /// <param name="alpha">The grid budget exponent: <c>B(n) = max(n^alpha, 4)</c> for alpha in (0, 1], or
    /// <c>B = min(alpha, n)</c> for alpha of at least 4 (paper and minepy default 0.6).</param>
    /// <param name="c">The clump factor (paper and minepy default 15).</param>
    /// <param name="columnIndices">The columns to operate on, or null for all columns.</param>
    public MaximalInformationSelector(
        int nFeaturesToSelect = 10,
        double alpha = MaximalInformationCoefficient.DefaultAlpha,
        double c = MaximalInformationCoefficient.DefaultClumps,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        MaximalInformationCoefficient.Validate(alpha, c, nameof(alpha), nameof(c));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _c = c;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MaximalInformationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        var scores = MaximalInformationCoefficient.Scores(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target), _alpha, _c);
        _micScores = scores.ToVector();
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformationSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MaximalInformationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformationSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
