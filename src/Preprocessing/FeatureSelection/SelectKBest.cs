using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Select K Best features using a pluggable scoring function.
/// </summary>
/// <remarks>
/// <para>
/// SelectKBest is a simple feature selector that computes a score for each feature
/// using a provided scoring function and selects the k features with highest scores.
/// </para>
/// <para><b>For Beginners:</b> This is a versatile "pick the best k features" tool.
/// You provide a scoring function (like correlation, F-statistic, or mutual information),
/// and it simply picks the top k scoring features. It's scikit-learn compatible in concept.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectKBest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _k;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _scoreFunc;
    private readonly UnivariateScoreFunction _defaultScoreFunc;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int K => _k;
    public Vector<T>? Scores => _scores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SelectKBest(
        int k = 10,
        Func<Matrix<T>, Vector<T>, Vector<T>>? scoreFunc = null,
        UnivariateScoreFunction defaultScoreFunc = UnivariateScoreFunction.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (k < 1)
            throw new ArgumentException("k must be at least 1.", nameof(k));

        _k = k;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectKBest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Compute scores
        Tensor<T> scores;
        if (_scoreFunc != null)
        {
            scores = Tensor<T>.FromVector(_scoreFunc(data, target));
        }
        else
        {
            var x = Tensor<T>.FromMatrix(data);
            Tensor<T> pValues;
            switch (_defaultScoreFunc)
            {
                case UnivariateScoreFunction.FRegression:
                    scores = SelectorTensorOps.FRegression(x, target, out pValues);
                    _pValues = pValues.ToVector();
                    break;
                case UnivariateScoreFunction.Chi2:
                    scores = SelectorTensorOps.Chi2(x, target, out pValues);
                    _pValues = pValues.ToVector();
                    break;
                case UnivariateScoreFunction.MutualInfoClassif:
                    scores = SelectorTensorOps.MutualInfoClassif(x, target);
                    break;
                default:
                    scores = SelectorTensorOps.FClassif(x, target, out pValues);
                    _pValues = pValues.ToVector();
                    break;
            }
        }

        _scores = scores.ToVector();

        // Select the top k features; like scikit-learn (stable ascending argsort, last k) a tie at the cut keeps
        // the higher-indexed features
        int kActual = Math.Min(_k, p);
        var order = PreprocessingTensorOps.StableOrder(scores, descending: true, lastAmongTies: true);
        _selectedIndices = order.Take(kActual).OrderBy(index => index).ToArray();

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
            throw new InvalidOperationException("SelectKBest has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectKBest does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectKBest has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
