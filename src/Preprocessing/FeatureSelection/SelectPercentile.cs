using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Select features based on percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// SelectPercentile is similar to SelectKBest but instead of selecting a fixed number
/// of features, it selects features that score in the top percentile (e.g., top 10%).
/// </para>
/// <para><b>For Beginners:</b> Instead of saying "give me the best 5 features", you say
/// "give me the best 10% of features". This is useful when you don't know the total
/// number of features ahead of time or want a proportional selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly double _percentile;
    private readonly Func<Matrix<T>, Vector<T>, Vector<T>>? _scoreFunc;
    private readonly UnivariateScoreFunction _defaultScoreFunc;

    private Vector<T>? _scores;
    private Vector<T>? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double Percentile => _percentile;
    public Vector<T>? Scores => _scores;
    public Vector<T>? PValues => _pValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SelectPercentile(
        double percentile = 10.0,
        Func<Matrix<T>, Vector<T>, Vector<T>>? scoreFunc = null,
        UnivariateScoreFunction defaultScoreFunc = UnivariateScoreFunction.FClassif,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile <= 0 || percentile > 100)
            throw new ArgumentException("Percentile must be in (0, 100].", nameof(percentile));

        _percentile = percentile;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectPercentile requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        if (_percentile >= 100 || p == 0)
        {
            _selectedIndices = Enumerable.Range(0, p).ToArray();
            IsFitted = true;
            return;
        }

        // scikit-learn SelectPercentile: keep the scores above the (100 - percentile)-th percentile (linear
        // interpolation), then fill with the lowest-indexed ties at the threshold up to int(p * percentile / 100)
        var engine = Engine;
        var threshold = engine.TensorQuantile(scores, (100 - _percentile) / 100, null, false, QuantileInterpolation.Linear);
        var thresholdRow = engine.TensorBroadcastTo(engine.Reshape(threshold, new[] { 1 }), new[] { p });
        var above = PreprocessingTensorOps.MaskPositions(engine.TensorGreaterThan(scores, thresholdRow));
        var ties = PreprocessingTensorOps.MaskPositions(engine.TensorEquals(scores, thresholdRow));
        int maxFeatures = (int)(p * _percentile / 100);
        int room = maxFeatures - above.Length;

        // ties[:room] in Python: a negative room drops that many ties from the end
        int keptTies = room >= 0 ? Math.Min(room, ties.Length) : Math.Max(ties.Length + room, 0);
        _selectedIndices = above.Concat(ties.Take(keptTies)).OrderBy(index => index).ToArray();

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
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectPercentile does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SelectPercentile has not been fitted.");

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
