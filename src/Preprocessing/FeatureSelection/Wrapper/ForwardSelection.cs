using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Forward Selection (Sequential Forward Selection) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Starts with an empty set and greedily adds one feature at a time that provides
/// the best improvement until the desired number of features is reached.
/// </para>
/// <para><b>For Beginners:</b> Forward Selection is like building a team one person
/// at a time. At each step, you try adding each remaining candidate and keep the one
/// who helps the most. You continue until your team is the desired size. Simple but
/// can miss feature interactions that only appear together.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ForwardSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scoringFunction;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ForwardSelection(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, int[], double>? scoringFunction = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scoringFunction = scoringFunction;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ForwardSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        var selected = new List<int>();
        var remaining = Enumerable.Range(0, p).ToHashSet();
        _featureScores = new double[p];

        while (selected.Count < nToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int candidate in remaining)
            {
                var testSet = selected.Concat(new[] { candidate }).ToArray();
                double score = EvaluateSubset(data, target, testSet);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = candidate;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _featureScores[bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (_scoringFunction is not null)
            return _scoringFunction(data, target, features);

        return DefaultScoring(data, target, features);
    }

    private double DefaultScoring(Matrix<T> data, Vector<T> target, int[] features)
    {
        // R-squared of the least-squares fit (with intercept) of the target on the subset, as scikit-learn's
        // LinearRegression().fit(X, y).score(X, y); the sum of squared univariate correlations it replaces is only
        // the R-squared when the features are uncorrelated
        int n = data.Rows;
        if (features.Length == 0 || n <= features.Length)
            return 0;

        return WrapperScoringOps.LinearRegressionRSquared(data, target, features);
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
            throw new InvalidOperationException("ForwardSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ForwardSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ForwardSelection has not been fitted.");

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
