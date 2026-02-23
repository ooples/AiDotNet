using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Backward Elimination (Sequential Backward Selection) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Starts with all features and greedily removes one feature at a time whose removal
/// causes the least degradation in performance until the desired number is reached.
/// </para>
/// <para><b>For Beginners:</b> Backward Elimination is like downsizing a team. You
/// start with everyone and repeatedly remove the person whose absence hurts performance
/// least. Unlike Forward Selection, this can find features that are only valuable
/// together, since they start included.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class BackwardElimination<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, int[], double>? _scoringFunction;

    private double[]? _removalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? RemovalScores => _removalScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BackwardElimination(
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

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BackwardElimination requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nToSelect = Math.Min(_nFeaturesToSelect, p);

        var remaining = Enumerable.Range(0, p).ToList();
        _removalScores = new double[p];

        // Initialize removal scores
        for (int j = 0; j < p; j++)
            _removalScores[j] = double.MaxValue;

        while (remaining.Count > nToSelect)
        {
            int worstFeature = -1;
            double bestScoreAfterRemoval = double.MinValue;

            foreach (int candidate in remaining)
            {
                var testSet = remaining.Where(x => x != candidate).ToArray();
                double score = EvaluateSubset(data, target, testSet);

                if (score > bestScoreAfterRemoval)
                {
                    bestScoreAfterRemoval = score;
                    worstFeature = candidate;
                }
            }

            if (worstFeature >= 0)
            {
                _removalScores[worstFeature] = bestScoreAfterRemoval;
                remaining.Remove(worstFeature);
            }
            else
            {
                break;
            }
        }

        _selectedIndices = remaining.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features)
    {
        if (features.Length == 0)
            return double.MinValue;

        if (_scoringFunction is not null)
            return _scoringFunction(data, target, features);

        return DefaultScoring(data, target, features);
    }

    private double DefaultScoring(Matrix<T> data, Vector<T> target, int[] features)
    {
        int n = data.Rows;
        if (features.Length == 0 || n <= features.Length)
            return 0;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double totalCorr = 0;
        foreach (int j in features)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
            {
                double corr = ssXY / Math.Sqrt(ssXX * ssYY);
                totalCorr += corr * corr;
            }
        }

        return totalCorr;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BackwardElimination has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BackwardElimination does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BackwardElimination has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
