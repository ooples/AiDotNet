using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Exhaustive Search Feature Selection (for small feature sets).
/// </summary>
/// <remarks>
/// <para>
/// Exhaustive search evaluates all possible feature subsets to find the optimal
/// combination. This guarantees finding the best subset but is only feasible
/// for small numbers of features (typically less than 20).
/// </para>
/// <para><b>For Beginners:</b> This is like trying every possible combination
/// of ingredients to find the perfect recipe. It's guaranteed to find the best
/// answer, but becomes impractical with many features because the number of
/// combinations grows exponentially (2^n for n features).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ExhaustiveSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxFeaturesForExhaustive;

    private double _bestScore;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double BestScore => _bestScore;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ExhaustiveSearch(
        int nFeaturesToSelect = 5,
        int maxFeaturesForExhaustive = 15,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxFeaturesForExhaustive < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeaturesForExhaustive));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxFeaturesForExhaustive = maxFeaturesForExhaustive;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ExhaustiveSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (p > _maxFeaturesForExhaustive)
            throw new InvalidOperationException(
                $"ExhaustiveSearch only supports up to {_maxFeaturesForExhaustive} features. " +
                $"Current data has {p} features. Use a different selection method.");

        int k = Math.Min(_nFeaturesToSelect, p);
        _bestScore = double.MinValue;
        _selectedIndices = null;

        // Generate all combinations of k features from p
        var indices = Enumerable.Range(0, p).ToArray();
        var combinations = GetCombinations(indices, k);

        foreach (var combo in combinations)
        {
            double score = EvaluateSubset(data, target, combo, n);
            if (score > _bestScore)
            {
                _bestScore = score;
                _selectedIndices = combo.ToArray();
            }
        }

        IsFitted = true;
    }

    private IEnumerable<int[]> GetCombinations(int[] elements, int k)
    {
        if (k == 0)
        {
            yield return Array.Empty<int>();
            yield break;
        }

        if (elements.Length < k)
            yield break;

        for (int i = 0; i <= elements.Length - k; i++)
        {
            int first = elements[i];
            var rest = elements.Skip(i + 1).ToArray();

            foreach (var combo in GetCombinations(rest, k - 1))
            {
                var result = new int[k];
                result[0] = first;
                for (int j = 0; j < combo.Length; j++)
                    result[j + 1] = combo[j];
                yield return result;
            }
        }
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] featureIndices, int n)
    {
        // Compute average absolute correlation with target
        double totalScore = 0;
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        foreach (int j in featureIndices)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
            totalScore += corr;
        }

        // Penalize redundancy
        double redundancy = 0;
        for (int i = 0; i < featureIndices.Length; i++)
        {
            for (int j = i + 1; j < featureIndices.Length; j++)
            {
                redundancy += ComputeFeatureCorrelation(data, featureIndices[i], featureIndices[j], n);
            }
        }

        int pairCount = featureIndices.Length * (featureIndices.Length - 1) / 2;
        double avgRedundancy = pairCount > 0 ? redundancy / pairCount : 0;

        return totalScore / featureIndices.Length - 0.5 * avgRedundancy;
    }

    private double ComputeFeatureCorrelation(Matrix<T> data, int j1, int j2, int n)
    {
        double mean1 = 0, mean2 = 0;
        for (int i = 0; i < n; i++)
        {
            mean1 += NumOps.ToDouble(data[i, j1]);
            mean2 += NumOps.ToDouble(data[i, j2]);
        }
        mean1 /= n;
        mean2 /= n;

        double s12 = 0, s11 = 0, s22 = 0;
        for (int i = 0; i < n; i++)
        {
            double d1 = NumOps.ToDouble(data[i, j1]) - mean1;
            double d2 = NumOps.ToDouble(data[i, j2]) - mean2;
            s12 += d1 * d2;
            s11 += d1 * d1;
            s22 += d2 * d2;
        }

        return (s11 > 1e-10 && s22 > 1e-10) ? Math.Abs(s12 / Math.Sqrt(s11 * s22)) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExhaustiveSearch has not been fitted.");

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
        throw new NotSupportedException("ExhaustiveSearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ExhaustiveSearch has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
