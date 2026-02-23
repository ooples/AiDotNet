using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Sequential Floating Forward Selection (SFFS) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SFFS is an advanced wrapper method that performs forward selection with
/// conditional backward elimination. After each forward step, it attempts
/// to remove features that may have become redundant.
/// </para>
/// <para><b>For Beginners:</b> Regular forward selection adds features one
/// at a time but never reconsiders its choices. SFFS can "float" - after
/// adding a feature, it checks if any previously added features are now
/// unnecessary and removes them. This leads to better final selections.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FloatingSearch<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBacktrack;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxBacktrack => _maxBacktrack;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FloatingSearch(
        int nFeaturesToSelect = 10,
        int maxBacktrack = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (maxBacktrack < 0)
            throw new ArgumentException("Max backtrack must be non-negative.", nameof(maxBacktrack));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBacktrack = maxBacktrack;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FloatingSearch requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _featureScores = new double[p];
        var selected = new List<int>();
        var available = new HashSet<int>(Enumerable.Range(0, p));

        while (selected.Count < Math.Min(_nFeaturesToSelect, p))
        {
            // Forward step: add best feature
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int candidate in available)
            {
                var testSet = selected.Concat(new[] { candidate }).ToArray();
                double score = EvaluateSubset(data, target, testSet, n);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = candidate;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                available.Remove(bestFeature);
                _featureScores[bestFeature] = EvaluateFeature(data, target, bestFeature, n);
            }

            // Conditional exclusion (backtrack)
            int backtrackCount = 0;
            while (selected.Count > 2 && backtrackCount < _maxBacktrack)
            {
                int worstFeature = -1;
                double bestWithoutOne = EvaluateSubset(data, target, selected.ToArray(), n);

                foreach (int candidate in selected.Take(selected.Count - 1))
                {
                    var testSet = selected.Where(f => f != candidate).ToArray();
                    double score = EvaluateSubset(data, target, testSet, n);

                    if (score > bestWithoutOne)
                    {
                        bestWithoutOne = score;
                        worstFeature = candidate;
                    }
                }

                if (worstFeature >= 0)
                {
                    selected.Remove(worstFeature);
                    available.Add(worstFeature);
                    backtrackCount++;
                }
                else
                {
                    break;
                }
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, int[] features, int n)
    {
        if (features.Length == 0) return 0;

        double totalScore = 0;
        double[,] correlationMatrix = new double[features.Length, features.Length];

        for (int i = 0; i < features.Length; i++)
        {
            double score = EvaluateFeature(data, target, features[i], n);
            totalScore += score;

            for (int j = i + 1; j < features.Length; j++)
            {
                double corr = ComputeCorrelation(data, features[i], features[j], n);
                correlationMatrix[i, j] = corr;
                correlationMatrix[j, i] = corr;
            }
        }

        double redundancy = 0;
        for (int i = 0; i < features.Length; i++)
            for (int j = i + 1; j < features.Length; j++)
                redundancy += Math.Abs(correlationMatrix[i, j]);

        if (features.Length > 1)
            redundancy /= (features.Length * (features.Length - 1) / 2.0);

        return totalScore - 0.5 * redundancy;
    }

    private double EvaluateFeature(Matrix<T> data, Vector<T> target, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, j]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    private double ComputeCorrelation(Matrix<T> data, int j1, int j2, int n)
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

        return (s11 > 1e-10 && s22 > 1e-10) ? s12 / Math.Sqrt(s11 * s22) : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingSearch has not been fitted.");

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
        throw new NotSupportedException("FloatingSearch does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingSearch has not been fitted.");

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
