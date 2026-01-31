using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Sequential Floating Forward Selection (SFFS) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SFFS is an advanced sequential search that combines forward selection with
/// conditional backward steps. After adding a feature, it tries removing previously
/// added features to see if removing them improves performance. This allows
/// correction of earlier suboptimal decisions.
/// </para>
/// <para><b>For Beginners:</b> Regular forward selection adds features one at a time
/// without looking back. SFFS is smarter - after adding a new feature, it reconsiders
/// whether any previously added features are now redundant. It's like building a team
/// and periodically reconsidering earlier hires as the team evolves.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SFFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly Func<Matrix<T>, Vector<T>, double>? _scorer;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SFFS(
        int nFeaturesToSelect = 10,
        Func<Matrix<T>, Vector<T>, double>? scorer = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _scorer = scorer;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SFFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var scorer = _scorer ?? DefaultScorer;

        var selected = new HashSet<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        double currentScore = double.NegativeInfinity;

        _featureImportances = new double[p];

        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            // Step 1: Forward - find best feature to add
            int bestToAdd = -1;
            double bestAddScore = double.NegativeInfinity;

            foreach (int j in remaining)
            {
                var testSet = selected.Union([j]).ToList();
                double score = EvaluateSubset(data, target, testSet, scorer);

                if (score > bestAddScore)
                {
                    bestAddScore = score;
                    bestToAdd = j;
                }
            }

            if (bestToAdd < 0) break;

            // Add the best feature
            selected.Add(bestToAdd);
            remaining.Remove(bestToAdd);
            currentScore = bestAddScore;
            _featureImportances[bestToAdd] = bestAddScore;

            // Step 2: Backward - try removing each selected feature (except the one just added)
            bool improved = true;
            while (improved && selected.Count > 1)
            {
                improved = false;
                int bestToRemove = -1;
                double bestRemoveScore = currentScore;

                foreach (int j in selected)
                {
                    if (j == bestToAdd) continue; // Don't immediately remove what we just added

                    var testSet = selected.Except([j]).ToList();
                    double score = EvaluateSubset(data, target, testSet, scorer);

                    if (score > bestRemoveScore)
                    {
                        bestRemoveScore = score;
                        bestToRemove = j;
                    }
                }

                if (bestToRemove >= 0)
                {
                    selected.Remove(bestToRemove);
                    remaining.Add(bestToRemove);
                    currentScore = bestRemoveScore;
                    improved = true;
                }
            }
        }

        _selectedIndices = [.. selected.OrderBy(x => x)];
        IsFitted = true;
    }

    private double EvaluateSubset(Matrix<T> data, Vector<T> target, List<int> subset, Func<Matrix<T>, Vector<T>, double> scorer)
    {
        if (subset.Count == 0) return double.NegativeInfinity;

        int n = data.Rows;
        var subData = new T[n, subset.Count];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < subset.Count; j++)
                subData[i, j] = data[i, subset[j]];

        return scorer(new Matrix<T>(subData), target);
    }

    private double DefaultScorer(Matrix<T> data, Vector<T> target)
    {
        // Use average absolute correlation as score
        int n = data.Rows;
        int p = data.Columns;

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double totalCorr = 0;
        for (int j = 0; j < p; j++)
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
            totalCorr += corr;
        }

        return totalCorr / p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SFFS has not been fitted.");

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
        throw new NotSupportedException("SFFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SFFS has not been fitted.");

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
