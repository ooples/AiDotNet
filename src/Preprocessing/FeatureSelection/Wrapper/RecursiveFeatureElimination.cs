using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Recursive Feature Elimination (RFE) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// RFE recursively removes the least important features based on model weights or
/// feature importances. At each step, a model is trained and the feature with the
/// lowest importance is eliminated until the desired number of features remains.
/// </para>
/// <para><b>For Beginners:</b> RFE works like an elimination tournament. It trains
/// a model, finds the weakest feature, removes it, and repeats. This continues until
/// only the strongest features remain. It's more thorough than one-shot methods
/// because feature importance changes as features are removed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RecursiveFeatureElimination<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _step;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _getImportances;
    private readonly int? _randomState;

    private double[]? _ranking;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int Step => _step;
    public double[]? Ranking => _ranking;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RecursiveFeatureElimination(
        int nFeaturesToSelect = 10,
        int step = 1,
        Func<Matrix<T>, Vector<T>, double[]>? getImportances = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (step < 1)
            throw new ArgumentException("Step must be at least 1.", nameof(step));

        _nFeaturesToSelect = nFeaturesToSelect;
        _step = step;
        _getImportances = getImportances;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RecursiveFeatureElimination requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var getImportances = _getImportances ?? DefaultGetImportances;

        // Track remaining features
        var remaining = Enumerable.Range(0, p).ToList();
        _ranking = new double[p];

        int currentRank = p;
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        while (remaining.Count > numToSelect)
        {
            // Get subset of data with remaining features
            var subsetData = ExtractColumns(data, remaining);

            // Get importances for remaining features
            var importances = getImportances(subsetData, target);

            // Find features to eliminate (lowest importance)
            int numToEliminate = Math.Min(_step, remaining.Count - numToSelect);

            var ranked = importances
                .Select((imp, idx) => (Importance: imp, LocalIdx: idx))
                .OrderBy(x => x.Importance)
                .Take(numToEliminate)
                .Select(x => x.LocalIdx)
                .OrderByDescending(x => x) // Remove from end first
                .ToList();

            foreach (int localIdx in ranked)
            {
                int globalIdx = remaining[localIdx];
                _ranking[globalIdx] = currentRank--;
                remaining.RemoveAt(localIdx);
            }
        }

        // Assign top ranks to remaining features
        foreach (int idx in remaining)
            _ranking[idx] = currentRank--;

        _selectedIndices = remaining.OrderBy(x => x).ToArray();

        IsFitted = true;
    }

    private double[] DefaultGetImportances(Matrix<T> data, Vector<T> target)
    {
        // Use absolute correlation as importance measure
        int n = data.Rows;
        int p = data.Columns;
        var importances = new double[p];

        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

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

            double corr = (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
            importances[j] = Math.Abs(corr);
        }

        return importances;
    }

    private Matrix<T> ExtractColumns(Matrix<T> data, List<int> indices)
    {
        int n = data.Rows;
        int p = indices.Count;
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                result[i, j] = data[i, indices[j]];

        return new Matrix<T>(result);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RecursiveFeatureElimination has not been fitted.");

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
        throw new NotSupportedException("RecursiveFeatureElimination does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RecursiveFeatureElimination has not been fitted.");

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
