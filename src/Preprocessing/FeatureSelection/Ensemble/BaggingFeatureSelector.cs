using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Bagging-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap aggregation (bagging) to create multiple subsamples and
/// aggregates feature importance across all subsamples for robust selection.
/// </para>
/// <para><b>For Beginners:</b> Bagging creates many random subsets of your data,
/// evaluates feature importance on each subset, then combines the results. This
/// helps ensure that the selected features are consistently important across
/// different parts of your data, not just important by chance.
/// </para>
/// </remarks>
public class BaggingFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nEstimators;
    private readonly double _subsampleRatio;
    private readonly int? _randomState;

    private double[]? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NEstimators => _nEstimators;
    public double[]? AggregatedScores => _aggregatedScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public BaggingFeatureSelector(
        int nFeaturesToSelect = 10,
        int nEstimators = 10,
        double subsampleRatio = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nEstimators = nEstimators;
        _subsampleRatio = subsampleRatio;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BaggingFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int subsampleSize = (int)(n * _subsampleRatio);

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        _aggregatedScores = new double[p];
        var scoreCounts = new int[p];

        for (int est = 0; est < _nEstimators; est++)
        {
            // Bootstrap sample
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => rand.Next())
                .Take(subsampleSize)
                .ToList();

            // Compute correlation-based importance on subsample
            for (int j = 0; j < p; j++)
            {
                double xMean = 0, yMean = 0;
                foreach (int i in indices)
                {
                    xMean += X[i, j];
                    yMean += y[i];
                }
                xMean /= indices.Count;
                yMean /= indices.Count;

                double sxy = 0, sxx = 0, syy = 0;
                foreach (int i in indices)
                {
                    double xd = X[i, j] - xMean;
                    double yd = y[i] - yMean;
                    sxy += xd * yd;
                    sxx += xd * xd;
                    syy += yd * yd;
                }

                double corr = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
                _aggregatedScores[j] += corr;
                scoreCounts[j]++;
            }
        }

        // Average the scores
        for (int j = 0; j < p; j++)
            if (scoreCounts[j] > 0)
                _aggregatedScores[j] /= scoreCounts[j];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _aggregatedScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BaggingFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("BaggingFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BaggingFeatureSelector has not been fitted.");

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
