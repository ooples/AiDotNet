using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Linear Discriminant Analysis Projection based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to class separation using
/// the ratio of between-class to within-class variance.
/// </para>
/// <para><b>For Beginners:</b> LDA finds directions that best separate classes.
/// This selector measures how much each feature contributes to these separating
/// directions, keeping features that help distinguish between groups.
/// </para>
/// </remarks>
public class LDAProjectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _ldaScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? LDAScores => _ldaScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LDAProjectionSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LDAProjectionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();
        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        // Compute global mean
        var globalMean = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                globalMean[j] += X[i, j];
            globalMean[j] /= n;
        }

        // Compute class means
        var classMeans = new Dictionary<int, double[]>();
        foreach (var c in classes)
        {
            classMeans[c] = new double[p];
            var indices = classIndices[c];
            for (int j = 0; j < p; j++)
            {
                foreach (int i in indices)
                    classMeans[c][j] += X[i, j];
                classMeans[c][j] /= indices.Count;
            }
        }

        _ldaScores = new double[p];

        // For each feature, compute LDA criterion (Sw / Sb ratio approximation)
        for (int j = 0; j < p; j++)
        {
            double withinClassVar = 0;
            double betweenClassVar = 0;

            foreach (var c in classes)
            {
                var indices = classIndices[c];
                double classMean = classMeans[c][j];

                // Within-class variance
                foreach (int i in indices)
                    withinClassVar += (X[i, j] - classMean) * (X[i, j] - classMean);

                // Between-class variance
                betweenClassVar += indices.Count * (classMean - globalMean[j]) * (classMean - globalMean[j]);
            }

            _ldaScores[j] = withinClassVar > 1e-10 ? betweenClassVar / withinClassVar : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _ldaScores[j])
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
            throw new InvalidOperationException("LDAProjectionSelector has not been fitted.");

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
        throw new NotSupportedException("LDAProjectionSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LDAProjectionSelector has not been fitted.");

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
