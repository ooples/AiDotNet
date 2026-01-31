using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// ANOVA F-Test based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their ANOVA F-statistic, measuring the ratio of
/// between-group variance to within-group variance.
/// </para>
/// <para><b>For Beginners:</b> ANOVA checks if groups differ significantly. Features
/// with high F-scores have very different values across classes, making them
/// useful for classification. Higher F means more class separation.
/// </para>
/// </remarks>
public class ANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ANOVASelector(
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
            "ANOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _fStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Overall mean
            double grandMean = 0;
            for (int i = 0; i < n; i++)
                grandMean += X[i, j];
            grandMean /= n;

            // Between-group sum of squares
            double ssb = 0;
            foreach (var c in classes)
            {
                var indices = classIndices[c];
                double classMean = indices.Average(i => X[i, j]);
                ssb += indices.Count * (classMean - grandMean) * (classMean - grandMean);
            }

            // Within-group sum of squares
            double ssw = 0;
            foreach (var c in classes)
            {
                var indices = classIndices[c];
                double classMean = indices.Average(i => X[i, j]);
                foreach (int i in indices)
                    ssw += (X[i, j] - classMean) * (X[i, j] - classMean);
            }

            int dfBetween = classes.Count - 1;
            int dfWithin = n - classes.Count;

            if (dfBetween > 0 && dfWithin > 0 && ssw > 1e-10)
            {
                double msBetween = ssb / dfBetween;
                double msWithin = ssw / dfWithin;
                _fStatistics[j] = msBetween / msWithin;
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fStatistics[j])
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
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

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
        throw new NotSupportedException("ANOVASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVASelector has not been fitted.");

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
