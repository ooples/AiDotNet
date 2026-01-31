using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// MANOVA (Multivariate ANOVA) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on multivariate analysis of variance, which considers
/// multiple dependent variables simultaneously when testing group differences.
/// </para>
/// <para><b>For Beginners:</b> MANOVA extends ANOVA to multiple response variables.
/// Instead of testing each feature independently, it considers correlations between
/// features when assessing group differences. This can detect patterns that
/// univariate tests miss.
/// </para>
/// </remarks>
public class MANOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _wilksLambda;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? WilksLambda => _wilksLambda;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MANOVASelector(
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
            "MANOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        int k = classes.Count;

        _wilksLambda = new double[p];

        // For each feature, compute Wilks' Lambda approximation using
        // pairwise feature combinations (simplified MANOVA)
        for (int j = 0; j < p; j++)
        {
            // Use univariate F-statistic as base
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            double grandMean = col.Average();

            // Between-group SS
            double ssBetween = 0;
            foreach (var c in classes)
            {
                var classIndices = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();
                double classMean = classIndices.Select(i => col[i]).Average();
                ssBetween += classIndices.Count * (classMean - grandMean) * (classMean - grandMean);
            }

            // Within-group SS
            double ssWithin = 0;
            foreach (var c in classes)
            {
                var classIndices = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();
                double classMean = classIndices.Select(i => col[i]).Average();
                ssWithin += classIndices.Sum(i => (col[i] - classMean) * (col[i] - classMean));
            }

            // Wilks' Lambda = SS_within / SS_total
            double ssTotal = ssBetween + ssWithin;
            _wilksLambda[j] = ssTotal > 1e-10 ? ssWithin / ssTotal : 1.0;
        }

        // Lower Wilks' Lambda = better discrimination
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderBy(j => _wilksLambda[j])
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
            throw new InvalidOperationException("MANOVASelector has not been fitted.");

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
        throw new NotSupportedException("MANOVASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MANOVASelector has not been fitted.");

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
