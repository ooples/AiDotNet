using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// ANCOVA (Analysis of Covariance) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on ANCOVA, which combines ANOVA with regression
/// to control for confounding continuous covariates.
/// </para>
/// <para><b>For Beginners:</b> ANCOVA tests group differences while accounting
/// for the effect of one or more continuous covariates. This helps isolate the
/// true effect of categorical grouping by removing the influence of confounding
/// variables. Features with significant adjusted F-statistics are selected.
/// </para>
/// </remarks>
public class ANCOVASelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _covariateIndex;

    private double[]? _adjustedFStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int CovariateIndex => _covariateIndex;
    public double[]? AdjustedFStatistics => _adjustedFStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ANCOVASelector(
        int nFeaturesToSelect = 10,
        int covariateIndex = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _covariateIndex = covariateIndex;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANCOVASelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (_covariateIndex < 0 || _covariateIndex >= p)
            throw new ArgumentException("Covariate index is out of range.");

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Extract covariate
        var covariate = new double[n];
        for (int i = 0; i < n; i++)
            covariate[i] = X[i, _covariateIndex];

        var classes = y.Distinct().OrderBy(c => c).ToList();
        int k = classes.Count;

        _adjustedFStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            if (j == _covariateIndex)
            {
                _adjustedFStatistics[j] = 0; // Skip covariate itself
                continue;
            }

            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            // Regress out covariate effect
            double covMean = covariate.Average();
            double colMean = col.Average();

            double covVar = covariate.Sum(v => (v - covMean) * (v - covMean));
            double cov = 0;
            for (int i = 0; i < n; i++)
                cov += (covariate[i] - covMean) * (col[i] - colMean);

            double slope = covVar > 1e-10 ? cov / covVar : 0;

            // Adjusted values (residuals)
            var adjusted = new double[n];
            for (int i = 0; i < n; i++)
                adjusted[i] = col[i] - slope * (covariate[i] - covMean);

            // ANOVA on adjusted values
            double grandMean = adjusted.Average();

            double ssBetween = 0;
            foreach (var c in classes)
            {
                var classIndices = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();
                double classMean = classIndices.Select(i => adjusted[i]).Average();
                ssBetween += classIndices.Count * (classMean - grandMean) * (classMean - grandMean);
            }

            double ssWithin = 0;
            foreach (var c in classes)
            {
                var classIndices = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();
                double classMean = classIndices.Select(i => adjusted[i]).Average();
                ssWithin += classIndices.Sum(i => (adjusted[i] - classMean) * (adjusted[i] - classMean));
            }

            int dfBetween = k - 1;
            int dfWithin = n - k - 1; // -1 for covariate

            if (dfWithin <= 0 || dfBetween <= 0)
            {
                _adjustedFStatistics[j] = 0;
                continue;
            }

            double msBetween = ssBetween / dfBetween;
            double msWithin = ssWithin / dfWithin;

            _adjustedFStatistics[j] = msWithin > 1e-10 ? msBetween / msWithin : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p - 1); // Exclude covariate
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => j != _covariateIndex)
            .OrderByDescending(j => _adjustedFStatistics[j])
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
            throw new InvalidOperationException("ANCOVASelector has not been fitted.");

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
        throw new NotSupportedException("ANCOVASelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANCOVASelector has not been fitted.");

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
