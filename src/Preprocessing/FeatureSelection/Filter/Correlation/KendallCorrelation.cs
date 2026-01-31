using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Kendall Tau Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Kendall Tau measures the ordinal association between features and target
/// by counting concordant and discordant pairs. It's more robust than Spearman
/// for small samples and handles ties well.
/// </para>
/// <para><b>For Beginners:</b> Kendall's Tau looks at pairs of data points and asks:
/// "when one goes up, does the other also go up?" It counts how many pairs agree
/// versus disagree. It's particularly good when you have a small dataset or many
/// tied values, and gives a more intuitive probability interpretation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KendallCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _tauScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TauScores => _tauScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KendallCorrelation(
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
            "KendallCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _tauScores = new double[p];

        // Extract target values
        var targetValues = new double[n];
        for (int i = 0; i < n; i++)
            targetValues[i] = NumOps.ToDouble(target[i]);

        for (int j = 0; j < p; j++)
        {
            // Extract feature values
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = NumOps.ToDouble(data[i, j]);

            // Compute Kendall's Tau
            _tauScores[j] = ComputeKendallTau(featureValues, targetValues, n);
        }

        // Select top features by absolute Tau
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => Math.Abs(_tauScores[j]))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeKendallTau(double[] x, double[] y, int n)
    {
        int concordant = 0;
        int discordant = 0;
        int tiesX = 0;
        int tiesY = 0;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double xDiff = x[i] - x[j];
                double yDiff = y[i] - y[j];

                if (Math.Abs(xDiff) < 1e-10)
                {
                    tiesX++;
                    if (Math.Abs(yDiff) < 1e-10)
                        tiesY++;
                }
                else if (Math.Abs(yDiff) < 1e-10)
                {
                    tiesY++;
                }
                else if (xDiff * yDiff > 0)
                {
                    concordant++;
                }
                else
                {
                    discordant++;
                }
            }
        }

        int totalPairs = n * (n - 1) / 2;

        // Tau-b formula (handles ties)
        double n0 = totalPairs;
        double n1 = tiesX;
        double n2 = tiesY;

        double denom = Math.Sqrt((n0 - n1) * (n0 - n2));

        return denom > 0 ? (concordant - discordant) / denom : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallCorrelation has not been fitted.");

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
        throw new NotSupportedException("KendallCorrelation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallCorrelation has not been fitted.");

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
