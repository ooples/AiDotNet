using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Mann-Whitney U test for feature selection in binary classification.
/// </summary>
/// <remarks>
/// <para>
/// The Mann-Whitney U test is a non-parametric test that compares the rank distributions
/// of two groups. For binary classification, it tests whether one class tends to have
/// systematically higher values than the other for each feature.
/// </para>
/// <para><b>For Beginners:</b> Mann-Whitney U is the non-parametric version of the t-test
/// for comparing two groups. It ranks all values and checks if one class has consistently
/// higher or lower ranks than the other. It's robust to outliers and doesn't assume
/// your data follows any particular distribution.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MannWhitneyU<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _uStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? UStatistics => _uStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MannWhitneyU(
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
            "MannWhitneyU requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class (binary)
        var class0Indices = new List<int>();
        var class1Indices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (label == 0)
                class0Indices.Add(i);
            else
                class1Indices.Add(i);
        }

        if (class0Indices.Count == 0 || class1Indices.Count == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        int n1 = class0Indices.Count;
        int n2 = class1Indices.Count;

        _uStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get values and compute ranks
            var values = new (double Value, int Index, int Class)[n];
            for (int i = 0; i < n; i++)
            {
                int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
                values[i] = (NumOps.ToDouble(data[i, j]), i, classLabel);
            }

            // Sort by value
            Array.Sort(values, (a, b) => a.Value.CompareTo(b.Value));

            // Assign ranks (with ties getting average rank)
            var ranks = new double[n];
            int pos = 0;
            while (pos < n)
            {
                int tieStart = pos;
                while (pos < n - 1 && Math.Abs(values[pos].Value - values[pos + 1].Value) < 1e-10)
                    pos++;

                double avgRank = (tieStart + pos + 2.0) / 2.0;
                for (int i = tieStart; i <= pos; i++)
                    ranks[values[i].Index] = avgRank;

                pos++;
            }

            // Compute rank sum for class 0
            double R1 = 0;
            foreach (int i in class0Indices)
                R1 += ranks[i];

            // Compute U statistic for class 0
            double U1 = R1 - (n1 * (n1 + 1.0)) / 2;
            double U2 = (double)n1 * n2 - U1;

            // Use the smaller U for the test
            _uStatistics[j] = Math.Min(U1, U2);

            // Compute p-value using normal approximation
            double meanU = (double)n1 * n2 / 2;
            double stdU = Math.Sqrt((double)n1 * n2 * (n1 + n2 + 1) / 12);

            // Continuity correction
            double z = Math.Abs(_uStatistics[j] - meanU) / stdU;
            _pValues[j] = 2 * (1 - NormalCDF(z)); // Two-tailed
        }

        // Select features with smallest p-values (most significant)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private static double NormalCDF(double z)
    {
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private static double Erf(double x)
    {
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

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
        throw new NotSupportedException("MannWhitneyU does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

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
