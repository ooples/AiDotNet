using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Kruskal-Wallis H-test for non-parametric feature selection in classification.
/// </summary>
/// <remarks>
/// <para>
/// The Kruskal-Wallis test is a non-parametric version of one-way ANOVA. It compares
/// the rank distributions across classes rather than assuming normal distributions.
/// Features that lead to different rank distributions across classes are considered important.
/// </para>
/// <para><b>For Beginners:</b> While ANOVA assumes your data follows a bell curve,
/// Kruskal-Wallis makes no such assumption. It ranks all values and checks if different
/// classes tend to have systematically higher or lower ranks. It's more robust when
/// your data has outliers or isn't normally distributed.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KruskalWallis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _hStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? HStatistics => _hStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KruskalWallis(
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
            "KruskalWallis requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _hStatistics = new double[p];
        _pValues = new double[p];

        // Group samples by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = [];
            classGroups[label].Add(i);
        }

        int k = classGroups.Count;

        for (int j = 0; j < p; j++)
        {
            // Get values and compute ranks
            var values = new (double Value, int Index, int Class)[n];
            for (int i = 0; i < n; i++)
            {
                values[i] = (NumOps.ToDouble(data[i, j]), i, (int)Math.Round(NumOps.ToDouble(target[i])));
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

                double avgRank = (tieStart + pos + 2.0) / 2.0; // Ranks are 1-based
                for (int i = tieStart; i <= pos; i++)
                    ranks[values[i].Index] = avgRank;

                pos++;
            }

            // Compute sum of ranks for each group
            var rankSums = new Dictionary<int, double>();
            foreach (var kvp in classGroups)
            {
                double sum = 0;
                foreach (int i in kvp.Value)
                    sum += ranks[i];
                rankSums[kvp.Key] = sum;
            }

            // Compute H statistic
            // H = (12 / (n*(n+1))) * sum(R_i^2 / n_i) - 3*(n+1)
            double sumTerm = 0;
            foreach (var kvp in classGroups)
            {
                double Ri = rankSums[kvp.Key];
                int ni = kvp.Value.Count;
                sumTerm += (Ri * Ri) / ni;
            }

            _hStatistics[j] = (12.0 / (n * (n + 1))) * sumTerm - 3 * (n + 1);

            // Tie correction (optional for large samples)
            // For simplicity, we skip the exact tie correction

            // Approximate p-value using chi-square distribution with k-1 degrees of freedom
            _pValues[j] = ApproximateChiSquarePValue(_hStatistics[j], k - 1);
        }

        // Select top features by H-statistic
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _hStatistics
            .Select((h, idx) => (HStat: h, Index: idx))
            .OrderByDescending(x => x.HStat)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private static double ApproximateChiSquarePValue(double chiSquare, int df)
    {
        if (chiSquare <= 0 || df <= 0)
            return 1.0;

        // Use incomplete gamma function approximation
        // For chi-square: P(X > x) = Q(df/2, x/2) where Q is regularized gamma
        // Simple approximation using normal distribution for large df
        if (df > 30)
        {
            double z = Math.Pow(chiSquare / df, 1.0 / 3) - (1 - 2.0 / (9 * df));
            z /= Math.Sqrt(2.0 / (9 * df));
            return 0.5 * (1 - Erf(z / Math.Sqrt(2)));
        }

        // Simple exponential approximation for small df
        return Math.Exp(-chiSquare / 2);
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
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

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
        throw new NotSupportedException("KruskalWallis does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

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
