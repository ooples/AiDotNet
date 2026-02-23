using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// ANOVA F-test for feature selection in multi-class classification.
/// </summary>
/// <remarks>
/// <para>
/// Uses one-way Analysis of Variance to test if feature means differ significantly
/// across multiple classes. Features with high F-statistics indicate strong
/// discriminative power between classes.
/// </para>
/// <para><b>For Beginners:</b> ANOVA asks: "Does this feature have different average
/// values across the different classes?" If a feature has very similar averages
/// in all classes, it can't help tell them apart. Features with high F-scores
/// have means that vary significantly between classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ANOVAFTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _pValueThreshold;

    private double[]? _fStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ANOVAFTest(
        int nFeaturesToSelect = 10,
        double pValueThreshold = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _pValueThreshold = pValueThreshold;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ANOVAFTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(classLabel))
                classGroups[classLabel] = new List<int>();
            classGroups[classLabel].Add(i);
        }

        int k = classGroups.Count;
        if (k < 2)
            throw new ArgumentException("At least 2 classes are required for ANOVA.");

        _fStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Overall mean
            double grandMean = 0;
            for (int i = 0; i < n; i++)
                grandMean += NumOps.ToDouble(data[i, j]);
            grandMean /= n;

            // Between-group sum of squares
            double ssBetween = 0;
            // Within-group sum of squares
            double ssWithin = 0;

            foreach (var group in classGroups.Values)
            {
                double groupMean = 0;
                foreach (int i in group)
                    groupMean += NumOps.ToDouble(data[i, j]);
                groupMean /= group.Count;

                ssBetween += group.Count * Math.Pow(groupMean - grandMean, 2);

                foreach (int i in group)
                    ssWithin += Math.Pow(NumOps.ToDouble(data[i, j]) - groupMean, 2);
            }

            // Degrees of freedom
            int dfBetween = k - 1;
            int dfWithin = n - k;

            // Mean squares
            double msBetween = ssBetween / dfBetween;
            double msWithin = ssWithin / (dfWithin + 1e-10);

            // F-statistic
            _fStatistics[j] = msBetween / (msWithin + 1e-10);
            _pValues[j] = FDistributionPValue(_fStatistics[j], dfBetween, dfWithin);
        }

        // Select significant features
        var significant = new List<int>();
        for (int j = 0; j < p; j++)
            if (_pValues[j] < _pValueThreshold)
                significant.Add(j);

        if (significant.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = significant
                .OrderBy(j => _pValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _fStatistics
                .Select((f, idx) => (F: f, Index: idx))
                .OrderByDescending(x => x.F)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double FDistributionPValue(double f, int df1, int df2)
    {
        if (f <= 0 || df1 <= 0 || df2 <= 0) return 1;
        double x = df2 / (df2 + df1 * f);
        return IncompleteBeta(x, df2 / 2.0, df1 / 2.0);
    }

    private double IncompleteBeta(double x, double a, double b)
    {
        if (x < 0 || x > 1) return 1;
        if (x == 0) return 0;
        if (x == 1) return 1;

        double bt = Math.Exp(a * Math.Log(x) + b * Math.Log(1 - x));
        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(x, a, b) / a;
        else
            return 1 - bt * BetaCF(1 - x, b, a) / b;
    }

    private double BetaCF(double x, double a, double b)
    {
        double qab = a + b;
        double qap = a + 1;
        double qam = a - 1;
        double c = 1;
        double d = 1 - qab * x / qap;
        if (Math.Abs(d) < 1e-30) d = 1e-30;
        d = 1 / d;
        double h = d;

        for (int m = 1; m <= 100; m++)
        {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            h *= d * c;

            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1 + aa * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = 1 + aa / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1 / d;
            double del = d * c;
            h *= del;

            if (Math.Abs(del - 1) < 1e-7) break;
        }

        return h;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVAFTest has not been fitted.");

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
        throw new NotSupportedException("ANOVAFTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVAFTest has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
