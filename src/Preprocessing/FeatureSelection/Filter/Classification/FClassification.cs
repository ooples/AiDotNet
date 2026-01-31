using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Classification;

/// <summary>
/// F-statistic based feature selection for classification (ANOVA F-test).
/// </summary>
/// <remarks>
/// <para>
/// F-Classification uses ANOVA F-test to score features based on their ability
/// to discriminate between class means. Features with high F-scores have
/// significantly different means across classes.
/// </para>
/// <para><b>For Beginners:</b> The F-test checks if the average values of a feature
/// are different across classes. If class A has very different values than class B
/// for a feature, that feature gets a high score because it helps tell the classes apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FClassification<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fScores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FScores => _fScores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FClassification(
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
            "FClassification requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = new List<int>();
            classGroups[label].Add(i);
        }

        int k = classGroups.Count; // Number of classes
        _fScores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute overall mean
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            // Compute between-group and within-group sum of squares
            double ssBetween = 0;
            double ssWithin = 0;

            foreach (var group in classGroups.Values)
            {
                double groupMean = 0;
                foreach (int idx in group)
                    groupMean += NumOps.ToDouble(data[idx, j]);
                groupMean /= group.Count;

                // Between-group variance
                ssBetween += group.Count * Math.Pow(groupMean - overallMean, 2);

                // Within-group variance
                foreach (int idx in group)
                {
                    double diff = NumOps.ToDouble(data[idx, j]) - groupMean;
                    ssWithin += diff * diff;
                }
            }

            // Compute F-statistic
            double dfBetween = k - 1;
            double dfWithin = n - k;

            double msBetween = dfBetween > 0 ? ssBetween / dfBetween : 0;
            double msWithin = dfWithin > 0 ? ssWithin / dfWithin : 0;

            _fScores[j] = msWithin > 1e-10 ? msBetween / msWithin : 0;
            _pValues[j] = ComputePValue(_fScores[j], (int)dfBetween, (int)dfWithin);
        }

        // Select top features by F-score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputePValue(double fStatistic, int df1, int df2)
    {
        if (fStatistic <= 0 || df1 <= 0 || df2 <= 0) return 1.0;

        // Approximation using normal distribution for large df
        if (df2 >= 30)
        {
            double z = Math.Pow(fStatistic, 1.0 / 3.0) * (1 - 2.0 / (9 * df2)) - (1 - 2.0 / (9 * df1));
            double se = Math.Sqrt(2.0 / (9 * df1) + 2.0 / (9 * df2));
            double standardZ = z / se;

            return 0.5 * (1 - Erf(standardZ / Math.Sqrt(2)));
        }

        return Math.Exp(-0.5 * fStatistic / Math.Max(1, df1));
    }

    private double Erf(double x)
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
            throw new InvalidOperationException("FClassification has not been fitted.");

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
        throw new NotSupportedException("FClassification does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FClassification has not been fitted.");

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
