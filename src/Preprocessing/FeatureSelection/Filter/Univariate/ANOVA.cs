using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// ANOVA (Analysis of Variance) F-test for feature selection in classification problems.
/// </summary>
/// <remarks>
/// <para>
/// ANOVA F-test compares the variance between groups (classes) to the variance within groups.
/// A high F-statistic indicates that the feature discriminates well between classes.
/// This is equivalent to f_classif in scikit-learn.
/// </para>
/// <para><b>For Beginners:</b> ANOVA asks: "Is the difference between class means large
/// compared to the scatter within each class?" If class means are very different but
/// values within each class are similar, the feature is good at separating classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ANOVA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FStatistics => _fStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ANOVA(
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
            "ANOVA requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _fStatistics = new double[p];
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
        int dfBetween = k - 1;
        int dfWithin = n - k;

        for (int j = 0; j < p; j++)
        {
            // Compute overall mean
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            // Compute SS_between and SS_within
            double ssBetween = 0;
            double ssWithin = 0;

            foreach (var kvp in classGroups)
            {
                // Compute class mean
                double classMean = 0;
                foreach (int i in kvp.Value)
                    classMean += NumOps.ToDouble(data[i, j]);
                classMean /= kvp.Value.Count;

                // SS_between: sum of (class_mean - overall_mean)^2 * class_size
                ssBetween += kvp.Value.Count * (classMean - overallMean) * (classMean - overallMean);

                // SS_within: sum of (x - class_mean)^2 for each sample in class
                foreach (int i in kvp.Value)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean;
                    ssWithin += diff * diff;
                }
            }

            // Compute F-statistic
            double msBetween = ssBetween / Math.Max(1, dfBetween);
            double msWithin = ssWithin / Math.Max(1, dfWithin);

            _fStatistics[j] = msWithin > 1e-10 ? msBetween / msWithin : 0;

            // Approximate p-value (simplified)
            _pValues[j] = ApproximateFPValue(_fStatistics[j], dfBetween, dfWithin);
        }

        // Select top features by F-statistic
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _fStatistics
            .Select((f, idx) => (FStat: f, Index: idx))
            .OrderByDescending(x => x.FStat)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private static double ApproximateFPValue(double fValue, int df1, int df2)
    {
        if (fValue <= 0 || df1 <= 0 || df2 <= 0)
            return 1.0;

        // For large degrees of freedom, use chi-square approximation
        if (df1 > 30 && df2 > 30)
        {
            double z = Math.Sqrt(2 * fValue) - Math.Sqrt(2 * (double)df1 / df2 - 1);
            return 0.5 * (1 - Erf(z / Math.Sqrt(2)));
        }

        // Simple exponential tail approximation
        return Math.Exp(-0.5 * fValue * df1 / Math.Max(df2, 1));
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
            throw new InvalidOperationException("ANOVA has not been fitted.");

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
        throw new NotSupportedException("ANOVA does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ANOVA has not been fitted.");

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
