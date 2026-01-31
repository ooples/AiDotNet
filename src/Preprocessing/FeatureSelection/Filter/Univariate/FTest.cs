using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// F-Test for feature selection in regression problems.
/// </summary>
/// <remarks>
/// <para>
/// The F-test computes the ratio of the variance explained by the relationship between
/// a feature and the target to the unexplained variance. Higher F-statistics indicate
/// features with stronger linear relationships to the target.
/// </para>
/// <para><b>For Beginners:</b> The F-test asks: "How much of the target's variation
/// can be explained by this feature compared to random noise?" A high F-value means
/// the feature explains a significant portion of the target variability. It's used
/// for continuous targets (regression), not classification.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FTest<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public FTest(
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
            "FTest requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _fScores = new double[p];
        _pValues = new double[p];

        // Compute target statistics
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        double ssTotal = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = NumOps.ToDouble(target[i]) - yMean;
            ssTotal += diff * diff;
        }

        for (int j = 0; j < p; j++)
        {
            // Compute feature statistics
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            // Simple linear regression: y = a + b*x
            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }

            double slope = sxx > 1e-10 ? sxy / sxx : 0;
            double intercept = yMean - slope * xMean;

            // Compute SS regression and SS residual
            double ssReg = 0;
            double ssRes = 0;
            for (int i = 0; i < n; i++)
            {
                double predicted = intercept + slope * NumOps.ToDouble(data[i, j]);
                double actual = NumOps.ToDouble(target[i]);
                ssReg += (predicted - yMean) * (predicted - yMean);
                ssRes += (actual - predicted) * (actual - predicted);
            }

            // F-statistic = (SS_reg / df_reg) / (SS_res / df_res)
            // df_reg = 1, df_res = n - 2
            int dfReg = 1;
            int dfRes = n - 2;

            double msReg = ssReg / dfReg;
            double msRes = dfRes > 0 ? ssRes / dfRes : 1;

            _fScores[j] = msRes > 1e-10 ? msReg / msRes : 0;

            // Approximate p-value using F-distribution approximation
            // (simplified - for accurate p-values would need proper F-distribution CDF)
            _pValues[j] = ApproximateFPValue(_fScores[j], dfReg, dfRes);
        }

        // Select top features by F-score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _fScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ApproximateFPValue(double f, int df1, int df2)
    {
        // Simple approximation of F-distribution p-value
        // For accurate values, use proper statistical libraries
        if (f <= 0 || df2 <= 0)
            return 1.0;

        double x = df2 / (df2 + df1 * f);

        // Very rough approximation using normal approximation for large df
        if (df1 >= 30 && df2 >= 30)
        {
            double z = Math.Sqrt(2 * f) - Math.Sqrt(2 * df1 - 1);
            return 0.5 * (1 - Erf(z / Math.Sqrt(2)));
        }

        // For smaller df, use simpler approximation
        return Math.Exp(-0.5 * f * df1 / (df1 + df2));
    }

    private double Erf(double x)
    {
        // Approximation of error function
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
            throw new InvalidOperationException("FTest has not been fitted.");

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
        throw new NotSupportedException("FTest does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FTest has not been fitted.");

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
