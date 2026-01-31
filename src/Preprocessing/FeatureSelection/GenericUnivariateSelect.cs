using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Mode options for GenericUnivariateSelect.
/// </summary>
public enum SelectionMode
{
    /// <summary>Select top k features by score.</summary>
    KBest,
    /// <summary>Select top percentile of features by score.</summary>
    Percentile,
    /// <summary>Select features with p-value below alpha threshold.</summary>
    FPR,
    /// <summary>Select features with false discovery rate below alpha.</summary>
    FDR,
    /// <summary>Select features controlling family-wise error rate.</summary>
    FWE
}

/// <summary>
/// Generic univariate feature selector with configurable selection mode.
/// </summary>
/// <remarks>
/// <para>
/// GenericUnivariateSelect is a flexible feature selector that can operate in different
/// modes: k-best, percentile, FPR (false positive rate), FDR (false discovery rate),
/// or FWE (family-wise error rate). This provides a unified interface for various
/// univariate selection strategies.
/// </para>
/// <para><b>For Beginners:</b> This is like a Swiss Army knife for feature selection.
/// Instead of using different classes for different strategies, you pick a mode:
/// - KBest: "Give me the top 10 features"
/// - Percentile: "Give me the top 10% of features"
/// - FPR/FDR/FWE: "Give me features with statistical significance below 0.05"
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class GenericUnivariateSelect<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly SelectionMode _mode;
    private readonly double _param;
    private readonly Func<Matrix<T>, Vector<T>, (double[] Scores, double[] PValues)>? _scoreFunc;
    private readonly string _defaultScoreFunc;

    private double[]? _scores;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public SelectionMode Mode => _mode;
    public double Param => _param;
    public double[]? Scores => _scores;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GenericUnivariateSelect(
        SelectionMode mode = SelectionMode.KBest,
        double param = 10.0,
        Func<Matrix<T>, Vector<T>, (double[] Scores, double[] PValues)>? scoreFunc = null,
        string defaultScoreFunc = "f_classif",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _mode = mode;
        _param = param;
        _scoreFunc = scoreFunc;
        _defaultScoreFunc = defaultScoreFunc;

        ValidateParameters();
    }

    private void ValidateParameters()
    {
        switch (_mode)
        {
            case SelectionMode.KBest:
                if (_param < 1)
                    throw new ArgumentException("k must be at least 1 for KBest mode.");
                break;
            case SelectionMode.Percentile:
                if (_param <= 0 || _param > 100)
                    throw new ArgumentException("Percentile must be in (0, 100].");
                break;
            case SelectionMode.FPR:
            case SelectionMode.FDR:
            case SelectionMode.FWE:
                if (_param <= 0 || _param > 1)
                    throw new ArgumentException("Alpha must be in (0, 1] for statistical tests.");
                break;
        }
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GenericUnivariateSelect requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute scores and p-values
        if (_scoreFunc != null)
        {
            var result = _scoreFunc(data, target);
            _scores = result.Scores;
            _pValues = result.PValues;
        }
        else
        {
            (_scores, _pValues) = _defaultScoreFunc switch
            {
                "f_classif" => ComputeFClassif(data, target, n, p),
                "f_regression" => ComputeFRegression(data, target, n, p),
                _ => ComputeFClassif(data, target, n, p)
            };
        }

        // Select features based on mode
        _selectedIndices = _mode switch
        {
            SelectionMode.KBest => SelectKBest(p),
            SelectionMode.Percentile => SelectPercentile(p),
            SelectionMode.FPR => SelectFPR(),
            SelectionMode.FDR => SelectFDR(p),
            SelectionMode.FWE => SelectFWE(p),
            _ => SelectKBest(p)
        };

        IsFitted = true;
    }

    private int[] SelectKBest(int p)
    {
        int k = Math.Min((int)_param, p);
        return _scores!
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();
    }

    private int[] SelectPercentile(int p)
    {
        int numToSelect = Math.Max(1, (int)Math.Ceiling(p * _param / 100.0));
        return _scores!
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();
    }

    private int[] SelectFPR()
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Select features with p-value below alpha
        var selected = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .Where(x => x.PValue < _param)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        // If none selected, select the one with lowest p-value
        if (selected.Length == 0)
        {
            int bestIdx = Array.IndexOf(_pValues, _pValues.Min());
            selected = [bestIdx];
        }

        return selected;
    }

    private int[] SelectFDR(int p)
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Benjamini-Hochberg procedure for FDR control
        var sorted = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .OrderBy(x => x.PValue)
            .ToArray();

        var selected = new List<int>();
        for (int i = 0; i < p; i++)
        {
            double threshold = (i + 1) * _param / p;
            if (sorted[i].PValue <= threshold)
                selected.Add(sorted[i].Index);
            else
                break;
        }

        if (selected.Count == 0)
            selected.Add(sorted[0].Index);

        return selected.OrderBy(x => x).ToArray();
    }

    private int[] SelectFWE(int p)
    {
        if (_pValues is null || _pValues.Length == 0)
            return [0];

        // Bonferroni correction for FWE control
        double threshold = _param / p;
        var selected = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .Where(x => x.PValue < threshold)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        if (selected.Length == 0)
        {
            int bestIdx = Array.IndexOf(_pValues, _pValues.Min());
            selected = [bestIdx];
        }

        return selected;
    }

    private (double[] Scores, double[] PValues) ComputeFClassif(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        var pValues = new double[p];

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
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            double ssb = 0, ssw = 0;

            foreach (var kvp in classGroups)
            {
                double classMean = 0;
                foreach (int i in kvp.Value)
                    classMean += NumOps.ToDouble(data[i, j]);
                classMean /= kvp.Value.Count;

                ssb += kvp.Value.Count * (classMean - overallMean) * (classMean - overallMean);

                foreach (int i in kvp.Value)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean;
                    ssw += diff * diff;
                }
            }

            double msb = ssb / Math.Max(1, k - 1);
            double msw = ssw / Math.Max(1, n - k);
            scores[j] = msw > 1e-10 ? msb / msw : 0;

            // Approximate p-value using F-distribution approximation
            pValues[j] = ApproximateFPValue(scores[j], k - 1, n - k);
        }

        return (scores, pValues);
    }

    private (double[] Scores, double[] PValues) ComputeFRegression(Matrix<T> data, Vector<T> target, int n, int p)
    {
        var scores = new double[p];
        var pValues = new double[p];

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
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double sxy = 0, sxx = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = NumOps.ToDouble(data[i, j]) - xMean;
                double yDiff = NumOps.ToDouble(target[i]) - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
            }

            double beta = sxx > 1e-10 ? sxy / sxx : 0;
            double ssResidual = 0;

            for (int i = 0; i < n; i++)
            {
                double pred = yMean + beta * (NumOps.ToDouble(data[i, j]) - xMean);
                double residual = NumOps.ToDouble(target[i]) - pred;
                ssResidual += residual * residual;
            }

            double ssRegression = ssTotal - ssResidual;
            double msResidual = ssResidual / Math.Max(1, n - 2);
            scores[j] = msResidual > 1e-10 ? ssRegression / msResidual : 0;

            pValues[j] = ApproximateFPValue(scores[j], 1, n - 2);
        }

        return (scores, pValues);
    }

    private static double ApproximateFPValue(double fValue, int df1, int df2)
    {
        // Simple approximation using the F-distribution
        // For more accurate results, a proper statistical library would be needed
        if (fValue <= 0 || df1 <= 0 || df2 <= 0)
            return 1.0;

        // Use incomplete beta function relationship: P(F > f) = I_{x}(df2/2, df1/2)
        // where x = df2/(df2 + df1*f)
        double x = df2 / (df2 + df1 * fValue);

        // Simple approximation using normal distribution for large df
        if (df1 > 30 && df2 > 30)
        {
            double z = Math.Sqrt(2 * fValue) - Math.Sqrt(2 * (double)df1 / df2 - 1);
            return 0.5 * (1 - Erf(z / Math.Sqrt(2)));
        }

        // Fallback: use exponential approximation for F-distribution tail
        return Math.Exp(-0.5 * fValue * df1 / Math.Max(df2, 1));
    }

    private static double Erf(double x)
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
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");

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
        throw new NotSupportedException("GenericUnivariateSelect does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GenericUnivariateSelect has not been fitted.");

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
