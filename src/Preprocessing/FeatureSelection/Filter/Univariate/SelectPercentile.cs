using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Selects features according to a percentile of the highest scores.
/// </summary>
/// <remarks>
/// <para>
/// SelectPercentile selects the top percentile of features based on a scoring
/// function. For example, selecting the top 10% of features ranked by F-score.
/// </para>
/// <para>
/// This is similar to SelectKBest but uses a relative threshold (percentile)
/// instead of an absolute number of features.
/// </para>
/// <para><b>For Beginners:</b> Instead of specifying an exact number of features:
/// - SelectPercentile(50) keeps the top 50% of features
/// - The actual number depends on your original feature count
/// - Useful when you want a proportion, not an absolute count
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SelectPercentile<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _percentile;
    private readonly SelectKBestScoreFunc _scoringFunction;

    // Fitted parameters
    private double[]? _scores;
    private double[]? _pValues;
    private bool[]? _supportMask;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the percentile of features to select (0-100).
    /// </summary>
    public double Percentile => _percentile;

    /// <summary>
    /// Gets the scoring function used.
    /// </summary>
    public SelectKBestScoreFunc ScoringFunction => _scoringFunction;

    /// <summary>
    /// Gets the scores for each feature.
    /// </summary>
    public double[]? Scores => _scores;

    /// <summary>
    /// Gets the p-values for each feature.
    /// </summary>
    public double[]? PValues => _pValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SelectPercentile{T}"/>.
    /// </summary>
    /// <param name="percentile">Percentile of features to select (0-100). Defaults to 10.</param>
    /// <param name="scoringFunction">The scoring function to use. Defaults to FRegression.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SelectPercentile(
        double percentile = 10.0,
        SelectKBestScoreFunc scoringFunction = SelectKBestScoreFunc.FRegression,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (percentile < 0 || percentile > 100)
        {
            throw new ArgumentException("Percentile must be between 0 and 100.", nameof(percentile));
        }

        _percentile = percentile;
        _scoringFunction = scoringFunction;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SelectPercentile requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by computing feature scores.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to double arrays
        var X = new double[n, p];
        var y = new double[n];

        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Compute scores for each feature
        _scores = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
            {
                featureValues[i] = X[i, j];
            }

            var (score, pValue) = ComputeScore(featureValues, y);
            _scores[j] = score;
            _pValues[j] = pValue;
        }

        // Calculate number of features to keep
        int nFeatures = (int)Math.Ceiling(p * _percentile / 100.0);
        nFeatures = Math.Max(1, Math.Min(p, nFeatures));

        // Select top features by score
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(i => _scores[i])
            .Take(nFeatures)
            .OrderBy(i => i)
            .ToArray();

        // Create support mask
        _supportMask = new bool[p];
        foreach (int idx in _selectedIndices)
        {
            _supportMask[idx] = true;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    private (double Score, double PValue) ComputeScore(double[] x, double[] y)
    {
        int n = x.Length;

        switch (_scoringFunction)
        {
            case SelectKBestScoreFunc.FRegression:
                return ComputeFRegression(x, y, n);

            case SelectKBestScoreFunc.MutualInfoRegression:
                return (ComputeMutualInfo(x, y, n), 0);

            case SelectKBestScoreFunc.FClassif:
                return ComputeFClassif(x, y, n);

            case SelectKBestScoreFunc.Chi2:
                return ComputeChi2(x, y, n);

            default:
                return ComputeFRegression(x, y, n);
        }
    }

    private (double Score, double PValue) ComputeFRegression(double[] x, double[] y, int n)
    {
        double xMean = x.Average();
        double yMean = y.Average();

        double ssXY = 0, ssXX = 0, ssYY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - xMean;
            double dy = y[i] - yMean;
            ssXY += dx * dy;
            ssXX += dx * dx;
            ssYY += dy * dy;
        }

        if (ssXX < 1e-10 || ssYY < 1e-10)
        {
            return (0, 1);
        }

        double r = ssXY / Math.Sqrt(ssXX * ssYY);
        double r2 = r * r;

        int df1 = 1;
        int df2 = n - 2;

        if (df2 <= 0)
        {
            return (0, 1);
        }

        double fStat = (r2 / df1) / ((1 - r2) / df2);
        double pValue = 1.0 - FDistributionCdf(fStat, df1, df2);

        return (fStat, pValue);
    }

    private (double Score, double PValue) ComputeFClassif(double[] x, double[] y, int n)
    {
        var classes = y.Distinct().OrderBy(c => c).ToArray();
        int nClasses = classes.Length;

        if (nClasses < 2)
        {
            return (0, 1);
        }

        double grandMean = x.Average();

        double ssBetween = 0;
        double ssWithin = 0;

        foreach (double c in classes)
        {
            var classX = new List<double>();
            for (int i = 0; i < n; i++)
            {
                if (Math.Abs(y[i] - c) < 1e-10)
                {
                    classX.Add(x[i]);
                }
            }

            if (classX.Count == 0) continue;

            double classMean = classX.Average();
            ssBetween += classX.Count * (classMean - grandMean) * (classMean - grandMean);

            foreach (double v in classX)
            {
                ssWithin += (v - classMean) * (v - classMean);
            }
        }

        int df1 = nClasses - 1;
        int df2 = n - nClasses;

        if (df1 <= 0 || df2 <= 0 || ssWithin < 1e-10)
        {
            return (0, 1);
        }

        double fStat = (ssBetween / df1) / (ssWithin / df2);
        double pValue = 1.0 - FDistributionCdf(fStat, df1, df2);

        return (fStat, pValue);
    }

    private (double Score, double PValue) ComputeChi2(double[] x, double[] y, int n)
    {
        var classes = y.Distinct().OrderBy(c => c).ToArray();
        int nClasses = classes.Length;

        if (nClasses < 2)
        {
            return (0, 1);
        }

        double xMin = x.Min();
        if (xMin < 0)
        {
            return (0, 1);
        }

        double[] sumPerClass = new double[nClasses];
        int[] countPerClass = new int[nClasses];
        double totalSum = 0;

        for (int i = 0; i < n; i++)
        {
            int classIdx = Array.IndexOf(classes, y[i]);
            sumPerClass[classIdx] += x[i];
            countPerClass[classIdx]++;
            totalSum += x[i];
        }

        if (totalSum < 1e-10)
        {
            return (0, 1);
        }

        double chi2 = 0;
        for (int c = 0; c < nClasses; c++)
        {
            double expected = totalSum * countPerClass[c] / n;
            if (expected > 1e-10)
            {
                double diff = sumPerClass[c] - expected;
                chi2 += diff * diff / expected;
            }
        }

        int df = nClasses - 1;
        double pValue = 1.0 - ChiSquaredCdf(chi2, df);

        return (chi2, pValue);
    }

    private double ComputeMutualInfo(double[] x, double[] y, int n)
    {
        int nBins = (int)Math.Sqrt(n);
        nBins = Math.Max(2, Math.Min(20, nBins));

        double xMin = x.Min();
        double xMax = x.Max();
        double yMin = y.Min();
        double yMax = y.Max();

        if (xMax - xMin < 1e-10 || yMax - yMin < 1e-10)
        {
            return 0;
        }

        var joint = new int[nBins, nBins];
        var marginalX = new int[nBins];
        var marginalY = new int[nBins];

        for (int i = 0; i < n; i++)
        {
            int xBin = (int)((x[i] - xMin) / (xMax - xMin) * (nBins - 1));
            int yBin = (int)((y[i] - yMin) / (yMax - yMin) * (nBins - 1));
            xBin = Math.Max(0, Math.Min(nBins - 1, xBin));
            yBin = Math.Max(0, Math.Min(nBins - 1, yBin));

            joint[xBin, yBin]++;
            marginalX[xBin]++;
            marginalY[yBin]++;
        }

        double mi = 0;
        for (int i = 0; i < nBins; i++)
        {
            for (int j = 0; j < nBins; j++)
            {
                if (joint[i, j] > 0 && marginalX[i] > 0 && marginalY[j] > 0)
                {
                    double pxy = (double)joint[i, j] / n;
                    double px = (double)marginalX[i] / n;
                    double py = (double)marginalY[j] / n;
                    mi += pxy * Math.Log(pxy / (px * py));
                }
            }
        }

        return Math.Max(0, mi);
    }

    private double FDistributionCdf(double x, int df1, int df2)
    {
        if (x <= 0) return 0;
        double a = df1 / 2.0;
        double b = df2 / 2.0;
        double z = df1 * x / (df1 * x + df2);
        return IncompleteBeta(a, b, z);
    }

    private double ChiSquaredCdf(double x, int df)
    {
        if (x <= 0) return 0;
        return IncompleteGamma(df / 2.0, x / 2.0);
    }

    private double IncompleteBeta(double a, double b, double x)
    {
        if (x <= 0) return 0;
        if (x >= 1) return 1;

        double sum = 0;
        double term = 1;
        int maxIter = 100;

        for (int n = 0; n < maxIter; n++)
        {
            sum += term;
            term *= (a + n) / (a + b + n) * x * (n + 1) / (n + 1);
            if (Math.Abs(term) < 1e-10) break;
        }

        return Math.Pow(x, a) * Math.Pow(1 - x, b) * sum / (a * Beta(a, b));
    }

    private double Beta(double a, double b)
    {
        return Math.Exp(LogGamma(a) + LogGamma(b) - LogGamma(a + b));
    }

    private double IncompleteGamma(double a, double x)
    {
        if (x <= 0) return 0;

        double sum = 0;
        double term = 1.0 / a;
        int maxIter = 100;

        for (int n = 0; n < maxIter; n++)
        {
            sum += term;
            term *= x / (a + n + 1);
            if (Math.Abs(term) < 1e-10) break;
        }

        return Math.Exp(-x + a * Math.Log(x) - LogGamma(a)) * sum;
    }

    private double LogGamma(double x)
    {
        double[] c = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                       -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);

        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
        {
            y += 1;
            ser += c[j] / y;
        }

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    /// <summary>
    /// Transforms the data by selecting top percentile features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SelectPercentile has not been fitted.");
        }

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                result[i, j] = data[i, _selectedIndices[j]];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SelectPercentile does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the support mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_supportMask is null)
        {
            throw new InvalidOperationException("SelectPercentile has not been fitted.");
        }
        return (bool[])_supportMask.Clone();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
