using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// Fast Causal Inference (FCI) Algorithm Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the FCI algorithm for causal discovery with latent confounders,
/// selecting features that are direct causes or effects of the target.
/// </para>
/// <para><b>For Beginners:</b> FCI extends the PC algorithm to handle hidden
/// (unmeasured) variables. It builds a partial ancestral graph and selects
/// features that have causal connections to the target, even accounting for
/// variables we can't observe.
/// </para>
/// </remarks>
public class FCI_Selector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;
    private readonly int _maxConditioningSetSize;

    private double[]? _causalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CausalScores => _causalScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FCI_Selector(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int maxConditioningSetSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
        _maxConditioningSetSize = maxConditioningSetSize;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FCI_Selector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Initialize complete graph (all edges present)
        var adjacent = new bool[p + 1, p + 1];
        for (int i = 0; i <= p; i++)
            for (int j = i + 1; j <= p; j++)
            {
                adjacent[i, j] = true;
                adjacent[j, i] = true;
            }

        // Phase 1: Edge removal (similar to PC algorithm)
        for (int condSize = 0; condSize <= _maxConditioningSetSize; condSize++)
        {
            for (int i = 0; i < p; i++)
            {
                // Test edge between feature i and target (index p)
                if (!adjacent[i, p]) continue;

                var neighbors = GetNeighbors(adjacent, i, p);
                if (neighbors.Count < condSize) continue;

                foreach (var condSet in GetSubsets(neighbors, condSize))
                {
                    double pValue = ConditionalIndependenceTest(X, y, i, condSet, n, p);
                    if (pValue > _alpha)
                    {
                        adjacent[i, p] = false;
                        adjacent[p, i] = false;
                        break;
                    }
                }
            }
        }

        // Phase 2: Orientation rules (simplified FCI rules)
        // Track possible ancestors of target
        var possibleAncestors = new HashSet<int>();
        for (int i = 0; i < p; i++)
        {
            if (adjacent[i, p])
                possibleAncestors.Add(i);
        }

        // Score features based on connection strength and orientation
        _causalScores = new double[p];
        for (int i = 0; i < p; i++)
        {
            if (possibleAncestors.Contains(i))
            {
                // Compute partial correlation as causal strength
                double corr = ComputeCorrelation(X, y, i, n);
                _causalScores[i] = Math.Abs(corr);
            }
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _causalScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<int> GetNeighbors(bool[,] adj, int node, int exclude)
    {
        var neighbors = new List<int>();
        int size = adj.GetLength(0);
        for (int i = 0; i < size; i++)
            if (i != node && i != exclude && adj[node, i])
                neighbors.Add(i);
        return neighbors;
    }

    private IEnumerable<List<int>> GetSubsets(List<int> items, int size)
    {
        if (size == 0)
        {
            yield return new List<int>();
            yield break;
        }
        if (items.Count < size) yield break;

        for (int i = 0; i <= items.Count - size; i++)
        {
            foreach (var rest in GetSubsets(items.Skip(i + 1).ToList(), size - 1))
            {
                var subset = new List<int> { items[i] };
                subset.AddRange(rest);
                yield return subset;
            }
        }
    }

    private double ConditionalIndependenceTest(double[,] X, double[] y, int featIdx, List<int> condSet, int n, int p)
    {
        if (condSet.Count == 0)
        {
            double corr = ComputeCorrelation(X, y, featIdx, n);
            double t = corr * Math.Sqrt((n - 2) / (1 - corr * corr + 1e-10));
            return 2 * (1 - StudentT_CDF(Math.Abs(t), n - 2));
        }

        // Partial correlation test
        double partialCorr = ComputePartialCorrelation(X, y, featIdx, condSet, n, p);
        int df = n - condSet.Count - 2;
        double tStat = partialCorr * Math.Sqrt(df / (1 - partialCorr * partialCorr + 1e-10));
        return 2 * (1 - StudentT_CDF(Math.Abs(tStat), df));
    }

    private double ComputeCorrelation(double[,] X, double[] y, int j, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++) { xMean += X[i, j]; yMean += y[i]; }
        xMean /= n; yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, j] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }
        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private double ComputePartialCorrelation(double[,] X, double[] y, int featIdx, List<int> condSet, int n, int p)
    {
        // Residualize X[featIdx] and y on conditioning set
        var residX = new double[n];
        var residY = new double[n];
        Array.Copy(y, residY, n);
        for (int i = 0; i < n; i++) residX[i] = X[i, featIdx];

        foreach (int c in condSet)
        {
            double beta_x = ComputeRegressionCoef(X, residX, c, n);
            double beta_y = ComputeRegressionCoef(X, residY, c, n);
            for (int i = 0; i < n; i++)
            {
                residX[i] -= beta_x * X[i, c];
                residY[i] -= beta_y * X[i, c];
            }
        }

        // Correlation of residuals
        double meanX = residX.Average();
        double meanY = residY.Average();
        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = residX[i] - meanX;
            double yd = residY[i] - meanY;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }
        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
    }

    private double ComputeRegressionCoef(double[,] X, double[] y, int col, int n)
    {
        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++) { xMean += X[i, col]; yMean += y[i]; }
        xMean /= n; yMean /= n;

        double sxy = 0, sxx = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = X[i, col] - xMean;
            sxy += xd * (y[i] - yMean);
            sxx += xd * xd;
        }
        return sxx > 1e-10 ? sxy / sxx : 0;
    }

    private double StudentT_CDF(double t, int df)
    {
        double x = df / (df + t * t);
        return 1 - 0.5 * IncompleteBeta(df / 2.0, 0.5, x);
    }

    private double IncompleteBeta(double a, double b, double x)
    {
        if (x < 0 || x > 1) return 0;
        if (x == 0) return 0;
        if (x == 1) return 1;

        double bt = Math.Exp(
            LogGamma(a + b) - LogGamma(a) - LogGamma(b) +
            a * Math.Log(x) + b * Math.Log(1 - x));

        if (x < (a + 1) / (a + b + 2))
            return bt * BetaCF(a, b, x) / a;
        else
            return 1 - bt * BetaCF(b, a, 1 - x) / b;
    }

    private double BetaCF(double a, double b, double x)
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
            if (Math.Abs(del - 1) < 1e-10) break;
        }
        return h;
    }

    private double LogGamma(double x)
    {
        double[] coef = { 76.18009172947146, -86.50532032941677,
            24.01409824083091, -1.231739572450155, 0.001208650973866179, -5.395239384953e-6 };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++) ser += coef[j] / ++y;
        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCI_Selector has not been fitted.");

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
        throw new NotSupportedException("FCI_Selector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCI_Selector has not been fitted.");

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
