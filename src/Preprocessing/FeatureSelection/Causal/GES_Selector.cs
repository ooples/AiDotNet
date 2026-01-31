using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// Greedy Equivalence Search (GES) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the GES algorithm for causal structure learning, which searches
/// over equivalence classes of DAGs using a score-based approach.
/// </para>
/// <para><b>For Beginners:</b> GES finds the best causal structure by
/// greedily adding and removing edges to maximize a score (like BIC).
/// Features connected to the target in the learned structure are selected.
/// </para>
/// </remarks>
public class GES_Selector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _penaltyDiscount;

    private double[]? _causalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CausalScores => _causalScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public GES_Selector(
        int nFeaturesToSelect = 10,
        double penaltyDiscount = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _penaltyDiscount = penaltyDiscount;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GES_Selector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize empty graph (adjacency matrix for edges to target)
        var parents = new HashSet<int>();
        double currentScore = ComputeBIC(X, y, parents, n, p);

        // Forward phase: Add edges
        bool improved = true;
        while (improved)
        {
            improved = false;
            int bestAdd = -1;
            double bestScore = currentScore;

            for (int j = 0; j < p; j++)
            {
                if (parents.Contains(j)) continue;

                var testParents = new HashSet<int>(parents) { j };
                double score = ComputeBIC(X, y, testParents, n, p);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestAdd = j;
                }
            }

            if (bestAdd >= 0)
            {
                parents.Add(bestAdd);
                currentScore = bestScore;
                improved = true;
            }
        }

        // Backward phase: Remove edges
        improved = true;
        while (improved)
        {
            improved = false;
            int bestRemove = -1;
            double bestScore = currentScore;

            foreach (int j in parents)
            {
                var testParents = new HashSet<int>(parents);
                testParents.Remove(j);
                double score = ComputeBIC(X, y, testParents, n, p);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestRemove = j;
                }
            }

            if (bestRemove >= 0)
            {
                parents.Remove(bestRemove);
                currentScore = bestScore;
                improved = true;
            }
        }

        // Compute scores based on regression coefficients
        _causalScores = new double[p];
        foreach (int j in parents)
        {
            double corr = ComputeCorrelation(X, y, j, n);
            _causalScores[j] = Math.Abs(corr);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _causalScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeBIC(double[,] X, double[] y, HashSet<int> parents, int n, int p)
    {
        if (parents.Count == 0)
        {
            double yMean = y.Average();
            double emptyRss = y.Sum(yi => (yi - yMean) * (yi - yMean));
            return -n * Math.Log(emptyRss / n + 1e-10) - _penaltyDiscount * Math.Log(n);
        }

        // Multiple regression
        var parentList = parents.ToList();
        int k = parentList.Count;

        // Compute X'X and X'y
        var XtX = new double[k, k];
        var Xty = new double[k];

        for (int i = 0; i < k; i++)
        {
            int fi = parentList[i];
            for (int j = 0; j < k; j++)
            {
                int fj = parentList[j];
                for (int r = 0; r < n; r++)
                    XtX[i, j] += X[r, fi] * X[r, fj];
            }
            for (int r = 0; r < n; r++)
                Xty[i] += X[r, fi] * y[r];
        }

        // Solve for coefficients (regularized)
        var beta = SolveLinearSystem(XtX, Xty, k);

        // Compute RSS
        double rss = 0;
        for (int r = 0; r < n; r++)
        {
            double pred = 0;
            for (int i = 0; i < k; i++)
                pred += beta[i] * X[r, parentList[i]];
            double err = y[r] - pred;
            rss += err * err;
        }

        return -n * Math.Log(rss / n + 1e-10) - _penaltyDiscount * (k + 1) * Math.Log(n);
    }

    private double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        // Add regularization
        for (int i = 0; i < n; i++)
            A[i, i] += 1e-6;

        // Gaussian elimination with partial pivoting
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            for (int j = 0; j <= n; j++)
                (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            if (Math.Abs(aug[col, col]) < 1e-10)
                continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / aug[col, col];
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            x[i] /= (Math.Abs(aug[i, i]) > 1e-10 ? aug[i, i] : 1);
        }

        return x;
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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GES_Selector has not been fitted.");

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
        throw new NotSupportedException("GES_Selector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GES_Selector has not been fitted.");

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
