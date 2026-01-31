using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Filter-Wrapper Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines filter and wrapper methods: first uses a fast filter to reduce
/// the feature space, then applies a more accurate wrapper method on the
/// reduced set.
/// </para>
/// <para><b>For Beginners:</b> This method gets the best of both worlds.
/// First, it quickly eliminates obviously irrelevant features using simple
/// statistics (filter). Then, it carefully evaluates the remaining features
/// using model performance (wrapper). This is faster than pure wrapper methods
/// but more accurate than pure filter methods.
/// </para>
/// </remarks>
public class FilterWrapperHybridSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _filterRatio;
    private readonly int _wrapperIterations;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FilterWrapperHybridSelector(
        int nFeaturesToSelect = 10,
        int filterRatio = 3,
        int wrapperIterations = 50,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _filterRatio = filterRatio;
        _wrapperIterations = wrapperIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FilterWrapperHybridSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Phase 1: Filter - compute correlation-based scores
        var filterScores = new double[p];
        double yMean = y.Average();

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++) xMean += X[i, j];
            xMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xd = X[i, j] - xMean;
                double yd = y[i] - yMean;
                sxy += xd * yd;
                sxx += xd * xd;
                syy += yd * yd;
            }
            filterScores[j] = (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
        }

        // Select top features from filter
        int filterCount = Math.Min(_nFeaturesToSelect * _filterRatio, p);
        var filteredFeatures = Enumerable.Range(0, p)
            .OrderByDescending(j => filterScores[j])
            .Take(filterCount)
            .ToList();

        // Phase 2: Wrapper - forward selection on filtered features
        var selected = new List<int>();
        var remaining = new HashSet<int>(filteredFeatures);
        double bestOverallScore = double.MinValue;

        while (selected.Count < _nFeaturesToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int j in remaining)
            {
                var testSet = new List<int>(selected) { j };
                double score = EvaluateSubset(X, y, testSet, n, p, rand);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0 && bestScore > bestOverallScore - 0.001)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                bestOverallScore = bestScore;
            }
            else
            {
                break;
            }
        }

        // Compute final scores (combination of filter and selection order)
        _featureScores = new double[p];
        for (int i = 0; i < selected.Count; i++)
            _featureScores[selected[i]] = filterScores[selected[i]] + (selected.Count - i);

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double EvaluateSubset(double[,] X, double[] y, List<int> features, int n, int p, Random rand)
    {
        if (features.Count == 0) return 0;

        // Simple cross-validation with linear regression
        int folds = Math.Min(5, n);
        double totalScore = 0;

        var indices = Enumerable.Range(0, n).OrderBy(_ => rand.Next()).ToList();

        for (int fold = 0; fold < folds; fold++)
        {
            int testStart = fold * n / folds;
            int testEnd = (fold + 1) * n / folds;

            // Train on non-test indices
            var trainIdx = indices.Where((_, i) => i < testStart || i >= testEnd).ToList();
            var testIdx = indices.Skip(testStart).Take(testEnd - testStart).ToList();

            if (trainIdx.Count < features.Count + 1 || testIdx.Count == 0)
                continue;

            // Fit simple linear model
            var beta = FitLinearModel(X, y, features, trainIdx, n);

            // Evaluate on test
            double mse = 0;
            double ssTot = 0;
            double yMeanTest = testIdx.Average(i => y[i]);

            foreach (int i in testIdx)
            {
                double pred = beta[0];
                for (int k = 0; k < features.Count; k++)
                    pred += beta[k + 1] * X[i, features[k]];
                mse += (y[i] - pred) * (y[i] - pred);
                ssTot += (y[i] - yMeanTest) * (y[i] - yMeanTest);
            }

            if (ssTot > 1e-10)
                totalScore += 1 - mse / ssTot; // R-squared
        }

        return totalScore / folds;
    }

    private double[] FitLinearModel(double[,] X, double[] y, List<int> features, List<int> indices, int n)
    {
        int k = features.Count + 1;
        var XtX = new double[k, k];
        var Xty = new double[k];

        foreach (int i in indices)
        {
            var xi = new double[k];
            xi[0] = 1; // Intercept
            for (int j = 0; j < features.Count; j++)
                xi[j + 1] = X[i, features[j]];

            for (int j1 = 0; j1 < k; j1++)
            {
                for (int j2 = 0; j2 < k; j2++)
                    XtX[j1, j2] += xi[j1] * xi[j2];
                Xty[j1] += xi[j1] * y[i];
            }
        }

        // Add regularization and solve
        for (int j = 0; j < k; j++)
            XtX[j, j] += 1e-6;

        return SolveSystem(XtX, Xty, k);
    }

    private double[] SolveSystem(double[,] A, double[] b, int n)
    {
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

            if (Math.Abs(aug[col, col]) < 1e-10) continue;

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

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapperHybridSelector has not been fitted.");

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
        throw new NotSupportedException("FilterWrapperHybridSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FilterWrapperHybridSelector has not been fitted.");

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
