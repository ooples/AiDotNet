using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Embedded-Wrapper Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines embedded (L1 regularization) and wrapper methods: first uses
/// Lasso to identify a candidate set, then refines selection using sequential
/// feature addition/removal.
/// </para>
/// <para><b>For Beginners:</b> This method first uses Lasso regression to
/// automatically shrink unimportant features to zero, giving us a candidate
/// set. Then it fine-tunes this selection by trying to add or remove features
/// to improve model performance. This balances speed with accuracy.
/// </para>
/// </remarks>
public class EmbeddedWrapperHybridSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lassoAlpha;
    private readonly int _wrapperIterations;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public EmbeddedWrapperHybridSelector(
        int nFeaturesToSelect = 10,
        double lassoAlpha = 0.01,
        int wrapperIterations = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lassoAlpha = lassoAlpha;
        _wrapperIterations = wrapperIterations;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EmbeddedWrapperHybridSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Normalize features
        var means = new double[p];
        var stds = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++) means[j] += X[i, j];
            means[j] /= n;
            for (int i = 0; i < n; i++) stds[j] += (X[i, j] - means[j]) * (X[i, j] - means[j]);
            stds[j] = Math.Sqrt(stds[j] / (n - 1)) + 1e-10;
            for (int i = 0; i < n; i++) X[i, j] = (X[i, j] - means[j]) / stds[j];
        }

        double yMean = y.Average();
        var yNorm = y.Select(yi => yi - yMean).ToArray();

        // Phase 1: Embedded - Lasso with coordinate descent
        var beta = new double[p];
        int maxIterLasso = 100;

        for (int iter = 0; iter < maxIterLasso; iter++)
        {
            for (int j = 0; j < p; j++)
            {
                // Compute residual without feature j
                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double residual = yNorm[i];
                    for (int k = 0; k < p; k++)
                        if (k != j)
                            residual -= beta[k] * X[i, k];
                    rho += X[i, j] * residual;
                }
                rho /= n;

                // Soft thresholding
                beta[j] = SoftThreshold(rho, _lassoAlpha);
            }
        }

        // Get candidate set from Lasso
        var candidates = Enumerable.Range(0, p)
            .Where(j => Math.Abs(beta[j]) > 1e-10)
            .OrderByDescending(j => Math.Abs(beta[j]))
            .ToList();

        // Ensure we have at least some candidates
        if (candidates.Count < _nFeaturesToSelect)
        {
            var additional = Enumerable.Range(0, p)
                .Where(j => !candidates.Contains(j))
                .OrderByDescending(j => Math.Abs(beta[j]))
                .Take(_nFeaturesToSelect - candidates.Count);
            candidates.AddRange(additional);
        }

        // Phase 2: Wrapper - refine using forward/backward selection
        var selected = candidates.Take(_nFeaturesToSelect).ToList();
        double bestScore = EvaluateSubset(X, yNorm, selected, n);

        for (int iter = 0; iter < _wrapperIterations; iter++)
        {
            bool improved = false;

            // Try to remove features
            for (int i = selected.Count - 1; i >= 0 && selected.Count > 1; i--)
            {
                int feature = selected[i];
                selected.RemoveAt(i);
                double score = EvaluateSubset(X, yNorm, selected, n);
                if (score > bestScore)
                {
                    bestScore = score;
                    improved = true;
                }
                else
                {
                    selected.Insert(i, feature);
                }
            }

            // Try to add features
            foreach (int j in candidates.Where(j => !selected.Contains(j)))
            {
                if (selected.Count >= _nFeaturesToSelect) break;

                selected.Add(j);
                double score = EvaluateSubset(X, yNorm, selected, n);
                if (score > bestScore)
                {
                    bestScore = score;
                    improved = true;
                }
                else
                {
                    selected.RemoveAt(selected.Count - 1);
                }
            }

            if (!improved) break;
        }

        // Compute final scores
        _featureScores = new double[p];
        for (int j = 0; j < p; j++)
            _featureScores[j] = Math.Abs(beta[j]);
        for (int i = 0; i < selected.Count; i++)
            _featureScores[selected[i]] += 1.0; // Boost selected features

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double SoftThreshold(double x, double lambda)
    {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0;
    }

    private double EvaluateSubset(double[,] X, double[] y, List<int> features, int n)
    {
        if (features.Count == 0) return double.MinValue;

        // Compute R-squared with simple linear regression
        var XtX = new double[features.Count, features.Count];
        var Xty = new double[features.Count];

        for (int i = 0; i < n; i++)
        {
            for (int j1 = 0; j1 < features.Count; j1++)
            {
                for (int j2 = 0; j2 < features.Count; j2++)
                    XtX[j1, j2] += X[i, features[j1]] * X[i, features[j2]];
                Xty[j1] += X[i, features[j1]] * y[i];
            }
        }

        // Add regularization
        for (int j = 0; j < features.Count; j++)
            XtX[j, j] += 1e-6;

        var beta = SolveSystem(XtX, Xty, features.Count);

        // Compute R-squared
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++)
        {
            double pred = 0;
            for (int j = 0; j < features.Count; j++)
                pred += beta[j] * X[i, features[j]];
            ssRes += (y[i] - pred) * (y[i] - pred);
            ssTot += y[i] * y[i];
        }

        return ssTot > 1e-10 ? 1 - ssRes / ssTot : 0;
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
            throw new InvalidOperationException("EmbeddedWrapperHybridSelector has not been fitted.");

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
        throw new NotSupportedException("EmbeddedWrapperHybridSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EmbeddedWrapperHybridSelector has not been fitted.");

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
