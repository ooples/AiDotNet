using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Sequential;

/// <summary>
/// Sequential Floating Forward Selection (SFFS).
/// </summary>
/// <remarks>
/// <para>
/// Combines forward selection with conditional backward steps to escape local optima,
/// allowing previously added features to be removed if they become redundant.
/// </para>
/// <para><b>For Beginners:</b> Regular forward selection only adds features but never
/// removes them. SFFS adds features but can also remove them if doing so improves
/// the result. This "floating" behavior helps find better feature combinations.
/// </para>
/// </remarks>
public class FloatingForwardSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBacktracks;

    private double[]? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FeatureImportances => _featureImportances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FloatingForwardSelector(
        int nFeaturesToSelect = 10,
        int maxBacktracks = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBacktracks = maxBacktracks;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FloatingForwardSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var selected = new HashSet<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        _featureImportances = new double[p];

        double currentScore = 0;

        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            // Forward step: add best feature
            int bestAdd = -1;
            double bestAddScore = currentScore;

            foreach (int j in remaining)
            {
                selected.Add(j);
                double score = EvaluateSubset(X, y, selected, n);
                if (score > bestAddScore)
                {
                    bestAddScore = score;
                    bestAdd = j;
                }
                selected.Remove(j);
            }

            if (bestAdd < 0) break;

            selected.Add(bestAdd);
            remaining.Remove(bestAdd);
            _featureImportances[bestAdd] = bestAddScore;
            currentScore = bestAddScore;

            // Backward steps: try removing features
            for (int backtrack = 0; backtrack < _maxBacktracks && selected.Count > 1; backtrack++)
            {
                int bestRemove = -1;
                double bestRemoveScore = currentScore;

                foreach (int j in selected)
                {
                    if (j == bestAdd) continue; // Don't remove just-added feature

                    selected.Remove(j);
                    double score = EvaluateSubset(X, y, selected, n);
                    if (score > bestRemoveScore)
                    {
                        bestRemoveScore = score;
                        bestRemove = j;
                    }
                    selected.Add(j);
                }

                if (bestRemove >= 0)
                {
                    selected.Remove(bestRemove);
                    remaining.Add(bestRemove);
                    currentScore = bestRemoveScore;
                }
                else
                {
                    break; // No improvement from removing
                }
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // Normalize feature importances
        double maxImportance = _featureImportances.Max();
        if (maxImportance > 0)
        {
            for (int j = 0; j < p; j++)
                _featureImportances[j] /= maxImportance;
        }

        IsFitted = true;
    }

    private double EvaluateSubset(double[,] X, double[] y, HashSet<int> selected, int n)
    {
        if (selected.Count == 0) return 0;

        // Compute R-squared as evaluation metric
        var selectedList = selected.ToList();
        int k = selectedList.Count;

        // Fit simple regression
        var XtX = new double[k, k];
        var Xty = new double[k];

        for (int i = 0; i < k; i++)
        {
            int ji = selectedList[i];
            for (int j = 0; j < k; j++)
            {
                int jj = selectedList[j];
                for (int row = 0; row < n; row++)
                    XtX[i, j] += X[row, ji] * X[row, jj];
            }
            XtX[i, i] += 1e-6; // Regularization

            for (int row = 0; row < n; row++)
                Xty[i] += X[row, ji] * y[row];
        }

        var beta = SolveSystem(XtX, Xty, k);

        // Compute R-squared
        double yMean = y.Average();
        double ssTot = 0, ssRes = 0;

        for (int row = 0; row < n; row++)
        {
            double pred = 0;
            for (int i = 0; i < k; i++)
                pred += beta[i] * X[row, selectedList[i]];

            ssRes += (y[row] - pred) * (y[row] - pred);
            ssTot += (y[row] - yMean) * (y[row] - yMean);
        }

        return ssTot > 0 ? 1 - ssRes / ssTot : 0;
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
            throw new InvalidOperationException("FloatingForwardSelector has not been fitted.");

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
        throw new NotSupportedException("FloatingForwardSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingForwardSelector has not been fitted.");

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
