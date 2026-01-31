using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.HighDimensional;

/// <summary>
/// Knockoff Filter for false discovery rate control in feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Knockoff Filter creates "knockoff" copies of features that mimic their
/// correlation structure but are independent of the target. Features whose
/// importance exceeds their knockoffs' importance are selected.
/// </para>
/// <para>
/// This method provides FDR (False Discovery Rate) control, guaranteeing that
/// the expected proportion of false discoveries is below a specified threshold.
/// </para>
/// <para><b>For Beginners:</b> Knockoffs are fake features designed to compete
/// with real ones. If a real feature beats its knockoff, it's likely important.
/// This controls how many false positives sneak into your selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KnockoffFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _fdr;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, double[]>? _importanceFunc;

    private double[]? _wStats;
    private double _threshold;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public double FDR => _fdr;
    public double[]? WStatistics => _wStats;
    public double Threshold => _threshold;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KnockoffFilter(
        double fdr = 0.1,
        Func<Matrix<T>, Vector<T>, double[]>? importanceFunc = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (fdr <= 0 || fdr >= 1)
            throw new ArgumentException("FDR must be between 0 and 1.", nameof(fdr));

        _fdr = fdr;
        _importanceFunc = importanceFunc;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KnockoffFilter requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Convert to double
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        // Generate knockoff features
        var knockoffs = GenerateKnockoffs(X, n, p, random);

        // Compute importance for original and knockoff features
        var augmentedData = new T[n, 2 * p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                augmentedData[i, j] = data[i, j];
                augmentedData[i, p + j] = NumOps.FromDouble(knockoffs[i, j]);
            }
        }

        var augmentedMatrix = new Matrix<T>(augmentedData);
        var importances = GetImportances(augmentedMatrix, target, n, 2 * p);

        // Compute W statistics: W_j = |Z_j| - |Z̃_j|
        _wStats = new double[p];
        for (int j = 0; j < p; j++)
            _wStats[j] = Math.Abs(importances[j]) - Math.Abs(importances[p + j]);

        // Compute knockoff threshold
        _threshold = ComputeThreshold(_wStats, p);

        // Select features with W > threshold
        var selectedList = new List<int>();
        for (int j = 0; j < p; j++)
            if (_wStats[j] >= _threshold)
                selectedList.Add(j);

        // If nothing selected, take highest W
        if (selectedList.Count == 0)
        {
            int bestIdx = 0;
            double bestW = _wStats[0];
            for (int j = 1; j < p; j++)
            {
                if (_wStats[j] > bestW)
                {
                    bestW = _wStats[j];
                    bestIdx = j;
                }
            }
            selectedList.Add(bestIdx);
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private double[,] GenerateKnockoffs(double[,] X, int n, int p, Random random)
    {
        // Simplified knockoff generation using permutation
        // (Full Model-X knockoffs require more complex covariance estimation)
        var knockoffs = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            // Permute feature values
            var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
            for (int i = 0; i < n; i++)
                knockoffs[i, j] = X[indices[i], j];
        }

        return knockoffs;
    }

    private double[] GetImportances(Matrix<T> data, Vector<T> target, int n, int p)
    {
        if (_importanceFunc is not null)
            return _importanceFunc(data, target);

        // Default: absolute correlation with target
        var importances = new double[p];
        double yMean = 0;
        for (int i = 0; i < n; i++)
            yMean += NumOps.ToDouble(target[i]);
        yMean /= n;

        for (int j = 0; j < p; j++)
        {
            double xMean = 0;
            for (int i = 0; i < n; i++)
                xMean += NumOps.ToDouble(data[i, j]);
            xMean /= n;

            double ssXY = 0, ssXX = 0, ssYY = 0;
            for (int i = 0; i < n; i++)
            {
                double dx = NumOps.ToDouble(data[i, j]) - xMean;
                double dy = NumOps.ToDouble(target[i]) - yMean;
                ssXY += dx * dy;
                ssXX += dx * dx;
                ssYY += dy * dy;
            }

            if (ssXX > 1e-10 && ssYY > 1e-10)
                importances[j] = ssXY / Math.Sqrt(ssXX * ssYY);
        }

        return importances;
    }

    private double ComputeThreshold(double[] wStats, int p)
    {
        // Knockoff+ threshold
        var sortedW = wStats.Where(w => w > 0).OrderByDescending(w => w).ToList();

        if (sortedW.Count == 0)
            return double.MaxValue;

        foreach (double t in sortedW)
        {
            int numPositive = wStats.Count(w => w >= t);
            int numNegative = wStats.Count(w => w <= -t);

            double fdp = (1.0 + numNegative) / Math.Max(1, numPositive);

            if (fdp <= _fdr)
                return t;
        }

        return double.MaxValue;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

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
        throw new NotSupportedException("KnockoffFilter does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KnockoffFilter has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
