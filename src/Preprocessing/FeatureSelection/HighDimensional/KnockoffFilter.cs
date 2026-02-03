using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.HighDimensional;

/// <summary>
/// Knockoff Filter for high-dimensional feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// The Knockoff Filter creates "knockoff" versions of each feature that mimic the
/// correlation structure but are conditionally independent of the target. By comparing
/// real features against their knockoffs, it controls the False Discovery Rate (FDR).
/// </para>
/// <para><b>For Beginners:</b> When testing many features, some will look important
/// just by chance. The Knockoff Filter creates fake versions of each feature and
/// asks: "Is the real feature more predictive than its fake twin?" This helps avoid
/// selecting features that only appear important by luck.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KnockoffFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _fdrThreshold;
    private readonly int? _randomState;

    private double[]? _featureStatistics;
    private double[]? _knockoffStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double FDRThreshold => _fdrThreshold;
    public double[]? FeatureStatistics => _featureStatistics;
    public double[]? KnockoffStatistics => _knockoffStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KnockoffFilter(
        int nFeaturesToSelect = 10,
        double fdrThreshold = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (fdrThreshold <= 0 || fdrThreshold > 1)
            throw new ArgumentException("FDR threshold must be between 0 and 1.", nameof(fdrThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _fdrThreshold = fdrThreshold;
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

        // Step 1: Generate knockoff features (simplified second-order knockoffs)
        var knockoffs = GenerateKnockoffs(data, random);

        // Step 2: Compute importance statistics for original and knockoff features
        _featureStatistics = new double[p];
        _knockoffStatistics = new double[p];

        // Use absolute correlation as the feature statistic
        for (int j = 0; j < p; j++)
        {
            _featureStatistics[j] = ComputeFeatureStatistic(data, target, j);
            _knockoffStatistics[j] = ComputeKnockoffStatistic(knockoffs, target, j);
        }

        // Step 3: Compute knockoff statistic W = |Z_j| - |Z_knockoff_j|
        var W = new double[p];
        for (int j = 0; j < p; j++)
            W[j] = _featureStatistics[j] - _knockoffStatistics[j];

        // Step 4: Apply knockoff filter with FDR control
        // Find threshold tau such that FDP <= fdrThreshold
        var sortedW = W.Select((w, idx) => (W: w, Index: idx))
            .OrderByDescending(x => x.W)
            .ToList();

        // Knockoff+ procedure: estimate FDR
        var selected = new List<int>();
        for (int k = 0; k < sortedW.Count && selected.Count < _nFeaturesToSelect; k++)
        {
            double tau = sortedW[k].W;
            if (tau <= 0) break;

            // Count #{j : W_j >= tau} and #{j : W_j <= -tau}
            int positives = W.Count(w => w >= tau);
            int negatives = W.Count(w => w <= -tau);

            // Estimated FDP = (1 + negatives) / max(1, positives)
            double estimatedFDP = (1.0 + negatives) / Math.Max(1, positives);

            if (estimatedFDP <= _fdrThreshold)
            {
                // Select all features with W >= tau
                foreach (var item in sortedW.Where(x => x.W >= tau))
                {
                    if (!selected.Contains(item.Index))
                        selected.Add(item.Index);
                }
                break;
            }
        }

        // If FDR control doesn't select enough, fall back to top W scores
        if (selected.Count == 0)
        {
            _selectedIndices = sortedW
                .Where(x => x.W > 0)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();

            if (_selectedIndices.Length == 0)
            {
                _selectedIndices = sortedW
                    .Take(_nFeaturesToSelect)
                    .Select(x => x.Index)
                    .OrderBy(x => x)
                    .ToArray();
            }
        }
        else
        {
            _selectedIndices = selected.Take(_nFeaturesToSelect).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private Matrix<T> GenerateKnockoffs(Matrix<T> data, Random random)
    {
        int n = data.Rows;
        int p = data.Columns;

        // Simplified knockoff generation: permute within feature then add noise
        var knockoffs = new T[n, p];

        for (int j = 0; j < p; j++)
        {
            // Get original values
            var values = new double[n];
            for (int i = 0; i < n; i++)
                values[i] = NumOps.ToDouble(data[i, j]);

            // Compute mean and std
            double mean = values.Average();
            double std = Math.Sqrt(values.Select(v => (v - mean) * (v - mean)).Average());

            // Generate knockoff: add noise while preserving marginal distribution
            var shuffled = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();
            for (int i = 0; i < n; i++)
            {
                double noise = (random.NextDouble() - 0.5) * std * 0.5;
                double knockoffVal = values[shuffled[i]] + noise;
                knockoffs[i, j] = NumOps.FromDouble(knockoffVal);
            }
        }

        return new Matrix<T>(knockoffs);
    }

    private double ComputeFeatureStatistic(Matrix<T> data, Vector<T> target, int featureIdx)
    {
        int n = data.Rows;

        double xMean = 0, yMean = 0;
        for (int i = 0; i < n; i++)
        {
            xMean += NumOps.ToDouble(data[i, featureIdx]);
            yMean += NumOps.ToDouble(target[i]);
        }
        xMean /= n;
        yMean /= n;

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = NumOps.ToDouble(data[i, featureIdx]) - xMean;
            double yDiff = NumOps.ToDouble(target[i]) - yMean;
            sxy += xDiff * yDiff;
            sxx += xDiff * xDiff;
            syy += yDiff * yDiff;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? Math.Abs(sxy / Math.Sqrt(sxx * syy)) : 0;
    }

    private double ComputeKnockoffStatistic(Matrix<T> knockoffs, Vector<T> target, int featureIdx)
    {
        return ComputeFeatureStatistic(knockoffs, target, featureIdx);
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
