using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Interaction;

/// <summary>
/// Synergy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their synergistic information with other features,
/// measuring how features provide complementary information about the target.
/// </para>
/// <para><b>For Beginners:</b> Synergy occurs when two features together provide
/// more information than the sum of their individual information. This selector
/// finds features that work synergistically with others.
/// </para>
/// </remarks>
public class SynergySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _synergyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? SynergyScores => _synergyScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SynergySelector(
        int nFeaturesToSelect = 10,
        int nBins = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SynergySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize all features and target
        var discretizedX = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];
            Discretize(col, discretizedX, j, n);
        }

        var discretizedY = new int[n];
        DiscretizeTarget(y, discretizedY, n);

        _synergyScores = new double[p];

        // Compute individual mutual information I(X_j;Y)
        var individualMI = new double[p];
        for (int j = 0; j < p; j++)
            individualMI[j] = ComputeMI(discretizedX, j, discretizedY, n);

        // For each feature, compute maximum synergy with another feature
        for (int j1 = 0; j1 < p; j1++)
        {
            double maxSynergy = 0;

            for (int j2 = 0; j2 < Math.Min(p, 20); j2++) // Limit comparisons for efficiency
            {
                if (j1 == j2) continue;

                // Compute joint MI: I(X_j1, X_j2; Y)
                double jointMI = ComputeJointMI(discretizedX, j1, j2, discretizedY, n);

                // Synergy = I(X_j1, X_j2; Y) - max(I(X_j1;Y), I(X_j2;Y))
                double synergy = jointMI - Math.Max(individualMI[j1], individualMI[j2]);
                maxSynergy = Math.Max(maxSynergy, synergy);
            }

            _synergyScores[j1] = maxSynergy + individualMI[j1]; // Combine synergy with individual MI
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _synergyScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private void Discretize(double[] values, int[,] result, int j, int n)
    {
        double min = values.Min();
        double max = values.Max();
        double range = max - min;

        for (int i = 0; i < n; i++)
            result[i, j] = range > 1e-10
                ? Math.Min(_nBins - 1, (int)((values[i] - min) / range * _nBins))
                : 0;
    }

    private void DiscretizeTarget(double[] values, int[] result, int n)
    {
        double min = values.Min();
        double max = values.Max();
        double range = max - min;

        for (int i = 0; i < n; i++)
            result[i] = range > 1e-10
                ? Math.Min(_nBins - 1, (int)((values[i] - min) / range * _nBins))
                : 0;
    }

    private double ComputeMI(int[,] X, int j, int[] Y, int n)
    {
        var joint = new int[_nBins, _nBins];
        var xCounts = new int[_nBins];
        var yCounts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            joint[X[i, j], Y[i]]++;
            xCounts[X[i, j]]++;
            yCounts[Y[i]]++;
        }

        double mi = 0;
        for (int xb = 0; xb < _nBins; xb++)
        {
            for (int yb = 0; yb < _nBins; yb++)
            {
                if (joint[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                {
                    double pxy = (double)joint[xb, yb] / n;
                    double px = (double)xCounts[xb] / n;
                    double py = (double)yCounts[yb] / n;
                    mi += pxy * Math.Log(pxy / (px * py)) / Math.Log(2);
                }
            }
        }

        return Math.Max(0, mi);
    }

    private double ComputeJointMI(int[,] X, int j1, int j2, int[] Y, int n)
    {
        // Joint bin = j1 * nBins + j2
        int jointBins = _nBins * _nBins;
        var joint = new int[jointBins, _nBins];
        var xCounts = new int[jointBins];
        var yCounts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            int jointBin = X[i, j1] * _nBins + X[i, j2];
            joint[jointBin, Y[i]]++;
            xCounts[jointBin]++;
            yCounts[Y[i]]++;
        }

        double mi = 0;
        for (int xb = 0; xb < jointBins; xb++)
        {
            for (int yb = 0; yb < _nBins; yb++)
            {
                if (joint[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                {
                    double pxy = (double)joint[xb, yb] / n;
                    double px = (double)xCounts[xb] / n;
                    double py = (double)yCounts[yb] / n;
                    mi += pxy * Math.Log(pxy / (px * py)) / Math.Log(2);
                }
            }
        }

        return Math.Max(0, mi);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SynergySelector has not been fitted.");

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
        throw new NotSupportedException("SynergySelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SynergySelector has not been fitted.");

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
