using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Multivariate;

/// <summary>
/// Conditional Mutual Information Maximization (CMIM) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features by maximizing mutual information with the target conditioned
/// on already selected features, ensuring each new feature adds unique information.
/// </para>
/// <para><b>For Beginners:</b> CMIM asks "what new information does this feature
/// give me that I don't already have?" It selects features that add something new,
/// conditioned on what you've already picked. This avoids redundant features more
/// rigorously than simple redundancy penalties.
/// </para>
/// </remarks>
public class CMIMSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private double[]? _cmimScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double[]? CMIMScores => _cmimScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public CMIMSelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
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
            "CMIMSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize
        var XBins = DiscretizeFeatures(X, n, p);
        var yBins = DiscretizeTarget(y, n);

        // Compute MI(X_j, Y) for all features
        var miWithTarget = new double[p];
        for (int j = 0; j < p; j++)
            miWithTarget[j] = ComputeMI(XBins, yBins, j, n);

        // Greedy forward selection with CMIM criterion
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        var conditionalMI = new double[p];
        for (int j = 0; j < p; j++)
            conditionalMI[j] = miWithTarget[j];

        _cmimScores = new double[p];

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            int bestFeature = -1;
            double bestScore = double.MinValue;

            foreach (int j in remaining)
            {
                // CMIM: min over selected features of I(X_j; Y | X_s)
                if (selected.Count > 0)
                {
                    double minConditionalMI = double.MaxValue;
                    foreach (int s in selected)
                    {
                        double cmi = ComputeConditionalMI(XBins, yBins, j, s, n);
                        minConditionalMI = Math.Min(minConditionalMI, cmi);
                    }
                    conditionalMI[j] = minConditionalMI;
                }

                if (conditionalMI[j] > bestScore)
                {
                    bestScore = conditionalMI[j];
                    bestFeature = j;
                }
            }

            if (bestFeature >= 0)
            {
                selected.Add(bestFeature);
                remaining.Remove(bestFeature);
                _cmimScores[bestFeature] = bestScore;
            }
            else
            {
                break;
            }
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
        IsFitted = true;
    }

    private int[,] DiscretizeFeatures(double[,] X, int n, int p)
    {
        var XBins = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double minVal = double.MaxValue, maxVal = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                minVal = Math.Min(minVal, X[i, j]);
                maxVal = Math.Max(maxVal, X[i, j]);
            }
            double binWidth = (maxVal - minVal) / _nBins + 1e-10;

            for (int i = 0; i < n; i++)
                XBins[i, j] = Math.Min((int)((X[i, j] - minVal) / binWidth), _nBins - 1);
        }
        return XBins;
    }

    private int[] DiscretizeTarget(double[] y, int n)
    {
        double minVal = y.Min(), maxVal = y.Max();
        double binWidth = (maxVal - minVal) / _nBins + 1e-10;
        var yBins = new int[n];
        for (int i = 0; i < n; i++)
            yBins[i] = Math.Min((int)((y[i] - minVal) / binWidth), _nBins - 1);
        return yBins;
    }

    private double ComputeMI(int[,] XBins, int[] yBins, int j, int n)
    {
        var jointCounts = new int[_nBins, _nBins];
        var xCounts = new int[_nBins];
        var yCounts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[XBins[i, j], yBins[i]]++;
            xCounts[XBins[i, j]]++;
            yCounts[yBins[i]]++;
        }

        double mi = 0;
        for (int xb = 0; xb < _nBins; xb++)
        {
            for (int yb = 0; yb < _nBins; yb++)
            {
                if (jointCounts[xb, yb] > 0 && xCounts[xb] > 0 && yCounts[yb] > 0)
                {
                    double pxy = (double)jointCounts[xb, yb] / n;
                    double px = (double)xCounts[xb] / n;
                    double py = (double)yCounts[yb] / n;
                    mi += pxy * Math.Log(pxy / (px * py) + 1e-10) / Math.Log(2);
                }
            }
        }
        return Math.Max(0, mi);
    }

    private double ComputeConditionalMI(int[,] XBins, int[] yBins, int j, int s, int n)
    {
        // I(X_j; Y | X_s) = H(X_j | X_s) - H(X_j | Y, X_s)
        // Approximated as I(X_j; Y) - I(X_j; X_s) (chain rule approximation)
        double miJY = ComputeMI(XBins, yBins, j, n);
        double miJS = ComputeMI_XX(XBins, j, s, n);
        return Math.Max(0, miJY - miJS);
    }

    private double ComputeMI_XX(int[,] XBins, int j1, int j2, int n)
    {
        var jointCounts = new int[_nBins, _nBins];
        var x1Counts = new int[_nBins];
        var x2Counts = new int[_nBins];

        for (int i = 0; i < n; i++)
        {
            jointCounts[XBins[i, j1], XBins[i, j2]]++;
            x1Counts[XBins[i, j1]]++;
            x2Counts[XBins[i, j2]]++;
        }

        double mi = 0;
        for (int b1 = 0; b1 < _nBins; b1++)
        {
            for (int b2 = 0; b2 < _nBins; b2++)
            {
                if (jointCounts[b1, b2] > 0 && x1Counts[b1] > 0 && x2Counts[b2] > 0)
                {
                    double pxy = (double)jointCounts[b1, b2] / n;
                    double px = (double)x1Counts[b1] / n;
                    double py = (double)x2Counts[b2] / n;
                    mi += pxy * Math.Log(pxy / (px * py) + 1e-10) / Math.Log(2);
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
            throw new InvalidOperationException("CMIMSelector has not been fitted.");

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
        throw new NotSupportedException("CMIMSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CMIMSelector has not been fitted.");

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
