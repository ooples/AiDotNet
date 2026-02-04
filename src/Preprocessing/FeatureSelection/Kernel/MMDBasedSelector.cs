using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Maximum Mean Discrepancy (MMD) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses MMD to measure the distribution difference between classes in kernel space,
/// selecting features that maximize this discrepancy.
/// </para>
/// <para><b>For Beginners:</b> MMD measures how different two probability distributions
/// are. By computing MMD between different classes, we can find features that make
/// the classes most distinguishable. Features with high MMD are good at separating
/// classes.
/// </para>
/// </remarks>
public class MMDBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;

    private double[]? _mmdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public double[]? MMDScores => _mmdScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MMDBasedSelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MMDBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();

        _mmdScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var featureVals = new double[n];
            for (int i = 0; i < n; i++)
                featureVals[i] = X[i, j];

            // Compute MMD between each pair of classes and average
            double totalMMD = 0;
            int pairCount = 0;

            for (int c1Idx = 0; c1Idx < classes.Count; c1Idx++)
            {
                for (int c2Idx = c1Idx + 1; c2Idx < classes.Count; c2Idx++)
                {
                    int c1 = classes[c1Idx];
                    int c2 = classes[c2Idx];

                    var idx1 = Enumerable.Range(0, n).Where(i => y[i] == c1).ToList();
                    var idx2 = Enumerable.Range(0, n).Where(i => y[i] == c2).ToList();

                    if (idx1.Count == 0 || idx2.Count == 0) continue;

                    double mmd = ComputeMMD(featureVals, idx1, idx2);
                    totalMMD += mmd;
                    pairCount++;
                }
            }

            _mmdScores[j] = pairCount > 0 ? totalMMD / pairCount : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _mmdScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeMMD(double[] values, List<int> idx1, List<int> idx2)
    {
        int n1 = idx1.Count;
        int n2 = idx2.Count;

        // E[k(X, X')]
        double kXX = 0;
        for (int i = 0; i < n1; i++)
        {
            for (int i2 = 0; i2 < n1; i2++)
            {
                double diff = values[idx1[i]] - values[idx1[i2]];
                kXX += Math.Exp(-_gamma * diff * diff);
            }
        }
        kXX /= (n1 * n1);

        // E[k(Y, Y')]
        double kYY = 0;
        for (int i = 0; i < n2; i++)
        {
            for (int i2 = 0; i2 < n2; i2++)
            {
                double diff = values[idx2[i]] - values[idx2[i2]];
                kYY += Math.Exp(-_gamma * diff * diff);
            }
        }
        kYY /= (n2 * n2);

        // E[k(X, Y)]
        double kXY = 0;
        for (int i = 0; i < n1; i++)
        {
            for (int i2 = 0; i2 < n2; i2++)
            {
                double diff = values[idx1[i]] - values[idx2[i2]];
                kXY += Math.Exp(-_gamma * diff * diff);
            }
        }
        kXY /= (n1 * n2);

        // MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        double mmdSquared = kXX + kYY - 2 * kXY;
        return Math.Max(0, mmdSquared);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMDBasedSelector has not been fitted.");

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
        throw new NotSupportedException("MMDBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MMDBasedSelector has not been fitted.");

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
