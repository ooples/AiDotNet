using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel Fisher Discriminant Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Extends Fisher's discriminant analysis to kernel space, finding features
/// that maximize class separability in a non-linear projected space.
/// </para>
/// <para><b>For Beginners:</b> Fisher's method tries to find directions that
/// best separate different classes. Kernel Fisher does this but in a
/// transformed space where non-linear patterns become linear. Features
/// that help with this non-linear class separation are selected.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KernelFisherSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;
    private readonly double _regularization;

    private double[]? _fisherScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FisherScores => _fisherScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KernelFisherSelector(
        int nFeaturesToSelect = 10,
        double gamma = 1.0,
        double regularization = 0.01,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _gamma = gamma;
        _regularization = regularization;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KernelFisherSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Convert to arrays
        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Get class indices
        var classes = y.Distinct().OrderBy(c => c).ToList();
        var classIndices = new Dictionary<int, List<int>>();
        foreach (int c in classes)
            classIndices[c] = new List<int>();
        for (int i = 0; i < n; i++)
            classIndices[y[i]].Add(i);

        _fisherScores = new double[p];

        // Compute kernel Fisher score for each feature
        for (int j = 0; j < p; j++)
        {
            // Extract single feature column
            var xj = new double[n];
            for (int i = 0; i < n; i++)
                xj[i] = X[i, j];

            // Compute 1D kernel matrices
            var K = new double[n, n];
            for (int i1 = 0; i1 < n; i1++)
            {
                for (int i2 = i1; i2 < n; i2++)
                {
                    double diff = xj[i1] - xj[i2];
                    double k = Math.Exp(-_gamma * diff * diff);
                    K[i1, i2] = k;
                    K[i2, i1] = k;
                }
            }

            // Compute between-class scatter in kernel space
            double betweenScatter = 0;
            double totalMean = 0;
            for (int i = 0; i < n; i++)
                for (int i2 = 0; i2 < n; i2++)
                    totalMean += K[i, i2];
            totalMean /= (n * n);

            foreach (int c in classes)
            {
                var indices = classIndices[c];
                int nc = indices.Count;
                if (nc == 0) continue;

                double classMean = 0;
                foreach (int i1 in indices)
                    foreach (int i2 in indices)
                        classMean += K[i1, i2];
                classMean /= (nc * nc);

                betweenScatter += nc * (classMean - totalMean) * (classMean - totalMean);
            }

            // Compute within-class scatter in kernel space
            double withinScatter = 0;
            foreach (int c in classes)
            {
                var indices = classIndices[c];
                int nc = indices.Count;
                if (nc < 2) continue;

                double classMean = 0;
                foreach (int i1 in indices)
                    foreach (int i2 in indices)
                        classMean += K[i1, i2];
                classMean /= (nc * nc);

                foreach (int i in indices)
                {
                    double sampleMean = 0;
                    foreach (int i2 in indices)
                        sampleMean += K[i, i2];
                    sampleMean /= nc;
                    withinScatter += (sampleMean - classMean) * (sampleMean - classMean);
                }
            }

            _fisherScores[j] = betweenScatter / (withinScatter + _regularization);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _fisherScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelFisherSelector has not been fitted.");

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
        throw new NotSupportedException("KernelFisherSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KernelFisherSelector has not been fitted.");

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
