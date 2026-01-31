using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Kernel;

/// <summary>
/// Kernel Fisher Discriminant-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Applies the Fisher criterion in kernel space to find features that maximize
/// class separation using non-linear transformations.
/// </para>
/// <para><b>For Beginners:</b> Fisher's criterion tries to maximize the distance
/// between class means while minimizing the spread within each class. Doing this
/// in kernel space allows us to find non-linear boundaries between classes,
/// selecting features that help separate classes in complex ways.
/// </para>
/// </remarks>
public class KernelFisherSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _gamma;

    private double[]? _fisherScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Gamma => _gamma;
    public double[]? FisherScores => _fisherScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KernelFisherSelector(
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
            "KernelFisherSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var classes = y.Distinct().ToList();

        _fisherScores = new double[p];

        // For each feature, compute kernel Fisher score
        for (int j = 0; j < p; j++)
        {
            // Use only this feature for kernel computation
            var featureVals = new double[n];
            for (int i = 0; i < n; i++)
                featureVals[i] = X[i, j];

            // Compute kernel values and class-conditional means
            double overallMean = 0;
            var classMeans = new Dictionary<int, double>();
            var classKernelSums = new Dictionary<int, double>();
            var classCounts = new Dictionary<int, int>();

            foreach (var c in classes)
            {
                classMeans[c] = 0;
                classKernelSums[c] = 0;
                classCounts[c] = 0;
            }

            // Compute pairwise kernel values
            for (int i = 0; i < n; i++)
            {
                double kernelSum = 0;
                for (int i2 = 0; i2 < n; i2++)
                {
                    double diff = featureVals[i] - featureVals[i2];
                    double k = Math.Exp(-_gamma * diff * diff);
                    kernelSum += k;
                }
                classMeans[y[i]] += kernelSum;
                classCounts[y[i]]++;
                overallMean += kernelSum;
            }

            overallMean /= n;
            foreach (var c in classes)
                if (classCounts[c] > 0)
                    classMeans[c] /= classCounts[c];

            // Between-class scatter (in kernel space)
            double betweenScatter = 0;
            foreach (var c in classes)
            {
                double diff = classMeans[c] - overallMean;
                betweenScatter += classCounts[c] * diff * diff;
            }

            // Within-class scatter (in kernel space)
            double withinScatter = 0;
            for (int i = 0; i < n; i++)
            {
                double kernelSum = 0;
                for (int i2 = 0; i2 < n; i2++)
                {
                    double diff = featureVals[i] - featureVals[i2];
                    kernelSum += Math.Exp(-_gamma * diff * diff);
                }
                double deviation = kernelSum - classMeans[y[i]];
                withinScatter += deviation * deviation;
            }

            _fisherScores[j] = withinScatter > 1e-10 ? betweenScatter / withinScatter : 0;
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
