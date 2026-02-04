using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Naive Bayes based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their discriminative power using Naive Bayes
/// class-conditional probability estimates.
/// </para>
/// <para><b>For Beginners:</b> Naive Bayes assumes features are independent given
/// the class. This selector measures how much each feature helps distinguish
/// between classes by looking at how different the feature distributions are
/// for each class.
/// </para>
/// </remarks>
public class NaiveBayesSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _discriminativePower;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DiscriminativePower => _discriminativePower;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public NaiveBayesSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NaiveBayesSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var classIndices = new Dictionary<int, List<int>>();
        foreach (var c in classes)
            classIndices[c] = Enumerable.Range(0, n).Where(i => y[i] == c).ToList();

        _discriminativePower = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute class-conditional means and variances
            var classMeans = new Dictionary<int, double>();
            var classVars = new Dictionary<int, double>();

            foreach (var c in classes)
            {
                var indices = classIndices[c];
                double mean = indices.Average(i => X[i, j]);
                double variance = indices.Average(i => (X[i, j] - mean) * (X[i, j] - mean));
                classMeans[c] = mean;
                classVars[c] = Math.Max(variance, 1e-10);
            }

            // Compute discriminative power as KL divergence between class distributions
            double totalDivergence = 0;
            int pairCount = 0;

            for (int c1Idx = 0; c1Idx < classes.Count; c1Idx++)
            {
                for (int c2Idx = c1Idx + 1; c2Idx < classes.Count; c2Idx++)
                {
                    int c1 = classes[c1Idx];
                    int c2 = classes[c2Idx];

                    double kl = ComputeGaussianKL(
                        classMeans[c1], classVars[c1],
                        classMeans[c2], classVars[c2]);
                    totalDivergence += kl;
                    pairCount++;
                }
            }

            _discriminativePower[j] = pairCount > 0 ? totalDivergence / pairCount : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _discriminativePower[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeGaussianKL(double mean1, double var1, double mean2, double var2)
    {
        // Symmetric KL divergence between two Gaussians
        double kl12 = 0.5 * (Math.Log(var2 / var1) + var1 / var2 +
            (mean1 - mean2) * (mean1 - mean2) / var2 - 1);
        double kl21 = 0.5 * (Math.Log(var1 / var2) + var2 / var1 +
            (mean2 - mean1) * (mean2 - mean1) / var1 - 1);
        return (kl12 + kl21) / 2;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NaiveBayesSelector has not been fitted.");

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
        throw new NotSupportedException("NaiveBayesSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NaiveBayesSelector has not been fitted.");

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
