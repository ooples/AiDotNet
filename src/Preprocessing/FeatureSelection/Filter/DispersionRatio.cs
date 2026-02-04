using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Dispersion Ratio for comparing within-class to between-class variation.
/// </summary>
/// <remarks>
/// <para>
/// The dispersion ratio measures the ratio of between-class variance to within-class
/// variance. Features with high dispersion ratios have well-separated class means
/// relative to the variation within each class.
/// </para>
/// <para><b>For Beginners:</b> A good feature for classification should have class
/// groups that are far apart (high between-class variance) and tight clusters
/// (low within-class variance). The dispersion ratio captures this by dividing
/// the separation by the spread. High values mean easy-to-separate classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class DispersionRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minRatio;

    private double[]? _dispersionRatios;
    private double[]? _betweenClassVariances;
    private double[]? _withinClassVariances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? DispersionRatios => _dispersionRatios;
    public double[]? BetweenClassVariances => _betweenClassVariances;
    public double[]? WithinClassVariances => _withinClassVariances;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public DispersionRatio(
        int nFeaturesToSelect = 10,
        double minRatio = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minRatio = minRatio;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DispersionRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classGroups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classGroups.ContainsKey(label))
                classGroups[label] = new List<int>();
            classGroups[label].Add(i);
        }

        int k = classGroups.Count;
        if (k < 2)
            throw new ArgumentException("At least 2 classes are required.");

        _dispersionRatios = new double[p];
        _betweenClassVariances = new double[p];
        _withinClassVariances = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Overall mean
            double overallMean = 0;
            for (int i = 0; i < n; i++)
                overallMean += NumOps.ToDouble(data[i, j]);
            overallMean /= n;

            // Compute class means
            var classMeans = new Dictionary<int, double>();
            foreach (var kvp in classGroups)
            {
                double classMean = 0;
                foreach (int i in kvp.Value)
                    classMean += NumOps.ToDouble(data[i, j]);
                classMeans[kvp.Key] = classMean / kvp.Value.Count;
            }

            // Between-class variance
            double betweenVar = 0;
            foreach (var kvp in classGroups)
            {
                double diff = classMeans[kvp.Key] - overallMean;
                betweenVar += kvp.Value.Count * diff * diff;
            }
            betweenVar /= n;
            _betweenClassVariances[j] = betweenVar;

            // Within-class variance
            double withinVar = 0;
            foreach (var kvp in classGroups)
            {
                double classMean = classMeans[kvp.Key];
                foreach (int i in kvp.Value)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - classMean;
                    withinVar += diff * diff;
                }
            }
            withinVar /= n;
            _withinClassVariances[j] = withinVar;

            // Dispersion ratio
            _dispersionRatios[j] = withinVar > 1e-10 ? betweenVar / withinVar : 0;
        }

        // Select features above threshold or top by ratio
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_dispersionRatios[j] >= _minRatio)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _dispersionRatios[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _dispersionRatios
                .Select((dr, idx) => (DR: dr, Index: idx))
                .OrderByDescending(x => x.DR)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

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
            throw new InvalidOperationException("DispersionRatio has not been fitted.");

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
        throw new NotSupportedException("DispersionRatio does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DispersionRatio has not been fitted.");

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
