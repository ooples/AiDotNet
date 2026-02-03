using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Fisher Score for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Fisher Score measures the ratio of between-class variance to within-class variance.
/// Features with high Fisher Scores have good class separability, meaning samples from
/// different classes are far apart while samples from the same class are close together.
/// </para>
/// <para><b>For Beginners:</b> Imagine you're sorting fruits by color. A good feature (like
/// color) would make all apples similar to each other but different from oranges. Fisher
/// Score measures exactly this: how much a feature groups similar items together while
/// keeping different groups apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FisherScore<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _fisherScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? FisherScores => _fisherScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public FisherScore(
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
            "FisherScore requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!classSamples.ContainsKey(classLabel))
                classSamples[classLabel] = [];
            classSamples[classLabel].Add(i);
        }

        var classLabels = classSamples.Keys.ToList();
        int numClasses = classLabels.Count;

        _fisherScores = new double[p];

        // Compute global mean for each feature
        var globalMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            for (int i = 0; i < n; i++)
                globalMeans[j] += NumOps.ToDouble(data[i, j]);
            globalMeans[j] /= n;
        }

        for (int j = 0; j < p; j++)
        {
            // Compute class means and variances
            var classMeans = new Dictionary<int, double>();
            var classVars = new Dictionary<int, double>();

            foreach (int c in classLabels)
            {
                var samples = classSamples[c];
                double mean = 0;
                foreach (int i in samples)
                    mean += NumOps.ToDouble(data[i, j]);
                mean /= samples.Count;
                classMeans[c] = mean;

                double variance = 0;
                foreach (int i in samples)
                {
                    double diff = NumOps.ToDouble(data[i, j]) - mean;
                    variance += diff * diff;
                }
                variance /= samples.Count;
                classVars[c] = variance;
            }

            // Between-class variance: sum of n_k * (mean_k - global_mean)^2
            double betweenVar = 0;
            foreach (int c in classLabels)
            {
                double diff = classMeans[c] - globalMeans[j];
                betweenVar += classSamples[c].Count * diff * diff;
            }

            // Within-class variance: sum of n_k * variance_k
            double withinVar = 0;
            foreach (int c in classLabels)
            {
                withinVar += classSamples[c].Count * classVars[c];
            }

            // Fisher Score = between-class variance / within-class variance
            _fisherScores[j] = withinVar > 1e-10 ? betweenVar / withinVar : 0;
        }

        // Select top features by Fisher Score
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _fisherScores
            .Select((s, idx) => (Score: s, Index: idx))
            .OrderByDescending(x => x.Score)
            .Take(numToSelect)
            .Select(x => x.Index)
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
            throw new InvalidOperationException("FisherScore has not been fitted.");

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
        throw new NotSupportedException("FisherScore does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FisherScore has not been fitted.");

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
