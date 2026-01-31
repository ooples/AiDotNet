using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Imbalanced;

/// <summary>
/// Class-Weighted Feature Selection for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para>
/// Class-Weighted Feature Selection applies class weights when computing feature
/// scores to give more importance to minority class samples. This helps select
/// features that are discriminative for underrepresented classes.
/// </para>
/// <para><b>For Beginners:</b> In datasets where one class has few examples,
/// normal feature selection might favor features that only work for the common class.
/// This method weights the rare class more heavily, ensuring selected features
/// can identify both common and rare cases.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ClassWeightedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly string _weightingStrategy;

    private double[]? _featureScores;
    private double[]? _classWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public string WeightingStrategy => _weightingStrategy;
    public double[]? FeatureScores => _featureScores;
    public double[]? ClassWeights => _classWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public ClassWeightedFS(
        int nFeaturesToSelect = 10,
        string weightingStrategy = "balanced",
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        var validStrategies = new[] { "balanced", "sqrt", "log" };
        if (!validStrategies.Contains(weightingStrategy.ToLower()))
            throw new ArgumentException("Weighting strategy must be 'balanced', 'sqrt', or 'log'.", nameof(weightingStrategy));

        _nFeaturesToSelect = nFeaturesToSelect;
        _weightingStrategy = weightingStrategy.ToLower();
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ClassWeightedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Count class frequencies
        var classCounts = new Dictionary<int, int>();
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            classCounts[label] = classCounts.GetValueOrDefault(label, 0) + 1;
        }

        int nClasses = classCounts.Count;

        // Compute class weights
        _classWeights = ComputeClassWeights(classCounts, n, nClasses);

        // Compute weighted feature scores
        _featureScores = ComputeWeightedScores(data, target, classCounts, n, p);

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeClassWeights(Dictionary<int, int> classCounts, int n, int nClasses)
    {
        int maxClass = classCounts.Keys.Max();
        var weights = new double[maxClass + 1];

        foreach (var kvp in classCounts)
        {
            double weight;
            switch (_weightingStrategy)
            {
                case "balanced":
                    weight = (double)n / (nClasses * kvp.Value);
                    break;
                case "sqrt":
                    weight = Math.Sqrt((double)n / (nClasses * kvp.Value));
                    break;
                case "log":
                    weight = Math.Log(1 + (double)n / (nClasses * kvp.Value));
                    break;
                default:
                    weight = 1.0;
                    break;
            }
            weights[kvp.Key] = weight;
        }

        return weights;
    }

    private double[] ComputeWeightedScores(
        Matrix<T> data, Vector<T> target,
        Dictionary<int, int> classCounts,
        int n, int p)
    {
        var scores = new double[p];

        // Compute weighted means per class
        var classMeans = new Dictionary<int, double[]>();
        var classVars = new Dictionary<int, double[]>();

        foreach (int label in classCounts.Keys)
        {
            classMeans[label] = new double[p];
            classVars[label] = new double[p];
        }

        // Compute class means
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                classMeans[label][j] += NumOps.ToDouble(data[i, j]);
        }

        foreach (int label in classCounts.Keys)
        {
            for (int j = 0; j < p; j++)
                classMeans[label][j] /= classCounts[label];
        }

        // Compute class variances
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - classMeans[label][j];
                classVars[label][j] += diff * diff;
            }
        }

        foreach (int label in classCounts.Keys)
        {
            for (int j = 0; j < p; j++)
                classVars[label][j] /= classCounts[label];
        }

        // Compute weighted Fisher score for each feature
        for (int j = 0; j < p; j++)
        {
            double overallMean = 0;
            double totalWeight = 0;

            foreach (int label in classCounts.Keys)
            {
                double weight = _classWeights![label];
                overallMean += weight * classCounts[label] * classMeans[label][j];
                totalWeight += weight * classCounts[label];
            }
            overallMean /= totalWeight;

            double betweenVar = 0, withinVar = 0;

            foreach (int label in classCounts.Keys)
            {
                double weight = _classWeights![label];
                double meanDiff = classMeans[label][j] - overallMean;
                betweenVar += weight * classCounts[label] * meanDiff * meanDiff;
                withinVar += weight * classCounts[label] * classVars[label][j];
            }

            scores[j] = withinVar > 1e-10 ? betweenVar / withinVar : 0;
        }

        return scores;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClassWeightedFS has not been fitted.");

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
        throw new NotSupportedException("ClassWeightedFS does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClassWeightedFS has not been fitted.");

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
