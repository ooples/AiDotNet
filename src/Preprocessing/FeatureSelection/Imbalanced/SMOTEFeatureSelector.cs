using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Imbalanced;

/// <summary>
/// SMOTE-aware Feature Selection for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para>
/// This feature selector is designed for imbalanced datasets where one class is
/// significantly underrepresented. It evaluates features after synthetic minority
/// oversampling to ensure selected features work well on balanced data.
/// </para>
/// <para><b>For Beginners:</b> When you have few examples of one class (like rare diseases),
/// regular feature selection might ignore patterns specific to that class. This method
/// creates synthetic examples of the minority class first, then selects features that
/// distinguish both classes well.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SMOTEFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly double _samplingRatio;
    private readonly int? _randomState;

    private double[]? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double SamplingRatio => _samplingRatio;
    public double[]? FeatureScores => _featureScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SMOTEFeatureSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        double samplingRatio = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        if (samplingRatio <= 0 || samplingRatio > 2)
            throw new ArgumentException("Sampling ratio must be between 0 and 2.", nameof(samplingRatio));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _samplingRatio = samplingRatio;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SMOTEFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Split data by class
        var class0Indices = new List<int>();
        var class1Indices = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0Indices.Add(i);
            else
                class1Indices.Add(i);
        }

        // Determine minority class
        var (minorityIndices, majorityIndices) = class0Indices.Count < class1Indices.Count
            ? (class0Indices, class1Indices)
            : (class1Indices, class0Indices);

        // Generate synthetic samples using SMOTE
        var synthData = new List<double[]>();
        int nSynthetic = (int)((majorityIndices.Count - minorityIndices.Count) * _samplingRatio);

        if (nSynthetic > 0 && minorityIndices.Count >= _nNeighbors)
        {
            for (int s = 0; s < nSynthetic; s++)
            {
                // Pick random minority sample
                int baseIdx = minorityIndices[random.Next(minorityIndices.Count)];

                // Find k nearest neighbors among minority
                var neighbors = FindNeighbors(data, baseIdx, minorityIndices, p);

                // Pick random neighbor and interpolate
                int neighborIdx = neighbors[random.Next(Math.Min(_nNeighbors, neighbors.Count))];
                double lambda = random.NextDouble();

                var synthetic = new double[p];
                for (int j = 0; j < p; j++)
                {
                    double baseVal = NumOps.ToDouble(data[baseIdx, j]);
                    double neighborVal = NumOps.ToDouble(data[neighborIdx, j]);
                    synthetic[j] = baseVal + lambda * (neighborVal - baseVal);
                }
                synthData.Add(synthetic);
            }
        }

        // Compute feature scores on balanced data
        _featureScores = ComputeBalancedScores(data, target, synthData, minorityIndices, majorityIndices, n, p);

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _featureScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private List<int> FindNeighbors(Matrix<T> data, int baseIdx, List<int> candidates, int p)
    {
        var distances = new List<(int Index, double Distance)>();

        foreach (int idx in candidates)
        {
            if (idx == baseIdx) continue;

            double dist = 0;
            for (int j = 0; j < p; j++)
            {
                double diff = NumOps.ToDouble(data[baseIdx, j]) - NumOps.ToDouble(data[idx, j]);
                dist += diff * diff;
            }
            distances.Add((idx, Math.Sqrt(dist)));
        }

        return distances
            .OrderBy(d => d.Distance)
            .Take(_nNeighbors)
            .Select(d => d.Index)
            .ToList();
    }

    private double[] ComputeBalancedScores(
        Matrix<T> data, Vector<T> target,
        List<double[]> synthData,
        List<int> minorityIndices, List<int> majorityIndices,
        int n, int p)
    {
        var scores = new double[p];

        // Combine original majority with original + synthetic minority
        int totalN = majorityIndices.Count + minorityIndices.Count + synthData.Count;

        for (int j = 0; j < p; j++)
        {
            // Compute mean for each class
            double mean0 = 0, mean1 = 0;
            int count0 = majorityIndices.Count, count1 = minorityIndices.Count + synthData.Count;

            // Majority class mean
            foreach (int idx in majorityIndices)
                mean0 += NumOps.ToDouble(data[idx, j]);
            mean0 /= count0;

            // Minority class mean (original + synthetic)
            foreach (int idx in minorityIndices)
                mean1 += NumOps.ToDouble(data[idx, j]);
            foreach (var synth in synthData)
                mean1 += synth[j];
            mean1 /= count1;

            // Compute variance for each class
            double var0 = 0, var1 = 0;

            foreach (int idx in majorityIndices)
            {
                double diff = NumOps.ToDouble(data[idx, j]) - mean0;
                var0 += diff * diff;
            }
            var0 /= count0;

            foreach (int idx in minorityIndices)
            {
                double diff = NumOps.ToDouble(data[idx, j]) - mean1;
                var1 += diff * diff;
            }
            foreach (var synth in synthData)
            {
                double diff = synth[j] - mean1;
                var1 += diff * diff;
            }
            var1 /= count1;

            // Fisher's discriminant ratio
            double denom = var0 + var1;
            scores[j] = denom > 1e-10 ? Math.Pow(mean0 - mean1, 2) / denom : 0;
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
            throw new InvalidOperationException("SMOTEFeatureSelector has not been fitted.");

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
        throw new NotSupportedException("SMOTEFeatureSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SMOTEFeatureSelector has not been fitted.");

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
