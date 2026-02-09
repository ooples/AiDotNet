using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// SMOTE-NC generator that creates synthetic minority samples by interpolating between
/// existing minority samples and their k-nearest neighbors, supporting both continuous
/// and categorical features.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SMOTE-NC (Nominal and Continuous) operates as follows:
/// 1. Extract all minority class samples from the training data
/// 2. For each minority sample, find its k nearest neighbors among other minority samples
/// 3. Generate synthetic samples by interpolating between samples and randomly chosen neighbors
/// </para>
/// <para>
/// <b>For Beginners:</b> SMOTE-NC generates new minority samples by "mixing" existing ones:
///
/// Given a minority sample and a nearby neighbor:
/// - For numbers (age, income): pick a random point between the two values
///   Example: if sample has age=30 and neighbor has age=40, synthetic might have age=35
/// - For categories (gender, region): use the most common value among the k neighbors
///   Example: if 3 of 5 neighbors have region="East", synthetic gets region="East"
///
/// This is simpler and faster than GANs, and works well for structured tabular data.
/// </para>
/// </remarks>
public class SMOTENCGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly SMOTENCOptions<T> _options;

    // Stored minority class samples from the original data
    private Matrix<T>? _minoritySamples;

    // Column metadata (numerical vs categorical)
    private IReadOnlyList<ColumnMetadata>? _columns;

    // Median standard deviation of continuous features (used for SMOTE-NC distance metric)
    private double _medianStdContinuous = 1.0;

    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="SMOTENCGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the SMOTE-NC model.</param>
    public SMOTENCGenerator(SMOTENCOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;
        _columns = columns;

        // Extract minority samples
        int labelCol = _options.LabelColumnIndex;
        if (labelCol < 0 || labelCol >= _numFeatures) labelCol = _numFeatures - 1;

        var minorityRows = new List<int>();
        for (int i = 0; i < data.Rows; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(data[i, labelCol]));
            if (label == _options.MinorityClassValue)
                minorityRows.Add(i);
        }

        // If no minority samples found, use all data
        if (minorityRows.Count == 0)
        {
            _minoritySamples = new Matrix<T>(data.Rows, _numFeatures);
            for (int i = 0; i < data.Rows; i++)
                for (int j = 0; j < _numFeatures; j++)
                    _minoritySamples[i, j] = data[i, j];
        }
        else
        {
            _minoritySamples = new Matrix<T>(minorityRows.Count, _numFeatures);
            for (int i = 0; i < minorityRows.Count; i++)
                for (int j = 0; j < _numFeatures; j++)
                    _minoritySamples[i, j] = data[minorityRows[i], j];
        }

        // Compute median standard deviation of continuous features for SMOTE-NC distance metric.
        // Per Chawla et al., the categorical mismatch cost should be (medianStd)^2 per mismatch.
        ComputeMedianStdOfContinuousFeatures();
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_minoritySamples is null || _columns is null)
            throw new InvalidOperationException("Generator is not fitted.");

        int nMinority = _minoritySamples.Rows;
        int k = Math.Min(_options.K, nMinority - 1);
        if (k <= 0) k = 1;

        var result = new Matrix<T>(numSamples, _numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            // Pick a random minority sample
            int sampleIdx = Random.Next(nMinority);

            // Find k nearest neighbors
            var neighbors = FindKNearestNeighbors(sampleIdx, k);

            // Pick a random neighbor
            int neighborIdx = neighbors[Random.Next(neighbors.Length)];

            // Generate synthetic sample by interpolation
            for (int j = 0; j < _numFeatures; j++)
            {
                if (_columns[j].IsCategorical)
                {
                    // Categorical: majority vote among k nearest neighbors
                    result[i, j] = MajorityVoteAmongNeighbors(j, neighbors);
                }
                else
                {
                    // Continuous: linear interpolation
                    double val1 = NumOps.ToDouble(_minoritySamples[sampleIdx, j]);
                    double val2 = NumOps.ToDouble(_minoritySamples[neighborIdx, j]);
                    double lambda = Random.NextDouble();
                    result[i, j] = NumOps.FromDouble(val1 + lambda * (val2 - val1));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Finds the k nearest neighbors for a given minority sample using mixed-type distance.
    /// </summary>
    private int[] FindKNearestNeighbors(int sampleIndex, int k)
    {
        if (_minoritySamples is null || _columns is null) return Array.Empty<int>();

        int n = _minoritySamples.Rows;
        var distances = new (double dist, int idx)[n - 1];
        int count = 0;

        for (int i = 0; i < n; i++)
        {
            if (i == sampleIndex) continue;
            double dist = ComputeDistance(sampleIndex, i);
            distances[count++] = (dist, i);
        }

        // Partial sort to find k smallest
        Array.Sort(distances, 0, count, Comparer<(double dist, int idx)>.Create((a, b) => a.dist.CompareTo(b.dist)));

        int numNeighbors = Math.Min(k, count);
        var result = new int[numNeighbors];
        for (int i = 0; i < numNeighbors; i++)
            result[i] = distances[i].idx;

        return result;
    }

    /// <summary>
    /// Computes the median standard deviation of continuous features from the minority samples.
    /// This is used in the SMOTE-NC distance metric per Chawla et al.
    /// </summary>
    private void ComputeMedianStdOfContinuousFeatures()
    {
        if (_minoritySamples is null || _columns is null) return;

        int n = _minoritySamples.Rows;
        var stds = new List<double>();

        for (int j = 0; j < _numFeatures; j++)
        {
            if (_columns[j].IsCategorical) continue;

            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += NumOps.ToDouble(_minoritySamples[i, j]);
            double mean = sum / Math.Max(n, 1);

            double sumSqDiff = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(_minoritySamples[i, j]) - mean;
                sumSqDiff += diff * diff;
            }

            double std = n > 1 ? Math.Sqrt(sumSqDiff / (n - 1)) : 1.0;
            if (std < 1e-10) std = 1e-10;
            stds.Add(std);
        }

        if (stds.Count > 0)
        {
            stds.Sort();
            _medianStdContinuous = stds[stds.Count / 2];
        }
    }

    /// <summary>
    /// Computes the mixed-type distance between two samples using the SMOTE-NC formula.
    /// For continuous features: squared difference (Euclidean component).
    /// For categorical features: each mismatch contributes (medianStd)^2 to the squared distance.
    /// </summary>
    private double ComputeDistance(int idx1, int idx2)
    {
        if (_minoritySamples is null || _columns is null) return double.MaxValue;

        double sumSq = 0;
        double catPenaltySq = _medianStdContinuous * _medianStdContinuous;

        for (int j = 0; j < _numFeatures; j++)
        {
            if (_columns[j].IsCategorical)
            {
                double v1 = NumOps.ToDouble(_minoritySamples[idx1, j]);
                double v2 = NumOps.ToDouble(_minoritySamples[idx2, j]);
                if (Math.Abs(v1 - v2) > 0.5)
                {
                    // Per SMOTE-NC: categorical mismatch contributes (medianStd)^2
                    sumSq += catPenaltySq;
                }
            }
            else
            {
                double v1 = NumOps.ToDouble(_minoritySamples[idx1, j]);
                double v2 = NumOps.ToDouble(_minoritySamples[idx2, j]);
                double diff = v1 - v2;
                sumSq += diff * diff;
            }
        }

        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// Returns the majority vote value for a categorical feature among the given neighbors.
    /// </summary>
    private T MajorityVoteAmongNeighbors(int featureIndex, int[] neighbors)
    {
        if (_minoritySamples is null) return NumOps.Zero;

        var counts = new Dictionary<double, int>();
        foreach (int idx in neighbors)
        {
            double val = NumOps.ToDouble(_minoritySamples[idx, featureIndex]);
            double key = Math.Round(val);
            if (counts.ContainsKey(key))
                counts[key]++;
            else
                counts[key] = 1;
        }

        // Find the most common value
        double bestVal = 0;
        int bestCount = -1;
        foreach (var kvp in counts)
        {
            if (kvp.Value > bestCount)
            {
                bestCount = kvp.Value;
                bestVal = kvp.Key;
            }
        }

        return NumOps.FromDouble(bestVal);
    }
}
