using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements SMOTE-ENN combination for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SMOTE-ENN combines SMOTE oversampling with Edited Nearest Neighbors (ENN)
/// cleaning. ENN removes samples whose class differs from the majority of their k nearest neighbors,
/// cleaning both majority and minority samples to improve class separation.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Apply SMOTE to generate synthetic minority samples</item>
/// <item>For each sample, find its k nearest neighbors</item>
/// <item>If the majority of neighbors have a different class, remove the sample</item>
/// </list>
/// </para>
///
/// <para><b>Benefits over SMOTE-Tomek:</b>
/// <list type="bullet">
/// <item>More aggressive cleaning than Tomek links</item>
/// <item>Removes both misclassified majority AND minority samples</item>
/// <item>Better noise removal in overlapping regions</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Batista et al., "A Study of the Behavior of Several Methods for Balancing
/// Machine Learning Training Data" (2004)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SmoteEnnAugmenter<T> : TabularAugmenterBase<T>
{
    private readonly SmoteAugmenter<T> _smote;
    private readonly int _ennKNeighbors;

    /// <summary>
    /// Gets the number of nearest neighbors used by SMOTE.
    /// </summary>
    public int SmoteKNeighbors => _smote.KNeighbors;

    /// <summary>
    /// Gets the number of neighbors used by ENN.
    /// </summary>
    public int EnnKNeighbors => _ennKNeighbors;

    /// <summary>
    /// Gets the SMOTE sampling ratio.
    /// </summary>
    public double SamplingRatio => _smote.SamplingRatio;

    /// <summary>
    /// Creates a new SMOTE-ENN augmenter.
    /// </summary>
    /// <param name="smoteKNeighbors">Number of nearest neighbors for SMOTE (default: 5).</param>
    /// <param name="ennKNeighbors">Number of neighbors for ENN cleaning (default: 3).</param>
    /// <param name="samplingRatio">SMOTE sampling ratio (default: 1.0).</param>
    /// <param name="probability">Probability of applying augmentation (default: 1.0).</param>
    public SmoteEnnAugmenter(
        int smoteKNeighbors = 5,
        int ennKNeighbors = 3,
        double samplingRatio = 1.0,
        double probability = 1.0) : base(probability)
    {
        if (ennKNeighbors < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(ennKNeighbors),
                "ENN k neighbors must be at least 1.");
        }

        _smote = new SmoteAugmenter<T>(smoteKNeighbors, samplingRatio, 1.0);
        _ennKNeighbors = ennKNeighbors;
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // SMOTE-ENN on raw data - just apply SMOTE
        return _smote.Apply(data, context);
    }

    /// <summary>
    /// Applies SMOTE-ENN to a labeled dataset.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Tuple of (balanced data, balanced labels) after SMOTE and ENN cleaning.</returns>
    public (Matrix<T> Data, Vector<T> Labels) ApplySmoteEnn(
        Matrix<T> data,
        Vector<T> labels,
        AugmentationContext<T> context)
    {
        // Step 1: Identify minority class
        var classCounts = GetClassCounts(labels, out var classLabelsList);

        if (classCounts.Count > 2)
        {
            throw new ArgumentException(
                "SMOTE-ENN only supports binary classification. " +
                $"Found {classCounts.Count} classes. For multi-class problems, apply SMOTE-ENN to each minority class separately.");
        }

        var (minorityClass, majorityClass, _, _) = IdentifyClasses(classCounts, classLabelsList);

        // Step 2: Extract minority class samples
        var (minorityData, minorityLabels) = ExtractClassSamples(data, labels, minorityClass);

        // Step 3: Apply SMOTE to minority class
        var (smoteData, smoteLabels) = _smote.ApplySmoteWithLabels(
            minorityData, minorityLabels, context);

        // Step 4: Combine with majority class
        var (majorityData, majorityLabels) = ExtractClassSamples(data, labels, majorityClass);
        var (combinedData, combinedLabels) = CombineDatasets(
            smoteData, smoteLabels,
            majorityData, majorityLabels);

        // Step 5: Apply ENN cleaning
        return ApplyEnnCleaning(combinedData, combinedLabels);
    }

    /// <summary>
    /// Applies Edited Nearest Neighbors cleaning to remove noisy samples.
    /// </summary>
    private (Matrix<T> Data, Vector<T> Labels) ApplyEnnCleaning(Matrix<T> data, Vector<T> labels)
    {
        int n = data.Rows;
        int cols = data.Columns;
        int k = Math.Min(_ennKNeighbors, n - 1);

        // Compute distance matrix
        var distances = ComputeDistanceMatrix(data);

        // Identify samples to keep
        var keepIndices = new List<int>();

        // Build class index lookup
        var classLabelsLocal = new List<T>();
        for (int i = 0; i < n; i++)
        {
            bool found = false;
            for (int c = 0; c < classLabelsLocal.Count; c++)
            {
                if (NumOps.Compare(classLabelsLocal[c], labels[i]) == 0)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                classLabelsLocal.Add(labels[i]);
            }
        }

        for (int i = 0; i < n; i++)
        {
            // Find k nearest neighbors
            var neighbors = GetKNearestNeighbors(distances, i, k);

            // Count neighbor class votes using int indices
            var voteCounts = new Dictionary<int, int>();
            foreach (int neighborIdx in neighbors)
            {
                int classIdx = GetClassIndex(labels[neighborIdx], classLabelsLocal);
                if (!voteCounts.ContainsKey(classIdx))
                {
                    voteCounts[classIdx] = 0;
                }
                voteCounts[classIdx]++;
            }

            // Find majority class among neighbors
            int majorityClassIdx = 0;
            int maxVoteCount = int.MinValue;
            foreach (var kv in voteCounts)
            {
                if (kv.Value > maxVoteCount)
                {
                    maxVoteCount = kv.Value;
                    majorityClassIdx = kv.Key;
                }
            }
            int sampleClassIdx = GetClassIndex(labels[i], classLabelsLocal);

            // Keep sample if its class matches majority of neighbors
            if (sampleClassIdx == majorityClassIdx)
            {
                keepIndices.Add(i);
            }
        }

        // Guard against empty results from aggressive ENN cleaning
        if (keepIndices.Count == 0)
        {
            throw new InvalidOperationException(
                "ENN cleaning removed all samples. Consider reducing ennKNeighbors or reviewing data quality.");
        }

        // Create cleaned dataset
        var cleanedData = new Matrix<T>(keepIndices.Count, cols);
        var cleanedLabels = new Vector<T>(keepIndices.Count);

        for (int i = 0; i < keepIndices.Count; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                cleanedData[i, j] = data[keepIndices[i], j];
            }
            cleanedLabels[i] = labels[keepIndices[i]];
        }

        return (cleanedData, cleanedLabels);
    }

    private double[,] ComputeDistanceMatrix(Matrix<T> data)
    {
        int n = data.Rows;
        int cols = data.Columns;
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int c = 0; c < cols; c++)
                {
                    double diff = NumOps.ToDouble(data[i, c]) - NumOps.ToDouble(data[j, c]);
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return distances;
    }

    private int[] GetKNearestNeighbors(double[,] distances, int sampleIdx, int k)
    {
        int n = distances.GetLength(0);
        var indexedDistances = new List<(int Index, double Distance)>();

        for (int i = 0; i < n; i++)
        {
            if (i != sampleIdx)
            {
                indexedDistances.Add((i, distances[sampleIdx, i]));
            }
        }

        return indexedDistances
            .OrderBy(x => x.Distance)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();
    }

    private int GetClassIndex(T label, List<T> classLabels)
    {
        for (int i = 0; i < classLabels.Count; i++)
        {
            if (NumOps.Compare(classLabels[i], label) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    private Dictionary<int, int> GetClassCounts(Vector<T> labels, out List<T> classLabels)
    {
        var counts = new Dictionary<int, int>();
        classLabels = new List<T>();

        for (int i = 0; i < labels.Length; i++)
        {
            int classIdx = -1;
            for (int c = 0; c < classLabels.Count; c++)
            {
                if (NumOps.Compare(classLabels[c], labels[i]) == 0)
                {
                    classIdx = c;
                    break;
                }
            }

            if (classIdx < 0)
            {
                classIdx = classLabels.Count;
                classLabels.Add(labels[i]);
                counts[classIdx] = 0;
            }
            counts[classIdx]++;
        }
        return counts;
    }

    private (T Minority, T Majority, int MinCount, int MajCount) IdentifyClasses(
        Dictionary<int, int> classCounts, List<T> classLabels)
    {
        if (classCounts.Count == 0)
        {
            throw new InvalidOperationException("Cannot identify classes from empty class counts.");
        }

        if (classCounts.Count == 1)
        {
            throw new InvalidOperationException(
                "SMOTE-ENN requires at least two classes. Only one class found in the data.");
        }

        // Use the first entry as a real (non-default) baseline for both min and max.
        // Initializing both minority and majority to the same class is intentional:
        // the loop below will update them when it finds smaller or larger counts.
        var firstEntry = classCounts.First();
        T minority = classLabels[firstEntry.Key];
        T majority = classLabels[firstEntry.Key];
        int minCount = firstEntry.Value;
        int maxCount = firstEntry.Value;

        foreach (var kvp in classCounts)
        {
            if (kvp.Value < minCount)
            {
                minCount = kvp.Value;
                minority = classLabels[kvp.Key];
            }
            if (kvp.Value > maxCount)
            {
                maxCount = kvp.Value;
                majority = classLabels[kvp.Key];
            }
        }

        return (minority, majority, minCount, maxCount);
    }

    private (Matrix<T>, Vector<T>) ExtractClassSamples(Matrix<T> data, Vector<T> labels, T targetClass)
    {
        var indices = new List<int>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (NumOps.Compare(labels[i], targetClass) == 0)
            {
                indices.Add(i);
            }
        }

        var extractedData = new Matrix<T>(indices.Count, data.Columns);
        var extractedLabels = new Vector<T>(indices.Count);

        for (int i = 0; i < indices.Count; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                extractedData[i, j] = data[indices[i], j];
            }
            extractedLabels[i] = labels[indices[i]];
        }

        return (extractedData, extractedLabels);
    }

    private (Matrix<T>, Vector<T>) CombineDatasets(
        Matrix<T> data1, Vector<T> labels1,
        Matrix<T> data2, Vector<T> labels2)
    {
        if (data1.Columns != data2.Columns)
        {
            throw new ArgumentException(
                $"Cannot combine datasets with different column counts: {data1.Columns} vs {data2.Columns}.");
        }

        int totalRows = data1.Rows + data2.Rows;
        int cols = data1.Columns;

        var combined = new Matrix<T>(totalRows, cols);
        var labels = new Vector<T>(totalRows);

        for (int i = 0; i < data1.Rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[i, j] = data1[i, j];
            }
            labels[i] = labels1[i];
        }

        for (int i = 0; i < data2.Rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[data1.Rows + i, j] = data2[i, j];
            }
            labels[data1.Rows + i] = labels2[i];
        }

        return (combined, labels);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["smoteKNeighbors"] = SmoteKNeighbors;
        parameters["ennKNeighbors"] = EnnKNeighbors;
        parameters["samplingRatio"] = SamplingRatio;
        return parameters;
    }
}
