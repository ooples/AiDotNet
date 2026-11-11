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
        // Validate inputs
        if (data.Rows == 0 || labels.Length == 0)
        {
            throw new ArgumentException("Data and labels must be non-empty.");
        }
        if (data.Rows != labels.Length)
        {
            throw new ArgumentException(
                $"Row count mismatch: data has {data.Rows} rows but labels has {labels.Length} items.");
        }

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
        int k = Math.Min(_ennKNeighbors, n - 1);

        // Compute distance matrix
        var distances = ComputeDistanceMatrix(data);

        // Identify samples to keep
        var keepIndices = new List<int>();

        // Build class index lookup
        var classOf = TabularEngineOps<T>.ClassIndices(labels, out _, out _);

        for (int i = 0; i < n; i++)
        {
            // Find k nearest neighbors
            var neighbors = GetKNearestNeighbors(distances, i, k);

            // Count neighbor class votes using int indices
            var voteCounts = new Dictionary<int, int>();
            foreach (int neighborIdx in neighbors)
            {
                int classIdx = classOf[neighborIdx];
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
            int sampleClassIdx = classOf[i];

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
        var (cleanedData, cleanedLabels) = TabularEngineOps<T>.SelectRows(data, labels, keepIndices.ToArray());

        return (cleanedData, cleanedLabels);
    }

    private static Tensor<double> ComputeDistanceMatrix(Matrix<T> data)
    {
        return TabularEngineOps<T>.PairwiseDistances(data);
    }

    private static int[] GetKNearestNeighbors(Tensor<double> distances, int sampleIdx, int k)
    {
        return TabularEngineOps<T>.NearestNeighbors(distances, sampleIdx, k);
    }

    private Dictionary<int, int> GetClassCounts(Vector<T> labels, out List<T> classLabels)
    {
        TabularEngineOps<T>.ClassIndices(labels, out classLabels, out var perClass);
        var counts = new Dictionary<int, int>();
        for (int c = 0; c < perClass.Length; c++)
        {
            counts[c] = perClass[c];
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

        // For binary classification, ensure minority and majority are always distinct
        // even when classes have equal counts
        if (classCounts.Count == 2)
        {
            var keys = classCounts.Keys.ToList();
            if (classCounts[keys[0]] <= classCounts[keys[1]])
            {
                return (classLabels[keys[0]], classLabels[keys[1]],
                        classCounts[keys[0]], classCounts[keys[1]]);
            }
            else
            {
                return (classLabels[keys[1]], classLabels[keys[0]],
                        classCounts[keys[1]], classCounts[keys[0]]);
            }
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
        var indices = TabularEngineOps<T>.IndicesEqualTo(labels, targetClass);
        return TabularEngineOps<T>.SelectRows(data, labels, indices);
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

        var combined = TabularEngineOps<T>.ConcatenateRows(data1, data2);
        var labels = TabularEngineOps<T>.ConcatenateLabels(labels1, data1.Rows, labels2, data2.Rows);

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
