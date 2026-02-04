using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Tabular;

/// <summary>
/// Implements SMOTE-Tomek combination for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> SMOTE-Tomek combines SMOTE oversampling with Tomek links cleaning.
/// First, SMOTE generates synthetic minority samples, then Tomek links are removed to clean
/// the class boundary and reduce noise.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Apply SMOTE to generate synthetic minority samples</item>
/// <item>Identify Tomek links (pairs of samples from different classes that are mutual nearest neighbors)</item>
/// <item>Remove the majority class samples from Tomek links</item>
/// </list>
/// </para>
///
/// <para><b>Benefits:</b>
/// <list type="bullet">
/// <item>Balances classes while cleaning the decision boundary</item>
/// <item>Reduces noise introduced by SMOTE near the boundary</item>
/// <item>More robust than SMOTE alone</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Batista et al., "A Study of the Behavior of Several Methods for Balancing
/// Machine Learning Training Data" (2004)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SmoteTomekAugmenter<T> : TabularAugmenterBase<T>
{
    private readonly SmoteAugmenter<T> _smote;
    private readonly TomekLinksAugmenter<T> _tomekLinks;

    /// <summary>
    /// Gets the number of nearest neighbors used by SMOTE.
    /// </summary>
    public int KNeighbors => _smote.KNeighbors;

    /// <summary>
    /// Gets the SMOTE sampling ratio.
    /// </summary>
    public double SamplingRatio => _smote.SamplingRatio;

    /// <summary>
    /// Creates a new SMOTE-Tomek augmenter.
    /// </summary>
    /// <param name="kNeighbors">Number of nearest neighbors for SMOTE (default: 5).</param>
    /// <param name="samplingRatio">SMOTE sampling ratio (default: 1.0).</param>
    /// <param name="probability">Probability of applying augmentation (default: 1.0).</param>
    public SmoteTomekAugmenter(
        int kNeighbors = 5,
        double samplingRatio = 1.0,
        double probability = 1.0) : base(probability)
    {
        _smote = new SmoteAugmenter<T>(kNeighbors, samplingRatio, 1.0);
        _tomekLinks = new TomekLinksAugmenter<T>(TomekLinksAugmenter<T>.RemovalStrategy.RemoveMajority, 1.0);
    }

    /// <inheritdoc />
    protected override Matrix<T> ApplyAugmentation(Matrix<T> data, AugmentationContext<T> context)
    {
        // SMOTE-Tomek on raw data - just apply SMOTE
        return _smote.Apply(data, context);
    }

    /// <summary>
    /// Applies SMOTE-Tomek to a labeled dataset.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="labels">Class labels.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>Tuple of (balanced data, balanced labels) after SMOTE and Tomek links removal.</returns>
    public (Matrix<T> Data, Vector<T> Labels) ApplySmoteTomek(
        Matrix<T> data,
        Vector<T> labels,
        AugmentationContext<T> context)
    {
        // Step 1: Identify minority and majority classes
        var classCounts = GetClassCounts(labels, out var classLabels);
        var (minorityClass, majorityClass, minorityCount, majorityCount) =
            IdentifyClasses(classCounts, classLabels);

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

        // Step 5: Remove Tomek links
        return _tomekLinks.RemoveTomekLinks(combinedData, combinedLabels, minorityClass);
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

        // Use first entry as initial values to avoid default!
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
        int totalRows = data1.Rows + data2.Rows;
        int cols = data1.Columns;

        var combined = new Matrix<T>(totalRows, cols);
        var labels = new Vector<T>(totalRows);

        // Copy first dataset
        for (int i = 0; i < data1.Rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                combined[i, j] = data1[i, j];
            }
            labels[i] = labels1[i];
        }

        // Copy second dataset
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
        parameters["kNeighbors"] = KNeighbors;
        parameters["samplingRatio"] = SamplingRatio;
        return parameters;
    }
}
