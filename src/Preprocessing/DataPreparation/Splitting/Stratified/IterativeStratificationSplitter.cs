using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;

/// <summary>
/// Iterative stratification splitter for multi-label classification problems.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Standard stratification works well for single-label classification
/// where each sample belongs to exactly one class. But what about multi-label problems
/// where each sample can have multiple labels (like a movie being both "Action" AND "Comedy")?
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Calculate label frequencies across all samples
/// 2. Start with rarest label combinations
/// 3. Iteratively assign samples to folds, prioritizing balance for rare labels
/// 4. Continue until all samples are assigned
/// </para>
/// <para>
/// <b>Example:</b>
/// Document classification where each document can have multiple topics:
/// - Doc1: [Politics, Economy]
/// - Doc2: [Sports]
/// - Doc3: [Politics, Sports]
/// - Doc4: [Economy]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class IterativeStratificationSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private Matrix<T>? _labelMatrix;

    /// <summary>
    /// Creates a new Iterative Stratification splitter for multi-label data.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="labelMatrix">Optional matrix where each row is a sample and each column is a binary label indicator.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public IterativeStratificationSplitter(
        double testSize = 0.2,
        Matrix<T>? labelMatrix = null,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
        _labelMatrix = labelMatrix;
    }

    /// <summary>
    /// Sets the multi-label matrix.
    /// </summary>
    /// <param name="labelMatrix">Matrix where labelMatrix[i,j] = 1 if sample i has label j, 0 otherwise.</param>
    public IterativeStratificationSplitter<T> WithLabelMatrix(Matrix<T> labelMatrix)
    {
        _labelMatrix = labelMatrix;
        return this;
    }

    /// <inheritdoc/>
    public override bool RequiresLabels => false; // Uses label matrix instead

    /// <inheritdoc/>
    public override string Description => $"Iterative Stratification ({_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;

        if (_labelMatrix == null)
        {
            throw new InvalidOperationException(
                "Label matrix must be set for iterative stratification. Use WithLabelMatrix() method.");
        }

        if (_labelMatrix.Rows != nSamples)
        {
            throw new ArgumentException(
                $"Label matrix rows ({_labelMatrix.Rows}) must match X rows ({nSamples}).");
        }

        int nLabels = _labelMatrix.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));
        int targetTrainSize = nSamples - targetTestSize;

        // Calculate label frequencies
        var labelCounts = new double[nLabels];
        for (int j = 0; j < nLabels; j++)
        {
            for (int i = 0; i < nSamples; i++)
            {
                labelCounts[j] += Convert.ToDouble(_labelMatrix[i, j]);
            }
        }

        // Sort labels by frequency (rarest first)
        var labelOrder = Enumerable.Range(0, nLabels)
            .OrderBy(j => labelCounts[j])
            .ToArray();

        // Initialize sample availability and fold assignment
        var available = new HashSet<int>(Enumerable.Range(0, nSamples));
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        // Calculate desired label counts per fold
        var trainLabelCounts = new double[nLabels];
        var testLabelCounts = new double[nLabels];

        // Iterate through labels from rarest to most common
        foreach (int labelIdx in labelOrder)
        {
            if (available.Count == 0) break;

            // Get samples with this label that are still available
            var samplesWithLabel = new List<int>();
            foreach (int i in available)
            {
                if (Convert.ToDouble(_labelMatrix[i, labelIdx]) > 0)
                {
                    samplesWithLabel.Add(i);
                }
            }

            if (samplesWithLabel.Count == 0) continue;

            if (_shuffle)
            {
                ShuffleList(samplesWithLabel);
            }

            // Assign samples to balance this label between train/test
            double totalLabelCount = labelCounts[labelIdx];
            double targetTrainForLabel = totalLabelCount * (1 - _testSize);
            double targetTestForLabel = totalLabelCount * _testSize;

            foreach (int sampleIdx in samplesWithLabel)
            {
                if (!available.Contains(sampleIdx)) continue;

                // Decide which fold needs this label more
                double trainDeficit = targetTrainForLabel - trainLabelCounts[labelIdx];
                double testDeficit = targetTestForLabel - testLabelCounts[labelIdx];

                // Also consider overall size constraints
                double trainSizeRatio = (double)trainIndices.Count / targetTrainSize;
                double testSizeRatio = (double)testIndices.Count / targetTestSize;

                bool assignToTest;
                if (testIndices.Count >= targetTestSize)
                {
                    assignToTest = false;
                }
                else if (trainIndices.Count >= targetTrainSize)
                {
                    assignToTest = true;
                }
                else
                {
                    // Balance based on label deficit and size ratio
                    double trainScore = trainDeficit / Math.Max(1, targetTrainForLabel) - trainSizeRatio;
                    double testScore = testDeficit / Math.Max(1, targetTestForLabel) - testSizeRatio;
                    assignToTest = testScore > trainScore;
                }

                if (assignToTest)
                {
                    testIndices.Add(sampleIdx);
                    for (int j = 0; j < nLabels; j++)
                    {
                        testLabelCounts[j] += Convert.ToDouble(_labelMatrix[sampleIdx, j]);
                    }
                }
                else
                {
                    trainIndices.Add(sampleIdx);
                    for (int j = 0; j < nLabels; j++)
                    {
                        trainLabelCounts[j] += Convert.ToDouble(_labelMatrix[sampleIdx, j]);
                    }
                }

                available.Remove(sampleIdx);
            }
        }

        // Assign any remaining samples
        foreach (int sampleIdx in available)
        {
            if (trainIndices.Count < targetTrainSize)
            {
                trainIndices.Add(sampleIdx);
            }
            else
            {
                testIndices.Add(sampleIdx);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }

    private void ShuffleList(List<int> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
