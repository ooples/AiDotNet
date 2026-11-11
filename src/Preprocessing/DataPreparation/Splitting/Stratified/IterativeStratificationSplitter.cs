using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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

        var engine = AiDotNetEngine.Current;
        var labels = Tensor<T>.FromMatrix(_labelMatrix);

        // Calculate label frequencies
        var labelCounts = engine.ReduceSum(labels, new[] { 0 }, false);

        // Sort labels by frequency (rarest first, ties in label order)
        var labelOrder = engine.TensorArgsort(labelCounts, 0, false).AsSpan().ToArray();

        // Membership flags (label value > 0), copied once for the sequential assignment below
        var membership = engine.TensorGreaterThan(labels, NumOps.Zero).Cast<int>().AsSpan().ToArray();

        // Initialize sample availability and fold assignment
        var available = new HashSet<int>(Enumerable.Range(0, nSamples));
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        // Running label counts per fold
        var trainLabelCounts = new Tensor<T>(new[] { nLabels });
        var testLabelCounts = new Tensor<T>(new[] { nLabels });

        // Iterate through labels from rarest to most common
        foreach (int labelIdx in labelOrder)
        {
            if (available.Count == 0) break;

            // Get samples with this label that are still available
            var samplesWithLabel = new List<int>();
            foreach (int i in available)
            {
                if (membership[i * nLabels + labelIdx] != 0)
                {
                    samplesWithLabel.Add(i);
                }
            }

            if (samplesWithLabel.Count == 0) continue;

            if (_shuffle)
            {
                ShuffleList(samplesWithLabel);
            }

            foreach (int sampleIdx in samplesWithLabel)
            {
                if (!available.Contains(sampleIdx)) continue;

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
                    assignToTest = TestNeedsLabelMore(
                        labelCounts, trainLabelCounts, testLabelCounts, labelIdx,
                        (double)trainIndices.Count / targetTrainSize, (double)testIndices.Count / targetTestSize);
                }

                var sampleLabels = engine.Reshape(engine.TensorNarrow(labels, 0, sampleIdx, 1), new[] { nLabels });
                if (assignToTest)
                {
                    testIndices.Add(sampleIdx);
                    testLabelCounts = engine.TensorAdd(testLabelCounts, sampleLabels);
                }
                else
                {
                    trainIndices.Add(sampleIdx);
                    trainLabelCounts = engine.TensorAdd(trainLabelCounts, sampleLabels);
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

    /// <summary>
    /// Balances one label between the folds: with targets total (1 - testSize) and total testSize, each fold scores
    /// (target - count) / max(1, target) - its size ratio, and the sample goes to test when test scores higher.
    /// </summary>
    private bool TestNeedsLabelMore(
        Tensor<T> labelCounts, Tensor<T> trainLabelCounts, Tensor<T> testLabelCounts, int labelIdx,
        double trainSizeRatio, double testSizeRatio)
    {
        var ops = NumOps;
        T total = labelCounts[labelIdx];
        T targetTrain = ops.Multiply(total, ops.FromDouble(1 - _testSize));
        T targetTest = ops.Multiply(total, ops.FromDouble(_testSize));

        // Decide which fold needs this label more, also considering overall size constraints
        T trainDeficit = ops.Subtract(targetTrain, trainLabelCounts[labelIdx]);
        T testDeficit = ops.Subtract(targetTest, testLabelCounts[labelIdx]);
        T trainScore = ops.Subtract(ops.Divide(trainDeficit, MaxWithOne(targetTrain)), ops.FromDouble(trainSizeRatio));
        T testScore = ops.Subtract(ops.Divide(testDeficit, MaxWithOne(targetTest)), ops.FromDouble(testSizeRatio));
        return ops.GreaterThan(testScore, trainScore);
    }

    private static T MaxWithOne(T value) => NumOps.GreaterThan(value, NumOps.One) ? value : NumOps.One;

    private void ShuffleList(List<int> list)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
