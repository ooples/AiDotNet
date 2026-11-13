using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;

/// <summary>
/// Covariate shift splitter that creates intentional distribution shift between train and test.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Covariate shift occurs when the distribution of input features
/// differs between training and test data, even though the relationship between inputs
/// and outputs remains the same. This splitter intentionally creates such a shift.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// 1. Identify a primary feature dimension for the shift
/// 2. Select training samples from one range of that feature
/// 3. Select test samples from a different range
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Testing model robustness to distribution shift
/// - Simulating real-world deployment scenarios
/// - Evaluating domain adaptation techniques
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CovariateShiftSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly double _shiftStrength;
    private readonly int _shiftDimension;

    /// <summary>
    /// Creates a new covariate shift splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shiftStrength">Strength of the distribution shift (0-1). Default is 0.5.</param>
    /// <param name="shiftDimension">Feature index to use for shift. Default is 0 (first feature).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public CovariateShiftSplitter(
        double testSize = 0.2,
        double shiftStrength = 0.5,
        int shiftDimension = 0,
        int randomSeed = 42)
        : base(shuffle: true, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        if (shiftStrength < 0 || shiftStrength > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(shiftStrength), "Shift strength must be between 0 and 1.");
        }

        if (shiftDimension < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(shiftDimension), "Shift dimension must be non-negative.");
        }

        _testSize = testSize;
        _shiftStrength = shiftStrength;
        _shiftDimension = shiftDimension;
    }

    /// <inheritdoc/>
    public override string Description => $"Covariate Shift split (strength={_shiftStrength:F2}, {_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        int shiftDim = Math.Min(_shiftDimension, nFeatures - 1);

        // Get values along the shift dimension
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var values = PreprocessingTensorOps.Column(X, shiftDim);

        // Calculate percentiles: order statistics with the NaNs first, as master's OrderBy over doubles sorts them
        var numberPositions = PreprocessingTensorOps.MaskPositions(engine.TensorEquals(values, values));
        int nanCount = nSamples - numberPositions.Length;
        var sortedNumbers = numberPositions.Length == 0
            ? new Tensor<T>(new[] { 0 })
            : engine.TensorSort(engine.TensorIndexSelect(
                values, new Tensor<int>(numberPositions, new[] { numberPositions.Length }), 0)).Item1;
        T OrderStatistic(int index) =>
            index < nanCount ? NumOps.FromDouble(double.NaN) : sortedNumbers.GetFlat(index - nanCount);
        T median = OrderStatistic(nSamples / 2);
        T q25 = OrderStatistic((int)(nSamples * 0.25));
        T q75 = OrderStatistic((int)(nSamples * 0.75));

        // Determine shift threshold based on shift strength
        // Higher shift strength means more separation
        T strength = NumOps.FromDouble(_shiftStrength);
        T trainThreshold = NumOps.Subtract(median, NumOps.Multiply(NumOps.Subtract(median, q25), strength));
        T testThreshold = NumOps.Add(median, NumOps.Multiply(NumOps.Subtract(q75, median), strength));

        // Categorize samples: at most the train threshold, else at least the test threshold, else middle. A NaN is
        // neither (every comparison with it is false, as in master), so it is a middle sample.
        var trainMask = engine.TensorLessEqual(values, PreprocessingTensorOps.Filled(new[] { nSamples }, trainThreshold));
        var testMask = engine.TensorMultiply(PreprocessingTensorOps.AtLeast(values, testThreshold), PreprocessingTensorOps.Not(trainMask));
        var middleMask = PreprocessingTensorOps.Not(engine.TensorAdd(trainMask, testMask));
        var trainCandidates = PreprocessingTensorOps.MaskPositions(trainMask).ToList();
        var testCandidates = PreprocessingTensorOps.MaskPositions(testMask).ToList();
        var middleSamples = PreprocessingTensorOps.MaskPositions(middleMask).ToList();

        // Shuffle candidates
        var trainArray = trainCandidates.ToArray();
        var testArray = testCandidates.ToArray();
        var middleArray = middleSamples.ToArray();
        ShuffleIndices(trainArray);
        ShuffleIndices(testArray);
        ShuffleIndices(middleArray);

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        // Add test candidates first
        foreach (int idx in testArray)
        {
            if (testIndices.Count < targetTestSize)
            {
                testIndices.Add(idx);
            }
            else
            {
                trainIndices.Add(idx);
            }
        }

        // Add middle samples to fill as needed
        foreach (int idx in middleArray)
        {
            if (testIndices.Count < targetTestSize)
            {
                testIndices.Add(idx);
            }
            else
            {
                trainIndices.Add(idx);
            }
        }

        // Add train candidates
        trainIndices.AddRange(trainArray);

        // If we still need more test samples, take from train
        while (testIndices.Count < targetTestSize && trainIndices.Count > 1)
        {
            int idx = trainIndices[trainIndices.Count - 1];
            trainIndices.RemoveAt(trainIndices.Count - 1);
            testIndices.Add(idx);
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
