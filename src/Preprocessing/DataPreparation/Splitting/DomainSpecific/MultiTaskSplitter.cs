using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Multi-task splitter that ensures consistent splits across multiple related tasks.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In multi-task learning, we train a model on multiple related tasks
/// simultaneously. It's important that the same samples appear in train/test consistently
/// across all tasks to fairly evaluate transfer learning benefits.
/// </para>
/// <para>
/// <b>Example:</b>
/// For a model predicting both "sentiment" and "topic" from text:
/// - Same documents should be in training for both tasks
/// - Same documents should be in testing for both tasks
/// </para>
/// <para>
/// <b>When to Use:</b>
/// - Multi-task neural networks
/// - Transfer learning experiments
/// - Multi-output regression/classification
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class MultiTaskSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _taskColumn;
    private readonly bool _stratifyByTask;

    /// <summary>
    /// Creates a new multi-task splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="taskColumn">Column index containing task IDs. Default is -1 (no task column, all samples share same tasks).</param>
    /// <param name="stratifyByTask">Whether to stratify by task. Default is false.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MultiTaskSplitter(
        double testSize = 0.2,
        int taskColumn = -1,
        bool stratifyByTask = false,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
        _taskColumn = taskColumn;
        _stratifyByTask = stratifyByTask;
    }

    /// <inheritdoc/>
    public override string Description => $"Multi-Task split ({_testSize * 100:F0}% test{(_stratifyByTask ? ", stratified" : "")})";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;
        int targetTestSize = Math.Max(1, (int)(nSamples * _testSize));

        if (_taskColumn >= 0 && _stratifyByTask)
        {
            return SplitStratifiedByTask(X, y, nSamples, nFeatures, targetTestSize);
        }
        else
        {
            return SplitSimple(X, y, nSamples, targetTestSize);
        }
    }

    private DataSplitResult<T> SplitSimple(Matrix<T> X, Vector<T>? y, int nSamples, int targetTestSize)
    {
        var indices = GetShuffledIndices(nSamples);
        var trainIndices = indices.Take(nSamples - targetTestSize).ToArray();
        var testIndices = indices.Skip(nSamples - targetTestSize).ToArray();

        return BuildResult(X, y, trainIndices, testIndices);
    }

    private DataSplitResult<T> SplitStratifiedByTask(Matrix<T> X, Vector<T>? y, int nSamples, int nFeatures, int targetTestSize)
    {
        if (_taskColumn >= nFeatures)
        {
            throw new ArgumentException(
                $"Task column ({_taskColumn}) exceeds feature count ({nFeatures}).");
        }

        // Group samples by task
        var taskSamples = new Dictionary<int, List<int>>();
        for (int i = 0; i < nSamples; i++)
        {
            int taskId = (int)Convert.ToDouble(X[i, _taskColumn]);
            if (!taskSamples.TryGetValue(taskId, out var list))
            {
                list = new List<int>();
                taskSamples[taskId] = list;
            }
            list.Add(i);
        }

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        // Stratified split within each task
        foreach (var kvp in taskSamples)
        {
            var taskIndices = kvp.Value.ToArray();
            if (_shuffle)
            {
                ShuffleIndices(taskIndices);
            }

            int taskTestSize = Math.Max(1, (int)(taskIndices.Length * _testSize));
            int taskTrainSize = taskIndices.Length - taskTestSize;

            for (int i = 0; i < taskTrainSize; i++)
            {
                trainIndices.Add(taskIndices[i]);
            }

            for (int i = taskTrainSize; i < taskIndices.Length; i++)
            {
                testIndices.Add(taskIndices[i]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
