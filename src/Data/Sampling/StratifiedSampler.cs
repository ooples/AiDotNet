using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Data.Sampling;

/// <summary>
/// A sampler that ensures each class is represented proportionally in each epoch.
/// </summary>
/// <remarks>
/// <para>
/// StratifiedSampler maintains the class distribution from the original dataset
/// during sampling. This is especially important for imbalanced datasets where
/// some classes have many more samples than others.
/// </para>
/// <para><b>For Beginners:</b> When your dataset has unequal class sizes (e.g., 90% cats, 10% dogs),
/// random sampling might sometimes produce batches with only cats. Stratified sampling
/// ensures every batch has a similar ratio of cats to dogs as the full dataset.
///
/// Example:
/// <code>
/// // Labels: [0, 0, 0, 1, 1, 2] (3 samples of class 0, 2 of class 1, 1 of class 2)
/// var sampler = new StratifiedSampler(labels, numClasses: 3);
/// // Each epoch will maintain the 3:2:1 ratio while shuffling within classes
/// </code>
/// </para>
/// </remarks>
public class StratifiedSampler : DataSamplerBase, IStratifiedSampler
{
    private int[] _labels;
    private readonly int _numClasses;
    private Dictionary<int, List<int>>? _classIndices;

    /// <summary>
    /// Initializes a new instance of the StratifiedSampler class.
    /// </summary>
    /// <param name="labels">The class label for each sample (0-indexed).</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <exception cref="ArgumentNullException">Thrown when labels is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when numClasses is less than 2.</exception>
    public StratifiedSampler(IEnumerable<int> labels, int numClasses, int? seed = null)
        : base(seed)
    {
        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (numClasses < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 2.");
        }

        _labels = labels.ToArray();
        _numClasses = numClasses;
        BuildClassIndices();
    }

    /// <inheritdoc/>
    public override int Length => _labels.Length;

    /// <inheritdoc/>
    public int NumClasses => _numClasses;

    /// <inheritdoc/>
    public IReadOnlyList<int> Labels
    {
        get => _labels;
        set
        {
            Guard.NotNull(value);
            _labels = value.ToArray();
            BuildClassIndices();
        }
    }

    /// <summary>
    /// Builds the index mapping for each class.
    /// </summary>
    private void BuildClassIndices()
    {
        _classIndices = new Dictionary<int, List<int>>();
        for (int c = 0; c < _numClasses; c++)
        {
            _classIndices[c] = new List<int>();
        }

        for (int i = 0; i < _labels.Length; i++)
        {
            int label = _labels[i];
            if (label >= 0 && label < _numClasses)
            {
                _classIndices[label].Add(i);
            }
        }
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        if (_classIndices == null)
        {
            BuildClassIndices();
        }

        // Shuffle indices within each class
        var shuffledClassIndices = new Dictionary<int, int[]>();
        foreach (var kvp in _classIndices!)
        {
            int classLabel = kvp.Key;
            List<int> indices = kvp.Value;

            int[] shuffled = indices.ToArray();
            ShuffleIndices(shuffled);
            shuffledClassIndices[classLabel] = shuffled;
        }

        // Interleave classes proportionally
        var classPositions = new Dictionary<int, int>();
        foreach (int c in shuffledClassIndices.Keys)
        {
            classPositions[c] = 0;
        }

        // Collect all indices in stratified order
        var allIndices = new List<int>();
        int totalSamples = _labels.Length;
        int samplesCollected = 0;

        while (samplesCollected < totalSamples)
        {
            foreach (int c in shuffledClassIndices.Keys)
            {
                var classArray = shuffledClassIndices[c];
                int pos = classPositions[c];

                if (pos < classArray.Length)
                {
                    allIndices.Add(classArray[pos]);
                    classPositions[c] = pos + 1;
                    samplesCollected++;

                    if (samplesCollected >= totalSamples)
                    {
                        break;
                    }
                }
            }
        }

        // Shuffle the final order to avoid class-based patterns
        var finalArray = allIndices.ToArray();
        ShuffleIndices(finalArray);

        foreach (int index in finalArray)
        {
            yield return index;
        }
    }
}

/// <summary>
/// A batch sampler that ensures each batch contains samples from all classes.
/// </summary>
/// <remarks>
/// <para>
/// StratifiedBatchSampler creates batches where each batch has approximately
/// the same class distribution. This is useful for batch normalization layers
/// or when batch-level statistics are important.
/// </para>
/// <para><b>For Beginners:</b> While StratifiedSampler ensures the overall epoch
/// has the right class balance, StratifiedBatchSampler ensures EACH BATCH
/// has balanced classes. This is helpful when:
/// - Using batch normalization (needs balanced statistics per batch)
/// - Doing contrastive learning (needs diverse samples in each batch)
/// </para>
/// </remarks>
public class StratifiedBatchSampler : DataSamplerBase, IBatchSampler, IStratifiedSampler
{
    private int[] _labels;
    private readonly int _numClasses;
    private int _batchSize;
    private bool _dropLast;
    private Dictionary<int, List<int>>? _classIndices;

    /// <summary>
    /// Initializes a new instance of the StratifiedBatchSampler class.
    /// </summary>
    /// <param name="labels">The class label for each sample (0-indexed).</param>
    /// <param name="numClasses">The total number of classes.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="dropLast">Whether to drop the last incomplete batch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public StratifiedBatchSampler(
        IEnumerable<int> labels,
        int numClasses,
        int batchSize,
        bool dropLast = false,
        int? seed = null)
        : base(seed)
    {
        if (labels == null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (numClasses < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(numClasses), "Number of classes must be at least 2.");
        }

        if (batchSize < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Batch size must be at least 1.");
        }

        _labels = labels.ToArray();
        _numClasses = numClasses;
        _batchSize = batchSize;
        _dropLast = dropLast;
        BuildClassIndices();
    }

    /// <inheritdoc/>
    public override int Length => _labels.Length;

    /// <inheritdoc/>
    public int NumClasses => _numClasses;

    /// <inheritdoc/>
    public int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    /// <inheritdoc/>
    public bool DropLast
    {
        get => _dropLast;
        set => _dropLast = value;
    }

    /// <inheritdoc/>
    public IReadOnlyList<int> Labels
    {
        get => _labels;
        set
        {
            Guard.NotNull(value);
            _labels = value.ToArray();
            BuildClassIndices();
        }
    }

    private void BuildClassIndices()
    {
        _classIndices = new Dictionary<int, List<int>>();
        for (int c = 0; c < _numClasses; c++)
        {
            _classIndices[c] = new List<int>();
        }

        for (int i = 0; i < _labels.Length; i++)
        {
            int label = _labels[i];
            if (label >= 0 && label < _numClasses)
            {
                _classIndices[label].Add(i);
            }
        }
    }

    /// <inheritdoc/>
    protected override IEnumerable<int> GetIndicesCore()
    {
        foreach (var batch in GetBatchIndices())
        {
            foreach (int index in batch)
            {
                yield return index;
            }
        }
    }

    /// <inheritdoc/>
    public IEnumerable<int[]> GetBatchIndices()
    {
        if (_classIndices == null)
        {
            BuildClassIndices();
        }

        // Shuffle indices within each class
        var shuffledQueues = new Dictionary<int, Queue<int>>();
        foreach (var kvp in _classIndices!)
        {
            int classLabel = kvp.Key;
            List<int> indices = kvp.Value;

            int[] shuffled = indices.ToArray();
            ShuffleIndices(shuffled);
            shuffledQueues[classLabel] = new Queue<int>(shuffled);
        }

        // Calculate samples per class per batch
        int samplesPerClassPerBatch = Math.Max(1, _batchSize / _numClasses);
        int totalSamples = _labels.Length;
        int numFullBatches = totalSamples / _batchSize;

        // Generate batches
        for (int b = 0; b < numFullBatches; b++)
        {
            var batch = new List<int>();

            // Take samples from each class
            foreach (var queue in shuffledQueues.Values)
            {
                for (int i = 0; i < samplesPerClassPerBatch && queue.Count > 0 && batch.Count < _batchSize; i++)
                {
                    batch.Add(queue.Dequeue());
                }
            }

            // Fill remaining slots from any available class
            while (batch.Count < _batchSize)
            {
                var availableQueue = shuffledQueues.Values.FirstOrDefault(q => q.Count > 0);
                if (availableQueue is null)
                {
                    break;
                }
                batch.Add(availableQueue.Dequeue());
            }

            if (batch.Count == _batchSize)
            {
                // Shuffle within batch
                var batchArray = batch.ToArray();
                ShuffleIndices(batchArray);
                yield return batchArray;
            }
        }

        // Handle remaining samples
        if (!_dropLast)
        {
            var remaining = new List<int>();
            foreach (int c in shuffledQueues.Keys)
            {
                while (shuffledQueues[c].Count > 0)
                {
                    remaining.Add(shuffledQueues[c].Dequeue());
                }
            }

            if (remaining.Count > 0)
            {
                var batchArray = remaining.ToArray();
                ShuffleIndices(batchArray);
                yield return batchArray;
            }
        }
    }
}
