using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific;

/// <summary>
/// Sequence splitter for sequential data like text, DNA, or user sessions.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sequential data has an inherent order that matters.
/// Examples include:
/// - Text documents (sequence of words)
/// - DNA sequences (sequence of nucleotides)
/// - User clickstreams (sequence of page visits)
/// </para>
/// <para>
/// <b>Split Strategies:</b>
/// - By sequence: Keep entire sequences together
/// - By position: Split within sequences at a certain position
/// - By time: For time-stamped sequences
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SequenceSplitter<T> : DataSplitterBase<T>
{
    private readonly double _testSize;
    private readonly int _sequenceColumn;
    private int[]? _sequenceIds;

    /// <summary>
    /// Creates a new sequence splitter.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="sequenceColumn">Column index containing sequence IDs. Default is 0.</param>
    /// <param name="shuffle">Whether to shuffle sequences before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public SequenceSplitter(
        double testSize = 0.2,
        int sequenceColumn = 0,
        bool shuffle = true,
        int randomSeed = 42)
        : base(shuffle, randomSeed)
    {
        if (testSize <= 0 || testSize >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(testSize), "Test size must be between 0 and 1.");
        }

        _testSize = testSize;
        _sequenceColumn = sequenceColumn;
    }

    /// <summary>
    /// Sets explicit sequence IDs for each sample.
    /// </summary>
    /// <param name="sequenceIds">Array where sequenceIds[i] gives the sequence ID for sample i.</param>
    public SequenceSplitter<T> WithSequenceIds(int[] sequenceIds)
    {
        _sequenceIds = sequenceIds;
        return this;
    }

    /// <inheritdoc/>
    public override string Description => $"Sequence split ({_testSize * 100:F0}% test)";

    /// <inheritdoc/>
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        int nSamples = X.Rows;
        int nFeatures = X.Columns;

        // Get sequence IDs
        int[] seqIds;
        if (_sequenceIds != null)
        {
            if (_sequenceIds.Length != nSamples)
            {
                throw new ArgumentException(
                    $"Sequence IDs length ({_sequenceIds.Length}) must match samples ({nSamples}).");
            }
            seqIds = _sequenceIds;
        }
        else
        {
            if (_sequenceColumn >= nFeatures)
            {
                throw new ArgumentException(
                    $"Sequence column ({_sequenceColumn}) exceeds feature count ({nFeatures}).");
            }

            seqIds = new int[nSamples];
            for (int i = 0; i < nSamples; i++)
            {
                seqIds[i] = (int)Convert.ToDouble(X[i, _sequenceColumn]);
            }
        }

        // Group samples by sequence
        var sequenceIndices = new Dictionary<int, List<int>>();
        for (int i = 0; i < nSamples; i++)
        {
            int seqId = seqIds[i];
            if (!sequenceIndices.TryGetValue(seqId, out var list))
            {
                list = new List<int>();
                sequenceIndices[seqId] = list;
            }
            list.Add(i);
        }

        var uniqueSequences = sequenceIndices.Keys.ToArray();
        int nSequences = uniqueSequences.Length;

        if (nSequences < 2)
        {
            throw new ArgumentException("Need at least 2 sequences for splitting.");
        }

        if (_shuffle)
        {
            ShuffleIndices(uniqueSequences);
        }

        int targetTestSequences = Math.Max(1, (int)(nSequences * _testSize));

        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        for (int i = 0; i < nSequences; i++)
        {
            int seqId = uniqueSequences[i];
            if (i < targetTestSequences)
            {
                testIndices.AddRange(sequenceIndices[seqId]);
            }
            else
            {
                trainIndices.AddRange(sequenceIndices[seqId]);
            }
        }

        return BuildResult(X, y, trainIndices.ToArray(), testIndices.ToArray());
    }
}
