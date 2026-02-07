using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text;

/// <summary>
/// In-memory dataset of pre-tokenized text sequences for language model training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Stores token ID sequences where each sample is a fixed-length array of integer token IDs.
/// This is the standard input format for transformer-based language models (GPT, BERT, etc.).
/// </para>
/// <para><b>For Beginners:</b> After tokenizing your text (converting words to numbers),
/// use this dataset to hold the token sequences for training:
/// <code>
/// int[][] tokenIds = { new[] {1, 5, 3, 2}, new[] {1, 7, 8, 2} };
/// int[] labels = { 0, 1 };
/// var dataset = new TokenizedTextDataset&lt;float&gt;(tokenIds, labels, sequenceLength: 4);
/// </code>
/// </para>
/// </remarks>
public class TokenizedTextDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly int[][] _tokenIds;
    private readonly int[] _labels;
    private readonly int _sequenceLength;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => "TokenizedText";

    /// <inheritdoc/>
    public override string Description => "Pre-tokenized text dataset for LLM training";

    /// <inheritdoc/>
    public override int TotalCount => _tokenIds.Length;

    /// <inheritdoc/>
    public override int FeatureCount => _sequenceLength;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Creates a tokenized text dataset from token ID sequences and labels.
    /// </summary>
    /// <param name="tokenIds">Array of token ID sequences. Each is padded/truncated to sequenceLength.</param>
    /// <param name="labels">Class labels for each sequence.</param>
    /// <param name="sequenceLength">Fixed sequence length. Sequences will be padded or truncated.</param>
    /// <param name="paddingTokenId">Token ID used for padding shorter sequences. Default is 0.</param>
    public TokenizedTextDataset(int[][] tokenIds, int[] labels, int sequenceLength, int paddingTokenId = 0)
    {
        if (tokenIds is null)
        {
            throw new ArgumentNullException(nameof(tokenIds));
        }

        if (labels is null)
        {
            throw new ArgumentNullException(nameof(labels));
        }

        if (tokenIds.Length != labels.Length)
        {
            throw new ArgumentException(
                $"Token sequences count ({tokenIds.Length}) must match labels count ({labels.Length}).",
                nameof(labels));
        }

        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be positive.");
        }

        _sequenceLength = sequenceLength;
        _numClasses = labels.Length > 0 ? labels.Max() + 1 : 0;

        // Pad/truncate to fixed length
        _tokenIds = new int[tokenIds.Length][];
        for (int i = 0; i < tokenIds.Length; i++)
        {
            _tokenIds[i] = new int[sequenceLength];
            if (tokenIds[i] is not null)
            {
                int copyLen = Math.Min(tokenIds[i].Length, sequenceLength);
                Array.Copy(tokenIds[i], _tokenIds[i], copyLen);
                for (int j = copyLen; j < sequenceLength; j++)
                {
                    _tokenIds[i][j] = paddingTokenId;
                }
            }
        }

        _labels = (int[])labels.Clone();
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        int n = _tokenIds.Length;
        var featuresData = new T[n * _sequenceLength];
        var labelsData = new T[n * _numClasses];

        for (int i = 0; i < n; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int featureOffset = i * _sequenceLength;
            for (int j = 0; j < _sequenceLength; j++)
            {
                featuresData[featureOffset + j] = NumOps.FromDouble(_tokenIds[i][j]);
            }

            int labelOffset = i * _numClasses;
            if (_labels[i] >= 0 && _labels[i] < _numClasses)
            {
                labelsData[labelOffset + _labels[i]] = NumOps.One;
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { n, _sequenceLength });
        LoadedLabels = new Tensor<T>(labelsData, new[] { n, _numClasses });
        InitializeIndices(n);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");

        return (ExtractTensorBatch(features, indices), ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        int n = TotalCount;
        var (trainSize, valSize, _) = ComputeSplitSizes(n, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();

        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures!, shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels!, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
