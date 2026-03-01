using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text;

/// <summary>
/// A streaming text dataset that lazily reads and tokenizes text files for language model training.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Reads text files sequentially, splits into fixed-length token sequences using a simple
/// word-level tokenizer. Input is tokens[0..n-1], target is tokens[1..n] (next-token prediction).
/// Suitable for pre-training language models on large text corpora.
/// </para>
/// </remarks>
public class StreamingTextDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly StreamingTextDatasetOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "StreamingText";
    public override string Description => "Streaming text dataset for LLM training";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.SequenceLength;
    public override int OutputDimension => _options.SequenceLength;

    public StreamingTextDataset(StreamingTextDatasetOptions? options = null)
    {
        _options = options ?? new StreamingTextDatasetOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("streaming_text");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        if (!Directory.Exists(_dataPath))
            throw new DirectoryNotFoundException($"Text data not found at {_dataPath}.");

        var files = Directory.GetFiles(_dataPath, _options.FilePattern, SearchOption.AllDirectories);
        if (files.Length == 0)
            throw new FileNotFoundException($"No files matching '{_options.FilePattern}' found at {_dataPath}.");

        if (_options.ShuffleFiles)
        {
            var rng = _options.Seed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
                : RandomHelper.CreateSecureRandom();
            files = files.OrderBy(_ => rng.Next()).ToArray();
        }
        else
        {
            Array.Sort(files, StringComparer.OrdinalIgnoreCase);
        }

        // Build vocabulary from all files (simple word-level)
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal) { ["<pad>"] = 0, ["<unk>"] = 1 };
        var allTokenIds = new List<int>();

        foreach (var file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string text = await FilePolyfill.ReadAllTextAsync(file, cancellationToken);
            var words = System.Text.RegularExpressions.Regex.Split(text.ToLowerInvariant(), @"\s+");

            foreach (string word in words)
            {
                if (string.IsNullOrEmpty(word)) continue;
                if (!vocab.TryGetValue(word, out int id))
                {
                    if (vocab.Count < _options.VocabularySize)
                    {
                        id = vocab.Count;
                        vocab[word] = id;
                    }
                    else
                    {
                        id = 1; // <unk>
                    }
                }
                allTokenIds.Add(id);
            }
        }

        // Split into fixed-length sequences (input = tokens[0..n-1], target = tokens[1..n])
        int seqLen = _options.SequenceLength;
        int totalSeqs = allTokenIds.Count / (seqLen + 1); // +1 for the shifted target
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSeqs)
            totalSeqs = _options.MaxSamples.Value;

        if (totalSeqs == 0)
            throw new InvalidOperationException(
                $"Text corpus is too small to produce any sequences of length {seqLen + 1}. " +
                $"Total tokens: {allTokenIds.Count}, required: {seqLen + 1}.");

        _sampleCount = totalSeqs;
        var featuresData = new T[totalSeqs * seqLen];
        var labelsData = new T[totalSeqs * seqLen];

        for (int i = 0; i < totalSeqs; i++)
        {
            int offset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                int tokenIdx = offset + j;
                featuresData[i * seqLen + j] = tokenIdx < allTokenIds.Count
                    ? NumOps.FromDouble(allTokenIds[tokenIdx])
                    : NumOps.Zero;

                int targetIdx = offset + j + 1;
                labelsData[i * seqLen + j] = targetIdx < allTokenIds.Count
                    ? NumOps.FromDouble(allTokenIds[targetIdx])
                    : NumOps.Zero;
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSeqs, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSeqs, seqLen });
        InitializeIndices(totalSeqs);
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape.Clone(); nls[0] = indices.Length;
        var bf = new Tensor<T>(nfs);
        var bl = new Tensor<T>(nls);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, bf, indices[i], i);
            TensorCopyHelper.CopySample(labels, bl, indices[i], i);
        }
        return (bf, bl);
    }

    public override (IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Train,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Validation,
        IInputOutputDataLoader<T, Tensor<T>, Tensor<T>> Test) Split(
        double trainRatio = 0.7, double validationRatio = 0.15, int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);
        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        return (
            CreateSplit(shuffled.Take(trainSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSplit(int[] indices)
    {
        var (bf, bl) = ExtractBatch(indices);
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(bf, bl);
    }
}
