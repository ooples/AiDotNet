using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the WikiText-103 language modeling dataset (100M+ tokens from Wikipedia articles).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// WikiText-103 expects:
/// <code>
/// {DataPath}/wikitext-103/
///   wiki.train.tokens
///   wiki.valid.tokens
///   wiki.test.tokens
/// </code>
/// Features are input token sequences Tensor[N, SequenceLength].
/// Labels are next-token target sequences Tensor[N, SequenceLength] (shifted by 1).
/// </para>
/// </remarks>
public class WikiText103DataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly WikiText103DataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "WikiText-103";
    /// <inheritdoc/>
    public override string Description => "WikiText-103 language modeling dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.SequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.SequenceLength;

    /// <summary>Creates a new WikiText-103 data loader.</summary>
    public WikiText103DataLoader(WikiText103DataLoaderOptions? options = null)
    {
        _options = options ?? new WikiText103DataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("wikitext-103");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = Path.Combine(_dataPath, "wikitext-103");
        if (!Directory.Exists(dataDir))
            dataDir = _dataPath;

        string splitFile = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "wiki.test.tokens",
            Geometry.DatasetSplit.Validation => "wiki.valid.tokens",
            _ => "wiki.train.tokens"
        };
        string filePath = Path.Combine(dataDir, splitFile);
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException(
                $"WikiText-103 data not found at {filePath}. " +
                "Download from https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/.");
        }

        string text = await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);

        // Tokenize entire text
        var tokens = TextLoaderHelper.Tokenize(text);
        if (tokens.Count < 2) return;

        // Build vocabulary
        var vocabulary = TextLoaderHelper.BuildVocabulary(tokens, tokens.Count, _options.VocabularySize);

        // Encode all tokens
        int seqLen = _options.SequenceLength;
        int numSequences = (tokens.Count - 1) / seqLen; // -1 for the target offset

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < numSequences)
            numSequences = _options.MaxSamples.Value;

        _sampleCount = numSequences;
        var featuresData = new T[numSequences * seqLen];
        var labelsData = new T[numSequences * seqLen];

        for (int i = 0; i < numSequences; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int startIdx = i * seqLen;
            int featureOffset = i * seqLen;

            for (int j = 0; j < seqLen; j++)
            {
                int tokenIdx = startIdx + j;
                string inputToken = tokenIdx < tokens.Count ? tokens[tokenIdx] : string.Empty;
                string targetToken = (tokenIdx + 1) < tokens.Count ? tokens[tokenIdx + 1] : string.Empty;

                int inputId = vocabulary.TryGetValue(inputToken, out int iid) ? iid : 1;
                int targetId = vocabulary.TryGetValue(targetToken, out int tid) ? tid : 1;

                featuresData[featureOffset + j] = NumOps.FromDouble(inputId);
                labelsData[featureOffset + j] = NumOps.FromDouble(targetId);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { numSequences, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { numSequences, seqLen });
        InitializeIndices(numSequences);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (TextLoaderHelper.ExtractTensorBatch(features, indices), TextLoaderHelper.ExtractTensorBatch(labels, indices));
    }

    /// <inheritdoc/>
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
        var features = LoadedFeatures ?? throw new InvalidOperationException("Not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Not loaded.");
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }
}
