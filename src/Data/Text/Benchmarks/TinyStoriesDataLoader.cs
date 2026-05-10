using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the TinyStories synthetic LM corpus (Eldan &amp; Li, 2023):
/// ≈ 2.1M GPT-generated short stories with deliberately small vocabulary
/// for small-scale language-model research.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   TinyStories-train.txt
///   TinyStories-valid.txt
/// </code>
/// Auto-download fetches both files from the canonical Hugging Face
/// repository <c>roneneldan/TinyStories</c>. Stories are concatenated and
/// chunked into fixed-length input/target sequences for LM training.
/// </para>
/// </remarks>
public class TinyStoriesDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TinyStoriesDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "TinyStories";
    /// <inheritdoc/>
    public override string Description => "TinyStories synthetic narrative LM corpus";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.SequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.SequenceLength;

    /// <summary>Creates a new TinyStories data loader.</summary>
    public TinyStoriesDataLoader(TinyStoriesDataLoaderOptions? options = null)
    {
        _options = options ?? new TinyStoriesDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("tinystories");
    }

    private static readonly string TrainUrl =
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt";
    private static readonly string ValidUrl =
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // TinyStories has no public test split; reuse the validation file when Test
        // is requested so consumers always get a stable held-out partition.
        bool useValid = _options.Split == Geometry.DatasetSplit.Validation
                     || _options.Split == Geometry.DatasetSplit.Test;
        string fileName = useValid ? "TinyStories-valid.txt" : "TinyStories-train.txt";
        string url = useValid ? ValidUrl : TrainUrl;
        string filePath = Path.Combine(_dataPath, fileName);

        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                Directory.CreateDirectory(_dataPath);
                await DatasetDownloader.DownloadFileAsync(url, filePath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to download TinyStories from {url}.", ex);
            }
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException(
                $"TinyStories not found at {filePath}. Enable AutoDownload or place the file manually.");
        }

        string text = await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);
        var tokens = TextLoaderHelper.Tokenize(text);
        if (tokens.Count < 2)
            throw new InvalidDataException(
                $"TinyStories split file at {filePath} is empty or truncated " +
                $"(got {tokens.Count} tokens, need ≥ 2 for next-token prediction).");

        var vocabulary = TextLoaderHelper.BuildVocabulary(tokens, tokens.Count, _options.VocabularySize);
        int seqLen = _options.SequenceLength;
        int numSequences = (tokens.Count - 1) / seqLen;
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
