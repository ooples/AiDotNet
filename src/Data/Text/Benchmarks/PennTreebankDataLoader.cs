using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the Penn Treebank language modeling dataset (Mikolov-preprocessed split).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/simple-examples/data/
///   ptb.train.txt
///   ptb.valid.txt
///   ptb.test.txt
/// </code>
/// Auto-download fetches the canonical Mikolov tarball.
/// Features are input token sequences Tensor[N, SequenceLength].
/// Labels are next-token targets Tensor[N, SequenceLength] (shifted by 1).
/// </para>
/// </remarks>
public class PennTreebankDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly PennTreebankDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "Penn Treebank";
    /// <inheritdoc/>
    public override string Description => "Penn Treebank (PTB) language modeling dataset (Mikolov split)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.SequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.SequenceLength;

    /// <summary>Creates a new PTB data loader.</summary>
    public PennTreebankDataLoader(PennTreebankDataLoaderOptions? options = null)
    {
        _options = options ?? new PennTreebankDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("ptb");
    }

    private static readonly string DownloadUrl =
        "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz";

    private static string SplitFileName(Geometry.DatasetSplit split) => split switch
    {
        Geometry.DatasetSplit.Train => "ptb.train.txt",
        Geometry.DatasetSplit.Test => "ptb.test.txt",
        Geometry.DatasetSplit.Validation => "ptb.valid.txt",
        // Reject unknown enum values rather than silently coercing to the
        // train split — same defensive pattern WikiText-2 / WikiText-103 use.
        _ => throw new ArgumentOutOfRangeException(
            nameof(split),
            split,
            $"Unsupported {nameof(Geometry.DatasetSplit)} for Penn Treebank (only Train / Validation / Test).")
    };

    /// <summary>
    /// Loads the raw, unprocessed text content for the requested PTB split,
    /// auto-downloading via <see cref="PennTreebankDataLoaderOptions.AutoDownload"/>
    /// if the file is not already cached. Lets consumers run their own
    /// tokenizer (BPE, SentencePiece, etc.) instead of the built-in
    /// whitespace tokenization that <see cref="LoadAsync(CancellationToken)"/>
    /// applies internally.
    /// </summary>
    /// <param name="split">Which PTB split to read.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The raw text contents of the requested split file.</returns>
    /// <remarks>
    /// Honors the same <see cref="PennTreebankDataLoaderOptions.DataPath"/> and
    /// <see cref="PennTreebankDataLoaderOptions.AutoDownload"/> options as
    /// <see cref="LoadAsync(CancellationToken)"/>. Independent of the loader's
    /// load-state, so it is safe to call without first calling LoadAsync.
    /// PTB's vocab-from-train convention is irrelevant here — this method
    /// returns the requested split's raw text regardless of which split's
    /// vocab the caller would build elsewhere.
    /// </remarks>
    public async Task<string> LoadRawTextAsync(
        Geometry.DatasetSplit split,
        CancellationToken cancellationToken = default)
    {
        string filePath = await EnsureSplitFileAsync(SplitFileName(split), cancellationToken);
        return await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);
    }

    private async Task<string> EnsureSplitFileAsync(string splitFile, CancellationToken cancellationToken)
    {
        string filePath = Path.Combine(ResolveDataDir(), splitFile);

        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                await DatasetDownloader.DownloadAndExtractTarGzAsync(
                    DownloadUrl, _dataPath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to download PTB from {DownloadUrl}. " +
                    $"Check your network connection or download manually to {_dataPath}.",
                    ex);
            }
            filePath = Path.Combine(ResolveDataDir(), splitFile);
        }

        if (!File.Exists(filePath))
        {
            string hint = _options.AutoDownload
                ? $"Auto-download completed but the expected file was not found in {_dataPath}."
                : $"Enable AutoDownload or extract {DownloadUrl} to {_dataPath}.";
            throw new FileNotFoundException(
                $"PTB data not found at {filePath}. {hint}");
        }

        return filePath;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string filePath = await EnsureSplitFileAsync(SplitFileName(_options.Split), cancellationToken);

        // Build the vocabulary from the train split regardless of which split was requested.
        // PTB convention: vocab is fit on ptb.train.txt and reused for valid/test.
        string trainPath = await EnsureSplitFileAsync("ptb.train.txt", cancellationToken);
        string trainText = await FilePolyfill.ReadAllTextAsync(trainPath, cancellationToken);
        var trainTokens = TextLoaderHelper.Tokenize(trainText);
        if (trainTokens.Count < 2)
            throw new InvalidDataException(
                $"PTB train split at {trainPath} is empty or truncated (got {trainTokens.Count} tokens, need ≥ 2).");
        var vocabulary = TextLoaderHelper.BuildVocabulary(trainTokens, trainTokens.Count, _options.VocabularySize);

        // Tokenize the requested split for actual sample data.
        string text = filePath == trainPath
            ? trainText
            : await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);
        var tokens = filePath == trainPath ? trainTokens : TextLoaderHelper.Tokenize(text);
        if (tokens.Count < 2)
            throw new InvalidDataException(
                $"PTB split file at {filePath} is empty or truncated (got {tokens.Count} tokens, need ≥ 2).");

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
        var trainIndices = shuffled.Take(trainSize).ToArray();
        var valIndices = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffled.Skip(trainSize + valSize).ToArray();
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, trainIndices), TextLoaderHelper.ExtractTensorBatch(labels, trainIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, valIndices), TextLoaderHelper.ExtractTensorBatch(labels, valIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, testIndices), TextLoaderHelper.ExtractTensorBatch(labels, testIndices))
        );
    }

    private string ResolveDataDir()
    {
        // Mikolov's tarball extracts to simple-examples/data/
        string subDir = Path.Combine(_dataPath, "simple-examples", "data");
        if (Directory.Exists(subDir) &&
            Directory.EnumerateFiles(subDir, "ptb.*.txt").Any())
            return subDir;
        return _dataPath;
    }
}
