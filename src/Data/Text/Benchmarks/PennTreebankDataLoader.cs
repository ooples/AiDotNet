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

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitFile = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "ptb.test.txt",
            Geometry.DatasetSplit.Validation => "ptb.valid.txt",
            _ => "ptb.train.txt"
        };
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

        string text = await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);
        var tokens = TextLoaderHelper.Tokenize(text);
        if (tokens.Count < 2) return;

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
