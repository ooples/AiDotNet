using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the MBPP (Mostly Basic Python Problems) benchmark — 1,000 entry-level
/// Python coding problems with natural-language descriptions and unit tests
/// (Austin et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects <c>{DataPath}/mbpp.jsonl</c>. Auto-download fetches the canonical
/// Google research GitHub release. Each record has <c>text</c> (problem),
/// <c>code</c> (canonical solution), <c>test_list</c>, and <c>task_id</c>.
/// Standard splits use task_id ranges: 1–10 prompts (skipped here), 11–510
/// test, 511–600 val, 601–974 train.
/// </para>
/// </remarks>
public class MbppDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly MbppDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "MBPP";
    /// <inheritdoc/>
    public override string Description => "Mostly Basic Python Problems benchmark";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxPromptLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.MaxSolutionLength;

    /// <summary>Creates a new MBPP data loader.</summary>
    public MbppDataLoader(MbppDataLoaderOptions? options = null)
    {
        _options = options ?? new MbppDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("mbpp");
    }

    private static readonly string DownloadUrl =
        "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string filePath = Path.Combine(_dataPath, "mbpp.jsonl");
        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                Directory.CreateDirectory(_dataPath);
                await DatasetDownloader.DownloadFileAsync(DownloadUrl, filePath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException($"Failed to download MBPP from {DownloadUrl}.", ex);
            }
        }
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"MBPP not found at {filePath}.");

        var prompts = new List<string>();
        var solutions = new List<string>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(line)) continue;
            JObject obj;
            try { obj = JObject.Parse(line); } catch { continue; }
            int taskId = obj["task_id"]?.Value<int>() ?? 0;
            // Apply canonical splits per Austin et al. 2021: 11..510 test, 511..600 val, 601..974 train.
            bool include = _options.Split switch
            {
                Geometry.DatasetSplit.Test => taskId >= 11 && taskId <= 510,
                Geometry.DatasetSplit.Validation => taskId >= 511 && taskId <= 600,
                _ => taskId >= 601 && taskId <= 974
            };
            if (!include) continue;
            string? text = obj["text"]?.ToString();
            string? code = obj["code"]?.ToString();
            if (string.IsNullOrEmpty(text) || string.IsNullOrEmpty(code)) continue;
            prompts.Add(text!);
            solutions.Add(code!);
        }

        int totalSamples = prompts.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++) { allTexts.Add(prompts[i]); allTexts.Add(solutions[i]); }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int pLen = _options.MaxPromptLength, sLen = _options.MaxSolutionLength;
        var featuresData = new T[totalSamples * pLen];
        var labelsData = new T[totalSamples * sLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] pTok = TextLoaderHelper.TokenizeAndEncode(prompts[i], vocabulary, pLen);
            int[] sTok = TextLoaderHelper.TokenizeAndEncode(solutions[i], vocabulary, sLen);
            int pOff = i * pLen, sOff = i * sLen;
            for (int j = 0; j < pLen; j++) featuresData[pOff + j] = NumOps.FromDouble(pTok[j]);
            for (int j = 0; j < sLen; j++) labelsData[sOff + j] = NumOps.FromDouble(sTok[j]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, pLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, sLen });
        InitializeIndices(totalSamples);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore() { LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0; }

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
