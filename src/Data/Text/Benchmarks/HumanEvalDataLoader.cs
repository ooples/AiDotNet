using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the HumanEval Python code-generation benchmark (Chen et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects <c>{DataPath}/HumanEval.jsonl</c>. Auto-download fetches the
/// canonical OpenAI release. Each record has <c>prompt</c>,
/// <c>canonical_solution</c>, <c>test</c>, <c>entry_point</c>; this loader
/// uses prompt as features and canonical_solution as the target. Scoring
/// against the unit tests requires running the model output through a
/// Python sandbox and is out of scope for the loader.
/// </para>
/// </remarks>
public class HumanEvalDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly HumanEvalDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "HumanEval";
    /// <inheritdoc/>
    public override string Description => "HumanEval Python code-generation benchmark (164 problems)";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxPromptLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.MaxSolutionLength;

    /// <summary>Creates a new HumanEval data loader.</summary>
    public HumanEvalDataLoader(HumanEvalDataLoaderOptions? options = null)
    {
        _options = options ?? new HumanEvalDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("humaneval");
    }

    private static readonly string DownloadUrl =
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string filePath = Path.Combine(_dataPath, "HumanEval.jsonl");
        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                Directory.CreateDirectory(_dataPath);
                await DatasetDownloader.DownloadAndDecompressGzipAsync(DownloadUrl, filePath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException($"Failed to download HumanEval from {DownloadUrl}.", ex);
            }
        }

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"HumanEval not found at {filePath}.");

        var prompts = new List<string>();
        var solutions = new List<string>();
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(line)) continue;
            JObject obj;
            try { obj = JObject.Parse(line); }
            catch { continue; }
            string? p = obj["prompt"]?.ToString();
            string? s = obj["canonical_solution"]?.ToString();
            if (string.IsNullOrEmpty(p) || s is null) continue;
            prompts.Add(p!);
            solutions.Add(s);
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
