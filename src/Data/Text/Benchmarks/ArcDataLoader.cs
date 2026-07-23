using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the AI2 Reasoning Challenge (ARC) multiple-choice science QA
/// benchmark (Clark et al. 2018) — Easy or Challenge variant.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/ARC-V1-Feb2018-2/ARC-{Easy,Challenge}/
///   ARC-{Easy,Challenge}-Train.jsonl
///   ARC-{Easy,Challenge}-Dev.jsonl
///   ARC-{Easy,Challenge}-Test.jsonl
/// </code>
/// Auto-download fetches the canonical AllenAI release. Handles both
/// 4-way and 5-way questions (ARC has a small fraction of 5-choice items);
/// the 5th choice is treated as an out-of-distribution distractor when
/// truncated. Features Tensor[N, 4, MaxSequenceLength]; one-hot labels
/// Tensor[N, 4].
/// </para>
/// </remarks>
public class ArcDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly ArcDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private const int Choices = 4;

    /// <inheritdoc/>
    public override string Name => $"ARC-{_options.Variant}";
    /// <inheritdoc/>
    public override string Description => $"AI2 Reasoning Challenge ({_options.Variant}) multiple-choice science QA";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => Choices * _options.MaxSequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => Choices;

    /// <summary>Creates a new ARC data loader.</summary>
    public ArcDataLoader(ArcDataLoaderOptions? options = null)
    {
        _options = options ?? new ArcDataLoaderOptions();
        _options.Validate();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("arc");
    }

    private static readonly string DownloadUrl =
        "https://ai2-public-datasets.s3.amazonaws.com/arc/ARC-V1-Feb2018.zip";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string variantDir = _options.Variant == ArcVariant.Challenge ? "ARC-Challenge" : "ARC-Easy";
        string variantPrefix = variantDir;
        string splitSuffix = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "Test",
            Geometry.DatasetSplit.Validation => "Dev",
            _ => "Train"
        };
        string fileName = $"{variantPrefix}-{splitSuffix}.jsonl";
        string filePath = Path.Combine(ResolveDataDir(variantDir), fileName);

        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                await DatasetDownloader.DownloadAndExtractZipAsync(DownloadUrl, _dataPath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException($"Failed to download ARC from {DownloadUrl}.", ex);
            }
            filePath = Path.Combine(ResolveDataDir(variantDir), fileName);
        }

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"ARC not found at {filePath}.");

        var questions = new List<string>();
        var choices = new List<string[]>();
        var labels = new List<int>();

        int lineNum = 0;
        foreach (string line in await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();
            lineNum++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            JObject obj;
            try { obj = JObject.Parse(line); }
            catch (Newtonsoft.Json.JsonException ex)
            {
                string preview = line.Length > 100 ? line.Substring(0, 100) + "..." : line;
                throw new InvalidDataException(
                    $"Malformed JSON in ARC file '{filePath}' at line {lineNum}: {preview}", ex);
            }
            string? stem = obj["question"]?["stem"]?.ToString();
            var choicesArr = obj["question"]?["choices"] as JArray;
            string? answerKey = obj["answerKey"]?.ToString();
            if (string.IsNullOrEmpty(stem) || choicesArr is null || choicesArr.Count < 2 || string.IsNullOrEmpty(answerKey))
                continue;

            var choiceStrs = new string[Choices];
            int correctIdx = -1;
            int n = Math.Min(choicesArr.Count, Choices);
            for (int i = 0; i < n; i++)
            {
                choiceStrs[i] = choicesArr[i]?["text"]?.ToString() ?? string.Empty;
                string? lbl = choicesArr[i]?["label"]?.ToString();
                if (lbl == answerKey) correctIdx = i;
            }
            for (int i = n; i < Choices; i++) choiceStrs[i] = string.Empty;
            if (correctIdx < 0 || correctIdx >= Choices) continue;
            questions.Add(stem!);
            choices.Add(choiceStrs);
            labels.Add(correctIdx);
        }

        int totalSamples = questions.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * (Choices + 1));
        for (int i = 0; i < totalSamples; i++)
        {
            allTexts.Add(questions[i]);
            for (int c = 0; c < Choices; c++) allTexts.Add(choices[i][c]);
        }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int seqLen = _options.MaxSequenceLength;
        var featuresData = new T[totalSamples * Choices * seqLen];
        var labelsData = new T[totalSamples * Choices];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int c = 0; c < Choices; c++)
            {
                int[] tok = TextLoaderHelper.TokenizeAndEncode(questions[i] + " " + choices[i][c], vocabulary, seqLen);
                int offset = (i * Choices + c) * seqLen;
                for (int j = 0; j < seqLen; j++) featuresData[offset + j] = NumOps.FromDouble(tok[j]);
            }
            labelsData[i * Choices + labels[i]] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, Choices, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, Choices });
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

    private string ResolveDataDir(string variantDir)
    {
        // Zip extracts to ARC-V1-Feb2018-2/{variantDir}/
        string root = Path.Combine(_dataPath, "ARC-V1-Feb2018-2", variantDir);
        if (Directory.Exists(root)) return root;
        // Fallback: directly under DataPath
        string flat = Path.Combine(_dataPath, variantDir);
        if (Directory.Exists(flat)) return flat;
        return _dataPath;
    }
}
