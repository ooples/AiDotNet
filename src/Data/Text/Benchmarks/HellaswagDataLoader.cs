using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the HellaSwag 4-way commonsense NLI benchmark (Zellers et al. 2019).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   hellaswag_train.jsonl
///   hellaswag_val.jsonl
///   hellaswag_test.jsonl  (test labels are hidden upstream)
/// </code>
/// Auto-download fetches the canonical Rowan Zellers GitHub release.
/// Features Tensor[N, 4 * MaxSequenceLength] holds the 4 candidate
/// (context + ending) sequences concatenated; labels are one-hot
/// 4-class vectors Tensor[N, 4]. Test split labels are filled with the
/// uniform prior since they aren't released publicly.
/// </para>
/// </remarks>
public class HellaswagDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly HellaswagDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "HellaSwag";
    /// <inheritdoc/>
    public override string Description => "HellaSwag 4-way commonsense NLI multiple choice";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxSequenceLength * 4;
    /// <inheritdoc/>
    public override int OutputDimension => 4;

    /// <summary>Creates a new HellaSwag data loader.</summary>
    public HellaswagDataLoader(HellaswagDataLoaderOptions? options = null)
    {
        _options = options ?? new HellaswagDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("hellaswag");
    }

    private static readonly string BaseUrl =
        "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string fileName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "hellaswag_test.jsonl",
            Geometry.DatasetSplit.Validation => "hellaswag_val.jsonl",
            _ => "hellaswag_train.jsonl"
        };
        string url = BaseUrl + fileName;
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
                throw new InvalidOperationException($"Failed to download HellaSwag from {url}.", ex);
            }
        }

        if (!File.Exists(filePath))
            throw new FileNotFoundException($"HellaSwag not found at {filePath}.");

        var contexts = new List<string>();
        var endings = new List<string[]>();
        var labels = new List<int>();
        bool isTestSplit = _options.Split == Geometry.DatasetSplit.Test;

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
                    $"Malformed JSONL in HellaSwag file '{filePath}' at line {lineNum}: {preview}", ex);
            }
            string? ctx = obj["ctx"]?.ToString() ?? obj["ctx_a"]?.ToString();
            var endingsArr = obj["endings"] as JArray;
            if (string.IsNullOrEmpty(ctx) || endingsArr is null || endingsArr.Count != 4) continue;
            var endingsStrs = new string[4];
            for (int i = 0; i < 4; i++) endingsStrs[i] = endingsArr[i]?.ToString() ?? string.Empty;

            // Test split: labels are hidden upstream — sentinel -1 → uniform-prior label tensor.
            // Train/Validation: missing or out-of-range label is a data integrity failure.
            int label;
            if (isTestSplit)
            {
                label = -1;
            }
            else if (obj["label"]?.Type == JTokenType.Integer)
            {
                label = obj["label"]!.Value<int>();
                if (label < 0 || label > 3)
                    throw new InvalidDataException(
                        $"Invalid label {label} in HellaSwag file '{filePath}' at line {lineNum}. Expected 0..3.");
            }
            else
            {
                throw new InvalidDataException(
                    $"Missing or non-integer label in HellaSwag file '{filePath}' at line {lineNum} for non-test split.");
            }

            contexts.Add(ctx!);
            endings.Add(endingsStrs);
            labels.Add(label);
        }

        int totalSamples = contexts.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 5);
        for (int i = 0; i < totalSamples; i++)
        {
            allTexts.Add(contexts[i]);
            for (int j = 0; j < 4; j++) allTexts.Add(endings[i][j]);
        }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int seqLen = _options.MaxSequenceLength;
        var featuresData = new T[totalSamples * 4 * seqLen];
        var labelsData = new T[totalSamples * 4];

        T uniformPrior = NumOps.FromDouble(0.25);
        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int e = 0; e < 4; e++)
            {
                int[] tok = TextLoaderHelper.TokenizeAndEncode(contexts[i] + " " + endings[i][e], vocabulary, seqLen);
                int offset = (i * 4 + e) * seqLen;
                for (int j = 0; j < seqLen; j++) featuresData[offset + j] = NumOps.FromDouble(tok[j]);
            }
            if (isTestSplit)
            {
                // Test labels are hidden upstream — emit the uniform prior so consumers
                // see the documented [0.25, 0.25, 0.25, 0.25] sentinel rather than a false one-hot.
                int baseIdx = i * 4;
                for (int c = 0; c < 4; c++) labelsData[baseIdx + c] = uniformPrior;
            }
            else
            {
                labelsData[i * 4 + labels[i]] = NumOps.One;
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, 4, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 4 });
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
