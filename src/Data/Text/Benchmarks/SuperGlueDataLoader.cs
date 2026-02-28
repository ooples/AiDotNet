using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads SuperGLUE benchmark sub-tasks (BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SuperGLUE expects JSONL files:
/// <code>
/// {DataPath}/SuperGLUE/{TaskName}/
///   train.jsonl
///   val.jsonl
///   test.jsonl
/// </code>
/// Features are token-index encoded text Tensor[N, MaxSequenceLength].
/// Labels are one-hot Tensor[N, NumClasses].
/// </para>
/// </remarks>
public class SuperGlueDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly SuperGlueDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => $"SuperGLUE-{_options.Task}";
    /// <inheritdoc/>
    public override string Description => $"SuperGLUE benchmark ({_options.Task})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxSequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new SuperGLUE data loader.</summary>
    public SuperGlueDataLoader(SuperGlueDataLoaderOptions? options = null)
    {
        _options = options ?? new SuperGlueDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("superglue");
        _numClasses = _options.Task == SuperGlueTask.CB ? 3 : 2;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string taskName = _options.Task.ToString();
        string taskDir = Path.Combine(_dataPath, "SuperGLUE", taskName);
        if (!Directory.Exists(taskDir))
            taskDir = Path.Combine(_dataPath, taskName);

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "val",
            _ => "train"
        };
        string jsonlFile = Path.Combine(taskDir, $"{splitName}.jsonl");
        if (!File.Exists(jsonlFile))
        {
            throw new FileNotFoundException(
                $"SuperGLUE {taskName} data not found at {jsonlFile}. " +
                "Download from https://super.gluebenchmark.com/tasks.");
        }

        var lines = await FilePolyfill.ReadAllLinesAsync(jsonlFile, cancellationToken);
        var texts = new List<string>();
        var labelValues = new List<int>();

        foreach (var line in lines)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            // Simple JSON parsing for text and label fields
            string text = ExtractJsonField(line, "passage") + " [SEP] " + ExtractJsonField(line, "question");
            if (text.Length <= 7) // only " [SEP] "
                text = ExtractJsonField(line, "premise") + " [SEP] " + ExtractJsonField(line, "hypothesis");
            if (text.Length <= 7)
                text = ExtractJsonField(line, "sentence1") + " [SEP] " + ExtractJsonField(line, "sentence2");
            if (text.Length <= 7)
                text = ExtractJsonField(line, "sentence") + " [SEP] " + ExtractJsonField(line, "word");

            string labelStr = ExtractJsonField(line, "label");
            int label = 0;
            if (int.TryParse(labelStr, out int parsedLabel))
                label = parsedLabel;
            else if (labelStr.Equals("true", StringComparison.OrdinalIgnoreCase))
                label = 1;

            if (text.Length > 7)
            {
                texts.Add(text);
                labelValues.Add(label);
            }
        }

        int totalSamples = texts.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        var vocabulary = TextLoaderHelper.BuildVocabulary(texts, totalSamples, _options.VocabularySize);

        int seqLen = _options.MaxSequenceLength;
        var featuresData = new T[totalSamples * seqLen];
        var labelsData = new T[totalSamples * _numClasses];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] tokenIds = TextLoaderHelper.TokenizeAndEncode(texts[i], vocabulary, seqLen);
            int featureOffset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
                featuresData[featureOffset + j] = NumOps.FromDouble(tokenIds[j]);

            int label = Math.Min(labelValues[i], _numClasses - 1);
            if (label >= 0)
                labelsData[i * _numClasses + label] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, _numClasses });
        InitializeIndices(totalSamples);
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

    private static string ExtractJsonField(string json, string fieldName)
    {
        string key = $"\"{fieldName}\"";
        int idx = json.IndexOf(key, StringComparison.OrdinalIgnoreCase);
        if (idx < 0) return string.Empty;

        int colonIdx = json.IndexOf(':', idx + key.Length);
        if (colonIdx < 0) return string.Empty;

        int start = colonIdx + 1;
        while (start < json.Length && json[start] == ' ') start++;
        if (start >= json.Length) return string.Empty;

        if (json[start] == '"')
        {
            int end = json.IndexOf('"', start + 1);
            return end > start ? json.Substring(start + 1, end - start - 1) : string.Empty;
        }

        // Numeric or boolean value
        int valEnd = start;
        while (valEnd < json.Length && json[valEnd] != ',' && json[valEnd] != '}' && json[valEnd] != ' ')
            valEnd++;
        return json.Substring(start, valEnd - start).Trim();
    }
}
