using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads GLUE benchmark sub-tasks (CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE, WNLI).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// GLUE expects TSV files:
/// <code>
/// {DataPath}/glue_data/{TaskName}/
///   train.tsv
///   dev.tsv
///   test.tsv
/// </code>
/// Features are token-index encoded text Tensor[N, MaxSequenceLength].
/// Labels are one-hot Tensor[N, NumClasses] (binary for most tasks, 3-class for MNLI, regression for STS-B).
/// </para>
/// </remarks>
public class GlueDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly GlueDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _numClasses;

    /// <inheritdoc/>
    public override string Name => $"GLUE-{_options.Task}";
    /// <inheritdoc/>
    public override string Description => $"GLUE benchmark ({_options.Task})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxSequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>Creates a new GLUE data loader.</summary>
    public GlueDataLoader(GlueDataLoaderOptions? options = null)
    {
        _options = options ?? new GlueDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("glue");
        _numClasses = _options.Task switch
        {
            GlueTask.MNLI => 3,
            GlueTask.STSB => 1, // STS-B is a regression task (score 0-5)
            _ => 2
        };
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string taskName = GetTaskDirectoryName(_options.Task);
        string taskDir = Path.Combine(_dataPath, "glue_data", taskName);
        if (!Directory.Exists(taskDir))
            taskDir = Path.Combine(_dataPath, taskName);

        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "dev",
            _ => "train"
        };
        string tsvFile = Path.Combine(taskDir, $"{splitName}.tsv");
        if (!File.Exists(tsvFile))
        {
            throw new FileNotFoundException(
                $"GLUE {taskName} data not found at {tsvFile}. " +
                "Download from https://gluebenchmark.com/tasks.");
        }

        var lines = await FilePolyfill.ReadAllLinesAsync(tsvFile, cancellationToken);
        var texts = new List<string>();
        var labelValues = new List<double>();
        bool isRegression = _options.Task == GlueTask.STSB;

        // Parse TSV (skip header). Column layout depends on task.
        for (int i = 1; i < lines.Length; i++)
        {
            var parts = lines[i].Split('\t');
            var (text, label) = ExtractTextAndLabel(parts, _options.Task);
            if (text.Length > 0)
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

            if (isRegression)
            {
                // STS-B: store regression score directly (0-5 range)
                labelsData[i] = NumOps.FromDouble(labelValues[i]);
            }
            else
            {
                // Classification: one-hot encoding
                int label = Math.Min((int)labelValues[i], _numClasses - 1);
                if (label >= 0)
                    labelsData[i * _numClasses + label] = NumOps.One;
            }
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

    private static string GetTaskDirectoryName(GlueTask task) => task switch
    {
        GlueTask.CoLA => "CoLA",
        GlueTask.SST2 => "SST-2",
        GlueTask.MRPC => "MRPC",
        GlueTask.QQP => "QQP",
        GlueTask.STSB => "STS-B",
        GlueTask.MNLI => "MNLI",
        GlueTask.QNLI => "QNLI",
        GlueTask.RTE => "RTE",
        GlueTask.WNLI => "WNLI",
        _ => task.ToString()
    };

    private static (string Text, double Label) ExtractTextAndLabel(string[] parts, GlueTask task)
    {
        // Column layouts vary by task
        return task switch
        {
            GlueTask.CoLA when parts.Length >= 4 => (parts[3].Trim(), int.TryParse(parts[1].Trim(), out int l) ? l : 0),
            GlueTask.SST2 when parts.Length >= 2 => (parts[0].Trim(), int.TryParse(parts[1].Trim(), out int l) ? l : 0),
            GlueTask.MRPC when parts.Length >= 5 => (parts[3].Trim() + " [SEP] " + parts[4].Trim(), int.TryParse(parts[0].Trim(), out int l) ? l : 0),
            GlueTask.QQP when parts.Length >= 6 => (parts[3].Trim() + " [SEP] " + parts[4].Trim(), int.TryParse(parts[5].Trim(), out int l) ? l : 0),
            // STS-B: regression label (float 0.0-5.0) in last column
            GlueTask.STSB when parts.Length >= 10 => (parts[7].Trim() + " [SEP] " + parts[8].Trim(),
                double.TryParse(parts[9].Trim(), System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out double score) ? score : 0.0),
            GlueTask.MNLI when parts.Length >= 10 => (parts[8].Trim() + " [SEP] " + parts[9].Trim(), ParseMnliLabel(parts.Length > 10 ? parts[10].Trim() : "")),
            GlueTask.QNLI when parts.Length >= 4 => (parts[1].Trim() + " [SEP] " + parts[2].Trim(), parts[3].Trim() == "entailment" ? 1 : 0),
            GlueTask.RTE when parts.Length >= 4 => (parts[1].Trim() + " [SEP] " + parts[2].Trim(), parts[3].Trim() == "entailment" ? 1 : 0),
            GlueTask.WNLI when parts.Length >= 4 => (parts[1].Trim() + " [SEP] " + parts[2].Trim(), int.TryParse(parts[3].Trim(), out int l) ? l : 0),
            _ => (string.Empty, 0)
        };
    }

    private static int ParseMnliLabel(string label) => label switch
    {
        "entailment" => 0,
        "neutral" => 1,
        "contradiction" => 2,
        _ => 0
    };
}
