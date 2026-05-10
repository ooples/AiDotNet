using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the AG News topic classification dataset (4 classes: World, Sports, Business, Sci/Tech).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects CSV files <c>train.csv</c> and <c>test.csv</c> with columns
/// <c>class_index,title,description</c>. Auto-download fetches the canonical
/// release from the FastAI mirror.
/// Features are tokenized title+description sequences Tensor[N, MaxSequenceLength].
/// Labels are one-hot 4-class vectors Tensor[N, 4].
/// </para>
/// </remarks>
public class AgNewsDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly AgNewsDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "AG News";
    /// <inheritdoc/>
    public override string Description => "AG News 4-class topic classification dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxSequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => 4;

    /// <summary>Creates a new AG News data loader.</summary>
    public AgNewsDataLoader(AgNewsDataLoaderOptions? options = null)
    {
        _options = options ?? new AgNewsDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("ag_news");
    }

    private static readonly string DownloadUrl =
        "https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string splitFile = _options.Split == Geometry.DatasetSplit.Test ? "test.csv" : "train.csv";
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
                    $"Failed to download AG News from {DownloadUrl}.", ex);
            }
            filePath = Path.Combine(ResolveDataDir(), splitFile);
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException(
                $"AG News data not found at {filePath}. Enable AutoDownload or extract manually.");
        }

        // CSV: "class_index","title","description"  (1-indexed class_index in [1..4])
        var texts = new List<string>();
        var labelClasses = new List<int>();
        ParseAgNewsCsv(await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken), texts, labelClasses);

        int totalSamples = texts.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;

        var vocabulary = TextLoaderHelper.BuildVocabulary(texts, totalSamples, _options.VocabularySize);
        int seqLen = _options.MaxSequenceLength;
        var featuresData = new T[totalSamples * seqLen];
        var labelsData = new T[totalSamples * 4];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] tokenIds = TextLoaderHelper.TokenizeAndEncode(texts[i], vocabulary, seqLen);
            int featureOffset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
                featuresData[featureOffset + j] = NumOps.FromDouble(tokenIds[j]);

            // Class index is 1..4 in the source; convert to 0..3 for one-hot.
            int classIdx = Math.Clamp(labelClasses[i] - 1, 0, 3);
            labelsData[i * 4 + classIdx] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 4 });
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

    private string ResolveDataDir()
    {
        // Tarball extracts to ag_news_csv/{train,test}.csv
        string subDir = Path.Combine(_dataPath, "ag_news_csv");
        if (Directory.Exists(subDir) &&
            (File.Exists(Path.Combine(subDir, "train.csv")) || File.Exists(Path.Combine(subDir, "test.csv"))))
            return subDir;
        return _dataPath;
    }

    /// <summary>
    /// Parses the AG News CSV format. Each row is <c>"class","title","description"</c>
    /// where embedded quotes are doubled per RFC 4180.
    /// </summary>
    private static void ParseAgNewsCsv(string csv, List<string> texts, List<int> labels)
    {
        int pos = 0;
        var sb = new System.Text.StringBuilder();
        while (pos < csv.Length)
        {
            // Field 1: class index (quoted)
            string classField = ReadCsvField(csv, ref pos, sb);
            if (classField.Length == 0 && pos >= csv.Length) break;
            // Field 2: title
            string title = ReadCsvField(csv, ref pos, sb);
            // Field 3: description
            string desc = ReadCsvField(csv, ref pos, sb);
            // Skip newline
            while (pos < csv.Length && (csv[pos] == '\r' || csv[pos] == '\n')) pos++;

            if (!int.TryParse(classField.Trim(), out int classIdx)) continue;
            string combined = (title + " " + desc).Trim();
            if (combined.Length == 0) continue;
            texts.Add(combined);
            labels.Add(classIdx);
        }
    }

    /// <summary>Reads one quoted CSV field handling RFC4180 double-quote escapes.</summary>
    private static string ReadCsvField(string csv, ref int pos, System.Text.StringBuilder sb)
    {
        sb.Clear();
        if (pos >= csv.Length) return string.Empty;
        if (csv[pos] != '"')
        {
            // Unquoted field — read until comma or newline
            while (pos < csv.Length && csv[pos] != ',' && csv[pos] != '\r' && csv[pos] != '\n')
                sb.Append(csv[pos++]);
            if (pos < csv.Length && csv[pos] == ',') pos++;
            return sb.ToString();
        }
        pos++; // skip opening quote
        while (pos < csv.Length)
        {
            char c = csv[pos];
            if (c == '"')
            {
                if (pos + 1 < csv.Length && csv[pos + 1] == '"')
                {
                    sb.Append('"');
                    pos += 2;
                    continue;
                }
                pos++; // closing quote
                if (pos < csv.Length && csv[pos] == ',') pos++;
                return sb.ToString();
            }
            sb.Append(c);
            pos++;
        }
        return sb.ToString();
    }
}
