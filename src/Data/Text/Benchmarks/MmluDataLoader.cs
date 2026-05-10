using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads MMLU — Massive Multitask Language Understanding (Hendrycks et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects the canonical AI release layout:
/// <code>
/// {DataPath}/data/
///   dev/{subject}_dev.csv
///   val/{subject}_val.csv
///   test/{subject}_test.csv
/// </code>
/// Each CSV has columns <c>question, A, B, C, D, answer</c> with no header.
/// Auto-downloads the canonical Hendrycks tarball. 4-way multiple choice
/// shape: features [N, 4, MaxQuestionLength], one-hot labels [N, 4].
/// </para>
/// </remarks>
public class MmluDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const int Choices = 4;
    private static readonly string DownloadUrl =
        "https://people.eecs.berkeley.edu/~hendrycks/data.tar";

    private readonly MmluDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "MMLU";
    public override string Description => "Massive Multitask Language Understanding (57-subject benchmark)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => Choices * _options.MaxQuestionLength;
    public override int OutputDimension => Choices;

    public MmluDataLoader(MmluDataLoaderOptions? options = null)
    {
        _options = options ?? new MmluDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("mmlu");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            // The canonical release is a plain .tar (not gz). DatasetDownloader's
            // tar.gz extractor handles plain tar via stream sniffing — but to be
            // safe we download then extract via the tar path explicitly.
            try
            {
                await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to download MMLU from {DownloadUrl}. " +
                    "Note: archive is plain tar (.tar) not gzipped — fetcher must support both.", ex);
            }
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"MMLU not found at {_dataPath}.");

        string splitDir = _options.Split switch
        {
            Geometry.DatasetSplit.Train => Path.Combine(root, "dev"),       // dev set used for few-shot prompts
            Geometry.DatasetSplit.Validation => Path.Combine(root, "val"),
            _ => Path.Combine(root, "test")
        };
        if (!Directory.Exists(splitDir))
            throw new DirectoryNotFoundException($"MMLU split directory not found: {splitDir}");

        var questions = new List<string>();
        var choicesAll = new List<string[]>();
        var labels = new List<int>();

        foreach (string csvFile in Directory.GetFiles(splitDir, "*.csv").OrderBy(f => f, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            string subjectFile = Path.GetFileNameWithoutExtension(csvFile);
            if (!string.IsNullOrEmpty(_options.SubjectFilter) &&
                !subjectFile.Contains(_options.SubjectFilter, StringComparison.OrdinalIgnoreCase))
                continue;

            foreach (string line in await FilePolyfill.ReadAllLinesAsync(csvFile, cancellationToken))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var fields = SplitCsvLine(line);
                if (fields.Length < 6) continue;
                string q = fields[0]; string a = fields[1]; string b = fields[2];
                string c = fields[3]; string d = fields[4]; string ans = fields[5].Trim();
                if (string.IsNullOrEmpty(q) || string.IsNullOrEmpty(ans)) continue;
                int label = ans.Length > 0 ? char.ToUpper(ans[0]) - 'A' : -1;
                if (label < 0 || label >= Choices) continue;
                questions.Add(q);
                choicesAll.Add(new[] { a, b, c, d });
                labels.Add(label);
            }
        }

        int totalSamples = questions.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * (Choices + 1));
        for (int i = 0; i < totalSamples; i++)
        {
            allTexts.Add(questions[i]);
            for (int c = 0; c < Choices; c++) allTexts.Add(choicesAll[i][c]);
        }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int seqLen = _options.MaxQuestionLength;
        var featuresData = new T[totalSamples * Choices * seqLen];
        var labelsData = new T[totalSamples * Choices];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            for (int c = 0; c < Choices; c++)
            {
                int[] tok = TextLoaderHelper.TokenizeAndEncode(questions[i] + " " + choicesAll[i][c], vocabulary, seqLen);
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
        // Tar extracts to data/{dev,val,test}/
        string sub = Path.Combine(_dataPath, "data");
        return Directory.Exists(sub) ? sub : _dataPath;
    }

    private static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var sb = new System.Text.StringBuilder();
        bool inQuotes = false;
        for (int i = 0; i < line.Length; i++)
        {
            char ch = line[i];
            if (inQuotes)
            {
                if (ch == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(ch);
            }
            else
            {
                if (ch == '"') inQuotes = true;
                else if (ch == ',') { fields.Add(sb.ToString()); sb.Clear(); }
                else sb.Append(ch);
            }
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }
}
