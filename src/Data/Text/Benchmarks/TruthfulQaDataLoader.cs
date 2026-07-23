using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the TruthfulQA benchmark (Lin et al. 2022) — 817 truthfulness questions
/// across 38 categories. Generation-style: question → best_answer.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects <c>{DataPath}/TruthfulQA.csv</c>. Auto-download fetches the
/// canonical sylinrl/TruthfulQA GitHub release. CSV columns:
/// <c>Type,Category,Question,Best Answer,...</c>.
/// </para>
/// </remarks>
public class TruthfulQaDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly TruthfulQaDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "TruthfulQA";
    public override string Description => "TruthfulQA truthfulness benchmark (817 questions, 38 categories)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxQuestionLength;
    public override int OutputDimension => _options.MaxAnswerLength;

    public TruthfulQaDataLoader(TruthfulQaDataLoaderOptions? options = null)
    {
        _options = options ?? new TruthfulQaDataLoaderOptions();
        _options.Validate();
        // TruthfulQA ships only the canonical 817-question Test set — every paper
        // reports 0-shot results on this single split. Reject Train/Validation rather
        // than silently returning the same data.
        if (_options.Split != Geometry.DatasetSplit.Test)
            throw new ArgumentException(
                "TruthfulQA has only a Test split (817 questions). Set Options.Split = DatasetSplit.Test, or use Split() for synthetic partitions.",
                nameof(options));
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("truthfulqa");
    }

    private static readonly string DownloadUrl =
        "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv";

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string filePath = Path.Combine(_dataPath, "TruthfulQA.csv");
        if (!File.Exists(filePath) && _options.AutoDownload)
        {
            try
            {
                Directory.CreateDirectory(_dataPath);
                await DatasetDownloader.DownloadFileAsync(DownloadUrl, filePath, cancellationToken);
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException($"Failed to download TruthfulQA from {DownloadUrl}.", ex);
            }
        }
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"TruthfulQA not found at {filePath}.");

        // RFC4180-complete: parse full file as a CSV stream so quoted fields can span newlines.
        string raw = await FilePolyfill.ReadAllTextAsync(filePath, cancellationToken);
        var rows = ParseCsvRfc4180(raw);
        if (rows.Count < 2)
            throw new InvalidDataException(
                $"TruthfulQA.csv at {filePath} contains fewer than 2 rows (header + data).");

        // Header: Type,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source
        var header = rows[0];
        int qIdx = header.FindIndex(h => h.Trim().Equals("Question", StringComparison.OrdinalIgnoreCase));
        int aIdx = header.FindIndex(h => h.Trim().Equals("Best Answer", StringComparison.OrdinalIgnoreCase));
        if (qIdx < 0 || aIdx < 0)
            throw new InvalidDataException("TruthfulQA.csv missing required columns Question/Best Answer.");

        var questions = new List<string>();
        var answers = new List<string>();
        for (int i = 1; i < rows.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var fields = rows[i];
            if (fields.Count <= aIdx) continue;
            string q = fields[qIdx];
            string a = fields[aIdx];
            if (string.IsNullOrEmpty(q) || string.IsNullOrEmpty(a)) continue;
            questions.Add(q);
            answers.Add(a);
        }

        int totalSamples = questions.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++) { allTexts.Add(questions[i]); allTexts.Add(answers[i]); }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int qLen = _options.MaxQuestionLength, aLen = _options.MaxAnswerLength;
        var featuresData = new T[totalSamples * qLen];
        var labelsData = new T[totalSamples * aLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] qTok = TextLoaderHelper.TokenizeAndEncode(questions[i], vocabulary, qLen);
            int[] aTok = TextLoaderHelper.TokenizeAndEncode(answers[i], vocabulary, aLen);
            int qOff = i * qLen, aOff = i * aLen;
            for (int j = 0; j < qLen; j++) featuresData[qOff + j] = NumOps.FromDouble(qTok[j]);
            for (int j = 0; j < aLen; j++) labelsData[aOff + j] = NumOps.FromDouble(aTok[j]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, qLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, aLen });
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

    /// <summary>
    /// RFC4180-complete CSV parser: walks the entire input character stream so
    /// quoted fields may contain commas, CR, LF, and escaped quotes.
    /// </summary>
    private static List<List<string>> ParseCsvRfc4180(string input)
    {
        var rows = new List<List<string>>();
        var row = new List<string>();
        var sb = new System.Text.StringBuilder();
        bool inQuotes = false;
        int i = 0;
        while (i < input.Length)
        {
            char c = input[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < input.Length && input[i + 1] == '"') { sb.Append('"'); i += 2; continue; }
                    inQuotes = false;
                }
                else sb.Append(c);
                i++;
            }
            else
            {
                if (c == '"') { inQuotes = true; i++; }
                else if (c == ',') { row.Add(sb.ToString()); sb.Clear(); i++; }
                else if (c == '\r')
                {
                    row.Add(sb.ToString()); sb.Clear();
                    rows.Add(row); row = new List<string>();
                    if (i + 1 < input.Length && input[i + 1] == '\n') i += 2; else i++;
                }
                else if (c == '\n')
                {
                    row.Add(sb.ToString()); sb.Clear();
                    rows.Add(row); row = new List<string>();
                    i++;
                }
                else { sb.Append(c); i++; }
            }
        }
        // Flush final field/row if input doesn't end with a newline.
        if (sb.Length > 0 || row.Count > 0)
        {
            row.Add(sb.ToString());
            rows.Add(row);
        }
        return rows;
    }
}
