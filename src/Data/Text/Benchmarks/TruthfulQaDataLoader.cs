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

        var lines = await FilePolyfill.ReadAllLinesAsync(filePath, cancellationToken);
        if (lines.Length < 2) return;

        // Header: Type,Category,Question,Best Answer,Correct Answers,Incorrect Answers,Source
        // Find column indices
        var header = SplitCsvLine(lines[0]);
        int qIdx = Array.FindIndex(header, h => h.Trim().Equals("Question", StringComparison.OrdinalIgnoreCase));
        int aIdx = Array.FindIndex(header, h => h.Trim().Equals("Best Answer", StringComparison.OrdinalIgnoreCase));
        if (qIdx < 0 || aIdx < 0)
            throw new InvalidDataException("TruthfulQA.csv missing required columns Question/Best Answer.");

        var questions = new List<string>();
        var answers = new List<string>();
        for (int i = 1; i < lines.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrWhiteSpace(lines[i])) continue;
            var fields = SplitCsvLine(lines[i]);
            if (fields.Length <= aIdx) continue;
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
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Take(trainSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize).Take(valSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, shuffled.Skip(trainSize + valSize).ToArray()), TextLoaderHelper.ExtractTensorBatch(labels, shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    /// <summary>RFC4180-compatible CSV row split (handles quoted fields with commas/newlines).</summary>
    private static string[] SplitCsvLine(string line)
    {
        var fields = new List<string>();
        var sb = new System.Text.StringBuilder();
        bool inQuotes = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"') { sb.Append('"'); i++; }
                    else inQuotes = false;
                }
                else sb.Append(c);
            }
            else
            {
                if (c == '"') inQuotes = true;
                else if (c == ',') { fields.Add(sb.ToString()); sb.Clear(); }
                else sb.Append(c);
            }
        }
        fields.Add(sb.ToString());
        return fields.ToArray();
    }
}
