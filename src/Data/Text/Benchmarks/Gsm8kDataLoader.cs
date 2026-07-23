using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the GSM8K grade-school math word-problem dataset (Cobbe et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/
///   train.jsonl
///   test.jsonl
/// </code>
/// Each line is <c>{"question": "...", "answer": "... #### N"}</c>.
/// Auto-download fetches the canonical OpenAI release. Features are
/// tokenized question sequences Tensor[N, MaxQuestionLength]; labels are
/// tokenized answer (chain-of-thought + final number) sequences
/// Tensor[N, MaxAnswerLength].
/// </para>
/// </remarks>
public class Gsm8kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly Gsm8kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "GSM8K";
    /// <inheritdoc/>
    public override string Description => "GSM8K grade-school math word problems with chain-of-thought solutions";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxQuestionLength;
    /// <inheritdoc/>
    public override int OutputDimension => _options.MaxAnswerLength;

    /// <summary>Creates a new GSM8K data loader.</summary>
    public Gsm8kDataLoader(Gsm8kDataLoaderOptions? options = null)
    {
        _options = options ?? new Gsm8kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("gsm8k");
    }

    private static readonly string TrainUrl =
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl";
    private static readonly string TestUrl =
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl";

    private async Task<(List<string> Questions, List<string> Answers)> ReadJsonlAsync(string filePath, CancellationToken cancellationToken)
    {
        var questions = new List<string>();
        var answers = new List<string>();
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
                    $"Malformed JSONL in GSM8K file '{filePath}' at line {lineNum}: {preview}", ex);
            }
            string? q = obj["question"]?.ToString();
            string? a = obj["answer"]?.ToString();
            if (string.IsNullOrEmpty(q) || string.IsNullOrEmpty(a)) continue;
            questions.Add(q!);
            answers.Add(a!);
        }
        return (questions, answers);
    }

    private async Task EnsureDownloadedAsync(string filePath, string url, CancellationToken cancellationToken)
    {
        if (File.Exists(filePath)) return;
        if (!_options.AutoDownload)
            throw new FileNotFoundException(
                $"GSM8K not found at {filePath}. Enable AutoDownload or place manually.");
        try
        {
            Directory.CreateDirectory(_dataPath);
            await DatasetDownloader.DownloadFileAsync(url, filePath, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            throw new InvalidOperationException($"Failed to download GSM8K from {url}.", ex);
        }
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        bool useTest = _options.Split == Geometry.DatasetSplit.Test
                    || _options.Split == Geometry.DatasetSplit.Validation;
        string trainPath = Path.Combine(_dataPath, "train.jsonl");
        string requestedPath = useTest ? Path.Combine(_dataPath, "test.jsonl") : trainPath;
        string requestedUrl = useTest ? TestUrl : TrainUrl;

        // Always materialize train (vocabulary source) and the requested split file.
        await EnsureDownloadedAsync(trainPath, TrainUrl, cancellationToken);
        if (useTest) await EnsureDownloadedAsync(requestedPath, requestedUrl, cancellationToken);

        // Build vocabulary from train regardless of which split was requested.
        // Industry standard: the vocab is fit on train and reused at eval time.
        var (trainQuestions, trainAnswers) = await ReadJsonlAsync(trainPath, cancellationToken);
        var trainTexts = new List<string>(trainQuestions.Count * 2);
        for (int i = 0; i < trainQuestions.Count; i++)
        {
            trainTexts.Add(trainQuestions[i]);
            trainTexts.Add(trainAnswers[i]);
        }
        var vocabulary = TextLoaderHelper.BuildVocabulary(trainTexts, trainTexts.Count, _options.VocabularySize);

        // Now read the requested split for actual sample data.
        List<string> questions, answers;
        if (useTest)
        {
            (questions, answers) = await ReadJsonlAsync(requestedPath, cancellationToken);
        }
        else
        {
            questions = trainQuestions;
            answers = trainAnswers;
        }

        int totalSamples = questions.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;

        int qLen = _options.MaxQuestionLength;
        int aLen = _options.MaxAnswerLength;
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
        var trainIndices = shuffled.Take(trainSize).ToArray();
        var valIndices = shuffled.Skip(trainSize).Take(valSize).ToArray();
        var testIndices = shuffled.Skip(trainSize + valSize).ToArray();
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, trainIndices), TextLoaderHelper.ExtractTensorBatch(labels, trainIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, valIndices), TextLoaderHelper.ExtractTensorBatch(labels, valIndices)),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(TextLoaderHelper.ExtractTensorBatch(features, testIndices), TextLoaderHelper.ExtractTensorBatch(labels, testIndices))
        );
    }
}
