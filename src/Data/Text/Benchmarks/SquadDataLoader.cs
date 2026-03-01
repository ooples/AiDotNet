using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the SQuAD question answering dataset (100K+ Q&amp;A pairs on Wikipedia articles).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SQuAD expects JSON files:
/// <code>
/// {DataPath}/
///   train-v1.1.json (or train-v2.0.json)
///   dev-v1.1.json   (or dev-v2.0.json)
/// </code>
/// Features are concatenated context + question tokens Tensor[N, MaxContextLength + MaxQuestionLength].
/// Labels are answer start/end positions Tensor[N, 2].
/// </para>
/// </remarks>
public class SquadDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly SquadDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;
    private int _totalSeqLen;

    /// <inheritdoc/>
    public override string Name => _options.Version2 ? "SQuAD-2.0" : "SQuAD-1.1";
    /// <inheritdoc/>
    public override string Description => $"SQuAD question answering dataset ({(_options.Version2 ? "v2.0" : "v1.1")})";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _totalSeqLen;
    /// <inheritdoc/>
    public override int OutputDimension => 2; // start, end positions

    /// <summary>Creates a new SQuAD data loader.</summary>
    public SquadDataLoader(SquadDataLoaderOptions? options = null)
    {
        _options = options ?? new SquadDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("squad");
        _totalSeqLen = _options.MaxContextLength + _options.MaxQuestionLength;
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string version = _options.Version2 ? "2.0" : "1.1";
        string splitName = _options.Split == Geometry.DatasetSplit.Test ? "dev" : "train";
        string jsonFile = Path.Combine(_dataPath, $"{splitName}-v{version}.json");
        if (!File.Exists(jsonFile))
        {
            jsonFile = Path.Combine(_dataPath, $"{splitName}.json");
            if (!File.Exists(jsonFile))
            {
                throw new FileNotFoundException(
                    $"SQuAD data not found at {_dataPath}. " +
                    "Download from https://rajpurkar.github.io/SQuAD-explorer/.");
            }
        }

        string jsonContent = await FilePolyfill.ReadAllTextAsync(jsonFile, cancellationToken);

        // Parse SQuAD JSON to extract context, question, answer_start
        var samples = new List<(string Context, string Question, int AnswerStart, int AnswerLength)>();
        ParseSquadJson(jsonContent, samples);

        int totalSamples = samples.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;

        // Build vocabulary from contexts and questions
        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++)
        {
            allTexts.Add(samples[i].Context);
            allTexts.Add(samples[i].Question);
        }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        var featuresData = new T[totalSamples * _totalSeqLen];
        var labelsData = new T[totalSamples * 2];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var (context, question, ansStart, ansLen) = samples[i];

            int[] contextTokens = TextLoaderHelper.TokenizeAndEncode(context, vocabulary, _options.MaxContextLength);
            int[] questionTokens = TextLoaderHelper.TokenizeAndEncode(question, vocabulary, _options.MaxQuestionLength);

            int featureOffset = i * _totalSeqLen;
            for (int j = 0; j < _options.MaxContextLength; j++)
                featuresData[featureOffset + j] = NumOps.FromDouble(contextTokens[j]);
            for (int j = 0; j < _options.MaxQuestionLength; j++)
                featuresData[featureOffset + _options.MaxContextLength + j] = NumOps.FromDouble(questionTokens[j]);

            // Convert character-level positions to approximate token positions
            // by counting word boundaries in the context up to the character offset
            int tokenStart = CharOffsetToTokenIndex(context, ansStart);
            int tokenEnd = CharOffsetToTokenIndex(context, ansStart + ansLen);
            labelsData[i * 2] = NumOps.FromDouble(Math.Min(tokenStart, _options.MaxContextLength - 1));
            labelsData[i * 2 + 1] = NumOps.FromDouble(Math.Min(tokenEnd, _options.MaxContextLength - 1));
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, _totalSeqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 2 });
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

    private static void ParseSquadJson(string json, List<(string Context, string Question, int AnswerStart, int AnswerLength)> samples)
    {
        // Simplified JSON parsing using string search for SQuAD format
        // Real format: {"data": [{"paragraphs": [{"context": "...", "qas": [{"question": "...", "answers": [{"answer_start": N, "text": "..."}]}]}]}]}
        int pos = 0;
        while (pos < json.Length)
        {
            int contextStart = FindJsonStringValue(json, "context", pos);
            if (contextStart < 0) break;
            string context = ExtractJsonString(json, contextStart);
            pos = contextStart + context.Length + 2;

            // Find all qas within this context block
            while (pos < json.Length)
            {
                int questionStart = FindJsonStringValue(json, "question", pos);
                if (questionStart < 0) break;

                // Check if we've moved past this context's scope (rough heuristic)
                int nextContextStart = FindJsonStringValue(json, "context", pos);
                if (nextContextStart >= 0 && nextContextStart < questionStart) break;

                string question = ExtractJsonString(json, questionStart);
                pos = questionStart + question.Length + 2;

                int answerStartPos = FindJsonStringValue(json, "answer_start", pos);
                int answerTextPos = FindJsonStringValue(json, "text", pos);

                int answerStart = 0;
                string answerText = string.Empty;

                if (answerStartPos >= 0)
                {
                    string ansStartStr = ExtractJsonValue(json, answerStartPos);
                    int.TryParse(ansStartStr, out answerStart);
                }
                if (answerTextPos >= 0)
                    answerText = ExtractJsonString(json, answerTextPos);

                if (context.Length > 0 && question.Length > 0)
                    samples.Add((context, question, answerStart, answerText.Length));
            }
        }
    }

    private static int FindJsonStringValue(string json, string key, int startPos)
    {
        string pattern = $"\"{key}\"";
        int idx = json.IndexOf(pattern, startPos, StringComparison.Ordinal);
        if (idx < 0) return -1;

        int colonIdx = json.IndexOf(':', idx + pattern.Length);
        if (colonIdx < 0) return -1;

        int valueStart = colonIdx + 1;
        while (valueStart < json.Length && (json[valueStart] == ' ' || json[valueStart] == '\n' || json[valueStart] == '\r'))
            valueStart++;

        return valueStart;
    }

    private static string ExtractJsonString(string json, int startPos)
    {
        if (startPos >= json.Length || json[startPos] != '"') return string.Empty;
        int end = startPos + 1;
        while (end < json.Length)
        {
            if (json[end] == '\\') { end += 2; continue; }
            if (json[end] == '"') break;
            end++;
        }
        return json.Substring(startPos + 1, end - startPos - 1);
    }

    /// <summary>
    /// Converts a character offset in text to an approximate token index
    /// by counting word boundaries (whitespace-separated tokens).
    /// </summary>
    private static int CharOffsetToTokenIndex(string text, int charOffset)
    {
        if (charOffset <= 0 || text.Length == 0) return 0;
        int clampedOffset = Math.Min(charOffset, text.Length);
        int tokenIndex = 0;
        bool inWord = false;
        for (int i = 0; i < clampedOffset; i++)
        {
            bool isAlpha = char.IsLetterOrDigit(text[i]);
            if (isAlpha && !inWord)
                tokenIndex++;
            inWord = isAlpha;
        }
        return Math.Max(0, tokenIndex - 1); // -1 because tokenIndex counts starts, we want 0-based
    }

    private static string ExtractJsonValue(string json, int startPos)
    {
        int end = startPos;
        while (end < json.Length && json[end] != ',' && json[end] != '}' && json[end] != ']' && json[end] != '\n')
            end++;
        return json.Substring(startPos, end - startPos).Trim().Trim('"');
    }
}
