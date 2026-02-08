using System.Text.RegularExpressions;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the IMDB 50k movie review sentiment analysis dataset (25k train / 25k test, binary classification).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The IMDB dataset contains 50,000 movie reviews for binary sentiment classification (positive/negative).
/// Reviews are tokenized into word indices using a simple whitespace/punctuation tokenizer,
/// with a configurable vocabulary size and sequence length.
/// </para>
/// </remarks>
public class Imdb50kDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";

    private static readonly TimeSpan TokenizerTimeout = TimeSpan.FromSeconds(1);
    private static readonly Regex TokenizerRegex = new Regex(
        @"[a-zA-Z]+", RegexOptions.Compiled, TokenizerTimeout);

    private readonly Imdb50kDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    /// <inheritdoc/>
    public override string Name => "IMDB-50k";
    /// <inheritdoc/>
    public override string Description => "IMDB 50k movie review sentiment analysis dataset";
    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;
    /// <inheritdoc/>
    public override int FeatureCount => _options.MaxSequenceLength;
    /// <inheritdoc/>
    public override int OutputDimension => 2;

    /// <summary>Creates a new IMDB 50k data loader.</summary>
    public Imdb50kDataLoader(Imdb50kDataLoaderOptions? options = null)
    {
        _options = options ?? new Imdb50kDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("imdb-50k");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string dataDir = FindDataDirectory(_dataPath);

        if (dataDir.Length == 0 && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            dataDir = FindDataDirectory(_dataPath);
        }

        if (dataDir.Length == 0)
            throw new FileNotFoundException($"IMDB data not found at {_dataPath}.");

        string splitDir = _options.Split == DatasetSplit.Test
            ? Path.Combine(dataDir, "test")
            : Path.Combine(dataDir, "train");

        if (!Directory.Exists(splitDir))
            throw new DirectoryNotFoundException($"IMDB split directory not found: {splitDir}");

        // Load review texts and labels
        var reviews = new List<string>();
        var labels = new List<int>();

        LoadReviews(Path.Combine(splitDir, "pos"), 1, reviews, labels, cancellationToken);
        LoadReviews(Path.Combine(splitDir, "neg"), 0, reviews, labels, cancellationToken);

        int totalSamples = reviews.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
        {
            // Shuffle before truncating to avoid single-class bias (positive reviews loaded first)
            int[] shuffleIndices = Enumerable.Range(0, reviews.Count).ToArray();
            var shuffleRandom = Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
            for (int i = shuffleIndices.Length - 1; i > 0; i--)
            {
                int j = shuffleRandom.Next(i + 1);
                (shuffleIndices[i], shuffleIndices[j]) = (shuffleIndices[j], shuffleIndices[i]);
            }

            var shuffledReviews = new List<string>(reviews.Count);
            var shuffledLabels = new List<int>(labels.Count);
            foreach (int idx in shuffleIndices)
            {
                shuffledReviews.Add(reviews[idx]);
                shuffledLabels.Add(labels[idx]);
            }
            reviews = shuffledReviews;
            labels = shuffledLabels;

            totalSamples = _options.MaxSamples.Value;
        }

        _sampleCount = totalSamples;

        // Build vocabulary from the reviews we will actually use
        var vocabulary = BuildVocabulary(reviews, totalSamples, _options.VocabularySize);

        // Tokenize and encode reviews
        int seqLen = _options.MaxSequenceLength;
        var featuresData = new T[totalSamples * seqLen];
        var labelsData = new T[totalSamples * 2];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            int[] tokenIds = TokenizeAndEncode(reviews[i], vocabulary, seqLen);
            int featureOffset = i * seqLen;
            for (int j = 0; j < seqLen; j++)
            {
                featuresData[featureOffset + j] = NumOps.FromDouble(tokenIds[j]);
            }

            // One-hot label: [negative, positive]
            labelsData[i * 2 + labels[i]] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, seqLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 2 });
        InitializeIndices(totalSamples);
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
    }

    /// <inheritdoc/>
    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        return (ExtractTensorBatch(features, indices), ExtractTensorBatch(labels, indices));
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
        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();
        var shuffled = Enumerable.Range(0, _sampleCount).OrderBy(_ => random.Next()).ToArray();
        return (
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Take(trainSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Take(trainSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize).Take(valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize).Take(valSize).ToArray())),
            new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(
                ExtractTensorBatch(LoadedFeatures ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize + valSize).ToArray()),
                ExtractTensorBatch(LoadedLabels ?? throw new InvalidOperationException("Not loaded."),
                    shuffled.Skip(trainSize + valSize).ToArray()))
        );
    }

    private static void LoadReviews(string directory, int label,
        List<string> reviews, List<int> labels, CancellationToken cancellationToken)
    {
        if (!Directory.Exists(directory)) return;

        string[] files = Directory.GetFiles(directory, "*.txt");
        Array.Sort(files, StringComparer.OrdinalIgnoreCase);

        foreach (string file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string text = File.ReadAllText(file);
            reviews.Add(text);
            labels.Add(label);
        }
    }

    private static Dictionary<string, int> BuildVocabulary(
        List<string> reviews, int sampleCount, int maxVocabSize)
    {
        var wordCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        int limit = Math.Min(sampleCount, reviews.Count);
        for (int i = 0; i < limit; i++)
        {
            var tokens = Tokenize(reviews[i]);
            foreach (string token in tokens)
            {
                if (wordCounts.ContainsKey(token))
                    wordCounts[token]++;
                else
                    wordCounts[token] = 1;
            }
        }

        // Reserve index 0 for padding, 1 for unknown
        var vocabulary = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        int idx = 2; // 0 = PAD, 1 = UNK

        foreach (var pair in wordCounts.OrderByDescending(p => p.Value))
        {
            if (idx >= maxVocabSize) break;
            vocabulary[pair.Key] = idx;
            idx++;
        }

        return vocabulary;
    }

    private static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        try
        {
            var matches = TokenizerRegex.Matches(text);
            foreach (Match match in matches)
            {
                tokens.Add(match.Value.ToLowerInvariant());
            }
        }
        catch (RegexMatchTimeoutException)
        {
            // If regex times out on malformed input, return empty tokens
        }
        return tokens;
    }

    private static int[] TokenizeAndEncode(string text, Dictionary<string, int> vocabulary, int maxLength)
    {
        var tokens = Tokenize(text);
        int[] encoded = new int[maxLength];

        int len = Math.Min(tokens.Count, maxLength);
        for (int i = 0; i < len; i++)
        {
            if (vocabulary.TryGetValue(tokens[i], out int idx))
                encoded[i] = idx;
            else
                encoded[i] = 1; // UNK token
        }
        // Remaining positions stay 0 (PAD)

        return encoded;
    }

    private static string FindDataDirectory(string rootPath)
    {
        if (!Directory.Exists(rootPath)) return string.Empty;
        string[] candidates = new[] { rootPath, Path.Combine(rootPath, "aclImdb") };
        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate) &&
                Directory.Exists(Path.Combine(candidate, "train")) &&
                Directory.Exists(Path.Combine(candidate, "test")))
                return candidate;
        }
        var dirs = Directory.GetDirectories(rootPath, "aclImdb", SearchOption.AllDirectories);
        return dirs.Length > 0 ? dirs[0] : string.Empty;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);
        for (int i = 0; i < indices.Length; i++)
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        return result;
    }
}
