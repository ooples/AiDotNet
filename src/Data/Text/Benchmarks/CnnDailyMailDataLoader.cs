using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Parquet;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the CNN/DailyMail abstractive-summarization dataset v3.0.0
/// (Hermann et al. 2015 / See et al. 2017) via HuggingFace parquet.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects parquet shards under <c>{DataPath}/cnn_dailymail/</c>. Auto-download
/// fetches a small sentinel set: validation parquet + a single train shard.
/// For the full 287k-train corpus, manually download all shards from
/// huggingface.co/datasets/abisee/cnn_dailymail and place them in DataPath.
/// </para>
/// </remarks>
public class CnnDailyMailDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const string HfBase =
        "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/refs%2Fconvert%2Fparquet/3.0.0/";

    private readonly CnnDailyMailDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "CNN/DailyMail";
    public override string Description => "CNN/DailyMail v3.0.0 abstractive summarization";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxArticleLength;
    public override int OutputDimension => _options.MaxSummaryLength;

    public CnnDailyMailDataLoader(CnnDailyMailDataLoaderOptions? options = null)
    {
        _options = options ?? new CnnDailyMailDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("cnn_dailymail");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(_dataPath);
        string splitName = _options.Split switch
        {
            Geometry.DatasetSplit.Test => "test",
            Geometry.DatasetSplit.Validation => "validation",
            _ => "train"
        };
        string splitDir = Path.Combine(_dataPath, splitName);
        Directory.CreateDirectory(splitDir);

        if (_options.AutoDownload && Directory.GetFiles(splitDir, "*.parquet").Length == 0)
        {
            // The HF parquet conversion serves files at /{split}/000{n}.parquet — we fetch shard 0
            // (the typical first shard). For full coverage, user downloads more shards manually.
            string url = $"{HfBase}{splitName}/0000.parquet";
            string dst = Path.Combine(splitDir, "0000.parquet");
            try { await DatasetDownloader.DownloadFileAsync(url, dst, cancellationToken); }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException(
                    $"Failed to download CNN/DailyMail parquet from {url}.", ex);
            }
        }

        var articles = new List<string>();
        var summaries = new List<string>();
        foreach (string parquetFile in Directory.EnumerateFiles(splitDir, "*.parquet")
                 .OrderBy(f => f, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            await using var stream = File.OpenRead(parquetFile);
            using var reader = await ParquetReader.CreateAsync(stream);
            var fields = reader.Schema.GetDataFields();
            var artField = fields.FirstOrDefault(f => f.Name.Equals("article", StringComparison.OrdinalIgnoreCase));
            var sumField = fields.FirstOrDefault(f => f.Name.Equals("highlights", StringComparison.OrdinalIgnoreCase));
            if (artField is null || sumField is null)
                throw new InvalidDataException($"{parquetFile} missing 'article' or 'highlights' columns.");

            for (int g = 0; g < reader.RowGroupCount; g++)
            {
                using var rg = reader.OpenRowGroupReader(g);
                var artCol = await rg.ReadColumnAsync(artField, cancellationToken);
                var sumCol = await rg.ReadColumnAsync(sumField, cancellationToken);
                if (artCol.Data is string[] arts && sumCol.Data is string[] sums)
                {
                    int n = Math.Min(arts.Length, sums.Length);
                    for (int i = 0; i < n; i++)
                    {
                        if (!string.IsNullOrEmpty(arts[i]) && !string.IsNullOrEmpty(sums[i]))
                        { articles.Add(arts[i]); summaries.Add(sums[i]); }
                    }
                }
            }
        }

        if (articles.Count == 0)
            throw new InvalidDataException($"No CNN/DailyMail records found in {splitDir}.");

        int totalSamples = articles.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++) { allTexts.Add(articles[i]); allTexts.Add(summaries[i]); }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int aLen = _options.MaxArticleLength, sLen = _options.MaxSummaryLength;
        var featuresData = new T[totalSamples * aLen];
        var labelsData = new T[totalSamples * sLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] aTok = TextLoaderHelper.TokenizeAndEncode(articles[i], vocabulary, aLen);
            int[] sTok = TextLoaderHelper.TokenizeAndEncode(summaries[i], vocabulary, sLen);
            int aOff = i * aLen, sOff = i * sLen;
            for (int j = 0; j < aLen; j++) featuresData[aOff + j] = NumOps.FromDouble(aTok[j]);
            for (int j = 0; j < sLen; j++) labelsData[sOff + j] = NumOps.FromDouble(sTok[j]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, aLen });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, sLen });
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
