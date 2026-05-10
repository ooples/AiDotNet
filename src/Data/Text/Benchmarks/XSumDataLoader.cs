using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Parquet;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the XSum extreme abstractive-summarization dataset (Narayan et al. 2018).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Mirrors CNN/DailyMail loader shape; reads the HuggingFace
/// <c>EdinburghNLP/xsum</c> parquet conversion. Schema: document, summary, id.
/// </para>
/// </remarks>
public class XSumDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private const string HfBase =
        "https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/refs%2Fconvert%2Fparquet/default/";

    private readonly XSumDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "XSum";
    public override string Description => "XSum extreme abstractive summarization (single-sentence summaries)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxDocumentLength;
    public override int OutputDimension => _options.MaxSummaryLength;

    public XSumDataLoader(XSumDataLoaderOptions? options = null)
    {
        _options = options ?? new XSumDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("xsum");
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
            string url = $"{HfBase}{splitName}/0000.parquet";
            string dst = Path.Combine(splitDir, "0000.parquet");
            try { await DatasetDownloader.DownloadFileAsync(url, dst, cancellationToken); }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                throw new InvalidOperationException($"Failed to download XSum parquet from {url}.", ex);
            }
        }

        var docs = new List<string>();
        var sums = new List<string>();
        foreach (string parquetFile in Directory.EnumerateFiles(splitDir, "*.parquet")
                 .OrderBy(f => f, StringComparer.Ordinal))
        {
            cancellationToken.ThrowIfCancellationRequested();
            await using var stream = File.OpenRead(parquetFile);
            using var reader = await ParquetReader.CreateAsync(stream);
            var fields = reader.Schema.GetDataFields();
            var docField = fields.FirstOrDefault(f => f.Name.Equals("document", StringComparison.OrdinalIgnoreCase));
            var sumField = fields.FirstOrDefault(f => f.Name.Equals("summary", StringComparison.OrdinalIgnoreCase));
            if (docField is null || sumField is null)
                throw new InvalidDataException($"{parquetFile} missing document/summary columns.");

            for (int g = 0; g < reader.RowGroupCount; g++)
            {
                using var rg = reader.OpenRowGroupReader(g);
                var dCol = await rg.ReadColumnAsync(docField, cancellationToken);
                var sCol = await rg.ReadColumnAsync(sumField, cancellationToken);
                if (dCol.Data is string[] d && sCol.Data is string[] s)
                {
                    int n = Math.Min(d.Length, s.Length);
                    for (int i = 0; i < n; i++)
                        if (!string.IsNullOrEmpty(d[i]) && !string.IsNullOrEmpty(s[i]))
                        { docs.Add(d[i]); sums.Add(s[i]); }
                }
            }
        }
        if (docs.Count == 0) throw new InvalidDataException($"No XSum records found in {splitDir}.");

        int totalSamples = docs.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++) { allTexts.Add(docs[i]); allTexts.Add(sums[i]); }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int dLen = _options.MaxDocumentLength, sLen = _options.MaxSummaryLength;
        var featuresData = new T[totalSamples * dLen];
        var labelsData = new T[totalSamples * sLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] dTok = TextLoaderHelper.TokenizeAndEncode(docs[i], vocabulary, dLen);
            int[] sTok = TextLoaderHelper.TokenizeAndEncode(sums[i], vocabulary, sLen);
            int dOff = i * dLen, sOff = i * sLen;
            for (int j = 0; j < dLen; j++) featuresData[dOff + j] = NumOps.FromDouble(dTok[j]);
            for (int j = 0; j < sLen; j++) labelsData[sOff + j] = NumOps.FromDouble(sTok[j]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, dLen });
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
