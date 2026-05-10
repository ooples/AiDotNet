using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Data.Text.Benchmarks;

/// <summary>
/// Loads the Hendrycks MATH benchmark — competition math problems (Hendrycks et al. 2021).
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects:
/// <code>
/// {DataPath}/MATH/
///   train/{subject}/NNNN.json
///   test/{subject}/NNNN.json
/// </code>
/// where <c>subject</c> is one of: algebra, counting_and_probability,
/// geometry, intermediate_algebra, number_theory, prealgebra, precalculus.
/// Each JSON has fields <c>problem</c>, <c>level</c>, <c>type</c>, <c>solution</c>.
/// </para>
/// </remarks>
public class MathDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private static readonly string DownloadUrl =
        "https://people.eecs.berkeley.edu/~hendrycks/MATH.tar";

    private readonly MathDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "MATH";
    public override string Description => "Hendrycks MATH competition math problems";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.MaxProblemLength;
    public override int OutputDimension => _options.MaxSolutionLength;

    public MathDataLoader(MathDataLoaderOptions? options = null)
    {
        _options = options ?? new MathDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("math-hendrycks");
    }

    /// <inheritdoc/>
    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string root = ResolveDataDir();
        if (!Directory.Exists(root) && _options.AutoDownload)
        {
            await DatasetDownloader.DownloadAndExtractTarGzAsync(DownloadUrl, _dataPath, cancellationToken);
            root = ResolveDataDir();
        }
        if (!Directory.Exists(root))
            throw new DirectoryNotFoundException($"MATH not found at {_dataPath}.");

        string splitDir = _options.Split == Geometry.DatasetSplit.Test || _options.Split == Geometry.DatasetSplit.Validation
            ? Path.Combine(root, "test") : Path.Combine(root, "train");
        if (!Directory.Exists(splitDir))
            throw new DirectoryNotFoundException($"MATH split dir not found: {splitDir}");

        var problems = new List<string>();
        var solutions = new List<string>();

        foreach (string subjectDir in Directory.EnumerateDirectories(splitDir).OrderBy(d => d, StringComparer.Ordinal))
        {
            string subject = Path.GetFileName(subjectDir);
            if (!string.IsNullOrEmpty(_options.SubjectFilter) &&
                !subject.Contains(_options.SubjectFilter, StringComparison.OrdinalIgnoreCase))
                continue;

            foreach (string jsonFile in Directory.EnumerateFiles(subjectDir, "*.json").OrderBy(f => f, StringComparer.Ordinal))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string content = await FilePolyfill.ReadAllTextAsync(jsonFile, cancellationToken);
                JObject obj;
                try { obj = JObject.Parse(content); } catch { continue; }

                if (_options.LevelFilter.HasValue)
                {
                    string? lvl = obj["level"]?.ToString();
                    if (string.IsNullOrEmpty(lvl)) continue;
                    // "level" is e.g. "Level 3"; extract integer.
                    int parsedLevel = -1;
                    foreach (var part in lvl!.Split(' '))
                        if (int.TryParse(part, out int n)) { parsedLevel = n; break; }
                    if (parsedLevel != _options.LevelFilter.Value) continue;
                }

                string? prob = obj["problem"]?.ToString();
                string? sol = obj["solution"]?.ToString();
                if (string.IsNullOrEmpty(prob) || string.IsNullOrEmpty(sol)) continue;
                problems.Add(prob!);
                solutions.Add(sol!);
            }
        }

        int totalSamples = problems.Count;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;
        _sampleCount = totalSamples;

        var allTexts = new List<string>(totalSamples * 2);
        for (int i = 0; i < totalSamples; i++) { allTexts.Add(problems[i]); allTexts.Add(solutions[i]); }
        var vocabulary = TextLoaderHelper.BuildVocabulary(allTexts, allTexts.Count, _options.VocabularySize);

        int pLen = _options.MaxProblemLength, sLen = _options.MaxSolutionLength;
        var featuresData = new T[totalSamples * pLen];
        var labelsData = new T[totalSamples * sLen];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] pTok = TextLoaderHelper.TokenizeAndEncode(problems[i], vocabulary, pLen);
            int[] sTok = TextLoaderHelper.TokenizeAndEncode(solutions[i], vocabulary, sLen);
            int pOff = i * pLen, sOff = i * sLen;
            for (int j = 0; j < pLen; j++) featuresData[pOff + j] = NumOps.FromDouble(pTok[j]);
            for (int j = 0; j < sLen; j++) labelsData[sOff + j] = NumOps.FromDouble(sTok[j]);
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, pLen });
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

    private string ResolveDataDir()
    {
        // Tar extracts to MATH/{train,test}/{subject}/NNNN.json
        string sub = Path.Combine(_dataPath, "MATH");
        return Directory.Exists(sub) ? sub : _dataPath;
    }
}
