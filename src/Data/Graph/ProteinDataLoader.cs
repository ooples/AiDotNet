using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Graph;

/// <summary>
/// Loads protein structure datasets as flattened feature/label tensors for graph-level classification.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Expects a directory structure with protein files in CSV or TSV format:
/// <code>
/// {DataPath}/
///   proteins/     (.csv files: one row per residue with features and contacts)
///   labels.csv    (protein_id, functional_class)
/// </code>
/// Features are flattened residue features Tensor[N, MaxResidues * FeatureDimension].
/// Labels are functional class index Tensor[N, 1].
/// </para>
/// </remarks>
public class ProteinDataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly ProteinDataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "Protein";
    public override string Description => "Protein structure graph classification";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.FeatureDimension;
    public override int OutputDimension => 1;

    public ProteinDataLoader(ProteinDataLoaderOptions? options = null)
    {
        _options = options ?? new ProteinDataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("proteins");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Look for CSV/TSV protein files
        string proteinDir = Path.Combine(_dataPath, "proteins");
        if (!Directory.Exists(proteinDir))
            proteinDir = _dataPath;

        if (!Directory.Exists(proteinDir))
            throw new DirectoryNotFoundException($"Protein data not found at {proteinDir}.");

        var csvFiles = Directory.GetFiles(proteinDir, "*.csv")
            .Concat(Directory.GetFiles(proteinDir, "*.tsv")).ToArray();
        Array.Sort(csvFiles, StringComparer.OrdinalIgnoreCase);

        // Read labels file if present
        var labelMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        string labelsFile = Path.Combine(_dataPath, "labels.csv");
        if (!File.Exists(labelsFile))
            labelsFile = Path.Combine(_dataPath, "labels.txt");

        if (File.Exists(labelsFile))
        {
            string[] labelLines = await FilePolyfill.ReadAllLinesAsync(labelsFile, cancellationToken);
            foreach (string line in labelLines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(new[] { ',', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && int.TryParse(parts[1].Trim(), out int cls))
                {
                    labelMap[parts[0].Trim()] = cls;
                }
            }
        }

        int totalSamples = csvFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int featDim = _options.FeatureDimension;
        var featuresData = new T[totalSamples * featDim];
        var labelsData = new T[totalSamples];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string[] lines = await FilePolyfill.ReadAllLinesAsync(csvFiles[i], cancellationToken);

            // Parse residue features (aggregate: mean pool over residues)
            var aggFeatures = new double[featDim];
            int residueCount = 0;
            foreach (string line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(new[] { ',', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                for (int f = 0; f < Math.Min(parts.Length, featDim); f++)
                {
                    if (double.TryParse(parts[f], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out double val))
                    {
                        aggFeatures[f] += val;
                    }
                }
                residueCount++;
            }

            int featOff = i * featDim;
            if (residueCount > 0)
            {
                for (int f = 0; f < featDim; f++)
                    featuresData[featOff + f] = NumOps.FromDouble(aggFeatures[f] / residueCount);
            }

            string proteinId = Path.GetFileNameWithoutExtension(csvFiles[i]);
            labelsData[i] = labelMap.TryGetValue(proteinId, out int label)
                ? NumOps.FromDouble(label % _options.NumClasses)
                : NumOps.FromDouble(-1); // -1 sentinel: no label found
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featDim });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, 1 });
        InitializeIndices(totalSamples);
    }

    protected override void UnloadDataCore()
    {
        LoadedFeatures = default; LoadedLabels = default; Indices = null; _sampleCount = 0;
    }

    protected override (Tensor<T> Features, Tensor<T> Labels) ExtractBatch(int[] indices)
    {
        var features = LoadedFeatures ?? throw new InvalidOperationException("Features not loaded.");
        var labels = LoadedLabels ?? throw new InvalidOperationException("Labels not loaded.");
        var nfs = (int[])features.Shape.ToArray().Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape.ToArray().Clone(); nls[0] = indices.Length;
        var bf = new Tensor<T>(nfs);
        var bl = new Tensor<T>(nls);
        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(features, bf, indices[i], i);
            TensorCopyHelper.CopySample(labels, bl, indices[i], i);
        }
        return (bf, bl);
    }

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
        return (
            CreateSplit(shuffled.Take(trainSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize).Take(valSize).ToArray()),
            CreateSplit(shuffled.Skip(trainSize + valSize).ToArray())
        );
    }

    private InMemoryDataLoader<T, Tensor<T>, Tensor<T>> CreateSplit(int[] indices)
    {
        var (bf, bl) = ExtractBatch(indices);
        return new InMemoryDataLoader<T, Tensor<T>, Tensor<T>>(bf, bl);
    }
}
