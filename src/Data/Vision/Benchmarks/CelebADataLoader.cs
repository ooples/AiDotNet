using AiDotNet.Data.Geometry;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Data.Vision.Benchmarks;

/// <summary>
/// Loads the CelebA face attributes dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// CelebA expects:
/// <code>
/// {DataPath}/
///   img_align_celeba/   (.jpg face images)
///   list_attr_celeba.txt  (attributes file: image_name attr1 attr2 ... attr40)
///   list_eval_partition.txt  (split file: image_name 0=train/1=val/2=test)
/// </code>
/// Features are flattened image pixels Tensor[N, H * W * 3].
/// Labels are binary attributes Tensor[N, 40].
/// </para>
/// </remarks>
public class CelebADataLoader<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly CelebADataLoaderOptions _options;
    private readonly string _dataPath;
    private int _sampleCount;

    public override string Name => "CelebA";
    public override string Description => "CelebA face attributes (40 binary attributes)";
    public override int TotalCount => _sampleCount;
    public override int FeatureCount => _options.ImageWidth * _options.ImageHeight * 3;
    public override int OutputDimension => _options.NumAttributes;

    public CelebADataLoader(CelebADataLoaderOptions? options = null)
    {
        _options = options ?? new CelebADataLoaderOptions();
        _dataPath = _options.DataPath ?? DatasetDownloader.GetDefaultDataPath("celeba");
    }

    protected override async Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string imgDir = Path.Combine(_dataPath, "img_align_celeba");
        if (!Directory.Exists(imgDir))
            imgDir = _dataPath;

        if (!Directory.Exists(imgDir))
            throw new DirectoryNotFoundException($"CelebA data not found at {imgDir}.");

        // Read partition file to determine split
        int targetPartition = _options.Split == DatasetSplit.Test ? 2
            : _options.Split == DatasetSplit.Validation ? 1
            : 0;

        var splitImages = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        string partFile = Path.Combine(_dataPath, "list_eval_partition.txt");
        if (File.Exists(partFile))
        {
            string[] partLines = await FilePolyfill.ReadAllLinesAsync(partFile, cancellationToken);
            foreach (string line in partLines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 2 && int.TryParse(parts[1].Trim(), out int part) && part == targetPartition)
                {
                    splitImages.Add(parts[0].Trim());
                }
            }
        }

        // Read attributes file
        var attrMap = new Dictionary<string, int[]>(StringComparer.OrdinalIgnoreCase);
        string attrFile = Path.Combine(_dataPath, "list_attr_celeba.txt");
        if (File.Exists(attrFile))
        {
            string[] attrLines = await FilePolyfill.ReadAllLinesAsync(attrFile, cancellationToken);
            // First line is count, second is header
            for (int lineIdx = 2; lineIdx < attrLines.Length; lineIdx++)
            {
                string line = attrLines[lineIdx];
                if (string.IsNullOrWhiteSpace(line)) continue;
                var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2) continue;

                string imgName = parts[0].Trim();
                int attrCount = Math.Min(parts.Length - 1, _options.NumAttributes);
                var attrs = new int[_options.NumAttributes];
                for (int a = 0; a < attrCount; a++)
                {
                    if (int.TryParse(parts[a + 1].Trim(), out int val))
                        attrs[a] = val > 0 ? 1 : 0; // CelebA uses -1/1, we convert to 0/1
                }
                attrMap[imgName] = attrs;
            }
        }

        // Get image files (filtered by split if partition file exists)
        var imageFiles = Directory.GetFiles(imgDir, "*.jpg")
            .Concat(Directory.GetFiles(imgDir, "*.png")).ToArray();
        Array.Sort(imageFiles, StringComparer.OrdinalIgnoreCase);

        if (splitImages.Count > 0)
        {
            imageFiles = imageFiles
                .Where(f => splitImages.Contains(Path.GetFileName(f)))
                .ToArray();
        }

        int totalSamples = imageFiles.Length;
        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < totalSamples)
            totalSamples = _options.MaxSamples.Value;

        _sampleCount = totalSamples;
        int featureSize = _options.ImageWidth * _options.ImageHeight * 3;
        int numAttrs = _options.NumAttributes;
        var featuresData = new T[totalSamples * featureSize];
        var labelsData = new T[totalSamples * numAttrs];

        for (int i = 0; i < totalSamples; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var pixels = VisionLoaderHelper.LoadAndResizeImage<T>(imageFiles[i], _options.ImageHeight, _options.ImageWidth, 3, _options.Normalize);
            int featOff = i * featureSize;
            int copyLen = Math.Min(pixels.Length, featureSize);
            Array.Copy(pixels, 0, featuresData, featOff, copyLen);

            // Labels from attributes map
            int lblOff = i * numAttrs;
            string imgName = Path.GetFileName(imageFiles[i]);
            if (attrMap.TryGetValue(imgName, out int[]? attrs))
            {
                for (int a = 0; a < numAttrs; a++)
                    labelsData[lblOff + a] = NumOps.FromDouble(attrs[a]);
            }
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { totalSamples, featureSize });
        LoadedLabels = new Tensor<T>(labelsData, new[] { totalSamples, numAttrs });
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
        var nfs = (int[])features.Shape._dims.Clone(); nfs[0] = indices.Length;
        var nls = (int[])labels.Shape._dims.Clone(); nls[0] = indices.Length;
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
