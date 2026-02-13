using AiDotNet.Data.Loaders;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;

namespace AiDotNet.Data.Video;

/// <summary>
/// Loads videos represented as directories of sequentially numbered image frames.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// This loader expects videos stored as frame directories (the standard preprocessing format
/// for ML video datasets like UCF-101, Kinetics, HMDB51, etc.). Each video is a directory
/// containing sequentially numbered image frames (BMP, PPM, or PGM format).
/// </para>
/// <para>
/// Expected structure:
/// <code>
/// root/
///   class1/
///     video1/
///       frame_001.bmp
///       frame_002.bmp
///       ...
///     video2/
///       frame_001.bmp
///       ...
///   class2/
///     video3/
///       ...
/// </code>
/// </para>
/// <para>
/// Frames are sampled uniformly across the video duration to produce a fixed number of frames
/// per video. Output tensor shape is [N, FramesPerVideo, Height, Width, Channels].
/// </para>
/// </remarks>
public class VideoFrameDataset<T> : InputOutputDataLoaderBase<T, Tensor<T>, Tensor<T>>
{
    private readonly VideoFrameDatasetOptions _options;
    private int _sampleCount;
    private int _featureCount;
    private int _numClasses;
    private List<string> _classNames = new List<string>();

    /// <inheritdoc/>
    public override string Name => "VideoFrame";

    /// <inheritdoc/>
    public override string Description => $"Video frame dataset from {_options.RootDirectory}";

    /// <inheritdoc/>
    public override int TotalCount => _sampleCount;

    /// <inheritdoc/>
    public override int FeatureCount => _featureCount;

    /// <inheritdoc/>
    public override int OutputDimension => _numClasses;

    /// <summary>
    /// Gets the class names discovered from directory names.
    /// </summary>
    public IReadOnlyList<string> ClassNames => _classNames;

    /// <summary>
    /// Creates a new VideoFrameDataset with the specified options.
    /// </summary>
    public VideoFrameDataset(VideoFrameDatasetOptions options)
    {
        Guard.NotNull(options);
        _options = options;

        if (string.IsNullOrWhiteSpace(options.RootDirectory))
        {
            throw new ArgumentException("Root directory cannot be empty.", nameof(options));
        }

        _featureCount = options.FramesPerVideo * options.FrameWidth * options.FrameHeight * options.Channels;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        string rootDir = _options.RootDirectory;
        if (!Directory.Exists(rootDir))
        {
            throw new DirectoryNotFoundException($"Video folder root directory not found: {rootDir}");
        }

        var frameExtensions = new HashSet<string>(
            _options.FrameExtensions.Select(e => e.StartsWith(".", StringComparison.Ordinal) ? e : "." + e),
            StringComparer.OrdinalIgnoreCase);

        // Discover videos: each video is a subdirectory containing frame images
        // Structure: root/class/video_dir/frame_*.ext
        List<(string VideoDir, int ClassIndex)> videos;

        if (_options.UseDirectoryLabels)
        {
            var classDirs = Directory.GetDirectories(rootDir)
                .OrderBy(d => Path.GetFileName(d), StringComparer.OrdinalIgnoreCase)
                .ToList();

            _classNames = classDirs
                .Select(d => Path.GetFileName(d) ?? string.Empty)
                .Where(n => n.Length > 0)
                .ToList();
            _numClasses = _classNames.Count;

            if (_numClasses == 0)
                throw new InvalidOperationException($"No class subdirectories found in {rootDir}.");

            videos = new List<(string, int)>();
            for (int classIdx = 0; classIdx < classDirs.Count; classIdx++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                foreach (string videoDir in Directory.GetDirectories(classDirs[classIdx]))
                {
                    // Verify the directory contains frame images
                    var framePaths = GetSortedFramePaths(videoDir, frameExtensions);
                    if (framePaths.Length > 0)
                    {
                        videos.Add((videoDir, classIdx));
                    }
                }
            }
        }
        else
        {
            _classNames = new List<string> { "default" };
            _numClasses = 1;

            videos = new List<(string, int)>();
            foreach (string videoDir in Directory.GetDirectories(rootDir))
            {
                var dirFrames = GetSortedFramePaths(videoDir, frameExtensions);
                if (dirFrames.Length > 0)
                {
                    videos.Add((videoDir, 0));
                }
            }
        }

        if (_options.MaxSamples.HasValue && _options.MaxSamples.Value < videos.Count)
        {
            var random = _options.RandomSeed.HasValue
                ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
                : RandomHelper.CreateSecureRandom();
            videos = videos.OrderBy(_ => random.Next()).Take(_options.MaxSamples.Value).ToList();
        }

        _sampleCount = videos.Count;
        if (_sampleCount == 0)
        {
            throw new InvalidOperationException("No video frame directories found.");
        }

        int frames = _options.FramesPerVideo;
        int h = _options.FrameHeight;
        int w = _options.FrameWidth;
        int c = _options.Channels;
        int pixelsPerFrame = h * w * c;

        var featuresData = new T[_sampleCount * frames * pixelsPerFrame];
        var labelsData = new T[_sampleCount * _numClasses];

        for (int i = 0; i < _sampleCount; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var (videoDir, classIndex) = videos[i];
            var framePaths = GetSortedFramePaths(videoDir, frameExtensions);

            // Sample frames uniformly across the video
            int totalAvailableFrames = framePaths.Length;
            for (int f = 0; f < frames; f++)
            {
                // Uniform sampling: pick frame index proportional to position
                int frameIdx = totalAvailableFrames <= frames
                    ? Math.Min(f, totalAvailableFrames - 1)
                    : (int)((long)f * (totalAvailableFrames - 1) / (frames - 1));

                T[] framePixels = LoadFramePixels(framePaths[frameIdx], w, h, c);
                int dstOffset = (i * frames + f) * pixelsPerFrame;
                Array.Copy(framePixels, 0, featuresData, dstOffset, pixelsPerFrame);
            }

            labelsData[i * _numClasses + classIndex] = NumOps.One;
        }

        LoadedFeatures = new Tensor<T>(featuresData, new[] { _sampleCount, frames, h, w, c });
        LoadedLabels = new Tensor<T>(labelsData, new[] { _sampleCount, _numClasses });
        InitializeIndices(_sampleCount);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        LoadedFeatures = default;
        LoadedLabels = default;
        Indices = null;
        _sampleCount = 0;
        _classNames = new List<string>();
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
        double trainRatio = 0.7,
        double validationRatio = 0.15,
        int? seed = null)
    {
        EnsureLoaded();
        ValidateSplitRatios(trainRatio, validationRatio);

        var (trainSize, valSize, _) = ComputeSplitSizes(_sampleCount, trainRatio, validationRatio);
        var random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
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

    private static string[] GetSortedFramePaths(string videoDir, HashSet<string> extensions)
    {
        if (!Directory.Exists(videoDir)) return Array.Empty<string>();

        var frames = Directory.GetFiles(videoDir)
            .Where(f => extensions.Contains(Path.GetExtension(f)))
            .OrderBy(f => f, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        return frames;
    }

    private T[] LoadFramePixels(string framePath, int targetWidth, int targetHeight, int targetChannels)
    {
        // Load image using ImageHelper which properly decodes BMP, PPM, PGM
        // ImageHelper returns [1, C, H, W] in CHW format
        Tensor<T> imageTensor = ImageHelper<T>.LoadImage(framePath, _options.NormalizePixels);

        int srcChannels = imageTensor.Shape[1];
        int srcHeight = imageTensor.Shape[2];
        int srcWidth = imageTensor.Shape[3];
        var srcSpan = imageTensor.AsSpan();

        int totalPixels = targetHeight * targetWidth * targetChannels;
        var pixels = new T[totalPixels];

        // Bilinear resize from source to target, converting CHW -> HWC
        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                double srcY = (double)y * (srcHeight - 1) / Math.Max(1, targetHeight - 1);
                double srcX = (double)x * (srcWidth - 1) / Math.Max(1, targetWidth - 1);

                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, srcHeight - 1);
                int x0 = (int)Math.Floor(srcX);
                int x1 = Math.Min(x0 + 1, srcWidth - 1);

                double dy = srcY - y0;
                double dx = srcX - x0;

                for (int ch = 0; ch < targetChannels; ch++)
                {
                    int srcCh = ch < srcChannels ? ch : (srcChannels == 1 ? 0 : ch);
                    int chOffset = srcCh * srcHeight * srcWidth;

                    double v00 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x0]);
                    double v01 = NumOps.ToDouble(srcSpan[chOffset + y0 * srcWidth + x1]);
                    double v10 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x0]);
                    double v11 = NumOps.ToDouble(srcSpan[chOffset + y1 * srcWidth + x1]);

                    double value = v00 * (1 - dx) * (1 - dy) +
                                   v01 * dx * (1 - dy) +
                                   v10 * (1 - dx) * dy +
                                   v11 * dx * dy;

                    pixels[(y * targetWidth + x) * targetChannels + ch] = NumOps.FromDouble(value);
                }
            }
        }

        return pixels;
    }

    private static Tensor<T> ExtractTensorBatch(Tensor<T> source, int[] indices)
    {
        var newShape = (int[])source.Shape.Clone();
        newShape[0] = indices.Length;
        var result = new Tensor<T>(newShape);

        for (int i = 0; i < indices.Length; i++)
        {
            TensorCopyHelper.CopySample(source, result, indices[i], i);
        }

        return result;
    }
}
