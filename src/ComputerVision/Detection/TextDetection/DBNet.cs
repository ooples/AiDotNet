using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.TextDetection;

/// <summary>
/// DBNet (Differentiable Binarization Network) text detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DBNet is a state-of-the-art text detector that uses
/// differentiable binarization to segment text regions. Unlike traditional methods
/// that use a fixed threshold, DBNet learns an adaptive threshold for each pixel,
/// making it more robust to varying text appearances.</para>
///
/// <para>Key features:
/// - Differentiable binarization for end-to-end training
/// - Adaptive thresholding per pixel
/// - Fast inference with single-pass architecture
/// - Works well for both regular and irregular text shapes
/// </para>
///
/// <para>Architecture, as in the paper and its reference implementation: a ResNet backbone; a feature
/// pyramid whose four levels are reduced to a common width by 1x1 lateral convolutions, merged top-down,
/// smoothed by 3x3 convolutions to a quarter of that width and upsampled to 1/4 resolution and
/// concatenated; then two identical heads - convolution, batch norm, ReLU, then two stride-2 transposed
/// convolutions back to full resolution - predicting the probability map and the threshold map.</para>
///
/// <para>Reference: Liao et al., "Real-time Scene Text Detection with Differentiable
/// Binarization", AAAI 2020</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Real-time Scene Text Detection with Differentiable Binarization",
    "https://arxiv.org/abs/1911.08947",
    Year = 2020,
    Authors = "Minghui Liao, Zhaoyi Wan, Cong Yao, Kai Chen, Xiang Bai")]
public partial class DBNet<T> : TextDetectorBase<T>
{
    private readonly DbFeaturePyramid<T> _pyramid;
    private readonly DbHead<T> _probabilityHead;
    private readonly DbHead<T> _thresholdHead;
    private readonly int _hiddenDim;
    private readonly double _k;

    /// <inheritdoc/>
    public override string Name => $"DBNet-{Options.Size}";

    /// <summary>
    /// Creates a new DBNet text detector.
    /// </summary>
    /// <param name="options">Text detection options.</param>
    /// <param name="k">Amplification factor for differentiable binarization (default 50).</param>
    public DBNet(TextDetectionOptions<T> options, double k = 50.0) : base(options)
    {
        _hiddenDim = GetHiddenDim(options.Size);
        _k = k;

        // ResNet backbone
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Feature pyramid (the paper's FPN, mmocr FPNC): the fused map has _hiddenDim channels at 1/4
        // resolution - 256 at the default size, as in the paper.
        _pyramid = new DbFeaturePyramid<T>(Backbone.OutputChannels, _hiddenDim);

        // Probability and threshold heads, each back to full input resolution.
        _probabilityHead = new DbHead<T>(_hiddenDim);
        _thresholdHead = new DbHead<T>(_hiddenDim);
    }

    /// <inheritdoc />
    /// <remarks>Also switches the heads' batch-norm layers.</remarks>
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
        _probabilityHead.SetTrainingMode(training);
        _thresholdHead.SetTrainingMode(training);
    }

    private static int GetHiddenDim(ModelSize size) => size switch
    {
        ModelSize.Nano => 64,
        ModelSize.Small => 128,
        ModelSize.Medium => 256,
        ModelSize.Large => 384,
        ModelSize.XLarge => 512,
        _ => 256
    };

    /// <inheritdoc/>
    public override TextDetectionResult<T> Detect(Tensor<T> image)
    {
        return Detect(image, NumOps.ToDouble(Options.ConfidenceThreshold));
    }

    /// <inheritdoc/>
    public override TextDetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold)
    {
        var startTime = DateTime.UtcNow;

        int originalHeight = image.Shape[2];
        int originalWidth = image.Shape[3];

        var input = Preprocess(image);
        var outputs = Forward(input);
        var textRegions = PostProcess(outputs, originalWidth, originalHeight, confidenceThreshold);

        return new TextDetectionResult<T>
        {
            TextRegions = textRegions,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight
        };
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        var fused = _pyramid.Forward(EnsureBackbone.ExtractFeatures(input));

        var probMap = _probabilityHead.Forward(fused);
        var threshMap = _thresholdHead.Forward(fused);

        // Apply differentiable binarization: DB = 1 / (1 + exp(-k * (P - T)))
        var binaryMap = ApplyDifferentiableBinarization(probMap, threshMap, _k);

        return new List<Tensor<T>> { probMap, threshMap, binaryMap };
    }

    /// <inheritdoc/>
    protected override List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold)
    {
        var probMap = outputs[0];
        var binaryMap = outputs[2];

        int mapH = binaryMap.Shape[2];
        int mapW = binaryMap.Shape[3];

        double scaleX = (double)imageWidth / mapW;
        double scaleY = (double)imageHeight / mapH;

        // Binarize the probability map
        double binThreshold = NumOps.ToDouble(Options.BinaryThreshold);
        var textMask = new bool[mapH, mapW];

        for (int h = 0; h < mapH; h++)
        {
            for (int w = 0; w < mapW; w++)
            {
                textMask[h, w] = NumOps.ToDouble(binaryMap[0, 0, h, w]) > binThreshold;
            }
        }

        // Find connected components
        var components = FindConnectedComponents(textMask, mapH, mapW);

        // Convert components to text regions
        var regions = new List<TextRegion<T>>();

        foreach (var component in components)
        {
            if (component.Count < 10)
                continue;

            // Compute bounding contour
            var contour = GetContour(component, textMask, mapH, mapW);

            if (contour.Count < 4)
                continue;

            // Compute average probability as confidence
            double avgProb = 0;
            foreach (var (h, w) in component)
            {
                avgProb += NumOps.ToDouble(probMap[0, 0, h, w]);
            }
            avgProb /= component.Count;

            if (avgProb < confidenceThreshold)
                continue;

            // Scale contour to original image coordinates
            var polygon = contour
                .Select(p => (X: p.W * scaleX, Y: p.H * scaleY))
                .ToList();

            // Simplify polygon
            polygon = SimplifyPolygon(polygon, Options.PolygonSimplificationEpsilon * Math.Max(scaleX, scaleY));

            if (polygon.Count >= 4)
            {
                var region = TextRegion<T>.FromPolygon(
                    polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                    NumOps.FromDouble(avgProb));

                region.RegionType = TextRegionType.Word;
                regions.Add(region);
            }
        }

        // Apply polygon NMS
        regions = ApplyPolygonNMS(regions, 0.3);

        // Limit to max detections
        if (regions.Count > Options.MaxDetections)
        {
            regions = regions
                .OrderByDescending(r => NumOps.ToDouble(r.Confidence))
                .Take(Options.MaxDetections)
                .ToList();
        }

        return regions;
    }

    internal static Tensor<T> ApplyDifferentiableBinarization(Tensor<T> prob, Tensor<T> thresh, double k)
    {
        // DB (Liao et al. 2020): B = 1 / (1 + exp(-k (P - T))). Engine ops, so the binarization step -
        // the whole point of DBNet - passes gradient to both the probability and threshold heads.
        var engine = AiDotNetEngine.Current;
        var scaled = engine.TensorMultiplyScalar(
            engine.TensorSubtract(prob, thresh), MathHelper.GetNumericOperations<T>().FromDouble(k));
        return engine.Sigmoid(scaled);
    }

    /// <inheritdoc/>
    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
        => _pyramid.ParameterCount + _probabilityHead.ParameterCount + _thresholdHead.ParameterCount;

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayWithCancellationAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            // Use Task.Run for net471 compatibility (ReadAllBytesAsync not available)
            data = await Task.Run(() => File.ReadAllBytes(pathOrUrl), cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x44424E54) // "DBNT" in ASCII
        {
            throw new InvalidDataException($"Invalid DBNet model file. Expected magic 0x44424E54, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 2)
        {
            throw new InvalidDataException(
                $"Unsupported DBNet model version: {version}. Version 2 is the paper architecture (feature " +
                "pyramid with two upsampling heads); version 1 files hold the earlier concatenation decoder, " +
                "whose layout no longer exists and cannot be loaded into it.");
        }

        string name = reader.ReadString();
        int hiddenDim = reader.ReadInt32();
        double k = reader.ReadDouble();

        if (name != Name)
        {
            throw new InvalidOperationException(
                $"DBNet configuration mismatch. Expected name={Name}, got name={name}");
        }

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"DBNet configuration mismatch. Expected hiddenDim={_hiddenDim}, got hiddenDim={hiddenDim}");
        }

        // Validate k parameter within tolerance for floating point comparison
        const double tolerance = 1e-12;
        if (Math.Abs(k - _k) > tolerance)
        {
            throw new InvalidOperationException(
                $"DBNet configuration mismatch. Expected k={_k}, got k={k}");
        }

        // Read component weights
        EnsureBackbone.ReadParameters(reader);
        _pyramid.ReadParameters(reader);
        _probabilityHead.ReadParameters(reader);
        _thresholdHead.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x44424E54); // "DBNT" in ASCII
        writer.Write(2); // Version 2: feature pyramid + upsampling heads
        writer.Write(Name);
        writer.Write(_hiddenDim);
        writer.Write(_k);

        // Write component weights
        EnsureBackbone.WriteParameters(writer);
        _pyramid.WriteParameters(writer);
        _probabilityHead.WriteParameters(writer);
        _thresholdHead.WriteParameters(writer);
    }

    private List<List<(int H, int W)>> FindConnectedComponents(bool[,] mask, int height, int width)
    {
        var components = new List<List<(int H, int W)>>();
        var visited = new bool[height, width];

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                if (mask[h, w] && !visited[h, w])
                {
                    var component = new List<(int H, int W)>();
                    FloodFill(mask, visited, h, w, height, width, component);
                    if (component.Count > 0)
                    {
                        components.Add(component);
                    }
                }
            }
        }

        return components;
    }

    private void FloodFill(
        bool[,] mask,
        bool[,] visited,
        int startH,
        int startW,
        int height,
        int width,
        List<(int H, int W)> component)
    {
        var stack = new Stack<(int H, int W)>();
        stack.Push((startH, startW));

        while (stack.Count > 0)
        {
            var (h, w) = stack.Pop();

            if (h < 0 || h >= height || w < 0 || w >= width)
                continue;

            if (visited[h, w] || !mask[h, w])
                continue;

            visited[h, w] = true;
            component.Add((h, w));

            stack.Push((h - 1, w));
            stack.Push((h + 1, w));
            stack.Push((h, w - 1));
            stack.Push((h, w + 1));
        }
    }

    private List<(int H, int W)> GetContour(
        List<(int H, int W)> component,
        bool[,] mask,
        int height,
        int width)
    {
        // Find boundary pixels (pixels with at least one non-text neighbor)
        var boundary = new HashSet<(int H, int W)>();

        foreach (var (h, w) in component)
        {
            bool isBoundary = false;

            // Check 4-connected neighbors
            if (h == 0 || !mask[h - 1, w]) isBoundary = true;
            if (h == height - 1 || !mask[h + 1, w]) isBoundary = true;
            if (w == 0 || !mask[h, w - 1]) isBoundary = true;
            if (w == width - 1 || !mask[h, w + 1]) isBoundary = true;

            if (isBoundary)
            {
                boundary.Add((h, w));
            }
        }

        // Order boundary points by angle from centroid
        double cx = boundary.Average(p => p.W);
        double cy = boundary.Average(p => p.H);

        return boundary
            .OrderBy(p => Math.Atan2(p.H - cy, p.W - cx))
            .ToList();
    }

    private List<TextRegion<T>> ApplyPolygonNMS(List<TextRegion<T>> regions, double iouThreshold)
    {
        if (regions.Count == 0)
            return regions;

        var sorted = regions.OrderByDescending(r => NumOps.ToDouble(r.Confidence)).ToList();
        var selected = new List<TextRegion<T>>();
        var used = new bool[sorted.Count];

        for (int i = 0; i < sorted.Count; i++)
        {
            if (used[i]) continue;

            selected.Add(sorted[i]);
            used[i] = true;

            for (int j = i + 1; j < sorted.Count; j++)
            {
                if (used[j]) continue;

                double iou = ComputeBoxIoU(sorted[i].Box, sorted[j].Box);
                if (iou > iouThreshold)
                {
                    used[j] = true;
                }
            }
        }

        return selected;
    }

    private double ComputeBoxIoU(BoundingBox<T> a, BoundingBox<T> b)
    {
        double ax1 = NumOps.ToDouble(a.X1);
        double ay1 = NumOps.ToDouble(a.Y1);
        double ax2 = NumOps.ToDouble(a.X2);
        double ay2 = NumOps.ToDouble(a.Y2);

        double bx1 = NumOps.ToDouble(b.X1);
        double by1 = NumOps.ToDouble(b.Y1);
        double bx2 = NumOps.ToDouble(b.X2);
        double by2 = NumOps.ToDouble(b.Y2);

        double intersectX1 = Math.Max(ax1, bx1);
        double intersectY1 = Math.Max(ay1, by1);
        double intersectX2 = Math.Min(ax2, bx2);
        double intersectY2 = Math.Min(ay2, by2);

        double intersectW = Math.Max(0, intersectX2 - intersectX1);
        double intersectH = Math.Max(0, intersectY2 - intersectY1);
        double intersect = intersectW * intersectH;

        double areaA = (ax2 - ax1) * (ay2 - ay1);
        double areaB = (bx2 - bx1) * (by2 - by1);
        double union = areaA + areaB - intersect;

        return union > 0 ? intersect / union : 0;
    }
}

/// <summary>
/// DBNet's feature pyramid (Liao et al. 2020; mmocr <c>FPNC</c>): 1x1 lateral convolutions to a common
/// width, a top-down merge, 3x3 smoothing to a quarter of that width per level, and all levels
/// upsampled to the finest level's resolution and concatenated.
/// </summary>
internal sealed class DbFeaturePyramid<T> : CvParameterModule<T>
{
    private readonly Conv2D<T>[] _lateral;
    private readonly Conv2D<T>[] _smooth;

    public DbFeaturePyramid(IReadOnlyList<int> stageChannels, int width)
    {
        if (width % 4 != 0)
        {
            throw new ArgumentException($"The pyramid width must be divisible by 4; got {width}.", nameof(width));
        }

        _lateral = stageChannels.Select(channels => new Conv2D<T>(channels, width, kernelSize: 1)).ToArray();
        _smooth = stageChannels.Select(_ => new Conv2D<T>(width, width / 4, kernelSize: 3, padding: 1)).ToArray();
    }

    public Tensor<T> Forward(List<Tensor<T>> features)
    {
        if (features.Count != _lateral.Length)
        {
            throw new ArgumentException(
                $"Expected {_lateral.Length} backbone stages, got {features.Count}.", nameof(features));
        }

        var engine = AiDotNetEngine.Current;
        int levels = features.Count;
        var merged = new Tensor<T>[levels];
        for (int i = levels - 1; i >= 0; i--)
        {
            var lateral = _lateral[i].Forward(features[i]);
            merged[i] = i == levels - 1
                ? lateral
                : engine.TensorAdd(lateral, CvTensorOps<T>.ResizeNearest(merged[i + 1], lateral.Shape[2], lateral.Shape[3]));
        }

        int height = merged[0].Shape[2], width = merged[0].Shape[3];
        var smoothed = new Tensor<T>[levels];
        for (int i = 0; i < levels; i++)
        {
            // Deepest level first, as the reference implementation concatenates them.
            var level = _smooth[levels - 1 - i].Forward(merged[levels - 1 - i]);
            smoothed[i] = i == levels - 1 ? level : CvTensorOps<T>.ResizeNearest(level, height, width);
        }

        return engine.TensorConcatenate(smoothed, 1);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => _lateral.Concat(_smooth);

    public void WriteParameters(BinaryWriter writer)
    {
        foreach (var conv in _lateral.Concat(_smooth))
        {
            conv.WriteParameters(writer);
        }
    }

    public void ReadParameters(BinaryReader reader)
    {
        foreach (var conv in _lateral.Concat(_smooth))
        {
            conv.ReadParameters(reader);
        }
    }
}

/// <summary>
/// One DBNet prediction head (probability or threshold): 3x3 convolution to a quarter of the width,
/// batch norm and ReLU, a stride-2 transposed convolution with batch norm and ReLU, and a stride-2
/// transposed convolution to one channel, then a sigmoid - from 1/4 resolution back to full.
/// </summary>
internal sealed class DbHead<T> : CvParameterModule<T>
{
    private readonly Conv2D<T> _conv;
    private readonly BatchNorm2D<T> _norm1;
    private readonly ConvTranspose2D<T> _up1;
    private readonly BatchNorm2D<T> _norm2;
    private readonly ConvTranspose2D<T> _up2;

    public DbHead(int width)
    {
        int inner = width / 4;
        _conv = new Conv2D<T>(width, inner, kernelSize: 3, padding: 1);
        _norm1 = new BatchNorm2D<T>(inner);
        _up1 = new ConvTranspose2D<T>(inner, inner, kernelSize: 2, stride: 2);
        _norm2 = new BatchNorm2D<T>(inner);
        _up2 = new ConvTranspose2D<T>(inner, 1, kernelSize: 2, stride: 2);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var h = engine.ReLU(_norm1.Forward(_conv.Forward(x)));
        h = engine.ReLU(_norm2.Forward(_up1.Forward(h)));
        return engine.Sigmoid(_up2.Forward(h));
    }

    public void SetTrainingMode(bool training)
    {
        _norm1.SetTrainingMode(training);
        _norm2.SetTrainingMode(training);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
        => new IParameterSource<T>?[] { _conv, _norm1, _up1, _norm2, _up2 };

    public void WriteParameters(BinaryWriter writer)
    {
        _conv.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _up1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
        _up2.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        _conv.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _up1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
        _up2.ReadParameters(reader);
    }
}
