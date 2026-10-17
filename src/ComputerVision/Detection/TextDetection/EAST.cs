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
/// EAST (Efficient and Accurate Scene Text) detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EAST is a fast and accurate text detector that directly
/// predicts word or text-line bounding boxes in one forward pass. It outputs a score map
/// showing where text is likely to be, plus geometry (box coordinates or rotated rectangles)
/// for each text region.</para>
///
/// <para>Key features:
/// - Single-shot detection (no region proposals)
/// - Supports both axis-aligned and rotated bounding boxes
/// - Fast inference suitable for real-time applications
/// - Works well for both horizontal and multi-oriented text
/// </para>
///
/// <para>Reference: Zhou et al., "EAST: An Efficient and Accurate Scene Text Detector", CVPR 2017</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("EAST: An Efficient and Accurate Scene Text Detector",
    "https://arxiv.org/abs/1704.03155",
    Year = 2017,
    Authors = "Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang")]
public partial class EAST<T> : TextDetectorBase<T>
{
    // Merging stages 2-4 (Zhou et al. 2017, Eq. 2): conv1x1 then conv3x3 over [unpool(h); f], 128, 64 and 32 channels.
    // One field per module: parameter discovery (training, serialization, counting) walks fields of parameter
    // sources, and does not look inside an array of tuples, so the merging branch held that way never trained.
    private readonly Conv2D<T> _reduce2, _reduce3, _reduce4, _fuse2, _fuse3, _fuse4;
    private readonly BatchNorm2D<T> _reduceNorm2, _reduceNorm3, _reduceNorm4, _fuseNorm2, _fuseNorm3, _fuseNorm4;

    private (Conv2D<T> Reduce, BatchNorm2D<T> ReduceNorm, Conv2D<T> Fuse, BatchNorm2D<T> FuseNorm)[] _stages => new[]
    {
        (_reduce2, _reduceNorm2, _fuse2, _fuseNorm2),
        (_reduce3, _reduceNorm3, _fuse3, _fuseNorm3),
        (_reduce4, _reduceNorm4, _fuse4, _fuseNorm4),
    };
    // g4 = conv3x3(h4), 32 channels (Eq. 1).
    private readonly Conv2D<T> _finalConv;
    private readonly BatchNorm2D<T> _finalNorm;
    private readonly Conv2D<T> _scoreHead;
    private readonly Conv2D<T> _geometryHead;
    private readonly bool _useRotatedBoxes;

    /// <summary>Output channels of merging stages 2, 3 and 4 (Fig. 3).</summary>
    private static readonly int[] StageChannels = { 128, 64, 32 };

    /// <inheritdoc/>
    public override string Name => "EAST";

    /// <summary>
    /// The minimum mean score-map value inside a detected box (after locality-aware NMS) for the box to be kept;
    /// the per-pixel threshold is the detection call's confidence threshold. The paper does not state either; 0.1 is
    /// the public reimplementation's box_thresh (argman/EAST eval.py).
    /// </summary>
    public double BoxScoreThreshold { get; set; } = 0.1;

    /// <summary>
    /// Creates a new EAST text detector: the paper's VGG16 stem, feature-merging branch and output layer.
    /// </summary>
    /// <param name="options">Text detection options.</param>
    /// <param name="useRotatedBoxes">Whether to predict rotated boxes (RBOX) or quadrilaterals (QUAD).</param>
    /// <remarks>
    /// <para>The stem is VGG16, the paper's commonly used base model (Tab. 2): the feature maps after pooling-2 to
    /// pooling-5 are f4 to f1 (Sec. 3.2). The merging branch unpools (bilinear x2) and concatenates as Eq. 1-2 state,
    /// with the channel counts of Fig. 3. The paper does not say how its convolutions are normalized; each is followed
    /// by batch normalization and ReLU, as in the public reimplementations (argman/EAST).</para>
    /// </remarks>
    public EAST(TextDetectionOptions<T> options, bool useRotatedBoxes = true) : base(options)
    {
        _useRotatedBoxes = useRotatedBoxes;

        Backbone = new VGG16Backbone<T>();
        var f = Backbone.OutputChannels;               // finest first: f4, f3, f2, f1
        // h1 = f1 (512 channels at 1/32); each stage concatenates the next finer stem map.
        _reduce2 = new Conv2D<T>(f[3] + f[2], StageChannels[0], kernelSize: 1);
        _reduceNorm2 = new BatchNorm2D<T>(StageChannels[0]);
        _fuse2 = new Conv2D<T>(StageChannels[0], StageChannels[0], kernelSize: 3, padding: 1);
        _fuseNorm2 = new BatchNorm2D<T>(StageChannels[0]);
        _reduce3 = new Conv2D<T>(StageChannels[0] + f[1], StageChannels[1], kernelSize: 1);
        _reduceNorm3 = new BatchNorm2D<T>(StageChannels[1]);
        _fuse3 = new Conv2D<T>(StageChannels[1], StageChannels[1], kernelSize: 3, padding: 1);
        _fuseNorm3 = new BatchNorm2D<T>(StageChannels[1]);
        _reduce4 = new Conv2D<T>(StageChannels[1] + f[0], StageChannels[2], kernelSize: 1);
        _reduceNorm4 = new BatchNorm2D<T>(StageChannels[2]);
        _fuse4 = new Conv2D<T>(StageChannels[2], StageChannels[2], kernelSize: 3, padding: 1);
        _fuseNorm4 = new BatchNorm2D<T>(StageChannels[2]);
        _finalConv = new Conv2D<T>(StageChannels[2], 32, kernelSize: 3, padding: 1);
        _finalNorm = new BatchNorm2D<T>(32);

        // Output layer: conv1x1 to the score map, and to 4 distances + 1 angle (RBOX) or 8 offsets (QUAD).
        _scoreHead = new Conv2D<T>(32, 1, kernelSize: 1);
        _geometryHead = new Conv2D<T>(32, useRotatedBoxes ? 5 : 8, kernelSize: 1);
    }

    /// <inheritdoc/>
    public override void SetTrainingMode(bool training)
    {
        base.SetTrainingMode(training);
        if (_finalNorm is null) return;
        foreach (var stage in _stages)
        {
            stage.ReduceNorm.SetTrainingMode(training);
            stage.FuseNorm.SetTrainingMode(training);
        }
        _finalNorm.SetTrainingMode(training);
    }

    /// <summary>
    /// The base preprocessing (resize, then pixels / 255), then the ImageNet normalization of the VGG16 stem's
    /// torchvision weights: (x - mean) / std per channel. The paper initializes the stem from ImageNet (Sec. 4.2) and does
    /// not state its input normalization; this is the domain those weights expect.
    /// </summary>
    protected override Tensor<T> Preprocess(Tensor<T> image)
    {
        var x = base.Preprocess(image);
        if (x.Shape[1] != 3) return x;
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(new[] { 0.485, 0.456, 0.406 }.Select(NumOps.FromDouble).ToArray()));
        var inverse = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(new[] { 1 / 0.229, 1 / 0.224, 1 / 0.225 }.Select(NumOps.FromDouble).ToArray()));
        var shape = x.Shape.ToArray();
        return Engine.TensorMultiply(Engine.TensorSubtract(x, Engine.TensorBroadcastTo(mean, shape)), Engine.TensorBroadcastTo(inverse, shape));
    }

    /// <summary>The score and geometry maps <c>[B, 1, H/4, W/4]</c> and <c>[B, 5 or 8, H/4, W/4]</c> of images
    /// <c>[B, 3, H, W]</c>.</summary>
    internal List<Tensor<T>> Maps(Tensor<T> images) => Forward(images);

    /// <summary>
    /// Loads a state dict named as the reference transcription names it (tools/reference-data/east_reference.py):
    /// <c>stem.features.{i}</c> for torchvision's VGG16, <c>merge.{k}.reduce</c>, <c>.reduce_norm</c>, <c>.fuse</c>,
    /// <c>.fuse_norm</c>, then <c>final</c>, <c>final_norm</c>, <c>score</c> and <c>geometry</c>.
    /// </summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read)
    {
        // The head convolutions size themselves on their first forward.
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
            Forward(new Tensor<T>(new[] { 1, 3, 32, 32 }));
        ((VGG16Backbone<T>)EnsureBackbone).LoadTorchWeights((name, shape) => read("stem." + name, shape));
        void Conv(Conv2D<T> conv, string name, int outputs, int inputs, int kernel)
            => conv.SetParameters(new Vector<T>(read(name + ".weight", new[] { outputs, inputs, kernel, kernel })
                .Concat(read(name + ".bias", new[] { outputs })).Select(NumOps.FromDouble).ToArray()));
        void Norm(BatchNorm2D<T> norm, string name, int channels)
        {
            var shape = new[] { channels };
            AiDotNet.TextToSpeech.CodecBased.TorchParameters.BatchNorm(norm.Layer, read(name + ".weight", shape), read(name + ".bias", shape),
                read(name + ".running_mean", shape), read(name + ".running_var", shape));
        }
        var f = EnsureBackbone.OutputChannels;
        int previous = f[f.Count - 1];
        for (int i = 0; i < _stages.Length; i++)
        {
            int skip = f[f.Count - 2 - i], channels = StageChannels[i];
            Conv(_stages[i].Reduce, $"merge.{i}.reduce", channels, previous + skip, 1);
            Norm(_stages[i].ReduceNorm, $"merge.{i}.reduce_norm", channels);
            Conv(_stages[i].Fuse, $"merge.{i}.fuse", channels, channels, 3);
            Norm(_stages[i].FuseNorm, $"merge.{i}.fuse_norm", channels);
            previous = channels;
        }
        Conv(_finalConv, "final", 32, previous, 3);
        Norm(_finalNorm, "final_norm", 32);
        Conv(_scoreHead, "score", 1, 32, 1);
        Conv(_geometryHead, "geometry", _useRotatedBoxes ? 5 : 8, 32, 1);
    }

    private Tensor<T> ConvNormRelu(Conv2D<T> conv, BatchNorm2D<T> norm, Tensor<T> x) => Engine.ReLU(norm.Forward(conv.Forward(x)));

    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        var features = EnsureBackbone.ExtractFeatures(input);     // f4 (1/4) .. f1 (1/32)
        var h = features[features.Count - 1];
        for (int i = 0; i < _stages.Length; i++)
        {
            var (reduce, reduceNorm, fuse, fuseNorm) = _stages[i];
            var skip = features[features.Count - 2 - i];
            var x = CvTensorOps<T>.ConcatChannels(BilinearUpsample(h, skip.Shape[2], skip.Shape[3]), skip);
            h = ConvNormRelu(fuse, fuseNorm, ConvNormRelu(reduce, reduceNorm, x));
        }
        var g = ConvNormRelu(_finalConv, _finalNorm, h);

        var score = ApplySigmoid(_scoreHead.Forward(g));
        var geometry = BoundGeometry(_geometryHead.Forward(g));
        return new List<Tensor<T>> { score, geometry };
    }

    /// <inheritdoc/>
    protected override List<TextRegion<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold)
    {
        var score = outputs[0];
        var geometry = outputs[1];

        int scoreH = score.Shape[2];
        int scoreW = score.Shape[3];

        double scaleX = (double)imageWidth / scoreW;
        double scaleY = (double)imageHeight / scoreH;

        var regions = new List<TextRegion<T>>();

        // Find text pixels and decode boxes
        for (int h = 0; h < scoreH; h++)
        {
            for (int w = 0; w < scoreW; w++)
            {
                double scoreVal = NumOps.ToDouble(score[0, 0, h, w]);

                if (scoreVal < confidenceThreshold)
                    continue;

                // Decode geometry
                var polygon = DecodeGeometry(geometry, h, w, scaleX, scaleY);

                if (polygon.Count >= 4)
                {
                    var region = TextRegion<T>.FromPolygon(
                        polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                        NumOps.FromDouble(scoreVal));

                    region.RegionType = TextRegionType.Word;

                    if (_useRotatedBoxes)
                    {
                        // Extract rotation angle
                        double angle = NumOps.ToDouble(geometry[0, 4, h, w]);
                        region.RotationAngle = angle * 180.0 / Math.PI;
                    }

                    regions.Add(region);
                }
            }
        }

        // Locality-aware NMS (Zhou et al. 2017, Algorithm 1), one of the paper's two contributions: the
        // per-pixel geometries arrive in row-major order, neighbours in a row that overlap are merged by
        // score-weighted averaging of their vertices, and only then does standard NMS run. Plain NMS over
        // thousands of per-pixel boxes, as before, keeps one pixel's box instead of the row's consensus.
        regions = ApplyTextNMS(LocalityAwareMerge(regions, 0.2), 0.2);

        // The merged score is a SUM (it orders the NMS above); report the box's mean score-map value, as the
        // reference does after LANMS, so confidences stay in [0, 1]. The reference filters that mean with its own,
        // lower threshold (box_thresh 0.1 against score_map_thresh 0.8 for pixels): the mean covers the whole box while
        // only its shrunk core is text in the score map, so a well-trained detector's box means sit near the core's
        // share of the box. Filtering them with the pixel threshold removed every box once training succeeded.
        foreach (var region in regions)
            region.Confidence = NumOps.FromDouble(MeanScoreInside(region, score, scaleX, scaleY));
        regions = regions.Where(r => NumOps.ToDouble(r.Confidence) > BoxScoreThreshold).ToList();

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

    /// <summary>
    /// The paper's RBOX geometry output (Zhou et al. 2017; reference model.py): the four edge distances are
    /// <c>sigmoid * text_scale</c> and the angle is <c>(sigmoid - 0.5) * pi / 2</c>. Distances are in map pixels,
    /// and the map side stands in for the 512-pixel text scale at stride 4. Without this bound the distances
    /// could go negative, and the IoU loss is undefined there. QUAD offsets stay linear, as in the paper.
    /// </summary>
    private Tensor<T> BoundGeometry(Tensor<T> raw)
    {
        if (!_useRotatedBoxes) return raw;
        int b = raw.Shape[0], h = raw.Shape[2], w = raw.Shape[3];
        var distances = Engine.TensorMultiplyScalar(
            Engine.Sigmoid(Engine.TensorSlice(raw, new[] { 0, 0, 0, 0 }, new[] { b, 4, h, w })), NumOps.FromDouble(Math.Max(h, w)));
        var angle = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.Sigmoid(Engine.TensorSlice(raw, new[] { 0, 4, 0, 0 }, new[] { b, 1, h, w })), NumOps.FromDouble(-0.5)),
            NumOps.FromDouble(Math.PI / 2));
        return Engine.TensorConcatenate(new[] { distances, angle }, 1);
    }

    /// <summary>The paper's optimizer (Zhou et al. 2017, Sec. 3.5): Adam starting at 1e-3, decayed to one tenth every
    /// 27300 minibatches, stopping at 1e-5.</summary>
    protected override AiDotNet.Interfaces.IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? CreateTrainingOptimizer()
        => PaperAdam(1e-3, new AiDotNet.LearningRateSchedulers.StepLRScheduler(1e-3, stepSize: 27300, gamma: 0.1, minLearningRate: 1e-5));

    /// <summary>
    /// EAST's loss (Zhou et al. 2017, Section 3.4): <c>L = L_s + lambda_g L_g</c> with lambda_g = 1. <c>L_s</c> is
    /// class-balanced cross-entropy on the score map. <c>L_g</c> is taken on positive pixels only: for RBOX it is
    /// <c>-log IoU + 10 (1 - cos(theta - theta*))</c>, and for QUAD it is smooth-L1 on the vertex offsets,
    /// normalised by 8 times the quad's shorter edge.
    /// </summary>
    protected override Tensor<T> TextDetectionLoss(List<Tensor<T>> outputs, TextDetectionTrainingBatch targets,
        int imageWidth, int imageHeight)
    {
        var score = outputs[0];
        var geometry = outputs[1];
        int batch = score.Shape[0], mapHeight = score.Shape[2], mapWidth = score.Shape[3];
        var scores = new List<double[][,]>();
        var geometries = new List<double[][,]>();
        var norms = new List<double[][,]>();
        for (int b = 0; b < batch; b++)
        {
            var (s, geometryTarget, n) = BuildTargets(targets[b], imageWidth, imageHeight, mapWidth, mapHeight, _useRotatedBoxes);
            scores.Add(new[] { s });
            geometries.Add(geometryTarget);
            norms.Add(new[] { n });
        }
        var y = MapTensor(scores);
        var g = MapTensor(geometries);

        // Class-balanced cross-entropy: beta = 1 - |Y*| / |Y|.
        double positives = 0;
        for (int i = 0; i < y.Length; i++) positives += NumOps.ToDouble(y[i]);
        double beta = 1.0 - (positives / y.Length);
        var p = Engine.TensorClamp(score, NumOps.FromDouble(1e-6), NumOps.FromDouble(1 - 1e-6));
        var ones = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(y, NumOps.Zero), NumOps.One);
        var positiveTerm = Engine.TensorMultiplyScalar(Engine.TensorMultiply(y, Engine.TensorLog(p)), NumOps.FromDouble(beta));
        var negativeTerm = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorSubtract(ones, y), Engine.TensorLog(Engine.TensorSubtract(ones, p))), NumOps.FromDouble(1 - beta));
        var scoreLoss = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorAdd(positiveTerm, negativeTerm), null), NumOps.FromDouble(-1.0 / y.Length));

        Tensor<T> Channel(Tensor<T> t, int c) => Engine.TensorSlice(t, new[] { 0, c, 0, 0 }, new[] { batch, 1, mapHeight, mapWidth });
        Tensor<T> geometryLoss;
        if (_useRotatedBoxes)
        {
            var (d0, d1, d2, d3, theta) = (Channel(geometry, 0), Channel(geometry, 1), Channel(geometry, 2), Channel(geometry, 3), Channel(geometry, 4));
            var (t0, t1, t2, t3, thetaT) = (Channel(g, 0), Channel(g, 1), Channel(g, 2), Channel(g, 3), Channel(g, 4));
            var predictedArea = Engine.TensorMultiply(Engine.TensorAdd(d0, d2), Engine.TensorAdd(d1, d3));
            var targetArea = Engine.TensorMultiply(Engine.TensorAdd(t0, t2), Engine.TensorAdd(t1, t3));
            var interWidth = Engine.TensorAdd(Engine.TensorMin(d1, t1), Engine.TensorMin(d3, t3));
            var interHeight = Engine.TensorAdd(Engine.TensorMin(d0, t0), Engine.TensorMin(d2, t2));
            var intersection = Engine.TensorMultiply(interWidth, interHeight);
            var union = Engine.TensorSubtract(Engine.TensorAdd(predictedArea, targetArea), intersection);
            var iouLoss = Engine.TensorMultiplyScalar(Engine.TensorLog(Engine.TensorDivide(
                Engine.TensorAddScalar(intersection, NumOps.One), Engine.TensorAddScalar(union, NumOps.One))), NumOps.FromDouble(-1.0));
            var angleLoss = Engine.TensorSubtract(ones, Engine.TensorCos(Engine.TensorSubtract(theta, thetaT)));
            geometryLoss = WeightedMean(Engine.TensorAdd(iouLoss, Engine.TensorMultiplyScalar(angleLoss, NumOps.FromDouble(10.0))), y);
        }
        else
        {
            var difference = Engine.TensorAbs(Engine.TensorSubtract(geometry, g));
            var clipped = Engine.TensorMin(difference, Engine.TensorAddScalar(Engine.TensorMultiplyScalar(difference, NumOps.Zero), NumOps.One));
            var smoothL1 = Engine.TensorAdd(Engine.TensorMultiplyScalar(Engine.TensorSquare(clipped), NumOps.FromDouble(0.5)), Engine.TensorSubtract(difference, clipped));
            var perPixel = Engine.ReduceSum(smoothL1, new[] { 1 }, keepDims: true);
            var weighted = Engine.TensorMultiply(perPixel, MapTensor(norms));
            geometryLoss = WeightedMean(weighted, y);
        }
        return Engine.TensorAdd(scoreLoss, geometryLoss);
    }

    /// <summary>
    /// EAST's targets for one image, in map pixels:
    /// <list type="bullet">
    /// <item>The score map is each quad offset inwards by 0.3 times its shortest edge. The paper moves each vertex by 0.3 r_i, where r_i is its shorter adjacent edge; for a rectangle the two are the same.</item>
    /// <item>For RBOX, a positive pixel's geometry is its distances to the top, right, bottom and left edges of
    /// the quad's minimum-area rectangle, plus that rectangle's angle in [-pi/4, pi/4).</item>
    /// <item>For QUAD, it is the offsets from the pixel centre to the four vertices, the first being the vertex
    /// with the smallest x + y.</item>
    /// </list>
    /// </summary>
    internal static (double[,] Score, double[][,] Geometry, double[,] Norm) BuildTargets(
        IReadOnlyList<TextPolygonTarget> polygons, int imageWidth, int imageHeight, int mapWidth, int mapHeight, bool rotatedBoxes)
    {
        int channels = rotatedBoxes ? 5 : 8;
        var score = new double[mapHeight, mapWidth];
        var norm = new double[mapHeight, mapWidth];
        var geometry = Enumerable.Range(0, channels).Select(_ => new double[mapHeight, mapWidth]).ToArray();
        foreach (var target in polygons)
            WritePolygonTargets(target, imageWidth, imageHeight, mapWidth, mapHeight, rotatedBoxes, score, geometry, norm);
        return (score, geometry, norm);
    }

    /// <summary>
    /// Writes one text polygon's score, geometry and normalisation targets into the maps: every pixel inside the
    /// polygon shrunk by 0.3 of its shortest edge is positive (Zhou et al. 2017, EAST §3.3.1).
    /// </summary>
    private static void WritePolygonTargets(
        TextPolygonTarget target, int imageWidth, int imageHeight, int mapWidth, int mapHeight, bool rotatedBoxes,
        double[,] score, double[][,] geometry, double[,] norm)
    {
        var polygon = ToMap(target, imageWidth, imageHeight, mapWidth, mapHeight);
        if (TextTargetGeometry.Area(polygon) < 1.0) return;
        var quad = polygon.Length == 4 ? polygon : TextTargetGeometry.MinAreaRectangle(polygon);
        double shortest = Enumerable.Range(0, 4).Min(i =>
            Math.Sqrt(Math.Pow(quad[(i + 1) % 4].X - quad[i].X, 2) + Math.Pow(quad[(i + 1) % 4].Y - quad[i].Y, 2)));
        var shrunk = TextTargetGeometry.Offset(quad, 0.3 * shortest);
        if (shrunk is null) return;

        var rect = TextTargetGeometry.MinAreaRectangle(quad);
        var (theta, width, height) = RotatedBoxFrame(rect);
        double cx = rect.Average(p => p.X), cy = rect.Average(p => p.Y);
        int start = TopLeftVertex(quad);

        double pixelNorm = 1.0 / (8.0 * Math.Max(shortest, 1e-6));
        var (x0, y0, x1, y1) = TextTargetGeometry.Bounds(shrunk, mapWidth, mapHeight);
        for (int py = y0; py <= y1; py++)
            for (int px = x0; px <= x1; px++)
            {
                double sx = px + 0.5, sy = py + 0.5;
                if (!TextTargetGeometry.Contains(shrunk, sx, sy)) continue;
                score[py, px] = 1.0;
                norm[py, px] = pixelNorm;
                if (rotatedBoxes)
                    WriteRotatedBoxGeometry(geometry, py, px, sx - cx, sy - cy, theta, width, height);
                else
                    WriteQuadGeometry(geometry, py, px, sx, sy, quad, start);
            }
    }

    /// <summary>
    /// The rotated box's angle, width and height, with the angle folded into [-pi/4, pi/4) by swapping width and
    /// height on each quarter turn, so every box has one canonical RBOX description.
    /// </summary>
    private static (double Theta, double Width, double Height) RotatedBoxFrame(IReadOnlyList<(double X, double Y)> rect)
    {
        double ex = rect[1].X - rect[0].X, ey = rect[1].Y - rect[0].Y;
        double theta = Math.Atan2(ey, ex);
        double width = Math.Sqrt((ex * ex) + (ey * ey));
        double height = Math.Sqrt(Math.Pow(rect[3].X - rect[0].X, 2) + Math.Pow(rect[3].Y - rect[0].Y, 2));
        while (theta >= Math.PI / 4) { theta -= Math.PI / 2; (width, height) = (height, width); }
        while (theta < -Math.PI / 4) { theta += Math.PI / 2; (width, height) = (height, width); }
        return (theta, width, height);
    }

    /// <summary>
    /// Index of the vertex with the smallest x + y, the QUAD target's starting (top-left) vertex.
    /// </summary>
    private static int TopLeftVertex(IReadOnlyList<(double X, double Y)> quad)
    {
        int start = 0;
        for (int i = 1; i < 4; i++)
        {
            if (quad[i].X + quad[i].Y < quad[start].X + quad[start].Y) start = i;
        }
        return start;
    }

    /// <summary>
    /// RBOX target at one pixel: distances to the top, right, bottom and left edges of the rotated box, then its
    /// angle. (<paramref name="dx"/>, <paramref name="dy"/>) is the pixel's offset from the box centre.
    /// </summary>
    private static void WriteRotatedBoxGeometry(
        double[][,] geometry, int py, int px, double dx, double dy, double theta, double width, double height)
    {
        double ux = Math.Cos(theta), uy = Math.Sin(theta);
        double pu = (dx * ux) + (dy * uy);
        double pv = (dx * -uy) + (dy * ux);
        geometry[0][py, px] = (height / 2) + pv; // top
        geometry[1][py, px] = (width / 2) - pu;  // right
        geometry[2][py, px] = (height / 2) - pv; // bottom
        geometry[3][py, px] = (width / 2) + pu;  // left
        geometry[4][py, px] = theta;
    }

    /// <summary>
    /// QUAD target at one pixel: the offset to each of the four vertices, starting from the top-left one.
    /// </summary>
    private static void WriteQuadGeometry(
        double[][,] geometry, int py, int px, double sx, double sy, IReadOnlyList<(double X, double Y)> quad, int start)
    {
        for (int v = 0; v < 4; v++)
        {
            var vertex = quad[(start + v) % 4];
            geometry[2 * v][py, px] = vertex.X - sx;
            geometry[(2 * v) + 1][py, px] = vertex.Y - sy;
        }
    }
    private List<(double X, double Y)> DecodeGeometry(
        Tensor<T> geometry,
        int h,
        int w,
        double scaleX,
        double scaleY)
    {
        double centerX = (w + 0.5) * scaleX;
        double centerY = (h + 0.5) * scaleY;

        if (_useRotatedBoxes)
        {
            // RBOX format: 4 distances from pixel to box edges + angle
            double d0 = NumOps.ToDouble(geometry[0, 0, h, w]) * scaleY; // top
            double d1 = NumOps.ToDouble(geometry[0, 1, h, w]) * scaleX; // right
            double d2 = NumOps.ToDouble(geometry[0, 2, h, w]) * scaleY; // bottom
            double d3 = NumOps.ToDouble(geometry[0, 3, h, w]) * scaleX; // left
            double angle = NumOps.ToDouble(geometry[0, 4, h, w]);

            // Compute rotated rectangle corners
            double boxHeight = d0 + d2;
            double boxWidth = d1 + d3;

            double cos = Math.Cos(angle);
            double sin = Math.Sin(angle);

            // Box center offset from pixel
            double offsetX = (d1 - d3) / 2;
            double offsetY = (d2 - d0) / 2;

            double cx = centerX + offsetX * cos - offsetY * sin;
            double cy = centerY + offsetX * sin + offsetY * cos;

            // Compute 4 corners
            double hw = boxWidth / 2;
            double hh = boxHeight / 2;

            return new List<(double X, double Y)>
            {
                (cx - hw * cos + hh * sin, cy - hw * sin - hh * cos), // top-left
                (cx + hw * cos + hh * sin, cy + hw * sin - hh * cos), // top-right
                (cx + hw * cos - hh * sin, cy + hw * sin + hh * cos), // bottom-right
                (cx - hw * cos - hh * sin, cy - hw * sin + hh * cos)  // bottom-left
            };
        }
        else
        {
            // QUAD format: 8 offsets to 4 corners
            var points = new List<(double X, double Y)>();
            for (int i = 0; i < 4; i++)
            {
                double offsetX = NumOps.ToDouble(geometry[0, i * 2, h, w]) * scaleX;
                double offsetY = NumOps.ToDouble(geometry[0, i * 2 + 1, h, w]) * scaleY;
                points.Add((centerX + offsetX, centerY + offsetY));
            }
            return points;
        }
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
        => _stages.Sum(s => s.Reduce.GetParameterCount() + s.ReduceNorm.GetParameterCount() + s.Fuse.GetParameterCount() + s.FuseNorm.GetParameterCount())
           + _finalConv.GetParameterCount() + _finalNorm.GetParameterCount() + _scoreHead.GetParameterCount() + _geometryHead.GetParameterCount();

    private IEnumerable<(Action<BinaryWriter> Write, Action<BinaryReader> Read)> HeadParts()
    {
        foreach (var (reduce, reduceNorm, fuse, fuseNorm) in _stages)
        {
            yield return (reduce.WriteParameters, reduce.ReadParameters);
            yield return (reduceNorm.WriteParameters, reduceNorm.ReadParameters);
            yield return (fuse.WriteParameters, fuse.ReadParameters);
            yield return (fuseNorm.WriteParameters, fuseNorm.ReadParameters);
        }
        yield return (_finalConv.WriteParameters, _finalConv.ReadParameters);
        yield return (_finalNorm.WriteParameters, _finalNorm.ReadParameters);
        yield return (_scoreHead.WriteParameters, _scoreHead.ReadParameters);
        yield return (_geometryHead.WriteParameters, _geometryHead.ReadParameters);
    }

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

        int magic = reader.ReadInt32();
        if (magic != 0x45415354) // "EAST" in ASCII
            throw new InvalidDataException($"Invalid EAST model file. Expected magic 0x45415354, got 0x{magic:X8}");
        int version = reader.ReadInt32();
        if (version != 2)
            throw new InvalidDataException($"Unsupported EAST model version: {version} (this build reads version 2, the VGG16 network).");
        bool useRotatedBoxes = reader.ReadBoolean();
        if (useRotatedBoxes != _useRotatedBoxes)
            throw new InvalidOperationException(
                $"EAST configuration mismatch. Expected useRotatedBoxes={_useRotatedBoxes}, got {useRotatedBoxes}.");

        EnsureBackbone.ReadParameters(reader);
        foreach (var part in HeadParts()) part.Read(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);
        writer.Write(0x45415354); // "EAST" in ASCII
        writer.Write(2);          // version 2: the VGG16 network
        writer.Write(_useRotatedBoxes);
        EnsureBackbone.WriteParameters(writer);
        foreach (var part in HeadParts()) part.Write(writer);
    }

    /// <summary>
    /// Elementwise Sigmoid, delegated to the engine.
    /// </summary>
    /// <remarks>
    /// This was a scalar loop that read each element out to <c>double</c> and wrote a fresh
    /// tensor. Arithmetically identical, but it severed the autodiff tape: the gradient chain
    /// stopped here, so every trainable layer UPSTREAM of this call received no gradient and
    /// silently never trained. The engine op records itself on the tape.
    /// </remarks>
    private Tensor<T> ApplySigmoid(Tensor<T> x) => Engine.Sigmoid(x);

    private Tensor<T> BilinearUpsample(Tensor<T> x, int targetH, int targetW)
        // Asymmetric bilinear (src = dst * in / out, no half-pixel offset), as the loop it replaces.
        => CvTensorOps<T>.ResizeBilinearAsymmetric(x, targetH, targetW);

    // Merges consecutive (row-major) regions whose boxes overlap by more than the threshold: vertices are
    // averaged by score, and the merged score is the sum of the parts.
    private List<TextRegion<T>> LocalityAwareMerge(List<TextRegion<T>> regions, double iouThreshold)
    {
        var merged = new List<TextRegion<T>>();
        List<(double X, double Y)>? polygon = null;
        double polygonScore = 0, angleSum = 0;
        // Unweighted fallbacks for a merge whose scores sum to zero (only reachable with a 0.0 confidence
        // threshold): a score-weighted average there divides by zero and yields non-finite vertices and angle.
        double angleUnweightedSum = 0;
        int parts = 0;

        void Flush()
        {
            if (polygon is null) return;
            var region = TextRegion<T>.FromPolygon(
                polygon.Select(p => (NumOps.FromDouble(p.X), NumOps.FromDouble(p.Y))).ToList(),
                NumOps.FromDouble(polygonScore));
            region.RegionType = TextRegionType.Word;
            if (_useRotatedBoxes)
                region.RotationAngle = polygonScore > 0 ? angleSum / polygonScore : angleUnweightedSum / parts;
            merged.Add(region);
        }

        foreach (var next in regions)
        {
            double s = NumOps.ToDouble(next.Confidence);
            var points = next.Polygon?.Select(v => (X: NumOps.ToDouble(v.X), Y: NumOps.ToDouble(v.Y))).ToList();
            if (points is null || points.Count == 0) continue;

            // Overlap of the quadrilaterals themselves, as LANMS does: their axis-aligned boxes overlap for rotated
            // neighbouring words whose quadrilaterals are disjoint, which merged them.
            if (polygon is not null && polygon.Count == points.Count
                && Metrics.TextDetectionMetrics<double>.PolygonIoU(polygon, points) > iouThreshold)
            {
                double total = polygonScore + s;
                if (total > 0)
                {
                    for (int k = 0; k < polygon.Count; k++)
                        polygon[k] = ((polygon[k].X * polygonScore + points[k].X * s) / total,
                                      (polygon[k].Y * polygonScore + points[k].Y * s) / total);
                }
                else
                {
                    // Every part so far scored zero: average the vertices with equal weight per part.
                    for (int k = 0; k < polygon.Count; k++)
                        polygon[k] = ((polygon[k].X * parts + points[k].X) / (parts + 1),
                                      (polygon[k].Y * parts + points[k].Y) / (parts + 1));
                }
                polygonScore = total;
                angleSum += s * next.RotationAngle;
                angleUnweightedSum += next.RotationAngle;
                parts++;
                continue;
            }

            Flush();
            polygon = points;
            polygonScore = s;
            angleSum = s * next.RotationAngle;
            angleUnweightedSum = next.RotationAngle;
            parts = 1;
        }

        Flush();
        return merged;
    }

    // The decoded quadrilateral in doubles, or null when a region carries no polygon.
    private List<(double X, double Y)>? PolygonOf(TextRegion<T> region)
    {
        var polygon = region.Polygon?.Select(v => (X: NumOps.ToDouble(v.X), Y: NumOps.ToDouble(v.Y))).ToList();
        return polygon is { Count: >= 3 } ? polygon : null;
    }

    // Polygon IoU when both regions carry their quadrilateral (always, for EAST's own output); box IoU otherwise.
    private double RegionIoU(TextRegion<T> a, TextRegion<T> b)
        => PolygonOf(a) is { } pa && PolygonOf(b) is { } pb
            ? Metrics.TextDetectionMetrics<double>.PolygonIoU(pa, pb)
            : ComputeBoxIoU(a.Box, b.Box);

    // Even-odd ray casting: whether a point lies inside a simple polygon.
    private static bool Contains(List<(double X, double Y)> polygon, double x, double y)
    {
        bool inside = false;
        for (int i = 0, j = polygon.Count - 1; i < polygon.Count; j = i++)
        {
            var (xi, yi) = polygon[i];
            var (xj, yj) = polygon[j];
            if ((yi > y) != (yj > y) && x < ((xj - xi) * (y - yi) / (yj - yi)) + xi)
                inside = !inside;
        }
        return inside;
    }

    // Mean score-map value over the cells whose centres fall inside the region's quadrilateral (its box when it has
    // none): averaging the whole axis-aligned box of a rotated word diluted the score with background corners.
    private double MeanScoreInside(TextRegion<T> region, Tensor<T> score, double scaleX, double scaleY)
    {
        var quadrilateral = PolygonOf(region);
        var (left, top, right, bottom) = region.Box.ToXYXY();
        double sum = 0;
        int count = 0;
        // Only cells whose centre (i + 0.5) * scale can lie in [low, high] are visited, instead of the whole map
        // for every region (up to 1000 regions x 6,400 cells at the default size). The inclusive centre test
        // below stays the source of truth; the bounds are widened by one cell so rounding cannot drop a cell.
        int hMin = Math.Max(0, (int)Math.Floor(top / scaleY - 0.5) - 1);
        int hMax = Math.Min(score.Shape[2] - 1, (int)Math.Ceiling(bottom / scaleY - 0.5) + 1);
        int wMin = Math.Max(0, (int)Math.Floor(left / scaleX - 0.5) - 1);
        int wMax = Math.Min(score.Shape[3] - 1, (int)Math.Ceiling(right / scaleX - 0.5) + 1);
        for (int h = hMin; h <= hMax; h++)
            for (int w = wMin; w <= wMax; w++)
            {
                double cx = (w + 0.5) * scaleX, cy = (h + 0.5) * scaleY;
                if (cx < left || cx > right || cy < top || cy > bottom) continue;
                if (quadrilateral is not null && !Contains(quadrilateral, cx, cy)) continue;
                sum += NumOps.ToDouble(score[0, 0, h, w]);
                count++;
            }
        return count > 0 ? sum / count : 0.0;
    }

    private List<TextRegion<T>> ApplyTextNMS(List<TextRegion<T>> regions, double iouThreshold)
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

                double iou = RegionIoU(sorted[i], sorted[j]);
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
