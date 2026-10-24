using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// SAM 2.1: Segment Anything Model 2.1, Meta's refreshed SAM 2 for images and videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SAM 2.1 cuts out an object from a few clicks, a box or a rough mask, in one image or through a
/// video. Call <see cref="SetImage"/> and then <see cref="SegmentFromPoints"/> (or box, mask) for an image, or
/// <see cref="TrackObject"/> for a video.
/// </para>
/// <para>
/// SAM 2.1 is SAM 2's network (Ravi et al. 2024; facebookresearch/sam2's <c>sam2.1_hiera_*</c> configs) with the two
/// changes its release made: the object pointers carry a projected temporal encoding, and the memories of frames predicted
/// to be occluded get a learned embedding. It shares the network, tracking and training with
/// <see cref="Video.Segmentation.SAM2{T}"/>, which match the official code at 1e-9: images go through the official image
/// predictor (resized to the encoder's square, points and boxes scaled to it, masks resized back) and the automatic mask
/// generator; videos through the video predictor. Released checkpoints load with <see cref="LoadPretrainedWeights"/>.
/// </para>
/// <para>
/// <b>Reference:</b> Ravi et al., "SAM 2: Segment Anything in Images and Videos", Meta AI, 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.BinaryClassification,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 1);
/// var sam = new SAM21&lt;float&gt;(architecture, options: new SAM21Options { ModelSize = SAM21ModelSize.Tiny });
/// sam.LoadPretrainedWeights("sam2.1-hiera-tiny/model.safetensors");
/// sam.SetImage(image);
/// var result = sam.SegmentFromPoints(points, labels);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Tracking)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SAM 2: Segment Anything in Images and Videos", "https://arxiv.org/abs/2408.00714", Year = 2024, Authors = "Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, Christoph Feichtenhofer")]
[PaperOptimizer(OptimizerKind.AdamW, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.1,
                ReferenceBatchSize = 256, WarmupSteps = 7500, MaxGradientNorm = 0.1,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.FineTuning,
                Source = "Ravi et al. 2024, Table 12b: full training runs 150k steps with AdamW (0.9, 0.999), weight decay "
                        + "0.1, gradient clipping at 0.1 and a cosine schedule after a 7.5k-iteration linear warmup, at "
                        + "6e-5 for the image encoder and 3.0e-4 for everything else with layer-wise decay (0.8 for T and "
                        + "S, 0.9 for B+, 0.925 for L). The model builds this optimizer itself (Sam2Optimizer): no single "
                        + "rate is declared because the rates differ per parameter.")]
public partial class SAM21<T> : Common.PromptableSegmentationBase<T>, ITrainingObjectiveProvider<T>
{
    private readonly SAM21Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _userOptimizer;
    private Sam2Runtime<T>? _runtime;

    // The image set by SetImage: its size and features (the embedding is also the base's field).
    private int _originalHeight, _originalWidth;
    private (Tensor<T> Embedding, Tensor<T>? HighRes0, Tensor<T>? HighRes1)? _features;

    internal SAM21ModelSize ModelSize => _options.ModelSize;

    private Sam2Runtime<T> Runtime => _runtime ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal Sam2Network<T> Network => Runtime.Network;

    /// <summary>Creates a trainable SAM 2.1.</summary>
    /// <param name="architecture">The architecture; its larger side is the encoder's square size unless
    /// <see cref="SAM21Options.ImageSize"/> sets it (1024 by default).</param>
    /// <param name="optimizer">The optimizer (default: the paper's AdamW recipe).</param>
    /// <param name="lossFunction">Reported by the base; training minimizes the paper's focal, dice, IoU and occlusion
    /// losses.</param>
    /// <param name="options">Model options.</param>
    public SAM21(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        SAM21Options? options = null)
        : base(architecture, optimizer, lossFunction ?? new BinaryCrossEntropyWithLogitsLoss<T>(), (options ??= new SAM21Options()).NumClasses)
    {
        _options = options;
        _options.Validate();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 1024;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 1024;
        if (_channels != 3) throw new ArgumentException("SAM 2.1 segments RGB images (3 channels).", nameof(architecture));
        _userOptimizer = optimizer;
        InitializeLayers();
    }

    private int SizeIndex => _options.ModelSize switch
    {
        SAM21ModelSize.Tiny => 0,
        SAM21ModelSize.Small => 1,
        SAM21ModelSize.BasePlus => 2,
        _ => 3,
    };

    protected override void InitializeLayers()
    {
        if (_options is null || _runtime is not null) return;
        if (!_useNativeMode) { ClearLayers(); return; }
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("SAM 2.1 builds the paper's network; custom layers are not supported.");
        _runtime = new Sam2Runtime<T>(Engine, _options, SizeIndex, sam21: true, Math.Max(_height, _width), new Sam2TrainingSettings(), _userOptimizer);
        Layers.AddRange(_runtime.Layers);
        _encoderLayerEnd = _runtime.Layers.Count;
    }

    /// <summary>Loads a released checkpoint in Hugging Face's format (the <c>model.safetensors</c> of
    /// <c>facebook/sam2.1-hiera-*</c>); the model must be built with that checkpoint's size.</summary>
    public void LoadPretrainedWeights(string path) => Network.LoadSafetensors(path);

    /// <summary>Loads a Hugging Face <c>Sam2VideoModel</c> state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc/>
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_runtime is not null) _runtime.Network.Training = isTraining;
    }

    // ------------------------------------------------------------------ images (SAM2ImagePredictor)

    /// <summary>The image embedding of an image <c>[3, H, W]</c> (values in 0..1) resized to the encoder's square.</summary>
    protected override Tensor<T> EncodeImage(Tensor<T> image) => Runtime.EncodeImage(Sam2Runtime<T>.Frame(image, nameof(image))).Embedding;

    /// <inheritdoc />
    public override void SetImage(Tensor<T> image)
    {
        var frame = Sam2Runtime<T>.Frame(image, nameof(image));
        _originalHeight = frame.Shape[2];
        _originalWidth = frame.Shape[3];
        _features = Runtime.EncodeImage(frame);
        _imageEmbedding = _features.Value.Embedding;
        _imageSet = true;
    }

    /// <summary>The official stability score: the IoU of the masks thresholded at ±offset around the mask threshold.</summary>
    private static double StabilityScore(IReadOnlyList<double> logits, double offset)
    {
        int high = 0, low = 0;
        foreach (double v in logits)
        {
            if (v > offset) high++;
            if (v > -offset) low++;
        }
        return low == 0 ? 0.0 : (double)high / low;
    }

    /// <summary>The decoder on the set image for prompts in the encoder frame; the masks resized to the image.</summary>
    private (Tensor<T> Logits, Tensor<T> Iou, Tensor<T> LowRes) Predict(Tensor<T>? points, Tensor<T>? labels, Tensor<T>? masks, bool multimask)
    {
        EnsureImageSet();
        var (low, iou, _) = Runtime.DecodeImage(_features!.Value, points, labels, masks, multimask);
        var logits = low.Shape[2] == _originalHeight && low.Shape[3] == _originalWidth
            ? low : SpatialResampling.Resize(Engine, low, _originalHeight, _originalWidth);
        return (logits, iou, low);
    }

    private PromptedSegmentationResult<T> Result(Tensor<T> logits, Tensor<T> iou, Tensor<T> low)
    {
        int count = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
        var masks = new Tensor<T>(new[] { count, h, w });
        var scores = new double[count];
        var stability = new double[count];
        var values = logits.ToArray();
        for (int m = 0; m < count; m++)
        {
            var plane = new double[h * w];
            for (int i = 0; i < h * w; i++)
            {
                plane[i] = NumOps.ToDouble(values[m * h * w + i]);
                masks[m * h * w + i] = plane[i] > 0 ? NumOps.One : NumOps.Zero;
            }
            scores[m] = NumOps.ToDouble(iou[0, m]);
            stability[m] = StabilityScore(plane, 1.0);
        }
        return new PromptedSegmentationResult<T> { Masks = masks, Scores = scores, StabilityScores = stability, LowResLogits = low };
    }

    private (double X, double Y) Point(Tensor<T> points, int i) => (NumOps.ToDouble(points[i, 0]), NumOps.ToDouble(points[i, 1]));

    /// <summary>The three ambiguity-aware masks for points <c>[k, 2]</c> (x, y in the image) with labels <c>[k]</c>
    /// (1 foreground, 0 background), as SAM2ImagePredictor gives them with multimask output.</summary>
    public override PromptedSegmentationResult<T> SegmentFromPoints(Tensor<T> points, Tensor<T> labels)
    {
        EnsureImageSet();
        int k = points.Shape[0];
        var prompt = Runtime.PointPrompt(Enumerable.Range(0, k).Select(i => Point(points, i)).ToList(),
            Enumerable.Range(0, k).Select(i => (int)Math.Round(NumOps.ToDouble(labels[i]))).ToList(), _originalHeight, _originalWidth);
        var (logits, iou, low) = Predict(prompt.Points, prompt.Labels, null, multimask: true);
        return Result(logits, iou, low);
    }

    /// <summary>The mask for a box <c>[4]</c> (x0, y0, x1, y1 in the image): a single mask.</summary>
    public override PromptedSegmentationResult<T> SegmentFromBox(Tensor<T> box)
    {
        EnsureImageSet();
        var prompt = Runtime.BoxPrompt(Enumerable.Range(0, 4).Select(i => NumOps.ToDouble(box[i])).ToList(), _originalHeight, _originalWidth);
        var (logits, iou, low) = Predict(prompt.Points, prompt.Labels, null, multimask: false);
        return Result(logits, iou, low);
    }

    /// <summary>
    /// The mask for a mask prompt of logits: low-resolution logits <c>[S/4, S/4]</c> (a previous result's
    /// <see cref="PromptedSegmentationResult{T}.LowResLogits"/> plane, the predictor's <c>mask_input</c>), or a map of the
    /// image's size <c>[H, W]</c>, resized to that resolution (bilinear, antialiased).
    /// </summary>
    public override PromptedSegmentationResult<T> SegmentFromMask(Tensor<T> mask)
    {
        EnsureImageSet();
        int side = Network.Configuration.ImageSize / 4;
        Tensor<T> m;
        if (mask.Length == side * side) m = Engine.Reshape(mask, new[] { 1, 1, side, side });
        else if (mask.Length == _originalHeight * _originalWidth)
            m = SpatialResampling.ResizeAntialias(Engine, Engine.Reshape(mask, new[] { 1, 1, _originalHeight, _originalWidth }), side, side);
        else
            throw new ArgumentException($"A mask prompt is a {side}x{side} plane of low-resolution logits or a {_originalHeight}x{_originalWidth} map of the image.", nameof(mask));
        var (logits, iou, low) = Predict(null, null, m, multimask: false);
        return Result(logits, iou, low);
    }

    /// <summary>
    /// The official SAM2AutomaticMaskGenerator with its default settings: a 32×32 grid of foreground points, three masks
    /// per point, kept when the predicted IoU exceeds 0.8 and the stability score (offset 1.0) reaches 0.95, then box
    /// non-maximum suppression at IoU 0.7, highest predicted IoU first.
    /// </summary>
    public override List<PromptedSegmentationResult<T>> SegmentEverything()
    {
        EnsureImageSet();
        const int perSide = 32, perBatch = 64;
        const double iouThreshold = 0.8, stabilityThreshold = 0.95, nmsThreshold = 0.7;
        var candidates = new List<(double Iou, double Stability, double[] Box, Tensor<T> Mask, Tensor<T> Low)>();
        var grid = new List<(double X, double Y)>();
        double offset = 1.0 / (2 * perSide), step = (1 - 2 * offset) / (perSide - 1);
        for (int i = 0; i < perSide; i++)
            for (int j = 0; j < perSide; j++)
                grid.Add(((offset + j * step) * _originalWidth, (offset + i * step) * _originalHeight));
        for (int start = 0; start < grid.Count; start += perBatch)
        {
            int n = Math.Min(perBatch, grid.Count - start);
            var scaled = Runtime.ScalePoints(grid.Skip(start).Take(n).ToList(), _originalHeight, _originalWidth);
            var points = Engine.Reshape(scaled, new[] { n, 1, 2 });
            var labels = new Tensor<T>(new[] { n, 1 });
            labels.AsWritableSpan().Fill(NumOps.One);
            var (logits, iou, low) = Predict(points, labels, null, multimask: true);
            int masks = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];
            var values = logits.ToArray();
            for (int b = 0; b < n; b++)
                for (int m = 0; m < masks; m++)
                {
                    double predicted = NumOps.ToDouble(iou[b, m]);
                    if (predicted <= iouThreshold) continue;
                    var plane = new double[h * w];
                    for (int i = 0; i < h * w; i++) plane[i] = NumOps.ToDouble(values[(b * masks + m) * h * w + i]);
                    double stability = StabilityScore(plane, 1.0);
                    if (stability < stabilityThreshold) continue;
                    var mask = new Tensor<T>(new[] { 1, h, w });
                    int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                        {
                            if (plane[y * w + x] <= 0) continue;
                            mask[0, y, x] = NumOps.One;
                            x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
                        }
                    // An empty mask's box is [0, 0, 0, 0], as batched_mask_to_box gives it.
                    var box = x1 < 0 ? new double[4] : new double[] { x0, y0, x1, y1 };
                    var lowPlane = Engine.Reshape(Engine.TensorSlice(low, new[] { b, m, 0, 0 }, new[] { 1, 1, low.Shape[2], low.Shape[3] }),
                        new[] { low.Shape[2], low.Shape[3] });
                    candidates.Add((predicted, stability, box, mask, lowPlane));
                }
        }
        // torchvision's batched_nms with one category: highest score first, drop boxes overlapping a kept one above the
        // threshold (areas without +1, as torchvision computes them).
        var order = Enumerable.Range(0, candidates.Count).OrderByDescending(i => candidates[i].Iou).ToList();
        var kept = new List<int>();
        foreach (int i in order)
        {
            var a = candidates[i].Box;
            bool suppressed = kept.Any(j =>
            {
                var b = candidates[j].Box;
                double iw = Math.Max(0, Math.Min(a[2], b[2]) - Math.Max(a[0], b[0]));
                double ih = Math.Max(0, Math.Min(a[3], b[3]) - Math.Max(a[1], b[1]));
                double inter = iw * ih;
                double union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
                return union > 0 && inter / union > nmsThreshold;
            });
            if (!suppressed) kept.Add(i);
        }
        return kept.Select(i => new PromptedSegmentationResult<T>
        {
            Masks = candidates[i].Mask, Scores = new[] { candidates[i].Iou }, StabilityScores = new[] { candidates[i].Stability },
            LowResLogits = candidates[i].Low,
        }).ToList();
    }

    // ------------------------------------------------------------------ videos (SAM2VideoPredictor)

    private List<Tensor<T>> Track(List<Tensor<T>> frames, Func<int, int, Dictionary<int, Sam2Prompt<T>>> prompts)
    {
        if (frames is null || frames.Count == 0) throw new ArgumentException("No frames.", nameof(frames));
        var batched = frames.Select(f => Sam2Runtime<T>.Frame(f, nameof(frames))).ToList();
        int height = batched[0].Shape[2], width = batched[0].Shape[3];
        if (batched.Any(f => f.Shape[2] != height || f.Shape[3] != width))
            throw new ArgumentException("Every frame must have the same size.", nameof(frames));
        return Runtime.Track(batched, prompts(height, width)).Select(o => Runtime.VideoProbabilities(o, height, width)).ToList();
    }

    /// <summary>Segments an object through a video (frames <c>[3, H, W]</c>, values in 0..1) from clicks <c>[k, 2]</c> on its
    /// first frame (labels 1 object, 0 background); each frame's mask probabilities <c>[H, W]</c>.</summary>
    public List<Tensor<T>> TrackObject(List<Tensor<T>> frames, float[,] points, int[] pointLabels)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.PointPrompt(points, pointLabels, h, w) });

    /// <summary>Segments an object through a video from a box <c>[x0, y0, x1, y1]</c> on its first frame.</summary>
    public List<Tensor<T>> TrackObjectWithBox(List<Tensor<T>> frames, float[] box)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.BoxPrompt(box.Select(v => (double)v).ToList(), h, w) });

    /// <summary>Segments an object through a video from its mask <c>[H, W]</c> on the first frame.</summary>
    public List<Tensor<T>> TrackObjectWithMask(List<Tensor<T>> frames, Tensor<T> mask)
        => Track(frames, (h, w) => new Dictionary<int, Sam2Prompt<T>> { [0] = Runtime.MaskPrompt(mask, h, w) });

    /// <summary>Segments an object through a video from clicks on any frames: each prompted frame is decoded from its clicks,
    /// and the object is propagated from the first prompted frame forward to the end and back to the start.</summary>
    public List<Tensor<T>> InteractiveVideoSegmentation(List<Tensor<T>> frames, Dictionary<int, (float[,] Points, int[] Labels)> framePrompts)
        => Track(frames, (h, w) => framePrompts.ToDictionary(kv => kv.Key, kv => Runtime.PointPrompt(kv.Value.Points, kv.Value.Labels, h, w)));

    // ------------------------------------------------------------------ the tensor contract and training

    /// <summary>The single-mask logits <c>[B, 1, H, W]</c> (or <c>[1, H, W]</c>) of images with no prompt: the image
    /// predictor's decoder on the model's empty prompt, resized to each image.</summary>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 4;
        if (!hasBatch) input = AddBatchDimension(input);
        int b = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var outputs = new Tensor<T>[b];
        for (int i = 0; i < b; i++)
        {
            var frame = Engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, input.Shape[1], h, w });
            var (low, _, _) = Runtime.DecodeImage(Runtime.EncodeImage(frame), null, null, null, multimask: false);
            outputs[i] = SpatialResampling.Resize(Engine, low, h, w);
        }
        var logits = b == 1 ? outputs[0] : Engine.TensorConcatenate(outputs, 0);
        return hasBatch ? logits : RemoveBatchDimension(logits);
    }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    private static IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> Images(Tensor<T> input, Tensor<T> target, IEngine engine)
    {
        if (input.Rank == 3) return new[] { (input, target) };
        if (input.Rank != 4) throw new ArgumentException("SAM 2.1 trains on images [C, H, W] or [B, C, H, W].", nameof(input));
        int b = input.Shape[0], c = input.Shape[1], h = input.Shape[2], w = input.Shape[3];
        if (target.Length != b * h * w) throw new ArgumentException($"The target is one {h}x{w} mask per image ({b}).", nameof(target));
        var flat = engine.Reshape(target, new[] { b, h * w });
        return Enumerable.Range(0, b).Select(i => (
            engine.Reshape(engine.TensorSlice(input, new[] { i, 0, 0, 0 }, new[] { 1, c, h, w }), new[] { c, h, w }),
            engine.Reshape(engine.TensorSlice(flat, new[] { i, 0 }, new[] { 1, h * w }), new[] { h, w }))).ToList();
    }

    /// <summary>
    /// One step of SAM 2's training (Ravi et al. 2024, Appendix D.2.2) on images <c>[C, H, W]</c> or <c>[B, C, H, W]</c> and
    /// their masks: each image is a one-frame example (a click or box prompt and seven corrective clicks), and the step
    /// minimizes the mean loss. Videos train through <see cref="TrainOnVideo"/>.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Step(Images(input, expectedOutput, Engine));
        return true;
    }

    /// <summary>One step of SAM 2's video training on one object: frames <c>[F, 3, H, W]</c> (values in 0..1) and their masks
    /// <c>[F, H, W]</c>.</summary>
    public void TrainOnVideo(Tensor<T> video, Tensor<T> masks)
    {
        if (video.Rank != 4) throw new ArgumentException("A video is [F, 3, H, W].", nameof(video));
        Step(new[] { (video, masks) });
    }

    private void Step(IReadOnlyList<(Tensor<T> Input, Tensor<T> Target)> examples)
    {
        SetTrainingMode(true);
        try
        {
            LastLoss = Runtime.Train(examples, gradients => PublishParameterGradients(gradients));
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>SAM 2.1 trains on a supervised objective: its losses against an object's masks.</summary>
    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    /// <inheritdoc/>
    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <summary>The training objective estimated with common random numbers (fixed draws), so two evaluations compare the
    /// same objective.</summary>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => Runtime.Evaluate(Images(input, target, Engine));

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "SAM21" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "ImageSize", Network.Configuration.ImageSize },
            { "ModelSize", _options.ModelSize.ToString() },
            { "MemoryBankSize", _options.MemoryBankSize },
            { "NumLayers", Layers.Count },
        },
        ModelDataProvider = () => SerializeForMetadata()
    };
}
