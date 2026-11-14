using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>One region a caption names, with its SAM 2 mask.</summary>
/// <param name="Detection">Grounding DINO's box, score and phrase.</param>
/// <param name="Mask">The mask <c>[H, W]</c> at the image's size: 1 inside, 0 outside.</param>
/// <param name="MaskScore">SAM 2's predicted IoU for the mask.</param>
public sealed record GroundedSegment<T>(GroundingDetection<T> Detection, Tensor<T> Mask, T MaskScore);

/// <summary>
/// Grounded SAM 2: Grounding DINO finds the boxes a caption names, and SAM 2 segments each box.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Grounded SAM (Ren et al., 2024) assembles open-world models rather than training a new one: an open-set detector turns
/// text into boxes, and a promptable segmenter turns each box into a mask. Grounded SAM 2 (IDEA-Research/Grounded-SAM-2)
/// pairs Grounding DINO with SAM 2. As the official demo runs them: Grounding DINO predicts with box and text thresholds
/// 0.35 and 0.25; its boxes, scaled to pixels, prompt SAM 2's image predictor (the image resized to 1024, single-mask
/// output); and the masks are SAM 2's logits resized to the image and thresholded at 0.
/// </para>
/// <para>
/// Both networks are the official ones (each checked against Hugging Face's port). The paper trains neither; the
/// detector's own objective trains it here (<see cref="TrainGrounding"/>), and SAM 2 is used as loaded.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks" (Ren et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> Describe what you want ("cat . remote control .") and Grounded SAM 2 returns a box and a
/// pixel mask for each match.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.ImageSegmentation,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 256);
/// var model = new GroundedSAM2&lt;double&gt;(architecture, new GroundedSAM2Options());
/// var segments = model.Segment(image, "cat . remote control .");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks",
    "https://arxiv.org/abs/2401.14159",
    Year = 2024,
    Authors = "Ren et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 1e-4,
                MaxGradientNorm = 0.1,
                Source = "Grounded SAM assembles pretrained models and trains neither. The detector trains with "
                        + "Grounding DINO's recipe (Liu et al. 2023, hyperparameter table): AdamW with a learning rate "
                        + "of 1e-4, a weight decay of 0.0001 and a clip max norm of 0.1.")]
public partial class GroundedSAM2<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>, IGroundingTrainable<T>
{
    private readonly GroundedSAM2Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private GroundingDinoDetector<T>? _detector;
    private Sam2Network<T>? _segmenter;

    private GroundingDinoDetector<T> Detector => _detector ?? throw new InvalidOperationException("The networks are not initialized.");

    /// <summary>The SAM 2 network.</summary>
    internal Sam2Network<T> Segmenter => _segmenter ?? throw new InvalidOperationException("The networks are not initialized.");

    /// <summary>The Grounding DINO network.</summary>
    internal GroundingDinoNetwork<T> DetectorNetwork => Detector.Network;

    public GroundedSAM2(
        NeuralNetworkArchitecture<T> architecture,
        GroundedSAM2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GroundedSAM2Options();
        Options = _options;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("Grounded SAM 2 declares its detector's recipe.");
        base.ImageSize = _options.SamImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.SamDim;
        InitializeLayers();
    }

    /// <summary>The channels of SAM 2's image embedding, which <see cref="EncodeImage"/> returns.</summary>
    public int EmbeddingDimension => _options.SamDim;
    int IVisualEncoder<T>.ImageSize => _options.SamImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <summary>The most detections <see cref="GroundText"/> returns (the highest scoring ones).</summary>
    public int MaxDetections => _options.Detector.MaxDetections;

    /// <summary>The SAM 2 configuration the options describe.</summary>
    private Sam2Configuration SegmenterConfiguration() => new()
    {
        ImageSize = _options.SamImageSize, BlocksPerStage = (int[])_options.SamBlocksPerStage.Clone(),
        EmbedDims = (int[])_options.SamEmbedDims.Clone(), HeadsPerStage = (int[])_options.SamHeadsPerStage.Clone(),
        WindowSizes = (int[])_options.SamWindowSizes.Clone(), GlobalAttentionBlocks = (int[])_options.SamGlobalAttentionBlocks.Clone(),
        Dim = _options.SamDim, MaskInChannels = _options.SamMaskInChannels, DecoderHeads = _options.SamDecoderHeads,
        DecoderMlp = _options.SamDecoderMlpDim, IouHidden = _options.SamIouHiddenDim,
    };

    protected override void InitializeLayers()
    {
        if (_options is null || _detector is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Grounded SAM 2 builds Grounding DINO and SAM 2; custom layers are not supported.");
        int seed = _options.Seed ?? 42;
        var layers = new List<LayerBase<T>>();
        _detector = new GroundingDinoDetector<T>(Engine, layers, _options.Detector, _options.Detector.Seed ?? seed);
        _segmenter = new Sam2Network<T>(Engine, layers, SegmenterConfiguration());
        _segmenter.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 2));
        Layers.AddRange(layers);
    }

    /// <summary>Loads an official Grounding DINO checkpoint's state dict (Hugging Face names).</summary>
    internal void LoadDetectorWeights(Func<string, int[], double[]> read) => Detector.Network.LoadTorchWeights(read);

    /// <summary>The same, straight from <paramref name="source"/> into the layers.</summary>
    internal void LoadDetectorWeights(AiDotNet.ModelLoading.TorchTensorSource source) => Detector.Network.LoadTorchWeights(source);

    /// <summary>Loads an official SAM 2 checkpoint's state dict (Hugging Face names).</summary>
    internal void LoadSegmenterWeights(Func<string, int[], double[]> read) => Segmenter.LoadTorchWeights(read);

    /// <summary>The same, straight from <paramref name="source"/> into the layers.</summary>
    internal void LoadSegmenterWeights(AiDotNet.ModelLoading.TorchTensorSource source) => Segmenter.LoadTorchWeights(source);

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_detector is not null) _detector.Network.Training = isTraining;
    }

    /// <summary>Images in 0..1 resized to SAM 2's input side (bilinear with antialiasing, as torchvision's Resize) and
    /// normalized by the ImageNet mean and deviation.</summary>
    private Tensor<T> SegmenterInput(Tensor<T> images)
    {
        int side = _options.SamImageSize;
        if (images.Shape[2] != side || images.Shape[3] != side)
            images = ComputerVision.Segmentation.SpatialResampling.ResizeAntialias(Engine, images, side, side);
        return Segmenter.Normalize(images);
    }

    /// <summary>
    /// SAM 2's image embedding <c>[SamDim, g, g]</c> (or <c>[B, SamDim, g, g]</c>), g = SamImageSize / 16, of images in
    /// 0..1 resized to SamImageSize.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var (embedding, _, _) = Segmenter.EncodeImage(SegmenterInput(GroundingDinoDetector<T>.Batched(Engine, image)));
        return image.Rank == 3 ? Engine.Reshape(embedding, embedding.Shape.ToArray().Skip(1).ToArray()) : embedding;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => EncodeImage(input);

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeImage(input);

    /// <summary>Grounding DINO's detections of the caption's phrases (the official <c>predict</c>).</summary>
    /// <param name="image">One RGB image <c>[3, H, W]</c> with values in 0..1.</param>
    /// <param name="caption">Period-separated phrases, such as "cat . remote control .".</param>
    public IReadOnlyList<GroundingDetection<T>> Ground(Tensor<T> image, string caption)
    {
        ThrowIfDisposed();
        return Detector.Ground(GroundingDinoDetector<T>.Batched(Engine, image), caption);
    }

    /// <summary>
    /// The official demo: Grounding DINO's detections of the caption's phrases, each box (in pixels) prompting SAM 2
    /// for a single mask, resized to the image and thresholded at <see cref="GroundedSAM2Options.MaskThreshold"/>.
    /// </summary>
    /// <param name="image">One RGB image <c>[3, H, W]</c> with values in 0..1.</param>
    /// <param name="caption">Period-separated phrases, such as "cat . remote control .".</param>
    public IReadOnlyList<GroundedSegment<T>> Segment(Tensor<T> image, string caption)
    {
        ThrowIfDisposed();
        var batch = GroundingDinoDetector<T>.Batched(Engine, image);
        var detections = Detector.Strongest(Detector.Ground(batch, caption));
        var segments = new List<GroundedSegment<T>>();
        if (detections.Count == 0) return segments;
        int height = batch.Shape[2], width = batch.Shape[3], side = _options.SamImageSize;
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var (embedding, s0, s1) = Segmenter.EncodeImage(SegmenterInput(batch));
        var positional = Segmenter.DensePositional();
        var corners = ScaledCorners(Engine, detections, side);
        for (int i = 0; i < detections.Count; i++)
        {
            var box = Engine.TensorSlice(corners, new[] { i, 0 }, new[] { 1, 4 });
            var (sparse, dense) = Segmenter.EncodePrompts(null, null, box, null);
            var (logits, iou, _) = Segmenter.DecodeMasks(embedding, s0, s1, positional, sparse, dense, false);
            var full = Segmenter.Postprocess(logits, height, width);
            // Thresholded exactly at the real MaskThreshold (a T-rounded threshold would move the boundary).
            var mask = MaskStatistics.GreaterThan(Engine.Reshape(
                Engine.TensorSlice(full, new[] { 0, 0, 0, 0 }, new[] { 1, 1, height, width }), new[] { height, width }), _options.MaskThreshold);
            segments.Add(new GroundedSegment<T>(detections[i], mask, iou.ToArray()[0]));
        }
        return segments;
    }

    /// <summary>
    /// The predictor's transform_boxes: each detection's pixel corners scaled to SAM's input side, <c>[n, 4]</c>.
    /// </summary>
    /// <remarks>
    /// The corners are computed from the model's (cx, cy, w, h) and scaled in double, then rounded to
    /// <typeparamref name="T"/> once. Scaling the corners already rounded to T rounded twice: in Half a corner of
    /// 0.27505 at side 1000 became 275.25 instead of 275.
    /// </remarks>
    internal static Tensor<T> ScaledCorners(IEngine engine, IReadOnlyList<GroundingDetection<T>> detections, int side)
    {
        int n = detections.Count;
        // Rows of (cx, cy, w, h) where the model's box is known, else the detection's own corners.
        var values = new Tensor<T>(new[] { n, 4 }, Vector<T>.FromArray(
            detections.SelectMany(d => d.CenterSize is { } c ? new[] { c.Cx, c.Cy, c.W, c.H } : new[] { d.X1, d.Y1, d.X2, d.Y2 }).ToArray())).Cast<double>();
        var fromBox = new Tensor<double>(new[] { n, 1 }, Vector<double>.FromArray(
            detections.Select(d => d.CenterSize is null ? 0.0 : 1.0).ToArray()));
        var centre = engine.TensorSlice(values, new[] { 0, 0 }, new[] { n, 2 });
        var half = engine.TensorMultiplyScalar(engine.TensorSlice(values, new[] { 0, 2 }, new[] { n, 2 }), 0.5);
        var boxCorners = engine.TensorConcatenate(new[] { engine.TensorSubtract(centre, half), engine.TensorAdd(centre, half) }, 1);
        var corners = engine.TensorWhere(engine.TensorBroadcastTo(fromBox, new[] { n, 4 }), boxCorners, values);
        return engine.TensorMultiplyScalar(corners, (double)side).Cast<T>();
    }

    /// <summary>
    /// Segments the named classes: the class names joined into one caption ("a. b", as <c>predict_with_classes</c>), each
    /// detection's class the first name its phrase contains, and per class the union of its detections' masks
    /// <c>[classes, H, W]</c> and its best detection score (0 when none). The semantic map holds, per pixel, the class of
    /// the highest-scoring mask covering it, or -1.
    /// </summary>
    public OpenVocabSegmentationResult<T> SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames)
    {
        if (classNames is null || classNames.Count == 0) throw new ArgumentException("Name at least one class.", nameof(classNames));
        var segments = Segment(image, string.Join(". ", classNames));
        return Combine(Engine, image, segments, classNames.ToArray(), s => GroundingDinoPrompt.ClassOf(s.Detection.Phrase, classNames));
    }

    /// <summary>Segments what one prompt describes: the union of every detection's mask, as one class.</summary>
    public OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt)
        => Combine(Engine, image, Segment(image, prompt), new[] { prompt }, _ => 0);

    /// <summary>How many detections' masks <see cref="Combine"/> holds at once.</summary>
    private const int CombineChunk = 16;

    /// <remarks>
    /// The segments are ranked by score, and by logit where scores tie, before the per-pixel argmax: in a narrow
    /// <typeparamref name="T"/> every confident detection's score rounds to 1, and the first such mask would win a
    /// pixel that a stronger later detection covers.
    /// </remarks>
    internal static OpenVocabSegmentationResult<T> Combine(IEngine engine, Tensor<T> image, IReadOnlyList<GroundedSegment<T>> segments,
        string[] classes, Func<GroundedSegment<T>, int> classOf)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var order = Comparer<T>.Default;
        var batch = GroundingDinoDetector<T>.Batched(engine, image);
        int height = batch.Shape[2], width = batch.Shape[3], pixels = height * width, count = classes.Length;
        var named = segments.Select(s => (Segment: s, Class: classOf(s))).Where(s => s.Class >= 0)
            .OrderByDescending(s => s.Segment.Detection.Score, order)
            .ThenByDescending(s => s.Segment.Detection.Logit, order)
            .ToList();
        if (named.Count == 0)
        {
            var none = Tensor<T>.CreateDefault(new[] { height, width }, numOps.FromDouble(-1));
            return new OpenVocabSegmentationResult<T>
            {
                Masks = new Tensor<T>(new[] { count, height, width }), ClassNames = classes,
                Scores = new Vector<T>(count), SemanticMap = none,
            };
        }

        int n = named.Count;
        var scoreOf = new Tensor<T>(new[] { n }, Vector<T>.FromArray(named.Select(s => s.Segment.Detection.Score).ToArray()));
        var classIds = new Tensor<int>(new[] { n }, Vector<int>.FromArray(named.Select(s => s.Class).ToArray()));

        // Per class: its best detection score (0 when it has none; scores are sigmoids).
        var scores = engine.ReduceMax(engine.TensorMultiply(engine.TensorOneHot<T>(classIds, count),
            engine.TensorBroadcastTo(scoreOf.Reshape(new[] { n, 1 }), new[] { n, count })), new[] { 0 }, false);

        // The masks are read CombineChunk detections at a time, so the [detections, pixels] intermediates stay bounded
        // however many detections there are. Per class: the union of its masks. Per pixel: the class of the
        // highest-scoring mask covering it, or -1; a later chunk takes a pixel only with a strictly higher score, so
        // ties still go to the first ranked detection.
        var classValues = engine.TensorArange<T>(0, count, 1);
        var unionSum = new Tensor<T>(new[] { count, pixels });
        var bestScore = Tensor<T>.CreateDefault(new[] { pixels }, numOps.MinValue);
        var semantic = Tensor<T>.CreateDefault(new[] { pixels }, numOps.FromDouble(-1));
        for (int start = 0; start < n; start += CombineChunk)
        {
            int c = Math.Min(CombineChunk, n - start);
            var chunk = named.GetRange(start, c);
            var masks = engine.Reshape(engine.TensorStack(chunk.Select(s => s.Segment.Mask).ToArray(), 0), new[] { c, pixels });
            var chunkScores = new Tensor<T>(new[] { c }, Vector<T>.FromArray(chunk.Select(s => s.Segment.Detection.Score).ToArray()));
            var chunkClasses = new Tensor<int>(new[] { c }, Vector<int>.FromArray(chunk.Select(s => s.Class).ToArray()));
            unionSum = engine.TensorAdd(unionSum,
                engine.TensorMatMul(engine.TensorTranspose(engine.TensorOneHot<T>(chunkClasses, count)), masks));

            var coverScores = engine.TensorWhere(engine.TensorGreaterThan(masks, numOps.Zero),
                engine.TensorBroadcastTo(chunkScores.Reshape(new[] { c, 1 }), new[] { c, pixels }),
                Tensor<T>.CreateDefault(new[] { c, pixels }, numOps.MinValue));
            var winner = engine.TensorArgMax(coverScores, 0).Reshape(new[] { pixels });
            var chunkBest = engine.ReduceMax(coverScores, new[] { 0 }, false);
            var winningClass = engine.TensorIndexSelect(engine.TensorIndexSelect(classValues, chunkClasses, 0), winner, 0);
            var takes = engine.TensorGreaterThan(chunkBest, bestScore);
            bestScore = engine.TensorWhere(takes, chunkBest, bestScore);
            semantic = engine.TensorWhere(takes, winningClass, semantic);
        }

        var union = engine.TensorMin(unionSum, numOps.One);
        return new OpenVocabSegmentationResult<T>
        {
            Masks = union.Reshape(new[] { count, height, width }), ClassNames = classes, Scores = scores.ToVector(),
            SemanticMap = semantic.Reshape(new[] { height, width }),
        };
    }

    /// <summary>
    /// <see cref="Ground"/> as a tensor <c>[n, 5]</c> of (x1, y1, x2, y2, score) rows, the highest-scoring
    /// <see cref="MaxDetections"/> in query order.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
        => GroundingRows<T>.Boxes(Detector.Strongest(Ground(image, textQuery)));

    /// <summary>
    /// The official <c>predict_with_classes</c>: the categories joined as "a. b" into one caption, and each detection's
    /// class the first category its phrase contains (-1 when none does). Rows <c>[n, 6]</c> of (x1, y1, x2, y2, class,
    /// score).
    /// </summary>
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        if (categories is null || categories.Count == 0) throw new ArgumentException("Name at least one category.", nameof(categories));
        return GroundingRows<T>.Classified(Detector.Strongest(Ground(image, string.Join(". ", categories))), categories);
    }

    /// <summary>
    /// One training step of the detector on an image, its caption and the boxes its phrases name, with Grounding DINO's
    /// loss. SAM 2 is not trained: Grounded SAM uses it as released.
    /// </summary>
    /// <param name="image">One RGB image <c>[3, H, W]</c> with values in 0..1.</param>
    /// <param name="caption">Period-separated phrases, such as "cat . remote control .".</param>
    /// <param name="boxes">The boxes, each naming one of the caption's phrases.</param>
    public void TrainGrounding(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes)
    {
        ThrowIfDisposed();
        var batch = GroundingDinoDetector<T>.Batched(Engine, image);
        var (prompt, target) = GroundingDinoDetector<T>.Supervision(caption, boxes);
        TrainWithCustomObjective(batch, batch, (x, _) => Detector.Loss(x, prompt, target), _optimizer);
    }

    /// <summary>The detector's loss on an image, its caption and boxes, without training (dropout off).</summary>
    public T EvaluateGroundingLoss(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes)
    {
        ThrowIfDisposed();
        var batch = GroundingDinoDetector<T>.Batched(Engine, image);
        var (prompt, target) = GroundingDinoDetector<T>.Supervision(caption, boxes);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return Detector.Loss(batch, prompt, target)[0];
    }

    /// <summary>Grounded SAM 2's detector trains on grounded phrases, which a tensor target cannot carry.</summary>
    /// <exception cref="NotSupportedException">Always: use <see cref="TrainGrounding"/>.</exception>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
        => throw new NotSupportedException(
            "Grounded SAM 2 trains its detector on an image, a caption and the boxes its phrases name: use TrainGrounding(image, caption, boxes).");

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    protected override bool SupportsParameterMutation => true;

    /// <summary>SAM 2's preprocessing: images in 0..1 resized to SamImageSize and normalized.</summary>
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => SegmenterInput(GroundingDinoDetector<T>.Batched(Engine, image));

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "GroundedSAM2",
            Description = "Grounded SAM 2 (Ren et al., 2024): Grounding DINO boxes segmented by SAM 2.",
            FeatureCount = _options.SamDim,
            Complexity = _options.Detector.NumVisionLayers + _options.Detector.NumDecoderLayers + _options.SamBlocksPerStage.Sum(),
        };
        m.AdditionalInfo["Architecture"] = "GroundedSAM2";
        m.AdditionalInfo["SamImageSize"] = _options.SamImageSize.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GroundedSAM2<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
