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
public sealed record GroundedSegment<T>(GroundingDetection Detection, Tensor<T> Mask, double MaskScore);

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

    public override ModelOptions GetOptions() => _options;

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

    /// <summary>Loads an official SAM 2 checkpoint's state dict (Hugging Face names).</summary>
    internal void LoadSegmenterWeights(Func<string, int[], double[]> read) => Segmenter.LoadTorchWeights(read);

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
    public IReadOnlyList<GroundingDetection> Ground(Tensor<T> image, string caption)
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
        foreach (var d in detections)
        {
            // The predictor's transform_boxes: pixel corners scaled to the resized image.
            var box = new Tensor<T>(new[] { 1, 4 }, new Vector<T>(new[]
            {
                NumOps.FromDouble(d.X1 * side), NumOps.FromDouble(d.Y1 * side), NumOps.FromDouble(d.X2 * side), NumOps.FromDouble(d.Y2 * side),
            }));
            var (sparse, dense) = Segmenter.EncodePrompts(null, null, box, null);
            var (logits, iou, _) = Segmenter.DecodeMasks(embedding, s0, s1, positional, sparse, dense, false);
            var full = Segmenter.Postprocess(logits, height, width);
            var mask = new Tensor<T>(new[] { height, width });
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    mask[y, x] = NumOps.ToDouble(full[0, 0, y, x]) > _options.MaskThreshold ? NumOps.One : NumOps.Zero;
            segments.Add(new GroundedSegment<T>(d, mask, NumOps.ToDouble(iou[0, 0])));
        }
        return segments;
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
        return Combine(image, segments, classNames.ToArray(), s => GroundingDinoPrompt.ClassOf(s.Detection.Phrase, classNames));
    }

    /// <summary>Segments what one prompt describes: the union of every detection's mask, as one class.</summary>
    public OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt)
        => Combine(image, Segment(image, prompt), new[] { prompt }, _ => 0);

    private OpenVocabSegmentationResult<T> Combine(Tensor<T> image, IReadOnlyList<GroundedSegment<T>> segments, string[] classes,
        Func<GroundedSegment<T>, int> classOf)
    {
        var batch = GroundingDinoDetector<T>.Batched(Engine, image);
        int height = batch.Shape[2], width = batch.Shape[3];
        var masks = new Tensor<T>(new[] { classes.Length, height, width });
        var scores = new double[classes.Length];
        var semantic = new Tensor<T>(new[] { height, width });
        var best = new double[height, width];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                semantic[y, x] = NumOps.FromDouble(-1);
                best[y, x] = double.NegativeInfinity;
            }
        foreach (var s in segments)
        {
            int c = classOf(s);
            if (c < 0) continue;
            scores[c] = Math.Max(scores[c], s.Detection.Score);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    if (NumOps.ToDouble(s.Mask[y, x]) == 0) continue;
                    masks[c, y, x] = NumOps.One;
                    if (s.Detection.Score > best[y, x])
                    {
                        best[y, x] = s.Detection.Score;
                        semantic[y, x] = NumOps.FromDouble(c);
                    }
                }
        }
        return new OpenVocabSegmentationResult<T> { Masks = masks, ClassNames = classes, Scores = scores, SemanticMap = semantic };
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
    public override void Train(Tensor<T> input, Tensor<T> expected)
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

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

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

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
