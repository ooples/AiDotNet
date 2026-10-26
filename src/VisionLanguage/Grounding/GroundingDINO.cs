using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// Grounding DINO: open-set detection combining DINO with grounded pre-training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Grounding DINO (Liu et al. 2023) finds the regions of an image that a caption's phrases name. A Swin image backbone
/// and a BERT text backbone feed a feature enhancer that fuses the two (vision-text bi-attention, text self-attention,
/// deformable image self-attention); language-guided query selection keeps the 900 encoder tokens most similar to the
/// text; a cross-modality decoder refines them into boxes; and each box is scored against every caption token. The
/// network is the official one (checked against Hugging Face's port of IDEA-Research/GroundingDINO to 1e-8).
/// </para>
/// <para>
/// A caption lists what to find as period-separated phrases ("cat . remote control ."). <see cref="Ground"/> returns
/// every query whose best token probability exceeds the box threshold (0.35), with the words whose tokens exceed the
/// text threshold (0.25), as the official <c>predict</c> does. Images are read at the size given, in 0..1, and
/// normalized with the ImageNet mean and deviation; the official demo resizes the shorter side to 800 first.
/// </para>
/// <para>
/// Training takes grounded phrases (<see cref="TrainGrounding"/>): the paper's loss of Hungarian matching, a
/// contrastive focal loss over the caption's tokens, and L1 and GIoU box losses, on every decoder layer and the
/// encoder's selected queries. A tensor target cannot carry phrases, so <see cref="Train"/> refuses one.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection" (Liu et al., 2023)</item></list></para>
/// <para><b>For Beginners:</b> Grounding DINO detects objects that you describe in words, so it is not limited to a
/// fixed list of categories. Default values follow the released model.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.ObjectDetection,
///     inputHeight: 800, inputWidth: 800, inputDepth: 3, outputSize: 256);
/// var model = new GroundingDINO&lt;double&gt;(architecture, new GroundingDINOOptions());
/// var detections = model.Ground(image, "cat . remote control .");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection",
    "https://arxiv.org/abs/2303.05499",
    Year = 2024,
    Authors = "Liu et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 1e-4,
                MaxGradientNorm = 0.1,
                Source = "Liu et al. 2023, hyperparameter table: AdamW with a learning rate of 1e-4, a "
                        + "weight decay of 0.0001 and a clip max norm of 0.1. The 1e-5 rates the table "
                        + "also lists apply to the image and text backbones separately, which one "
                        + "optimizer over all parameters cannot express.")]
public partial class GroundingDINO<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>, IGroundingTrainable<T>
{
    private readonly GroundingDINOOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private GroundingDinoDetector<T>? _detector;


    private GroundingDinoDetector<T> Detector => _detector ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The network.</summary>
    internal GroundingDinoNetwork<T> Network => Detector.Network;

    public GroundingDINO(
        NeuralNetworkArchitecture<T> architecture,
        GroundingDINOOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GroundingDINOOptions(); Options = _options;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("Grounding DINO declares its paper recipe.");
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        InitializeLayers();
    }

    /// <summary>The width of the image tokens <see cref="EncodeImage"/> returns.</summary>
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <summary>The most detections <see cref="GroundText"/> returns (the highest scoring ones).</summary>
    public int MaxDetections => _options.MaxDetections;

    protected override void InitializeLayers()
    {
        if (_options is null || _detector is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Grounding DINO builds the paper's network; custom layers are not supported.");
        var layers = new List<LayerBase<T>>();
        _detector = new GroundingDinoDetector<T>(Engine, layers, _options, _options.Seed ?? 42);
        Layers.AddRange(layers);
    }

    /// <summary>Loads an official checkpoint's state dict (Hugging Face names).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_detector is not null) _detector.Network.Training = isTraining;
    }

    /// <summary>
    /// The image tokens before they meet the text: every feature level (the backbone's and the extra stride-2 level),
    /// projected to the model width and group-normalized, flattened and concatenated, <c>[tokens, DecoderDim]</c> (or
    /// <c>[B, tokens, DecoderDim]</c>), from images in 0..1.
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var tokens = Detector.ImageTokens(GroundingDinoDetector<T>.Batched(Engine, image));
        return image.Rank == 3 ? Engine.Reshape(tokens, new[] { tokens.Shape[1], tokens.Shape[2] }) : tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => EncodeImage(input);

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeImage(input);

    /// <summary>
    /// The regions the caption's phrases name (the official <c>predict</c>): each query whose best token probability
    /// exceeds the box threshold (<see cref="GroundingVLMOptions.ConfidenceThreshold"/>), with that probability, its box
    /// as fractions of the image, and the words whose tokens exceed <see cref="GroundingDINOOptions.TextThreshold"/>.
    /// </summary>
    /// <param name="image">One RGB image <c>[3, H, W]</c> with values in 0..1.</param>
    /// <param name="caption">Period-separated phrases, such as "cat . remote control .".</param>
    public IReadOnlyList<GroundingDetection> Ground(Tensor<T> image, string caption)
    {
        ThrowIfDisposed();
        return Detector.Ground(GroundingDinoDetector<T>.Batched(Engine, image), caption);
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
    /// One training step on an image, its caption and the boxes its phrases name, with the paper's loss (Hungarian
    /// matching; contrastive focal, L1 and GIoU losses on every decoder layer and the encoder's selected queries).
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

    /// <summary>The paper's loss on an image, its caption and boxes, without training (dropout off).</summary>
    public T EvaluateGroundingLoss(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes)
    {
        ThrowIfDisposed();
        var batch = GroundingDinoDetector<T>.Batched(Engine, image);
        var (prompt, target) = GroundingDinoDetector<T>.Supervision(caption, boxes);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return Detector.Loss(batch, prompt, target)[0];
    }

    /// <summary>Grounding DINO trains on grounded phrases, which a tensor target cannot carry.</summary>
    /// <exception cref="NotSupportedException">Always: use <see cref="TrainGrounding"/>.</exception>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
        => throw new NotSupportedException(
            "Grounding DINO trains on an image, a caption and the boxes its phrases name: use TrainGrounding(image, caption, boxes).");

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    protected override bool SupportsParameterMutation => true;

    /// <summary>Normalizes images in 0..1 with the options' mean and deviation (ImageNet's).</summary>
    protected override Tensor<T> PreprocessImage(Tensor<T> image) => Detector.Normalize(GroundingDinoDetector<T>.Batched(Engine, image));

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "GroundingDINO",
            Description = "Grounding DINO (Liu et al., 2023): open-set detection of the regions a caption names.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "GroundingDINO";
        m.AdditionalInfo["NumQueryPositions"] = _options.NumQueryPositions.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GroundingDINO<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
