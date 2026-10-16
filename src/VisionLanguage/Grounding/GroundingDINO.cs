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
    private GroundingDinoNetwork<T>? _network;

    public override ModelOptions GetOptions() => _options;

    /// <summary>The network.</summary>
    internal GroundingDinoNetwork<T> Network => _network ?? throw new InvalidOperationException("The network is not initialized.");

    public GroundingDINO(
        NeuralNetworkArchitecture<T> architecture,
        GroundingDINOOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GroundingDINOOptions();
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

    /// <summary>The configuration the options describe.</summary>
    internal static GroundingDinoConfiguration Configuration(GroundingDINOOptions o) => new()
    {
        Backbone = new SwinConfiguration
        {
            EmbedDim = o.VisionDim, Depths = (int[])o.SwinDepths.Clone(), Heads = (int[])o.SwinHeads.Clone(),
            WindowSize = o.SwinWindowSize, DropPathRate = o.SwinDropPathRate,
        },
        Text = new BertConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.TextHiddenSize, Layers = o.TextLayers, Heads = o.TextHeads,
            Intermediate = o.TextIntermediateSize,
        },
        Dim = o.DecoderDim, EncoderLayers = o.NumVisionLayers, DecoderLayers = o.NumDecoderLayers, EncoderHeads = o.NumHeads,
        DecoderHeads = o.NumHeads, EncoderFfn = o.FeedForwardDim, DecoderFfn = o.FeedForwardDim, Levels = o.NumFeatureLevels,
        EncoderPoints = o.NumSamplingPoints, DecoderPoints = o.NumSamplingPoints, Queries = o.NumQueryPositions,
        MaxTextLength = o.MaxSequenceLength, Dropout = o.DropoutRate, FusionDropPath = o.FusionDropPathRate,
    };

    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("Grounding DINO builds the paper's network; custom layers are not supported.");
        if (_options.VocabSize < 30522)
            throw new ArgumentException("Grounding DINO reads BERT-base-uncased token ids; VocabSize must be at least 30522.");
        var layers = new List<LayerBase<T>>();
        _network = new GroundingDinoNetwork<T>(Engine, layers, Configuration(_options));
        int seed = _options.Seed ?? 42;
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        _network.SeedRandom(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1));
        Layers.AddRange(layers);
    }

    /// <summary>Loads an official checkpoint's state dict (Hugging Face names).</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_network is not null) _network.Training = isTraining;
    }

    private Tensor<T> Batched(Tensor<T> image)
    {
        if (image.Rank == 3) image = Engine.Reshape(image, new[] { 1, image.Shape[0], image.Shape[1], image.Shape[2] });
        if (image.Rank != 4 || image.Shape[1] != 3)
            throw new ArgumentException($"Grounding DINO reads RGB images [3, H, W] or [B, 3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
        return image;
    }

    /// <summary>
    /// The image tokens before they meet the text: every feature level (the backbone's and the extra stride-2 level),
    /// projected to the model width and group-normalized, flattened and concatenated, <c>[tokens, DecoderDim]</c> (or
    /// <c>[B, tokens, DecoderDim]</c>).
    /// </summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        bool single = image.Rank == 3;
        var maps = Network.ProjectImage(PreprocessImage(Batched(image)));
        int b = maps[0].Shape[0], d = maps[0].Shape[1];
        var flat = maps.Select(m => Engine.TensorPermute(Engine.Reshape(m, new[] { b, d, m.Shape[2] * m.Shape[3] }), new[] { 0, 2, 1 })).ToArray();
        var tokens = Engine.TensorConcatenate(flat, 1);
        return single ? Engine.Reshape(tokens, new[] { tokens.Shape[1], d }) : tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => EncodeImage(input);

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeImage(input);

    private GroundingDinoOutput<T> Run(Tensor<T> image, GroundingDinoPrompt prompt)
        => Network.Forward(PreprocessImage(image), prompt.IdMatrix, prompt.Mask);

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
        var batch = Batched(image);
        if (batch.Shape[0] != 1) throw new ArgumentException("Ground reads one image.", nameof(image));
        var prompt = GroundingDinoPrompt.Create(caption);
        GroundingDinoOutput<T> output;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
            output = Run(batch, prompt);
        return prompt.Detections(output, 0, _options.ConfidenceThreshold, _options.TextThreshold);
    }

    /// <summary>
    /// <see cref="Ground"/> as a tensor <c>[n, 5]</c> of (x1, y1, x2, y2, score) rows, the highest-scoring
    /// <see cref="MaxDetections"/> in query order.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        var detections = Strongest(Ground(image, textQuery));
        var result = new Tensor<T>(new[] { detections.Count, 5 });
        for (int i = 0; i < detections.Count; i++)
        {
            var d = detections[i];
            result[i, 0] = NumOps.FromDouble(d.X1);
            result[i, 1] = NumOps.FromDouble(d.Y1);
            result[i, 2] = NumOps.FromDouble(d.X2);
            result[i, 3] = NumOps.FromDouble(d.Y2);
            result[i, 4] = NumOps.FromDouble(d.Score);
        }
        return result;
    }

    /// <summary>
    /// The official <c>predict_with_classes</c>: the categories joined as "a. b" into one caption, and each detection's
    /// class the first category its phrase contains (-1 when none does). Rows <c>[n, 6]</c> of (x1, y1, x2, y2, class,
    /// score).
    /// </summary>
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        if (categories is null || categories.Count == 0) throw new ArgumentException("Name at least one category.", nameof(categories));
        var detections = Strongest(Ground(image, string.Join(". ", categories)));
        var result = new Tensor<T>(new[] { detections.Count, 6 });
        for (int i = 0; i < detections.Count; i++)
        {
            var d = detections[i];
            result[i, 0] = NumOps.FromDouble(d.X1);
            result[i, 1] = NumOps.FromDouble(d.Y1);
            result[i, 2] = NumOps.FromDouble(d.X2);
            result[i, 3] = NumOps.FromDouble(d.Y2);
            result[i, 4] = NumOps.FromDouble(GroundingDinoPrompt.ClassOf(d.Phrase, categories));
            result[i, 5] = NumOps.FromDouble(d.Score);
        }
        return result;
    }

    private List<GroundingDetection> Strongest(IReadOnlyList<GroundingDetection> detections)
    {
        if (detections.Count <= _options.MaxDetections) return detections.ToList();
        var keep = new HashSet<GroundingDetection>(detections.OrderByDescending(d => d.Score).Take(_options.MaxDetections));
        return detections.Where(keep.Contains).ToList();
    }

    /// <summary>The image, the caption and its boxes as the network and the loss read them.</summary>
    private (Tensor<T> Image, GroundingDinoPrompt Prompt, GroundingTarget Target) Supervision(Tensor<T> image, string caption,
        IReadOnlyList<GroundingBox> boxes)
    {
        if (boxes is null) throw new ArgumentNullException(nameof(boxes));
        var batch = Batched(image);
        if (batch.Shape[0] != 1) throw new ArgumentException("A grounding step reads one image and its caption.", nameof(image));
        var prompt = GroundingDinoPrompt.Create(caption);
        var targets = new double[boxes.Count][];
        var phrases = new int[boxes.Count];
        for (int i = 0; i < boxes.Count; i++)
        {
            var box = boxes[i];
            if (!(box.X2 > box.X1 && box.Y2 > box.Y1))
                throw new ArgumentException($"Box {i} (\"{box.Phrase}\") has no area.", nameof(boxes));
            targets[i] = new[] { (box.X1 + box.X2) / 2, (box.Y1 + box.Y2) / 2, box.X2 - box.X1, box.Y2 - box.Y1 };
            phrases[i] = prompt.PhraseIndex(box.Phrase);
        }
        return (batch, prompt, new GroundingTarget(targets, phrases));
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
        var (batch, prompt, target) = Supervision(image, caption, boxes);
        var loss = new GroundingDinoLoss<T>(Engine);
        var ids = prompt.IdMatrix;
        var mask = prompt.Mask;
        TrainWithCustomObjective(batch, batch, (x, _) => loss.Compute(Network.Forward(PreprocessImage(x), ids, mask), ids, mask, new[] { target }),
            _optimizer);
    }

    /// <summary>The paper's loss on an image, its caption and boxes, without training (dropout off).</summary>
    public T EvaluateGroundingLoss(Tensor<T> image, string caption, IReadOnlyList<GroundingBox> boxes)
    {
        ThrowIfDisposed();
        var (batch, prompt, target) = Supervision(image, caption, boxes);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return new GroundingDinoLoss<T>(Engine).Compute(Run(batch, prompt), prompt.IdMatrix, prompt.Mask, new[] { target })[0];
    }

    /// <summary>Grounding DINO trains on grounded phrases, which a tensor target cannot carry.</summary>
    /// <exception cref="NotSupportedException">Always: use <see cref="TrainGrounding"/>.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expected)
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

    /// <summary>Normalizes <c>[B, 3, H, W]</c> images in 0..1 with the options' mean and deviation (ImageNet's).</summary>
    protected override Tensor<T> PreprocessImage(Tensor<T> image)
    {
        var batch = Batched(image);
        int b = batch.Shape[0];
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 });
        var scale = new Tensor<T>(new[] { 1, 3, 1, 1 });
        for (int c = 0; c < 3; c++)
        {
            mean[0, c, 0, 0] = NumOps.FromDouble(_options.ImageMean[c]);
            scale[0, c, 0, 0] = NumOps.FromDouble(1.0 / _options.ImageStd[c]);
        }
        var shape = batch._shape;
        return Engine.TensorMultiply(Engine.TensorSubtract(batch, Engine.TensorBroadcastTo(mean, shape)), Engine.TensorBroadcastTo(scale, shape));
    }

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
