using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// OWLv2: self-training for scaling open-vocabulary detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OWLv2 (Minderer et al., 2023) scales open-vocabulary object detection through self-training
/// on web-scale image-text data. Building on OWL-ViT, it uses a stronger ViT-L/14 backbone
/// with 960px input resolution, pseudo-label self-training from an OWL-ViT teacher model, and
/// objectness-calibrated scoring for better separation of true objects from background patches.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Scaling Open-Vocabulary Object Detection" (Google, 2023)</item></list></para>
/// <para><b>For Beginners:</b> OWLv2 is an improved version of OWL-ViT that achieves better
/// open-vocabulary detection through self-training at web scale. Default values follow the
/// original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an OWLv2 model for scaled open-vocabulary detection
/// // with self-training and objectness-calibrated scoring
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new OWLv2&lt;double&gt;(architecture, "owlv2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new OWLv2&lt;double&gt;(architecture, new OWLv2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Scaling Open-Vocabulary Object Detection",
    "https://arxiv.org/abs/2306.09683",
    Year = 2023,
    Authors = "Minderer et al."
)]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 2e-5,
                Phase = TrainingPhase.FineTuning,
                Source = "Minderer et al. 2023, Sec. 4: fine-tuning switched to a learning rate of 2e-5 "
                        + "after 740,000 self-training steps, having overfit at the earlier setting. The "
                        + "optimizer is left unspecified because the paper names none of its own -- the "
                        + "Adam in its text is a reference to Kingma and Ba.")]
public partial class OWLv2<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly OWLv2Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private readonly List<ILayer<T>> _groundingHeadLayers = new();
    private OwlDetectionHead<T>? _groundingHead;

    public OWLv2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        OWLv2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new OWLv2Options();
        Options = _options;
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public OWLv2(
        NeuralNetworkArchitecture<T> architecture,
        OWLv2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new OWLv2Options();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <inheritdoc />
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    /// <inheritdoc />
    public int MaxDetections => _options.MaxDetections;

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        var c = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Grounds text using OWLv2's scaled open-vocabulary detection approach.
    /// Per the paper (Minderer et al., Google 2023), OWLv2 improves OWL-ViT
    /// through self-training on web-scale image-text data. Key innovations:
    /// (1) a stronger ViT-L/14 backbone with 960px input, (2) self-training
    /// with pseudo-labels from an OWL-ViT teacher, (3) objectness-calibrated
    /// scoring that better separates true objects from background, (4) larger
    /// NumClassEmbeddings (768) for richer class representations.
    /// The self-training pseudo-label refinement uses confidence recalibration.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        // OWLv2 (Minderer et al. 2023): the OWL-ViT per-token box and class heads plus an objectness head.
        var prediction = Detect(image, textQuery);
        var scores = Engine.TensorSigmoid(prediction.Logits.Reshape(new[] { prediction.Logits.Shape[0] }));
        return GroundingDetections<T>.Ranked(Engine, GroundingHeadModule<T>.Corners(Engine, prediction.Boxes), scores,
            _options.ConfidenceThreshold, _options.MaxDetections, _options.NmsThreshold);
    }

    /// <summary>
    /// The training loss of one image (OWL-ViT section 3.2, DETR's bipartite matching loss with a focal sigmoid class
    /// loss): the per-token predictions matched one-to-one to the ground-truth boxes of the query, then focal, L1 and GIoU.
    /// </summary>
    /// <param name="image">The image.</param>
    /// <param name="textQuery">The text query.</param>
    /// <param name="targetBoxes">The query's ground-truth boxes <c>[G, 4]</c>, normalized (cx, cy, w, h).</param>
    internal GroundingSetLossTerms<T> GroundingLoss(Tensor<T> image, string textQuery, Tensor<T> targetBoxes)
    {
        ThrowIfDisposed();
        if (targetBoxes is null) throw new ArgumentNullException(nameof(targetBoxes));
        var prediction = Detect(image, textQuery);
        var labels = Tensor<T>.CreateDefault(new[] { targetBoxes.Shape[0], 1 }, NumOps.One);
        return new GroundingSetLoss<T>(Engine).Compute(prediction.Boxes, prediction.Logits, targetBoxes, labels);
    }

    // ViT backbone with a class token (OWL-ViT section 3.1): the patch embedding, the learned class token ahead of the
    // patch tokens, then the transformer over the one sequence; then the head against the embedded query.
    private OwlPrediction<T> Detect(Tensor<T> image, string textQuery)
    {
        var head = _groundingHead ?? throw new InvalidOperationException("The grounding head is not initialized.");
        var patches = GroundingHeadModule<T>.AsRows(Engine, Layers[0].Forward(PreprocessImage(image)));
        var visualFeatures = Engine.TensorConcatenate(new[] { head.ClassToken(), patches }, 0);
        visualFeatures = visualFeatures.Reshape(new[] { 1, visualFeatures.Shape[0], visualFeatures.Shape[1] });
        for (int i = 1; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        return head.Forward(visualFeatures, head.EmbedQuery(TokenIds(textQuery)));
    }

    /// <inheritdoc />
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        ThrowIfDisposed();
        string combined = string.Join(". ", categories) + ".";
        return GroundText(image, combined);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultGroundingDetectionLayers(
                    _options.VisionDim,
                    _options.TextEmbeddingDim,
                    _options.DecoderDim,
                    _options.DetectionDim,
                    _options.NumVisionLayers,
                    _options.NumFusionLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            ComputeEncoderDecoderBoundary();
        }
        if (_groundingHead is null)
        {
            // CLIP's text transformer uses 64-wide attention heads.
            int textHeads = Math.Max(1, _options.TextEmbeddingDim / 64);
            _groundingHead = new OwlDetectionHead<T>(Engine, _groundingHeadLayers, _options.VisionDim, new OwlDetectionHeadConfiguration(
                _options.NumClassEmbeddings, _options.TextEmbeddingDim, _options.NumTextLayers, textHeads, _options.VocabSize,
                _options.MaxSequenceLength, Objectness: true));
            RegisterAuxiliaryEncoderStream(_groundingHeadLayers);
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        // Each vision block is one residual TransformerEncoderLayer (+ an optional DropoutLayer);
        // the encoder prefix is the linear input projection (patch-embedding stand-in) and the
        // trailing prefix is the final encoder LayerNorm — 2 non-block layers total.
        int lpb = _options.DropoutRate > 0 ? 2 : 1;
        _encoderLayerEnd = 2 + _options.NumVisionLayers * lpb;
    }

    private List<int> TokenIds(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = _tokenizer.Encode(text).TokenIds;
        return ids.Count > 0 ? ids : throw new ArgumentException("The text query has no tokens.", nameof(text));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "OWLv2-Native" : "OWLv2-ONNX",
            Description = "OWLv2: self-training for scaling open-vocabulary detection.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "OWLv2";
        m.AdditionalInfo["NumClassEmbeddings"] = _options.NumClassEmbeddings.ToString();
        m.AdditionalInfo["SelfTraining"] = _options.EnableSelfTraining.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(OWLv2<T>));
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
