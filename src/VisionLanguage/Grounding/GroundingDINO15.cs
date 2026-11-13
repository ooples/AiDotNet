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
/// Grounding DINO 1.5: enhanced open-set detection with improved architecture.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Grounding DINO 1.5 (Ren et al., 2024) advances open-set object detection with a stronger
/// ViT-H backbone, multi-scale deformable attention for better handling of objects at different
/// scales, improved text-visual alignment with contrastive learning, and EfficientSAM integration
/// for segmentation-aware features. It builds on the original Grounding DINO architecture with
/// enhanced cross-modal fusion and larger-scale pre-training.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Grounding DINO 1.5: Advance the Edge of Open-Set Object Detection" (IDEA, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Grounding DINO 1.5 is an enhanced version of Grounding DINO with
/// a stronger backbone and multi-scale attention for improved open-set detection. Default values
/// follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Grounding DINO 1.5 model for enhanced open-set detection
/// // with multi-scale deformable attention and EfficientSAM integration
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new GroundingDINO15&lt;double&gt;(architecture, "groundingdino15.onnx");
///
/// // Training mode with native layers
/// var trainModel = new GroundingDINO15&lt;double&gt;(architecture, new GroundingDINO15Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Grounding DINO 1.5: Advance the Edge of Open-Set Object Detection",
    "https://arxiv.org/abs/2405.10300",
    Year = 2024,
    Authors = "Ren et al."
)]
public partial class GroundingDINO15<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GroundingDINO15Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private readonly List<ILayer<T>> _groundingHeadLayers = new();
    private GroundingQueryDecoder<T>? _groundingHead;

    public GroundingDINO15(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GroundingDINO15Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new GroundingDINO15Options();
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

    public GroundingDINO15(
        NeuralNetworkArchitecture<T> architecture,
        GroundingDINO15Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GroundingDINO15Options();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    /// Grounds a text query using Grounding DINO 1.5's enhanced architecture.
    /// Per the paper (IDEA, 2024), 1.5 improves over the original by using
    /// a stronger backbone (Swin-L or ViT-H), multi-scale deformable attention
    /// in the feature enhancer for better handling of objects at different scales,
    /// and an improved text-visual alignment module with contrastive learning.
    /// The model also integrates EfficientSAM for segmentation-aware features.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var prediction = Detect(image, TokenIds(textQuery));
        return GroundingDetections<T>.Ranked(Engine, GroundingHeadModule<T>.Corners(Engine, prediction.Boxes),
            prediction.Scores(Engine), _options.ConfidenceThreshold, _options.MaxDetections, _options.NmsThreshold);
    }

    /// <summary>
    /// The training loss of one image (Grounding DINO section 3.5): the head's queries matched one-to-one to the
    /// ground-truth boxes by the Hungarian algorithm, then the contrastive focal loss, L1 and GIoU, on the last decoder
    /// layer, every earlier decoder layer (auxiliary losses) and the encoder proposals.
    /// </summary>
    /// <param name="image">The image.</param>
    /// <param name="textQuery">The text prompt.</param>
    /// <param name="targetBoxes">The ground-truth boxes <c>[G, 4]</c>, normalized (cx, cy, w, h).</param>
    /// <param name="targetTokens">Each box's phrase as 0/1 over the prompt's tokens <c>[G, L]</c>.</param>
    internal GroundingDetectorLoss<T> GroundingLoss(Tensor<T> image, string textQuery, Tensor<T> targetBoxes, Tensor<T> targetTokens)
    {
        ThrowIfDisposed();
        var prediction = Detect(image, TokenIds(textQuery));
        return new GroundingSetLoss<T>(Engine).Compute(prediction, targetBoxes, targetTokens);
    }

    // The backbone's image tokens, then the feature enhancer, language-guided query selection, cross-modality decoder
    // and box / contrastive heads.
    private GroundingQueryPrediction<T> Detect(Tensor<T> image, IReadOnlyList<int> tokenIds)
    {
        var visualFeatures = PreprocessImage(image);
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        var head = _groundingHead ?? throw new InvalidOperationException("The grounding head is not initialized.");
        return head.Forward(visualFeatures, tokenIds);
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
                    768,
                    _options.VisionDim,
                    256,
                    _options.NumVisionLayers,
                    6,
                    6,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            ComputeEncoderDecoderBoundary();
        }
        if (_groundingHead is null)
        {
            // Grounding DINO's head (Liu et al. 2023, §3): Grounding DINO-T's 6 enhancer and 6 decoder layers, 8 heads,
            // FFN 2048 and 900 queries at width 256 in the paper configuration.
            _groundingHead = new GroundingQueryDecoder<T>(Engine, _groundingHeadLayers, new GroundingQueryDecoderConfiguration(
                _options.DecoderDim, _options.NumHeads, 8 * _options.DecoderDim, _options.NumDecoderLayers, _options.NumDecoderLayers,
                _options.NumQueryPositions, _options.VocabSize, _options.MaxSequenceLength));
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
            Name = _useNativeMode ? "GroundingDINO-1.5-Native" : "GroundingDINO-1.5-ONNX",
            Description =
                "Grounding DINO 1.5: enhanced open-set detection with improved architecture.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "GroundingDINO-1.5";
        m.AdditionalInfo["BackboneType"] = _options.BackboneType;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GroundingDINO15<T>));
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
