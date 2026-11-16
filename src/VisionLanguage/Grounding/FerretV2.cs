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
/// Ferret-v2: improved referring and grounding with enhanced spatial understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ferret-v2 (Zhang et al., 2024) improves upon Ferret with multi-granularity visual encoding
/// that processes images at multiple resolutions, enhanced spatial understanding through DINOv2
/// features, and a flexible any-resolution scaling approach for handling diverse aspect ratios
/// and input sizes. The multi-granularity design captures both fine-grained details and global context.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Ferret-v2: An Improved Baseline for Referring and Grounding with Multi-Granularity Visual Encoding" (Apple, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Ferret-v2 is an improved vision-language model for referring and grounding
/// that handles images at multiple resolutions for better spatial understanding. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Ferret-v2 model for improved referring and grounding
/// // with multi-granularity visual encoding and enhanced spatial understanding
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new FerretV2&lt;double&gt;(architecture, "ferretv2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new FerretV2&lt;double&gt;(architecture, new FerretV2Options());
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
    "Ferret-v2: An Improved Baseline for Referring and Grounding with Multi-Granularity Visual Encoding",
    "https://arxiv.org/abs/2404.07973",
    Year = 2024,
    Authors = "Zhang et al."
)]
public partial class FerretV2<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly FerretV2Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private readonly List<ILayer<T>> _groundingHeadLayers = new();
    private CoordinateTokenDecoder<T>? _groundingHead;

    public FerretV2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FerretV2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new FerretV2Options();
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

    public FerretV2(
        NeuralNetworkArchitecture<T> architecture,
        FerretV2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new FerretV2Options();
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
    /// Grounds text using Ferret-v2's any-resolution high-res grounding approach.
    /// Per the paper (Zhang et al., Apple 2024), Ferret-v2 improves over Ferret by:
    /// (1) Any-resolution encoding: the image is processed at multiple scales - a
    /// global low-res view and multiple high-res sub-image crops, (2) DINOv2
    /// features are fused with CLIP features for better spatial awareness,
    /// (3) improved spatial sampler with multi-granularity region understanding.
    /// The multi-scale approach enables fine-grained grounding at higher resolutions
    /// while maintaining global context from the full image.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        // Backbone: the image tokens the grounding head reads.
        var visualFeatures = PreprocessImage(image);
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Ferret-v2 (Zhang et al. 2024): the language model writes each box as discrete coordinates in [0, 999].
        var head = _groundingHead ?? throw new InvalidOperationException("The grounding head is not initialized.");
        var (boxes, scores) = head.Ground(visualFeatures, TokenIds(textQuery), RunLanguageModel, _options.MaxDetections);
        return GroundingDetections<T>.Ranked(Engine, boxes, scores, _options.ConfidenceThreshold, _options.MaxDetections,
            _options.NmsThreshold);
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
                LayerHelper<T>.CreateDefaultLLaVAMLPProjectorLayers(
                    _options.VisionDim,
                    _options.VisionDim * 4,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            ComputeEncoderDecoderBoundary();
        }
        if (_groundingHead is null)
        {
            _groundingHead = new CoordinateTokenDecoder<T>(Engine, _groundingHeadLayers, new CoordinateTokenDecoderConfiguration(
                _options.DecoderDim, _options.VocabSize, _options.MaxSequenceLength, Bins: 1000, Divisor: 1000, CoordinateSpelling.Integer));
            RegisterAuxiliaryEncoderStream(_groundingHeadLayers);
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 3;
    }

    private List<int> TokenIds(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = _tokenizer.Encode(text).TokenIds;
        return ids.Count > 0 ? ids : throw new ArgumentException("The text query has no tokens.", nameof(text));
    }

    /// <summary>The language model (the decoder layers after the image projector) on <c>[n, DecoderDim]</c> rows.</summary>
    private Tensor<T> RunLanguageModel(Tensor<T> rows)
    {
        var hidden = Engine.Reshape(rows, new[] { 1, rows.Shape[0], rows.Shape[1] });
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            hidden = Layers[i].Forward(hidden);
        return hidden;
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
            Name = _useNativeMode ? "Ferret-v2-Native" : "Ferret-v2-ONNX",
            Description =
                "Ferret-v2: improved referring and grounding with enhanced spatial understanding.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Ferret-v2";
        m.AdditionalInfo["FreeFormRegions"] = _options.EnableFreeFormRegions.ToString();
        m.AdditionalInfo["HighResolution"] = _options.EnableHighResolution.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(FerretV2<T>));
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
