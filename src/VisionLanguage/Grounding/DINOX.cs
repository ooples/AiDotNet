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
/// DINO-X: strongest open-world perception model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DINO-X (Ren et al., 2024) extends Grounding DINO with a universal object prompt module
/// that supports text, visual exemplars, and custom prompts. Trained on Grounding-100M
/// (100M+ images with long-tailed open-world categories), it uses a foundation encoder
/// with prompt-guided deformable cross-attention for open-world object detection and
/// understanding across diverse domains.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding" (IDEA, 2024)</item></list></para>
/// <para><b>For Beginners:</b> DINOX is a vision-language model that can detect and understand
/// objects using text descriptions, visual examples, or custom prompts. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a DINO-X model for open-world object detection
/// // the strongest unified open-world perception model
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new DINOX&lt;double&gt;(architecture, "dinox.onnx");
///
/// // Training mode with native layers
/// var trainModel = new DINOX&lt;double&gt;(architecture, new DINOXOptions());
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
    "DINO-X: A Unified Vision Model for Open-World Object Detection and Understanding",
    "https://arxiv.org/abs/2411.14347",
    Year = 2024,
    Authors = "Ren et al."
)]
public partial class DINOX<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly DINOXOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public DINOX(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        DINOXOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new DINOXOptions();
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

    public DINOX(
        NeuralNetworkArchitecture<T> architecture,
        DINOXOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new DINOXOptions();
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
    /// Grounds a text query using DINO-X's universal perception architecture.
    /// Per the paper (IDEA, 2024), DINO-X extends Grounding DINO with a universal
    /// object prompt that supports text, visual exemplars, and custom prompts.
    /// It uses a foundation encoder trained on 100M+ images with a long-tailed
    /// open-world dataset (Grounding-100M). The key innovation is the universal
    /// prompt module that encodes text via multi-head prompt embeddings and fuses
    /// them with visual features through a prompt-guided deformable cross-attention.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int numQueries = _options.NumQueryPositions;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: Foundation encoder - extract multi-scale visual features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Universal prompt encoding via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Step 3: Decoder with prompt-guided cross-modal features
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        // Step 5: Query decoding with box prediction
        int maxDet = Math.Min(numQueries, _options.MaxDetections);
        if (maxDet <= 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);

        GroundingDetections<T>.QuarterBlockQueries(Engine, decoderOut, maxDet, out var boxes, out var confidences);
        var eligible = Engine.TensorMultiply(
            GroundingDetections<T>.AtLeast(Engine, confidences, NumOps.FromDouble(confThreshold)),
            GroundingDetections<T>.HasArea(Engine, boxes));

        // Step 6: NMS, highest confidence first
        return GroundingDetections<T>.Select(Engine, boxes, confidences, eligible, maxDet, nmsThreshold);
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
    }

    private void ComputeEncoderDecoderBoundary()
    {
        // Each vision block is one residual TransformerEncoderLayer (+ an optional DropoutLayer);
        // the encoder prefix is the linear input projection (patch-embedding stand-in) and the
        // trailing prefix is the final encoder LayerNorm — 2 non-block layers total.
        int lpb = _options.DropoutRate > 0 ? 2 : 1;
        _encoderLayerEnd = 2 + _options.NumVisionLayers * lpb;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = TokenIdsToTensor(encoding.TokenIds, seqLen);
        return tokens;
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
            Name = _useNativeMode ? "DINO-X-Native" : "DINO-X-ONNX",
            Description = "DINO-X: strongest open-world perception model.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "DINO-X";
        m.AdditionalInfo["UniversalPerception"] = _options.EnableUniversalPerception.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(DINOX<T>));
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
