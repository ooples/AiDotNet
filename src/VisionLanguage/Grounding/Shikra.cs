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
/// Shikra: referential dialogue model with coordinate output for grounding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Shikra (Chen et al., 2023) is a referential dialogue model that treats spatial coordinates
/// as plain-text numbers in the LLM output vocabulary. Instead of using special tokens or
/// external detection heads, it generates normalized bounding box coordinates as decimal text
/// tokens via CLIP ViT visual encoding interleaved with instruction tokens, enabling natural
/// language grounding through straightforward autoregressive generation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic" (SenseTime, 2023)</item></list></para>
/// <para><b>For Beginners:</b> Shikra is a vision-language model that grounds objects by generating
/// their coordinates as plain text numbers in natural dialogue. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Shikra model for referential dialogue with coordinate grounding
/// // generating bounding box coordinates as plain-text numbers
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Shikra&lt;double&gt;(architecture, "shikra.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Shikra&lt;double&gt;(architecture, new ShikraOptions());
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
    "Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic",
    "https://arxiv.org/abs/2306.15195",
    Year = 2023,
    Authors = "Chen et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, ReferenceBatchSize = 64,
                MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Chen et al. 2023, Sec. 4: AdamW under a cosine learning rate scheduler with "
                        + "an initial learning rate of 2e-5 and a global batch size of 64.")]
public partial class Shikra<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly ShikraOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public Shikra(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ShikraOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ShikraOptions();
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

    public Shikra(
        NeuralNetworkArchitecture<T> architecture,
        ShikraOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ShikraOptions();
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
    /// Grounds text using Shikra's referential dialogue with numeric coordinate output.
    /// Per the paper (Chen et al., SenseTime 2023), Shikra is a multimodal LLM that
    /// treats spatial coordinates as plain-text numbers in the output vocabulary. Instead
    /// of special tokens or external heads, coordinates [x1, y1, x2, y2] are generated
    /// as normalized (0-1) decimal numbers directly in the text stream. The model:
    /// (1) encodes the image via CLIP ViT into visual tokens, (2) interleaves with
    /// instruction tokens, (3) the LLM autoregressively generates coordinate values
    /// as text tokens when asked "where is [object]". The coordinate tokens are
    /// parsed from the LLM output to extract bounding boxes. For grounding, the
    /// LLM generates box coordinates for each referred object.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: CLIP ViT visual encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Interleave visual tokens with text instruction via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var multimodalSeq = visualFeatures.ConcatenateTensors(textTokens);

        // Step 3: LLM decoder to generate coordinate tokens
        var decoderOut = multimodalSeq;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 4: Parse coordinate tokens from LLM output
        // Shikra generates coordinates as groups of 4 values (x1,y1,x2,y2)
        // Each coordinate is sigmoid-bounded to [0,1]
        int coordsPerBox = 4;
        int maxDet = Math.Min(outDim / coordsPerBox, _options.MaxDetections);
        if (maxDet <= 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);

        var raw = Engine.TensorSlice(decoderOut.Reshape([outDim]), [0], [maxDet * coordsPerBox])
            .Reshape([maxDet, coordsPerBox]);
        var unit = Engine.TensorSigmoid(raw);
        Tensor<T> Column(int index) => Engine.TensorSlice(unit, [0, index], [maxDet, 1]);

        // Ensure x2 >= x1, y2 >= y1
        var boxes = Engine.TensorConcatenate(
            [
                Engine.TensorMin(Column(0), Column(2)),
                Engine.TensorMin(Column(1), Column(3)),
                Engine.TensorMax(Column(0), Column(2)),
                Engine.TensorMax(Column(1), Column(3))
            ], 1);

        // Confidence from coordinate token magnitudes: sigmoid(sum |raw| / 4 - 2)
        var coordMagnitude = Engine.ReduceSum(Engine.TensorAbs(raw), [1], false);
        var confidences = Engine.TensorSigmoid(Engine.TensorAddScalar(
            Engine.TensorDivideScalar(coordMagnitude, NumOps.FromDouble(4.0)), NumOps.FromDouble(-2.0)));

        // Valid box check: confident, positive extent and area above 0.001
        var width = Engine.TensorSubtract(Engine.TensorSlice(boxes, [0, 2], [maxDet, 1]), Engine.TensorSlice(boxes, [0, 0], [maxDet, 1]));
        var height = Engine.TensorSubtract(Engine.TensorSlice(boxes, [0, 3], [maxDet, 1]), Engine.TensorSlice(boxes, [0, 1], [maxDet, 1]));
        var largeEnough = Engine.TensorGreaterThan(Engine.TensorMultiply(width, height), NumOps.FromDouble(0.001)).Reshape([maxDet]);
        var eligible = Engine.TensorMultiply(Engine.TensorMultiply(
            GroundingDetections<T>.AtLeast(Engine, confidences, NumOps.FromDouble(confThreshold)),
            GroundingDetections<T>.HasArea(Engine, boxes)), largeEnough);

        // Step 5: NMS, highest confidence first
        return GroundingDetections<T>.Select(Engine, boxes, confidences, eligible, maxDet, nmsThreshold);
    }

    /// <inheritdoc />
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        ThrowIfDisposed();
        // Shikra treats each category as a referential dialogue question
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
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = 1 + _options.NumVisionLayers * lpb + 3;
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
            Name = _useNativeMode ? "Shikra-Native" : "Shikra-ONNX",
            Description =
                "Shikra: referential dialogue model with coordinate output for grounding.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Shikra";
        m.AdditionalInfo["CoordinateOutput"] = _options.EnableCoordinateOutput.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Shikra<T>));
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
