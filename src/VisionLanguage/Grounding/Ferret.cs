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
/// Ferret: spatial-aware visual sampler for free-form region referring and grounding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Ferret (You et al., 2023) introduces a spatial-aware visual sampler that can refer to and
/// ground anything at any granularity. It uses a hybrid region representation combining discrete
/// coordinates with continuous visual features and a spatial-aware visual sampler to handle
/// free-form referring regions of any shape including points, boxes, and scribbles.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Ferret: Refer and Ground Anything Anywhere at Any Granularity" (Apple, 2023)</item></list></para>
/// <para><b>For Beginners:</b> Ferret is a vision-language model that can locate objects in images
/// based on natural language descriptions and refer to regions of any shape. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Ferret model for region referring and grounding
/// // with spatial-aware visual sampler at any granularity
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Ferret&lt;double&gt;(architecture, "ferret.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Ferret&lt;double&gt;(architecture, new FerretOptions());
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
    "Ferret: Refer and Ground Anything Anywhere at Any Granularity",
    "https://arxiv.org/abs/2310.07704",
    Year = 2023,
    Authors = "You et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-5, ReferenceBatchSize = 128,
                Source = "You et al. 2023, Sec. 4: Ferret is trained on GRIT for three epochs at a "
                        + "learning rate of 2e-5 and a batch size of 128. The paper names its optimizer "
                        + "only by citation, Loshchilov and Hutter 2017, which is AdamW.")]
public partial class Ferret<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly FerretOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public Ferret(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FerretOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new FerretOptions();
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

    public Ferret(
        NeuralNetworkArchitecture<T> architecture,
        FerretOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new FerretOptions();
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
    /// Grounds text using Ferret's spatial-aware visual sampler for free-form referring.
    /// Per the paper (You et al., Apple 2023), Ferret introduces a hybrid region
    /// representation that handles points, boxes, and free-form shapes (scribbles,
    /// polygons). The key innovation is the spatial-aware visual sampler that:
    /// (1) samples visual features at discrete points within the referred region,
    /// (2) uses a spatial-aware pooling that weights features by their distance
    /// to the region center and boundary, (3) produces region tokens that are
    /// interleaved with text tokens for the LLM. For grounding (text-to-region),
    /// the LLM generates coordinate tokens [x1, y1, x2, y2] as part of its output.
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

        // Step 1: Visual encoder (CLIP ViT)
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Step 2: Text-conditioned feature fusion via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned spatial features
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;
        int maxDet = _options.MaxDetections;
        if (maxDet <= 0 || outDim == 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);

        // Step 3: Spatial attention from text-conditioned features, and Step 4: region proposals grown from its peaks
        GroundingDetections<T>.AttentionRegions(Engine, decoderOut, featureCap: 4, logitScale: 0.1, seedFactor: 1.5,
            growFactor: 0.7, confidenceOffset: 1.0, out var boxes, out var confidences);
        if (confidences.Length == 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);

        var eligible = GroundingDetections<T>.AtLeast(Engine, confidences, NumOps.FromDouble(confThreshold));

        // Step 5: NMS, highest confidence first
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
            Name = _useNativeMode ? "Ferret-Native" : "Ferret-ONNX",
            Description =
                "Ferret: spatial-aware visual sampler for free-form region referring and grounding.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Ferret";
        m.AdditionalInfo["FreeFormRegions"] = _options.EnableFreeFormRegions.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Ferret<T>));
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
