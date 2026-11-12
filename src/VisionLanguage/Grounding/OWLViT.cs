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
/// OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// OWL-ViT (Minderer et al., 2022) repurposes a CLIP ViT image encoder for open-vocabulary
/// detection by treating each patch token as a candidate object. Per-patch MLP heads predict
/// bounding boxes, while class scores come from cosine similarity between patch tokens and
/// CLIP text embeddings of category names. This is a query-free approach where every ViT
/// patch serves as a candidate detection, eliminating the need for learned object queries.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Simple Open-Vocabulary Object Detection with Vision Transformers" (Google, 2022)</item></list></para>
/// <para><b>For Beginners:</b> OWL-ViT is a vision-language model that detects objects using
/// text descriptions by leveraging CLIP's vision-language alignment at the patch level. Default
/// values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create an OWL-ViT model for open-vocabulary object detection
/// // using CLIP-aligned patch-level detection without learned queries
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new OWLViT&lt;double&gt;(architecture, "owlvit.onnx");
///
/// // Training mode with native layers
/// var trainModel = new OWLViT&lt;double&gt;(architecture, new OWLViTOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Detection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Simple Open-Vocabulary Object Detection with Vision Transformers",
    "https://arxiv.org/abs/2205.06230",
    Year = 2022,
    Authors = "Minderer et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.999, ReferenceBatchSize = 256,
                MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Minderer et al. 2022, Sec. 3: the Adam optimizer with beta1 0.9 and beta2 "
                        + "0.999 under cosine learning rate decay, fine-tuned at a batch size of 256 for "
                        + "at most 140,000 steps. No learning rate is declared because the paper sets it "
                        + "per model size -- 2e-3, 2e-4 and 2e-5 across its variants. The 0.3 that "
                        + "appears alongside is the focal loss alpha, not a rate.")]
public partial class OWLViT<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly OWLViTOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public OWLViT(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        OWLViTOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new OWLViTOptions();
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

    public OWLViT(
        NeuralNetworkArchitecture<T> architecture,
        OWLViTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new OWLViTOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this));
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
    /// Grounds text using OWL-ViT's CLIP-aligned patch-level detection approach.
    /// Per the paper (Minderer et al., Google 2022), OWL-ViT repurposes a
    /// CLIP ViT image encoder for detection by: (1) removing the final pooling
    /// token and using per-patch ViT output tokens as object representations,
    /// (2) each patch token gets a lightweight MLP box prediction head,
    /// (3) class prediction via cosine similarity between patch tokens and
    /// CLIP text embeddings of category names, (4) per-patch objectness score
    /// from a linear head. This is a query-free approach: every ViT patch is
    /// a candidate detection, no learned object queries needed.
    /// Output format: [x1, y1, x2, y2, confidence] per detection.
    /// </summary>
    public Tensor<T> GroundText(Tensor<T> image, string textQuery)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(PreprocessImage(image));

        var p = PreprocessImage(image);
        int dim = _options.DecoderDim;
        double confThreshold = _options.ConfidenceThreshold;
        double nmsThreshold = _options.NmsThreshold;

        // Step 1: ViT backbone - extract per-patch token features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Text-conditioned feature fusion via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedInput = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned detection features
        var decoderOut = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;

        // Step 3: Per-patch detection - each patch produces box + objectness
        int patchSize = 16;
        int gridSize = _options.ImageSize / patchSize;
        int numPatches = gridSize * gridSize;
        int actualPatches = Math.Min(numPatches, outDim / Math.Max(1, dim));
        if (actualPatches < 1)
            actualPatches = Math.Min(numPatches, outDim);
        int maxDet = Math.Min(actualPatches, _options.MaxDetections);
        if (maxDet <= 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);
        if (dim <= 0)
            throw new InvalidOperationException("OWL-ViT needs a positive DecoderDim.");

        // Patch p reads dim decoder values from (p * dim) % outDim, stopping at the end of the output (the rest
        // reads zero), and L2-normalizes them.
        var positions = new int[actualPatches * dim];
        for (int pIdx = 0; pIdx < actualPatches; pIdx++)
        {
            int featStart = (pIdx * dim) % outDim;
            for (int d = 0; d < dim; d++)
                positions[pIdx * dim + d] = featStart + d < outDim ? featStart + d : -1;
        }

        var features = GroundingDetections<T>.Take(Engine, decoderOut, positions, [actualPatches, dim]);
        var norms = GroundingDetections<T>.RowNorm(Engine, features).Reshape([actualPatches, 1]);
        var safeNorms = Engine.TensorWhere(Engine.TensorGreaterThan(norms, NumOps.Zero),
            Engine.TensorAddScalar(norms, NumOps.FromDouble(1e-8)),
            Tensor<T>.CreateDefault([actualPatches, 1], NumOps.One));
        var unit = Engine.TensorDivide(features, Engine.TensorBroadcastTo(safeNorms, [actualPatches, dim]));

        // Objectness score from text-conditioned patch features, then the box regression
        var confidences = Engine.TensorSigmoid(Engine.TensorDivideScalar(
            Engine.ReduceSum(unit, [1], false), NumOps.FromDouble(dim)));
        var boxes = GroundingDetections<T>.PatchOffsetBoxes(Engine, unit, gridSize);
        var eligible = Engine.TensorMultiply(
            GroundingDetections<T>.AtLeast(Engine, confidences, NumOps.FromDouble(confThreshold)),
            GroundingDetections<T>.HasArea(Engine, boxes));

        // Step 4: NMS, highest confidence first
        return GroundingDetections<T>.Select(Engine, boxes, confidences, eligible, maxDet, nmsThreshold);
    }

    /// <inheritdoc />
    public Tensor<T> DetectObjects(Tensor<T> image, IReadOnlyList<string> categories)
    {
        ThrowIfDisposed();
        // OWL-ViT encodes each category separately and matches against patches
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
            Name = _useNativeMode ? "OWL-ViT-Native" : "OWL-ViT-ONNX",
            Description = "OWL-ViT: open-vocabulary object detection via ViT + CLIP alignment.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "OWL-ViT";
        m.AdditionalInfo["NumClassEmbeddings"] = _options.NumClassEmbeddings.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(OWLViT<T>));
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
