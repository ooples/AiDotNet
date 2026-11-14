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
/// Groma: localized visual tokenization for grounded understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Groma (Ma et al., 2024) introduces localized visual tokenization for grounding multimodal LLMs.
/// It uses a detect-then-describe pipeline with a region proposal network to extract candidate
/// object regions, RoI-Align for fixed-size region feature extraction, and quantized region tokens
/// interleaved with text tokens, enabling the LLM to refer to and describe specific image regions
/// without requiring explicit coordinate output in the text stream.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models" (ByteDance, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Groma is a vision-language model that grounds language understanding
/// in specific image regions using localized visual tokens. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Groma model for localized visual grounding
/// // with region-level visual tokenization for region-text alignment
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Groma&lt;double&gt;(architecture, "groma.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Groma&lt;double&gt;(architecture, new GromaOptions());
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
    "Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models",
    "https://arxiv.org/abs/2404.13013",
    Year = 2024,
    Authors = "Ma et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, WeightDecay = 1e-4,
                ReferenceBatchSize = 64, Phase = TrainingPhase.PreTraining,
                Source = "Ma et al. 2024, training configuration table: detection pre-training uses "
                        + "AdamW for 12 epochs at a batch size of 64, a learning rate of 2e-4 and a "
                        + "weight decay of 1e-4.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, WeightDecay = 0,
                ReferenceBatchSize = 128, Phase = TrainingPhase.Alignment,
                Source = "Ma et al. 2024: alignment pre-training runs 2 epochs at a batch size of 128, "
                        + "a learning rate of 1e-4 and no weight decay. The zero decay is the table's "
                        + "value.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-5, WeightDecay = 0,
                ReferenceBatchSize = 128, Phase = TrainingPhase.FineTuning,
                Source = "Ma et al. 2024: instruction fine-tuning runs 1 epoch at a batch size of 128, "
                        + "a learning rate of 1e-5 and no weight decay.")]
public partial class Groma<T> : VisionLanguageModelBase<T>, IVisualGroundingModel<T>
{
    private readonly GromaOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public Groma(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GromaOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new GromaOptions();
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

    public Groma(
        NeuralNetworkArchitecture<T> architecture,
        GromaOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GromaOptions();
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
    /// Grounds text using Groma's localized visual tokenization approach.
    /// Per the paper (Ma et al., ByteDance 2024), Groma grounds multimodal LLMs
    /// by introducing region tokens via a detect-then-describe pipeline:
    /// (1) A region proposal network extracts candidate object regions from visual
    /// features using learned region queries, (2) RoI-Align extracts fixed-size
    /// features from each proposed region, (3) region tokens are quantized into
    /// discrete visual tokens and interleaved with text tokens for the LLM,
    /// (4) the LLM can refer to specific regions and generate localized descriptions.
    /// The localized tokenization enables region-level understanding without
    /// requiring explicit coordinate output in the text stream.
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

        // Step 1: Visual encoder
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        int visDim = visualFeatures.Length;

        // Step 2: Region proposal network with learned region queries. Query q reads its stride of visual features;
        // feature d sits at grid cell (d % grid, d / grid), and |feature| weights that cell position.
        int numRegionQueries = 64;
        int gridSize = (int)Math.Sqrt(visDim);
        if (gridSize < 4)
            gridSize = 4;
        int stride = Math.Max(1, visDim / numRegionQueries);
        var featurePositions = new int[numRegionQueries * stride];
        var cellColumns = new int[numRegionQueries * stride];
        var cellRows = new int[numRegionQueries * stride];
        for (int q = 0; q < numRegionQueries; q++)
        {
            for (int k = 0; k < stride; k++)
            {
                int d = q * stride + k;
                featurePositions[d] = d < visDim ? d : -1;
                cellColumns[d] = d % gridSize;
                cellRows[d] = d / gridSize;
            }
        }

        int[] queryShape = [numRegionQueries, stride];
        var weights = Engine.TensorAbs(GroundingDetections<T>.Take(Engine, visualFeatures, featurePositions, queryShape));
        var grid = NumOps.FromDouble(gridSize);
        var spatialX = Engine.TensorDivideScalar(GroundingDetections<T>.IntsAsT(cellColumns).Reshape(queryShape), grid);
        var spatialY = Engine.TensorDivideScalar(GroundingDetections<T>.IntsAsT(cellRows).Reshape(queryShape), grid);

        var attnSum = GroundingDetections<T>.SumRows(Engine, weights);
        var hasWeight = Engine.TensorGreaterThan(attnSum, NumOps.FromDouble(1e-8));
        var safeSum = Engine.TensorWhere(hasWeight, attnSum,
            Tensor<T>.CreateDefault([numRegionQueries], NumOps.One));
        var middle = Tensor<T>.CreateDefault([numRegionQueries], NumOps.FromDouble(0.5));
        var cx = Engine.TensorWhere(hasWeight,
            Engine.TensorDivide(GroundingDetections<T>.SumRows(Engine, Engine.TensorMultiply(spatialX, weights)), safeSum),
            middle);
        var cy = Engine.TensorWhere(hasWeight,
            Engine.TensorDivide(GroundingDetections<T>.SumRows(Engine, Engine.TensorMultiply(spatialY, weights)), safeSum),
            middle);

        // Predict width/height from the weighted spatial variance: max(0.05, 4 sqrt(var)), at most 1
        Tensor<T> Extent(Tensor<T> spatial, Tensor<T> centre)
        {
            var offset = Engine.TensorSubtract(spatial,
                Engine.TensorBroadcastTo(centre.Reshape([numRegionQueries, 1]), queryShape));
            var spread = GroundingDetections<T>.SumRows(Engine,
                Engine.TensorMultiply(weights, Engine.TensorMultiply(offset, offset)));
            var variance = Engine.TensorWhere(hasWeight, Engine.TensorDivide(spread, safeSum), spread);
            return Engine.TensorClampMax(Engine.TensorMax(
                Engine.TensorMultiplyScalar(Engine.TensorSqrt(variance), NumOps.FromDouble(4.0)), NumOps.FromDouble(0.05)),
                NumOps.One);
        }

        var regionW = Extent(spatialX, cx);
        var regionH = Extent(spatialY, cy);
        var regionScores = Engine.TensorDivideScalar(attnSum, NumOps.FromDouble(stride));

        // Step 3: Text-conditioned region scoring via ConcatenateTensors
        var textTokens = TokenizeText(textQuery);
        var fusedFeatures = visualFeatures.ConcatenateTensors(textTokens);

        // Run through decoder layers for text-conditioned scoring
        var decoderOut = fusedFeatures;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            decoderOut = Layers[i].Forward(decoderOut);

        int outDim = decoderOut.Length;
        int maxDet = _options.MaxDetections;
        if (maxDet <= 0)
            return new Tensor<T>([GroundingDetections<T>.FieldsPerDetection]);

        // Score each region against text-conditioned decoder features: the region of interest starts at
        // floor(cx * outDim) % outDim and spans max(1, floor(w * outDim / 2)) values, read cyclically. The
        // positions are integers, so they are derived without rounding the products to T.
        int[] roiStarts = GroundingDetections<T>.FloorScaled(Engine, cx, outDim);
        int[] roiSpans = GroundingDetections<T>.FloorScaled(Engine, regionW, outDim * 0.5);
        int longestSpan = 1;
        for (int q = 0; q < numRegionQueries; q++)
        {
            roiStarts[q] %= outDim;
            roiSpans[q] = Math.Max(1, roiSpans[q]);
            longestSpan = Math.Max(longestSpan, roiSpans[q]);
        }

        var roiPositions = new int[numRegionQueries * longestSpan];
        for (int q = 0; q < numRegionQueries; q++)
        {
            for (int d = 0; d < longestSpan; d++)
                roiPositions[q * longestSpan + d] = d < roiSpans[q] ? (roiStarts[q] + d) % outDim : -1;
        }

        var roiFeatSum = GroundingDetections<T>.SumRows(Engine, GroundingDetections<T>.Take(Engine, decoderOut,
            roiPositions, [numRegionQueries, longestSpan]));

        // Divide by each span through its reciprocal, formed in double and rounded to T once: a span past 2048
        // is not exact in Half, and one past 65504 overflows it.
        var inverseSpans = new double[numRegionQueries];
        for (int q = 0; q < numRegionQueries; q++)
            inverseSpans[q] = 1.0 / roiSpans[q];
        var roiConf = Engine.TensorSigmoid(Engine.TensorMultiply(roiFeatSum,
            new Tensor<double>(new[] { numRegionQueries }, Vector<double>.FromArray(inverseSpans)).Cast<T>()));
        var confidences = Engine.TensorSigmoid(Engine.TensorAddScalar(
            Engine.TensorMultiplyScalar(Engine.TensorMultiply(roiConf, regionScores), NumOps.FromDouble(5.0)),
            NumOps.FromDouble(-2.0)));

        var boxes = GroundingDetections<T>.ClippedCornerBoxes(Engine, cx, cy, regionW, regionH);
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
            Name = _useNativeMode ? "Groma-Native" : "Groma-ONNX",
            Description = "Groma: localized visual tokenization for grounded understanding.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Groma";
        m.AdditionalInfo["LocalizedTokenization"] = _options.EnableLocalizedTokenization.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Groma<T>));
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
