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
    private readonly List<ILayer<T>> _groundingHeadLayers = new();
    private GromaRegionGrounder<T>? _groundingHead;

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

        // Backbone: the image tokens the grounding head reads.
        var visualFeatures = PreprocessImage(image);
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Region proposer, region tokens, and the language model referring to them (Ma et al. 2024, section 3).
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
            // Groma's proposer is a Deformable DETR (width 256, 8 heads, 6 decoder layers) proposing 300 regions.
            _groundingHead = new GromaRegionGrounder<T>(Engine, _groundingHeadLayers, new GromaRegionGrounderConfiguration(
                _options.DecoderDim, ProposerWidth: 256, ProposerHeads: 8, ProposerLayers: 6, Proposals: 300,
                _options.VocabSize, _options.MaxSequenceLength));
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
