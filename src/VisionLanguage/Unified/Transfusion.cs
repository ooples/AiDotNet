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

namespace AiDotNet.VisionLanguage.Unified;

/// <summary>
/// Transfusion: combined autoregressive and diffusion loss in single transformer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Transfusion (Meta, 2024) combines autoregressive next-token prediction and continuous
/// diffusion loss within a single multi-modal transformer. Text tokens use cross-entropy loss
/// while image patches use diffusion denoising loss, allowing the model to both generate
/// text autoregressively and produce high-quality images through diffusion within a unified
/// architecture.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Meta, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Transfusion is a unified model that combines autoregressive
/// text generation with diffusion-based image generation. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Transfusion model combining autoregressive and diffusion objectives
/// // in a single transformer for unified text and image generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Transfusion&lt;double&gt;(architecture, "transfusion.onnx");
///
/// // Training mode with native layers
/// var trainModel = new Transfusion&lt;double&gt;(architecture, new TransfusionOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model",
    "https://arxiv.org/abs/2408.11039",
    Year = 2024,
    Authors = "Zhou et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.95,
                Epsilon = 1e-8, WarmupSteps = 4000, MinLearningRate = 1.5e-5,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Zhou et al. 2024, Sec. 3: all parameters are randomly initialized and "
                        + "optimized with AdamW at beta1 0.9, beta2 0.95 and epsilon 1e-8, at a learning "
                        + "rate of 3e-4 warmed up over 4000 steps and decaying to 1.5e-5 under a cosine "
                        + "scheduler. The stated batch of 4M tokens is not declared as a reference batch "
                        + "size, which counts examples rather than tokens.")]
public partial class Transfusion<T>
    : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>, ITrainingObjectiveProvider<T>
{
    private readonly TransfusionOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public Transfusion(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        TransfusionOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new TransfusionOptions();
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
        TryAutoEnableWeightStreaming();
    }

    public Transfusion(
        NeuralNetworkArchitecture<T> architecture,
        TransfusionOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new TransfusionOptions();
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
        TryAutoEnableWeightStreaming();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public bool SupportsGeneration => _options.SupportsGeneration;

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
    /// Generates text from image using Transfusion's mixed-modal transformer.
    /// Per the paper (Meta, 2024), Transfusion processes text with next-token
    /// prediction (autoregressive loss) and images with diffusion loss, all within
    /// one transformer. For understanding: image patches are treated as continuous
    /// vectors within the sequence, processed by the shared transformer, and text
    /// tokens are generated autoregressively using causal attention.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Encode image to continuous patch representations
        // Transfusion treats images as CONTINUOUS patches (not discrete tokens)
        var features = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            features = Layers[i].Forward(features);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = features.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = features;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    /// <summary>
    /// Generates an image from text with the model the objective trains (Zhou et al. 2024,
    /// Sec. 3): the caption is embedded, a BOI marker opens the image block, and the image's
    /// patches are denoised from Gaussian noise by the shared trunk under bidirectional
    /// attention, reading the noise prediction from the same head the diffusion term of
    /// <see cref="Train"/> minimizes.
    /// </summary>
    /// <remarks>
    /// Sampling is DDIM (Song et al. 2021) with <c>eta = 0</c> over a strided subset of the
    /// trained cosine schedule -- <see cref="TransfusionOptions.DiffusionInferenceSteps"/> steps
    /// of the <see cref="TransfusionOptions.DiffusionTimesteps"/> the objective trains -- so a
    /// training step that lowers the diffusion loss is a training step that changes this output.
    /// The result is <c>OutputImageSize * OutputImageSize * 3</c> values in <c>[0, 1]</c>,
    /// interleaved per pixel, resampled from the patch grid at the training resolution.
    /// </remarks>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        var tokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        int[] caption = NeuralNetworkBase<T>.TokenIds(
            tokens, ClampTokenId, _options.MaxSequenceLength);
        var patches = SampleImagePatches(caption, _options.DiffusionInferenceSteps);
        return Unpatchify(patches, Math.Max(1, _options.OutputImageSize));
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        InitializeTrainingHeads();
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultUnifiedBidirectionalLayers(
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.DecoderDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    _options.NumDecoderLayers / 2,
                    _options.NumDecoderLayers / 2,
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
        _encoderLayerEnd =
            1
            + _options.NumVisionLayers * lpb
            + (_options.VisionDim != _options.DecoderDim ? 1 : 0);
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        return Accelerate(
            input,
            () =>
            {
                var c = input;
                foreach (var l in Layers)
                    c = l.Forward(c);
                return c;
            }
        );
    }

    /// <inheritdoc />
    /// <remarks>
    /// One step on Transfusion's own objective, <c>L = L_LM + lambda * L_DDPM</c> (Zhou et al.
    /// 2024, Eq. 4): next-token cross-entropy over the text positions of the mixed sequence plus
    /// the weighted DDPM noise-prediction loss over the image's patches. <paramref name="input"/>
    /// is the image; <paramref name="expected"/> is the caption's token ids when it is rank-1, and
    /// any other shape carries no text, which leaves the diffusion term alone with no tokens to
    /// score. See <see cref="TrainMixedModal"/> for the masks each term is read under.
    /// </remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        TrainMixedModal(input, expected);

        return true;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    // Lazy decoder/vision blocks report ParameterCount == 0 until the first
    // forward, so weight streaming would engage too late and OOM a constrained
    // runner; a coarse structural estimate (~12·d² per transformer block plus
    // the token embedding) lets the auto-detector turn streaming on in the
    // constructor. ONNX mode keeps weights in the external runtime → no estimate.
    protected override long EstimateStructuralParameterCount()
    {
        if (!_useNativeMode)
            return 0L;
        long visionDim = _options.VisionDim;
        long decoderDim = _options.DecoderDim;
        long vision = (long)_options.NumVisionLayers * 12L * visionDim * visionDim;
        long decoder = (long)_options.NumDecoderLayers * 12L * decoderDim * decoderDim;
        long embeddings = (long)_options.VocabSize * decoderDim;
        // The mixed-modal training path adds a vocabulary head (d x V) and the linear
        // patchifier / noise head (patchDim x d each), all built in the constructor.
        long lmHead = (long)_options.VocabSize * decoderDim;
        long patchHeads = 2L * _patchDim * decoderDim;
        return vision + decoder + embeddings + lmHead + patchHeads;
    }

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Transfusion-Native" : "Transfusion-ONNX",
            Description =
                "Transfusion: combined autoregressive and diffusion loss in single transformer.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Transfusion";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["DiffusionLoss"] = _options.EnableDiffusionLoss.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Transfusion<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
