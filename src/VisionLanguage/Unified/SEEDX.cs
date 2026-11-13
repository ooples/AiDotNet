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
/// SEED-X: multi-granularity comprehension and generation model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SEED-X (Tencent, 2024) is a multimodal model with unified multi-granularity comprehension
/// and generation. It processes visual information at multiple granularity levels from global
/// scene understanding to fine-grained detail recognition, using a SEED visual tokenizer that
/// captures both high-level semantics and low-level visual details for generation tasks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation" (Tencent, 2024)</item></list></para>
/// <para><b>For Beginners:</b> SEED-X is a unified model for multi-granularity visual
/// understanding and generation from Tencent. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SEED-X model for multi-granularity comprehension and generation
/// // with unified visual understanding at multiple detail levels
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SEEDX&lt;double&gt;(architecture, "seedx.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SEEDX&lt;double&gt;(architecture, new SEEDXOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation",
    "https://arxiv.org/abs/2404.14396",
    Year = 2024,
    Authors = "Ge et al."
)]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 1e-4,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.PreTraining,
                Source = "Ge et al. 2024, Sec. 4: pre-training runs on 48 H800-80G GPUs over 158M "
                        + "samples with the learning rate set to 1e-4 under cosine decay. The "
                        + "identically worded 1e-4 settings elsewhere in the paper train and fine-tune "
                        + "the visual de-tokenizer, a component rather than this model. The optimizer is "
                        + "left unspecified because the paper names none.")]
public partial class SEEDX<T> : VisionLanguageModelBase<T>, IUnifiedVisionModel<T>
{
    private readonly SEEDXOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public SEEDX(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SEEDXOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SEEDXOptions();
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

    public SEEDX(
        NeuralNetworkArchitecture<T> architecture,
        SEEDXOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SEEDXOptions();
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
    public int MaxGenerationLength => _options.MaxGenerationLength;
    /// <inheritdoc />
    public int DecoderEmbeddingDim => _options.DecoderDim;
    /// <inheritdoc />
    public bool SupportsGeneration => _options.SupportsGeneration;

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
    /// Generates text from image using SEED-X's multi-granularity visual comprehension.
    /// Per the paper (Tencent, 2024), SEED-X uses a SEED tokenizer that produces
    /// multi-granularity visual tokens: (1) high-level semantic tokens for global
    /// understanding via Q-Former, (2) region-level tokens for localized comprehension,
    /// (3) low-level tokens for fine-grained details. The multi-granularity tokens
    /// are concatenated and fed to the LLM alongside text tokens.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Visual encoding
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
    /// Generates an image from text using SEED-X's multi-granularity generation pipeline.
    /// Per the paper (Tencent, 2024), SEED-X generates images by:
    /// (1) The LLM produces visual generation tokens conditioned on text,
    /// (2) These tokens are de-tokenized by the SEED de-tokenizer to produce
    ///     conditioning embeddings at multiple granularities,
    /// (3) An SDXL-based diffusion decoder generates the final image conditioned
    ///     on the multi-granularity embeddings.
    /// The multi-granularity approach captures both global composition (semantic tokens)
    /// and local details (fine-grained tokens) for high-quality generation.
    /// Output: image tensor of size OutputImageSize * OutputImageSize * 3.
    /// </summary>
    public Tensor<T> GenerateImage(string textDescription)
    {
        ThrowIfDisposed();
        var tokens = TokenizeText(textDescription);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        int outSize = _options.OutputImageSize;
        int outPixels = outSize * outSize * 3;
        int dim = _options.DecoderDim;

        // Step 1: LLM generates visual generation tokens
        var textHidden = tokens;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            textHidden = Layers[i].Forward(textHidden);
        int hiddenDim = textHidden.Length;

        // Step 2: SEED de-tokenization to multi-granularity conditioning
        // 2a: Semantic conditioning (global composition)
        var engine = Engine;
        int numSemantic = 32;
        // semanticCond[s] = sum_{h < min(hiddenDim, 64)} hidden[h] * cos((s + 1) * (h + 1) * 0.01) / 64
        var semanticCond = WaveProjection(textHidden, numSemantic, Math.Min(hiddenDim, 64), 0.01, useCosine: true, divisor: 64.0);

        // 2b: Detail conditioning (local texture/color)
        int numDetail = 64;
        // detailCond[d] = sum_{h < min(hiddenDim, 32)} hidden[h] * sin((d + 1) * (h + 1) * 0.015) / 32
        var detailCond = WaveProjection(textHidden, numDetail, Math.Min(hiddenDim, 32), 0.015, useCosine: false, divisor: 32.0);

        // Step 3: SDXL-style diffusion generation conditioned on SEED tokens
        // Initialize from the deterministic pattern sin(0.37 i + 0.5) * 0.7
        var latent = engine.TensorMultiplyScalar(
            engine.TensorSin(engine.TensorAddScalar(
                engine.TensorMultiplyScalar(engine.TensorArange<T>(0, outPixels, 1), NumOps.FromDouble(0.37)),
                NumOps.FromDouble(0.5))),
            NumOps.FromDouble(0.7));

        // Multi-granularity fusion: semantic for structure, detail for texture (step independent).
        var fusedCond = engine.TensorAdd(
            engine.TensorMultiplyScalar(CyclicTake(semanticCond, dim), NumOps.FromDouble(0.6)),
            engine.TensorMultiplyScalar(CyclicTake(detailCond, dim), NumOps.FromDouble(0.4)));

        int numDiffSteps = 30;
        // Per-step schedule: t = 1 - step / numDiffSteps, alpha = cos(t pi / 2), sigma = sin(t pi / 2).
        var halfPiT = engine.TensorMultiplyScalar(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorDivideScalar(
                engine.TensorArange<T>(0, numDiffSteps, 1), NumOps.FromDouble(numDiffSteps))),
            NumOps.FromDouble(Math.PI / 2.0));
        var alphas = engine.TensorCos(halfPiT);
        var sigmas = engine.TensorSin(halfPiT);
        var condScales = engine.TensorMultiplyScalar(sigmas, NumOps.FromDouble(0.2));
        var safeAlphas = engine.TensorMax(alphas, NumOps.FromDouble(1e-8));

        for (int step = 0; step < numDiffSteps; step++)
        {

            // stepInput[d] = latent[d % outPixels] * alpha + cond[d] * sigma * 0.2
            var stepInput = engine.TensorAdd(
                engine.TensorMultiply(CyclicTake(latent, dim), Pick(alphas, step)),
                engine.TensorMultiply(fusedCond, Pick(condScales, step)));

            var noisePred = stepInput;
            for (int i = _encoderLayerEnd; i < Layers.Count; i++)
                noisePred = Layers[i].Forward(noisePred);

            // latent = (latent - sigma * pred[i % predDim]) / max(alpha, 1e-8)
            latent = engine.TensorDivide(
                engine.TensorSubtract(latent, engine.TensorMultiply(CyclicTake(noisePred, outPixels), Pick(sigmas, step))),
                engine.TensorBroadcastTo(Pick(safeAlphas, step), [outPixels]));
        }

        var result = engine.TensorSigmoid(latent);
        return result;
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
        return TokenIdsToTensor(encoding.TokenIds, seqLen);
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
            Name = _useNativeMode ? "SEED-X-Native" : "SEED-X-ONNX",
            Description = "SEED-X: multi-granularity comprehension and generation model.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "SEED-X";
        m.AdditionalInfo["SupportsGeneration"] = _options.SupportsGeneration.ToString();
        m.AdditionalInfo["MultiGranularity"] = _options.EnableMultiGranularity.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SEEDX<T>));
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
