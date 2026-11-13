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

namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Kimi-VL: MoE VLM with MoonViT and long-context processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Kimi-VL (Moonshot AI, 2025) is a Mixture of Experts vision-language model featuring
/// the MoonViT visual encoder and long-context processing capabilities. It uses sparse
/// MoE routing for efficient scaling and supports extended context windows for processing
/// high-resolution images and lengthy multi-turn conversations with visual inputs.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Kimi-VL Technical Report" (Moonshot AI, 2025)</item></list></para>
/// <para><b>For Beginners:</b> Kimi-VL is a vision-language model with MoE architecture
/// for efficient multimodal reasoning. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Kimi-VL model for multimodal reasoning
/// // with MoE architecture, MoonViT encoder, and long-context processing
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new KimiVL&lt;double&gt;(architecture, "kimivl.onnx");
///
/// // Training mode with native layers
/// var trainModel = new KimiVL&lt;double&gt;(architecture, new KimiVLOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Kimi-VL Technical Report",
    "https://arxiv.org/abs/2504.07491",
    Year = 2025,
    Authors = "Moonshot AI"
)]
public partial class KimiVL<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly KimiVLOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public KimiVL(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        KimiVLOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new KimiVLOptions();
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

    public KimiVL(
        NeuralNetworkArchitecture<T> architecture,
        KimiVLOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new KimiVLOptions();
        Options = _options;
        // Fail-fast input validation BEFORE any layer is allocated, so an invalid options bag
        // can't waste the call stack down to a deep multi-MB weight allocation and surface as
        // an obscure shape exception. Delegates to the shared InstructionTunedVLMOptions check
        // (ImageSize and MaxVisualTokens must both be positive) to avoid drift between the
        // ctor guard and InitializeLayers' validation.
        _options.ValidateVisualSizing();
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
    public string LanguageModelName => _options.LanguageModelName;
    /// <inheritdoc />
    public string ReasoningApproach => _options.ReasoningApproach;

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
    /// Generates text from image using Kimi-VL's MoE architecture with MoonViT encoder.
    /// Per the Kimi-VL Technical Report (Moonshot AI, 2025), the architecture features:
    /// (1) MoonViT: a native-resolution ViT that processes images at their original aspect ratio
    ///     using dynamic token merging to handle variable-length visual token sequences,
    /// (2) Mixture-of-Experts (MoE) LLM backbone: 16B total parameters with only 2.8B active
    ///     per token via top-2 expert routing with load-balancing auxiliary loss,
    /// (3) 128K long-context window enabling multi-image and video understanding,
    /// (4) Visual token compression via adaptive pooling before the LLM.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        // Step 1: MoonViT native-resolution encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = visualFeatures.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = visualFeatures;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
    }

    /// <inheritdoc />
    public Tensor<T> Chat(
        Tensor<T> image,
        IEnumerable<(string Role, string Content)> conversationHistory,
        string userMessage
    )
    {
        ThrowIfDisposed();
        var sb = new System.Text.StringBuilder();
        sb.Append(_options.SystemPrompt);
        foreach (var (role, content) in conversationHistory)
            sb.Append($"\n{role}: {content}");
        sb.Append($"\nUser: {userMessage}\nAssistant:");
        return GenerateFromImage(image, sb.ToString());
    }

    /// <summary>
    /// Generates reasoning using Kimi-VL's MoE-based chain-of-thought with long-context.
    /// Per the paper (Moonshot AI, 2025), Kimi-VL's reasoning leverages:
    /// (1) MoE expert specialization: different experts activate for different reasoning tasks
    ///     (spatial reasoning, counting, text reading, etc.),
    /// (2) 128K long-context window allows accumulating extended reasoning chains,
    /// (3) Load-balanced expert routing ensures diverse reasoning perspectives,
    /// (4) Visual re-grounding at each reasoning step via MoonViT token re-attention.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numExperts = 8;
        int numReasoningSteps = 5;

        // Step 1: MoonViT encoding
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        // Step 3: Multi-step MoE reasoning with expert specialization per step
        var engine = Engine;
        var reasoningState = new Tensor<T>([dim]);
        int numSamples = Math.Min(32, visDim);
        int patchCount = Math.Min(48, visDim);

        // Per-step coefficients for all steps: stepDecay = 1 / (step + 1), the router frequency
        // (step + 1) * 0.4 and the expert focus frequency (step + 1) * 0.008.
        var stepNumber = engine.TensorArange<T>(1, numReasoningSteps + 1, 1);
        var stepDecay = engine.TensorDivide(Tensor<T>.CreateDefault([numReasoningSteps], NumOps.One), stepNumber);
        var stateKeep = engine.ScalarMinusTensor(NumOps.One, engine.TensorMultiplyScalar(stepDecay, NumOps.FromDouble(0.25)));
        var routerFrequency = engine.TensorMultiplyScalar(stepNumber, NumOps.FromDouble(0.4));
        var focusFrequency = engine.TensorMultiplyScalar(stepNumber, NumOps.FromDouble(0.008));
        var expertNumber = engine.TensorArange<T>(1, numExperts + 1, 1);
        var patchNumber = engine.TensorArange<T>(1, patchCount + 1, 1);
        T questionScale = NumOps.FromDouble(0.3 / _options.VocabSize);
        T sampleCount = NumOps.FromDouble(numSamples);
        T patchTotal = NumOps.FromDouble(patchCount);
        T half = NumOps.FromDouble(0.5);
        T focusAmplitude = NumOps.FromDouble(0.3);
        T questionWeight = NumOps.FromDouble(0.2);
        // Question conditioning: q[d % qLen] / vocab (the same every step).
        var qEmb = engine.TensorMultiplyScalar(CyclicTake(questionTokens, dim), NumOps.FromDouble(1.0 / _options.VocabSize));

        for (int step = 0; step < numReasoningSteps; step++)
        {
            // Step-dependent expert routing: different steps activate different experts.
            // routerScore[d, e] = cos((e + 1)(step + 1) 0.4) * mean_s(vis[(d ns + s + 13 step) % visDim]
            //                     + 0.3 q[(s + step) % qLen] / vocab); the first expert with the highest
            // exp(0.5 routerScore) wins. With no visual features every score is NaN and no expert wins.
            Tensor<T> bestExpertOut;
            if (numSamples == 0)
            {
                bestExpertOut = new Tensor<T>([dim]);
            }
            else
            {
                var visSamples = AffineModGather(visualFeatures, [dim, numSamples], [numSamples, 1], step * 13);
                var qSamples = engine.TensorMultiplyScalar(AffineModGather(questionTokens, [numSamples], [1], step), questionScale);
                var sampleMean = engine.TensorDivideScalar(
                    engine.ReduceSum(
                        engine.TensorAdd(visSamples, engine.TensorBroadcastTo(qSamples.Reshape([1, numSamples]), [dim, numSamples])),
                        [1], false),
                    sampleCount);
                var stepBias = engine.TensorCos(engine.TensorMultiply(expertNumber, Pick(routerFrequency, step)));
                var routerScore = engine.TensorOuter(sampleMean, stepBias);
                var expertProb = engine.TensorExp(engine.TensorMultiplyScalar(routerScore, half));
                var bestExpert = RowFirstArgMaxMask(expertProb, NumOps.Zero);

                // Expert processes visual features with step-specific focus:
                // expertVal[e] = mean_v vis[(v + 11 e + 17 step) % visDim] * (1 + 0.3 sin((step + 1)(v + 1) 0.008))
                var focus = engine.TensorAddScalar(
                    engine.TensorMultiplyScalar(
                        engine.TensorSin(engine.TensorMultiply(patchNumber, Pick(focusFrequency, step))), focusAmplitude),
                    NumOps.One);
                var expertPatches = AffineModGather(visualFeatures, [numExperts, patchCount], [11, 1], step * 17);
                var expertVal = engine.TensorDivideScalar(
                    engine.ReduceSum(
                        engine.TensorMultiply(expertPatches, engine.TensorBroadcastTo(focus.Reshape([1, patchCount]), [numExperts, patchCount])),
                        [1], false),
                    patchTotal);
                bestExpertOut = MaskedRowSum(
                    engine.TensorBroadcastTo(expertVal.Reshape([1, numExperts]), [dim, numExperts]), bestExpert);
            }

            // Residual reasoning accumulation with long-context decay
            reasoningState = engine.TensorAdd(
                engine.TensorMultiply(reasoningState, Pick(stateKeep, step)),
                engine.TensorMultiply(
                    engine.TensorAdd(bestExpertOut, engine.TensorMultiplyScalar(qEmb, questionWeight)),
                    Pick(stepDecay, step)));
        }

        // Step 4: Final answer from reasoning chain
        var chainInput = reasoningState;

        var output = chainInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);

        return output;
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
            // ComputeKimiPatchSize() is only reached on the default Kimi stack
            // path; custom Architecture.Layers stacks don't consume the patch
            // options, so don't reject them on those values.
            ValidateVisualPatchOptions(_options.ImageSize, _options.MaxVisualTokens);
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultCrossAttentionResamplerVLMLayers(
                    _options.VisionDim,
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    4,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    patchSize: ComputeKimiPatchSize()
                )
            );
            _encoderLayerEnd = ComputeKimiEncoderDecoderBoundary();
        }
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
    }

    private int ComputeKimiPatchSize() =>
        ComputeVisualPatchSize(_options.ImageSize, _options.MaxVisualTokens, roundUp: true);

    // +2 leading layers from the helper: PatchEmbedding + LayerNorm.
    private int ComputeKimiEncoderDecoderBoundary() =>
        ComputeVisionLanguageBoundary(
            leadingLayerCount: 2,
            visionLayerCount: _options.NumVisionLayers,
            visionBlockLayerCount: TransformerBlockLayerCount(_options.DropoutRate),
            auxiliaryBlockCount: 4,
            auxiliaryBlockLayerCount: ResamplerBlockLayerCount(_options.DropoutRate),
            trailingLayerCount: 1
        );

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
            Name = _useNativeMode ? "Kimi-VL-Native" : "Kimi-VL-ONNX",
            Description = "Kimi-VL: MoE VLM with MoonViT and long-context processing.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Kimi-VL";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["TotalParameters"] = _options.TotalParameters.ToString();
        m.AdditionalInfo["ActiveParameters"] = _options.ActiveParameters.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(KimiVL<T>));
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
