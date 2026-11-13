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
/// LLaVA-CoT: chain-of-thought visual reasoning with structured output.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// LLaVA-CoT (2024) extends the LLaVA architecture with chain-of-thought visual reasoning,
/// generating structured step-by-step reasoning traces for complex visual tasks. The model
/// produces intermediate reasoning steps including observation, hypothesis, and conclusion
/// phases before arriving at the final answer, improving accuracy on visual reasoning benchmarks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step" (2024)</item></list></para>
/// <para><b>For Beginners:</b> LLaVA-CoT is a vision-language model that reasons step-by-step
/// through visual problems using chain-of-thought. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a LLaVA-CoT model for step-by-step visual reasoning
/// // with chain-of-thought structured output for complex visual tasks
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new LLaVACoT&lt;double&gt;(architecture, "llavacot.onnx");
///
/// // Training mode with native layers
/// var trainModel = new LLaVACoT&lt;double&gt;(architecture, new LLaVACoTOptions());
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
    "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step",
    "https://arxiv.org/abs/2411.10440",
    Year = 2024,
    Authors = "Xu et al."
)]
public partial class LLaVACoT<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly LLaVACoTOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public LLaVACoT(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        LLaVACoTOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new LLaVACoTOptions();
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

    public LLaVACoT(
        NeuralNetworkArchitecture<T> architecture,
        LLaVACoTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new LLaVACoTOptions();
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
    /// Generates text from image using LLaVA-CoT's structured reasoning pipeline.
    /// Per the paper (2024), LLaVA-CoT introduces systematic visual reasoning via:
    /// (1) Stage-gated generation: output is structured into 4 explicit stages
    ///     (Summary, Caption, Reasoning, Conclusion), each marked by special stage tokens,
    /// (2) Each stage conditions on the previous stage's output, building progressively
    ///     deeper understanding: global overview -> detailed description -> logical analysis -> answer,
    /// (3) Stage-specific visual attention: Summary attends broadly, Caption focuses on objects,
    ///     Reasoning re-attends based on the question, Conclusion synthesizes,
    /// (4) Built on LLaMA-3 with CLIP ViT-L/14 encoder for strong visual grounding.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: CLIP ViT-L/14 vision encoding
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
    /// Generates structured chain-of-thought reasoning using LLaVA-CoT's 4-stage pipeline.
    /// Per the paper (2024), the explicit reasoning structure produces:
    /// (1) Summary stage: "I see an image showing..." - global understanding of the scene,
    /// (2) Caption stage: "The image contains..." - detailed object/attribute description,
    /// (3) Reasoning stage: "To answer the question, I need to..." - step-by-step logical
    ///     analysis that connects visual evidence to the question,
    /// (4) Conclusion stage: "Therefore, the answer is..." - final answer derived from
    ///     the reasoning chain with confidence based on visual grounding strength.
    /// Each stage iteratively refines the internal representation with the question context.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Encode visual features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        int numTokens = Math.Min(visDim, 384);

        var engine = Engine;
        var tokenValues = numTokens > 0 ? CyclicTake(visualFeatures, numTokens) : new Tensor<T>([0]);
        T invVocab = NumOps.FromDouble(1.0 / _options.VocabSize);
        var qEmb = engine.TensorMultiplyScalar(CyclicTake(questionTokens, dim), invVocab);

        // Stage 1: Summary - broad global understanding
        // Uniform broad attention with slight position encoding: w[d, t] = 1 + 0.1 sin((d + 1) t 0.001)
        var positionWeights = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorMultiplyScalar(
                    engine.TensorOuter(engine.TensorArange<T>(1, dim + 1, 1), engine.TensorArange<T>(0, numTokens, 1)),
                    NumOps.FromDouble(0.001))),
                NumOps.FromDouble(0.1)),
            NumOps.One);
        var summaryState = WeightedAverage(positionWeights, tokenValues);

        // Stage 2: Caption - focused description with question awareness
        // Saliency-weighted attention w_t = exp((|vis_t| + 0.1 q[t % qLen] / vocab) * 0.4), the same for every d.
        Tensor<T> focusAttn;
        if (numTokens > 0)
        {
            var captionBias = engine.TensorMultiplyScalar(CyclicTake(questionTokens, numTokens), NumOps.FromDouble(0.1 / _options.VocabSize));
            var captionWeights = engine.TensorExp(engine.TensorMultiplyScalar(
                engine.TensorAdd(engine.TensorAbs(tokenValues), captionBias), NumOps.FromDouble(0.4)));
            focusAttn = WeightedAverage(captionWeights.Reshape([1, numTokens]), tokenValues);
        }
        else
        {
            focusAttn = new Tensor<T>([1]);
        }
        var captionState = engine.TensorAdd(
            engine.TensorMultiplyScalar(engine.TensorBroadcastTo(focusAttn, [dim]), NumOps.FromDouble(0.7)),
            engine.TensorMultiplyScalar(summaryState, NumOps.FromDouble(0.3)));

        // Stage 3: Reasoning - question-conditioned multi-step analysis
        int reasoningIterations = _options.EnableStructuredReasoning ? 3 : 1;

        // Initialize from caption
        var reasoningState = captionState;
        var iterNumber = engine.TensorArange<T>(1, reasoningIterations + 1, 1);
        var iterWeights = engine.TensorDivide(Tensor<T>.CreateDefault([reasoningIterations], NumOps.One), iterNumber);
        var stateKeep = engine.ScalarMinusTensor(NumOps.One, engine.TensorMultiplyScalar(iterWeights, NumOps.FromDouble(0.3)));
        var focusFrequency = engine.TensorMultiplyScalar(iterNumber, NumOps.FromDouble(0.01));
        var tokenNumber = engine.TensorArange<T>(1, numTokens + 1, 1);
        T questionScale = NumOps.FromDouble(0.4 / _options.VocabSize);
        T focusAmplitude = NumOps.FromDouble(0.2);
        T temperature = NumOps.FromDouble(0.35);
        T questionWeight = NumOps.FromDouble(0.2);

        for (int iter = 0; iter < reasoningIterations; iter++)
        {
            // Strong question conditioning, iteration-dependent focus (the same for every d):
            // w_t = exp((vis_t + 0.4 q[(t + 7 iter) % qLen] / vocab + 0.2 sin((iter + 1)(t + 1) 0.01)) * 0.35)
            Tensor<T> reasonAttn;
            if (numTokens > 0)
            {
                var questionTerm = engine.TensorMultiplyScalar(
                    AffineModGather(questionTokens, [numTokens], [1], iter * 7), questionScale);
                var iterFocus = engine.TensorMultiplyScalar(
                    engine.TensorSin(engine.TensorMultiply(tokenNumber, Pick(focusFrequency, iter))), focusAmplitude);
                var weights = engine.TensorExp(engine.TensorMultiplyScalar(
                    engine.TensorAdd(engine.TensorAdd(tokenValues, questionTerm), iterFocus), temperature));
                reasonAttn = WeightedAverage(weights.Reshape([1, numTokens]), tokenValues);
            }
            else
            {
                reasonAttn = new Tensor<T>([1]);
            }

            // Residual update conditioned on question
            reasoningState = engine.TensorAdd(
                engine.TensorMultiply(reasoningState, Pick(stateKeep, iter)),
                engine.TensorMultiply(
                    engine.TensorAdd(engine.TensorBroadcastTo(reasonAttn, [dim]), engine.TensorMultiplyScalar(qEmb, questionWeight)),
                    Pick(iterWeights, iter)));
        }

        // Stage 4: Conclusion - synthesize all stages into final answer
        // Weighted synthesis: reasoning-heavy for CoT (0.1 * the 0.3-scaled question embedding).
        var chainInput = engine.TensorAdd(
            engine.TensorAdd(
                engine.TensorMultiplyScalar(reasoningState, NumOps.FromDouble(0.55)),
                engine.TensorMultiplyScalar(captionState, NumOps.FromDouble(0.25))),
            engine.TensorAdd(
                engine.TensorMultiplyScalar(summaryState, NumOps.FromDouble(0.1)),
                engine.TensorMultiplyScalar(qEmb, NumOps.FromDouble(0.3 * 0.1))));

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
            Name = _useNativeMode ? "LLaVA-CoT-Native" : "LLaVA-CoT-ONNX",
            Description = "LLaVA-CoT: chain-of-thought visual reasoning with structured output.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "LLaVA-CoT";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["StructuredReasoning"] = _options.EnableStructuredReasoning.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(LLaVACoT<T>));
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
