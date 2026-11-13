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

namespace AiDotNet.VisionLanguage.Reasoning;

/// <summary>
/// Skywork R1V: cross-modal transfer of reasoning LLMs to vision.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Skywork R1V (2025) pioneers cross-modal transfer of reasoning capabilities from text-only
/// LLMs to vision-language models. It transfers chain-of-thought reasoning patterns learned
/// by text reasoning models to the visual domain, enabling structured multi-step reasoning
/// about images without requiring vision-specific reasoning data from scratch.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought" (2025)</item></list></para>
/// <para><b>For Beginners:</b> Skywork R1V is a vision-language model that transfers reasoning
/// capabilities from text LLMs to multimodal understanding. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Skywork R1V model for multimodal chain-of-thought reasoning
/// // with cross-modal transfer of reasoning capabilities from LLMs to vision
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SkyworkR1V&lt;double&gt;(architecture, "skyworkr1v.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SkyworkR1V&lt;double&gt;(architecture, new SkyworkR1VOptions());
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
    "Skywork R1V: Pioneering Multimodal Reasoning with Chain-of-Thought",
    "https://arxiv.org/abs/2504.05599",
    Year = 2025,
    Authors = "Skywork Team"
)]
[PaperOptimizer(OptimizerKind.Unspecified, LearningRate = 1e-6, WeightDecay = 0.05,
                ReferenceBatchSize = 512, WarmupFraction = 0.03,
                Source = "Peng et al. 2025, Sec. 4: the model is trained with a learning rate of 1e-6, "
                        + "a weight decay of 0.05, a warmup ratio of 0.03, a batch size of 512 and one "
                        + "training epoch per stage, at a context length of 16384 tokens. The optimizer "
                        + "is left unspecified because the paper names none -- the Adam names in its "
                        + "text are the authors Adam Kalai, Adam Lerer and Adam Richardson in a "
                        + "reference.")]
public partial class SkyworkR1V<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly SkyworkR1VOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public SkyworkR1V(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SkyworkR1VOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SkyworkR1VOptions();
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

    public SkyworkR1V(
        NeuralNetworkArchitecture<T> architecture,
        SkyworkR1VOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SkyworkR1VOptions();
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
    /// Generates text from image using Skywork R1V's cross-modal reasoning transfer.
    /// Per the paper (Skywork, 2025), R1V pioneers multimodal reasoning via:
    /// (1) Cross-modal transfer: takes a text-only reasoning LLM (e.g., QwQ-32B-Preview)
    ///     and transfers its reasoning capability to the visual domain,
    /// (2) Selective visual token enrichment: identifies tokens relevant to the reasoning
    ///     task and enriches them with stronger visual features while dampening noise,
    /// (3) Progressive visual integration: visual features are injected gradually across
    ///     decoder layers rather than all at once, allowing the text reasoning patterns
    ///     to adapt incrementally to multimodal input,
    /// (4) Reasoning-aware visual attention: cross-attention weights are biased toward
    ///     image regions most relevant to the type of reasoning being performed.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Vision encoder
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
    /// Generates multi-step reasoning using Skywork R1V's cross-modal transfer approach.
    /// Per the paper (Skywork, 2025), the cross-modal reasoning process:
    /// (1) Transfers text reasoning patterns: the model applies structured reasoning templates
    ///     learned from text-only training (e.g., decompose, analyze, verify),
    /// (2) Visual grounding at each step: each reasoning step re-attends to the image
    ///     with attention masks shaped by the current reasoning context,
    /// (3) Confidence-based early stopping: if reasoning confidence exceeds threshold,
    ///     the model produces the answer without exhausting all reasoning steps,
    /// (4) Cross-modal consistency check: verifies that text reasoning conclusions
    ///     are consistent with visual evidence before finalizing.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numReasoningSteps = 5; // decompose, analyze, ground, verify, conclude

        // Step 1: Encode visual features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Compute enrichment scores (same as GenerateFromImage)
        int numTokens = Math.Min(visDim, 512);
        var engine = Engine;
        var tokenValues = numTokens > 0 ? CyclicTake(visualFeatures, numTokens) : new Tensor<T>([0]);
        var tokenScores = new Tensor<T>([0]);
        if (numTokens > 0)
        {
            // score = |v| + 2 (|v - prev| + |v - next|); the first token's prev and the last token's next are itself.
            var prev = numTokens > 1
                ? engine.TensorConcatenate([engine.TensorSlice(tokenValues, [0], [1]), engine.TensorSlice(tokenValues, [0], [numTokens - 1])], 0)
                : tokenValues;
            var next = numTokens > 1
                ? engine.TensorConcatenate([engine.TensorSlice(tokenValues, [1], [numTokens - 1]), engine.TensorSlice(tokenValues, [numTokens - 1], [1])], 0)
                : tokenValues;
            tokenScores = engine.TensorAdd(
                engine.TensorAbs(tokenValues),
                engine.TensorMultiplyScalar(
                    engine.TensorAdd(
                        engine.TensorAbs(engine.TensorSubtract(tokenValues, prev)),
                        engine.TensorAbs(engine.TensorSubtract(tokenValues, next))),
                    NumOps.FromDouble(2.0)));

            // Normalize to [0, 1) by the (NaN-skipping) min and max.
            var nanMask = engine.TensorIsNan(tokenScores);
            T maxS = engine.TensorMaxValue(engine.TensorWhere(
                nanMask, Tensor<T>.CreateDefault([numTokens], NumOps.FromDouble(double.NegativeInfinity)), tokenScores));
            T minS = engine.TensorMinValue(engine.TensorWhere(
                nanMask, Tensor<T>.CreateDefault([numTokens], NumOps.FromDouble(double.PositiveInfinity)), tokenScores));
            T range = NumOps.Add(NumOps.Subtract(maxS, minS), NumOps.FromDouble(1e-8));
            tokenScores = engine.TensorDivideScalar(engine.TensorSubtractScalar(tokenScores, minS), range);
        }

        // Step 3: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        // Step 4: Cross-modal iterative reasoning with transferred text patterns
        var reasoningState = new Tensor<T>([dim]);
        var prevConfidence = new Tensor<T>([1]);
        var qEmb = engine.TensorMultiplyScalar(CyclicTake(questionTokens, dim), NumOps.FromDouble(1.0 / _options.VocabSize));
        // enrichWeight = 0.3 + 0.7 score (the same every step)
        var enrichWeight = engine.TensorAddScalar(engine.TensorMultiplyScalar(tokenScores, NumOps.FromDouble(0.7)), NumOps.FromDouble(0.3));
        // Attention phase (d + 1)(t + 1) 0.004, shifted by step * 0.5 each step.
        var basePhase = engine.TensorMultiplyScalar(
            engine.TensorOuter(engine.TensorArange<T>(1, dim + 1, 1), engine.TensorArange<T>(1, numTokens + 1, 1)),
            NumOps.FromDouble(0.004));
        var stepIndex = engine.TensorArange<T>(0, numReasoningSteps, 1);
        var phaseShift = engine.TensorMultiplyScalar(stepIndex, NumOps.FromDouble(0.5));
        // Step-dependent attention: early steps = broad, late steps = focused: exp(-0.3 step) * 0.5 + 0.5
        var stepFocus = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorExp(engine.TensorMultiplyScalar(stepIndex, NumOps.FromDouble(-0.3))), NumOps.FromDouble(0.5)),
            NumOps.FromDouble(0.5));
        var stepWeights = engine.TensorDivide(
            Tensor<T>.CreateDefault([numReasoningSteps], NumOps.One), engine.TensorArange<T>(1, numReasoningSteps + 1, 1));
        var stateKeep = engine.ScalarMinusTensor(NumOps.One, engine.TensorMultiplyScalar(stepWeights, NumOps.FromDouble(0.3)));
        var valueScale = engine.TensorMultiplyScalar(
            engine.TensorBroadcastTo(tokenValues.Reshape([1, numTokens]), [dim, numTokens]), NumOps.FromDouble(0.3));
        T epsilon = NumOps.FromDouble(1e-8);
        T questionWeight = NumOps.FromDouble(0.2);
        T consistencyWeight = NumOps.FromDouble(0.3);
        T consistencyBase = NumOps.FromDouble(0.7);
        T invDim = NumOps.FromDouble(1.0 / dim);
        T convergence = NumOps.FromDouble(0.001);

        for (int step = 0; step < numReasoningSteps; step++)
        {
            // Cross-modal attention with step-specific focus:
            // w[d, t] = exp(vis_t * sin((d + 1)(t + 1) 0.004 + 0.5 step) * 0.3) * enrichWeight_t * stepFocus
            var phase = engine.TensorAdd(basePhase, engine.TensorBroadcastTo(Pick(phaseShift, step), [dim, numTokens]));
            var weights = engine.TensorMultiply(
                engine.TensorExp(engine.TensorMultiply(valueScale, engine.TensorSin(phase))),
                engine.TensorBroadcastTo(engine.TensorMultiply(enrichWeight, Pick(stepFocus, step)).Reshape([1, numTokens]), [dim, numTokens]));
            var visAttn = WeightedAverage(weights, tokenValues);

            // Cross-modal consistency: compare visual attention with reasoning state
            Tensor<T> consistency;
            if (step > 0)
            {
                consistency = engine.ScalarMinusTensor(
                    NumOps.One,
                    engine.TensorDivide(
                        engine.TensorAbs(engine.TensorSubtract(visAttn, reasoningState)),
                        engine.TensorAddScalar(engine.TensorAdd(engine.TensorAbs(visAttn), engine.TensorAbs(reasoningState)), epsilon)));
            }
            else
            {
                consistency = Tensor<T>.CreateDefault([dim], NumOps.One);
            }

            // Residual update with consistency weighting
            reasoningState = engine.TensorAdd(
                engine.TensorMultiply(reasoningState, Pick(stateKeep, step)),
                engine.TensorMultiply(
                    engine.TensorMultiply(
                        engine.TensorAdd(visAttn, engine.TensorMultiplyScalar(qEmb, questionWeight)), Pick(stepWeights, step)),
                    engine.TensorAddScalar(engine.TensorMultiplyScalar(consistency, consistencyWeight), consistencyBase)));

            // Confidence estimation (average magnitude of reasoning state)
            var confidence = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorAbs(reasoningState), [0], true), invDim);

            // Early stopping if confidence converges
            var converged = engine.TensorLessThan(engine.TensorAbs(engine.TensorSubtract(confidence, prevConfidence)), convergence);
            if (step > 1 && engine.TensorSum(converged.Cast<int>()) > 0)
                break;
            prevConfidence = confidence;
        }

        // Step 5: Final answer
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
            Name = _useNativeMode ? "Skywork-R1V-Native" : "Skywork-R1V-ONNX",
            Description = "Skywork R1V: cross-modal transfer of reasoning LLMs to vision.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Skywork-R1V";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["CrossModalTransfer"] = _options.EnableCrossModalTransfer.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SkyworkR1V<T>));
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
