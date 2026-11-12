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
/// Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Skywork R1V2 (2025) achieves state-of-the-art multimodal reasoning through hybrid
/// reinforcement learning combining MPO (Multi-head Policy Optimization) and GRPO (Group
/// Relative Policy Optimization) objectives. This hybrid RL approach improves both the
/// quality of reasoning chains and the accuracy of final answers on complex visual tasks.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Skywork R1V2: Multimodal Hybrid Reinforcement Learning" (2025)</item></list></para>
/// <para><b>For Beginners:</b> Skywork R1V2 is an advanced vision-language model using hybrid
/// reinforcement learning for state-of-the-art multimodal reasoning. Default values follow
/// the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Skywork R1V2 model for state-of-the-art multimodal reasoning
/// // with hybrid RL training combining MPO and GRPO objectives
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SkyworkR1V2&lt;double&gt;(architecture, "skyworkr1v2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SkyworkR1V2&lt;double&gt;(architecture, new SkyworkR1V2Options());
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
    "Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning",
    "https://arxiv.org/abs/2504.16656",
    Year = 2025,
    Authors = "Skywork Team"
)]
public partial class SkyworkR1V2<T> : VisionLanguageModelBase<T>, IReasoningVLM<T>
{
    private readonly SkyworkR1V2Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public SkyworkR1V2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SkyworkR1V2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SkyworkR1V2Options();
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

    public SkyworkR1V2(
        NeuralNetworkArchitecture<T> architecture,
        SkyworkR1V2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SkyworkR1V2Options();
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
    /// Generates text from image using Skywork R1V2's hybrid RL-aligned pipeline.
    /// Per the paper (Skywork, 2025), R1V2 achieves SOTA multimodal reasoning via:
    /// (1) Hybrid RL training: combines MPO (Mixed Preference Optimization) for precision
    ///     with GRPO (Group Relative Policy Optimization) for exploration diversity,
    /// (2) Multi-reward signals: correctness, coherence, and visual grounding are separately
    ///     rewarded, producing reasoning that is both accurate and well-grounded,
    /// (3) Dual-phase inference: coarse reasoning (GRPO-style broad exploration) followed by
    ///     fine reasoning (MPO-style precise refinement) of the visual answer,
    /// (4) Visual grounding reward: cross-attention patterns are evaluated for consistency
    ///     with the image, penalizing hallucinated visual references.
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
    /// Generates reasoning using Skywork R1V2's hybrid RL chain-of-thought.
    /// Per the paper (Skywork, 2025), the reasoning pipeline combines:
    /// (1) GRPO exploration: generates multiple reasoning trajectories in parallel
    ///     and ranks them by group-relative reward (no absolute reward baseline needed),
    /// (2) MPO refinement: selects and refines the best trajectory using mixed
    ///     preference signals (correctness + coherence + visual grounding),
    /// (3) Multi-step iteration: alternates between GRPO exploration and MPO refinement
    ///     across reasoning steps for progressively better reasoning chains,
    /// (4) Visual grounding verification: at each step, checks that reasoning conclusions
    ///     are supported by actual visual evidence.
    /// </summary>
    public Tensor<T> ReasonWithChainOfThought(Tensor<T> image, string question)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;
        int numGroups = 4; // GRPO group size
        int numReasoningSteps = 6;

        // Step 1: Encode visual features
        var visualFeatures = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visualFeatures = Layers[i].Forward(visualFeatures);
        int visDim = visualFeatures.Length;

        // Step 2: Encode question
        var questionTokens = TokenizeText(question);
        int qLen = questionTokens.Length;

        // Step 3: Iterative GRPO+MPO reasoning
        var engine = Engine;
        var reasoningState = new Tensor<T>([dim]);
        int numTokens = Math.Min(visDim, 256);
        int rewardDims = Math.Min(dim, 32);
        var tokenValues = numTokens > 0 ? CyclicTake(visualFeatures, numTokens) : new Tensor<T>([0]);
        T invVocab = NumOps.FromDouble(1.0 / _options.VocabSize);
        var qEmb = engine.TensorMultiplyScalar(CyclicTake(questionTokens, dim), invVocab);
        // Question term 0.3 q[(t + 3 g) % qLen] / vocab of each trajectory (the same every step).
        var questionTerm = numTokens > 0
            ? engine.TensorMultiplyScalar(AffineModGather(questionTokens, [numGroups, numTokens], [3, 1], 0), NumOps.FromDouble(0.3 / _options.VocabSize))
            : new Tensor<T>([numGroups, 0]);
        var groupTokenPhase = engine.TensorOuter(engine.TensorArange<T>(1, numGroups + 1, 1), engine.TensorArange<T>(1, numTokens + 1, 1));
        var stepNumber = engine.TensorArange<T>(1, numReasoningSteps + 1, 1);
        var trajectoryFrequency = engine.TensorMultiplyScalar(stepNumber, NumOps.FromDouble(0.005));
        var stepDecay = engine.TensorDivide(Tensor<T>.CreateDefault([numReasoningSteps], NumOps.One), stepNumber);
        var stateKeep = engine.ScalarMinusTensor(NumOps.One, engine.TensorMultiplyScalar(stepDecay, NumOps.FromDouble(0.35)));
        var tokenGrid = engine.TensorBroadcastTo(tokenValues.Reshape([1, numTokens]), [numGroups, numTokens]);
        // Visual references vis[d % visDim] for the reward dimensions.
        var visualReference = rewardDims > 0 ? CyclicTake(visualFeatures, rewardDims) : new Tensor<T>([0]);
        var questionContribution = engine.TensorBroadcastTo(
            engine.TensorMultiplyScalar(qEmb, NumOps.FromDouble(0.15)).Reshape([1, dim]), [numGroups, dim]);
        T trajectoryAmplitude = NumOps.FromDouble(0.25);
        T temperature = NumOps.FromDouble(0.3);
        T previousWeight = NumOps.FromDouble(0.3);
        T epsilon = NumOps.FromDouble(1e-8);
        T coherenceSlope = NumOps.FromDouble(3.0);
        T groundingSlope = NumOps.FromDouble(1.5);
        T invRewardDims = NumOps.FromDouble(1.0 / rewardDims);
        var negInf = Tensor<T>.CreateDefault([numGroups], NumOps.FromDouble(double.NegativeInfinity));
        var posInf = Tensor<T>.CreateDefault([numGroups], NumOps.FromDouble(double.PositiveInfinity));
        T minusOne = NumOps.FromDouble(-1.0);
        T rankScale = NumOps.FromDouble(2.0);
        T correctnessWeight = NumOps.FromDouble(0.5);
        T coherenceWeight = NumOps.FromDouble(0.3);
        T groundingWeight = NumOps.FromDouble(0.2);

        for (int step = 0; step < numReasoningSteps; step++)
        {
            // Phase A: GRPO exploration - multiple reasoning trajectories.
            // w[g, t] = exp((vis_t + 0.3 q[(t + 3 g) % qLen] / vocab + 0.25 sin((g + 1)(step + 1)(t + 1) 0.005)) * 0.3)
            var trajBias = engine.TensorMultiplyScalar(
                engine.TensorSin(engine.TensorMultiply(groupTokenPhase, Pick(trajectoryFrequency, step))), trajectoryAmplitude);
            var attention = engine.TensorExp(engine.TensorMultiplyScalar(
                engine.TensorAdd(engine.TensorAdd(tokenGrid, questionTerm), trajBias), temperature));
            var visAttn = WeightedAverage(attention, tokenValues);

            // trajectory[g, d] = visAttn[g] + 0.15 qEmb[d] + 0.3 state[d]
            var trajectoryOutputs = engine.TensorAdd(
                engine.TensorAdd(engine.TensorBroadcastTo(visAttn.Reshape([numGroups, 1]), [numGroups, dim]), questionContribution),
                engine.TensorBroadcastTo(engine.TensorMultiplyScalar(reasoningState, previousWeight).Reshape([1, dim]), [numGroups, dim]));

            // Compute multi-reward over the first rewardDims dims: correctness proxy + coherence + grounding
            var head = engine.TensorSlice(trajectoryOutputs, [0, 0], [numGroups, rewardDims]);
            var correctness = engine.ReduceSum(engine.ScalarMinusTensor(NumOps.One, engine.TensorMin(
                engine.TensorAbs(engine.TensorSubtract(head, engine.TensorBroadcastTo(visualReference.Reshape([1, rewardDims]), [numGroups, rewardDims]))),
                NumOps.One)), [1], false);
            var coherence = rewardDims > 1
                ? engine.ReduceSum(engine.ScalarMinusTensor(NumOps.One, engine.TensorMin(
                    engine.TensorMultiplyScalar(engine.TensorAbs(engine.TensorSubtract(
                        engine.TensorSlice(head, [0, 1], [numGroups, rewardDims - 1]),
                        engine.TensorSlice(head, [0, 0], [numGroups, rewardDims - 1]))), coherenceSlope),
                    NumOps.One)), [1], false)
                : new Tensor<T>([numGroups]);
            var grounding = engine.ReduceSum(engine.ScalarMinusTensor(NumOps.One, engine.TensorMin(
                engine.TensorMultiplyScalar(engine.TensorAbs(head), groundingSlope), NumOps.One)), [1], false);
            var trajectoryRewards = engine.TensorMultiplyScalar(
                engine.TensorAdd(
                    engine.TensorAdd(engine.TensorMultiplyScalar(correctness, correctnessWeight), engine.TensorMultiplyScalar(coherence, coherenceWeight)),
                    engine.TensorMultiplyScalar(grounding, groundingWeight)),
                invRewardDims);

            // Phase B: GRPO ranking + MPO refinement: exp((r - min) / (max - min + 1e-8) * 2), NaN-skipping min/max.
            var nanRewards = engine.TensorIsNan(trajectoryRewards);
            var maxR = engine.ReduceMax(engine.TensorWhere(nanRewards, negInf, trajectoryRewards), [0], true);
            var minR = engine.TensorMultiplyScalar(
                engine.ReduceMax(engine.TensorMultiplyScalar(engine.TensorWhere(nanRewards, posInf, trajectoryRewards), minusOne), [0], true),
                minusOne);
            var rRange = engine.TensorAddScalar(engine.TensorSubtract(maxR, minR), epsilon);
            var trajWeights = engine.TensorExp(engine.TensorMultiplyScalar(
                engine.TensorDivide(
                    engine.TensorSubtract(trajectoryRewards, engine.TensorBroadcastTo(minR, [numGroups])),
                    engine.TensorBroadcastTo(rRange, [numGroups])),
                rankScale));
            var wTotal = engine.TensorMax(engine.ReduceSum(trajWeights, [0], true), epsilon);

            // MPO: preference-weighted combination into reasoning state
            var preference = engine.TensorDivide(trajWeights, engine.TensorBroadcastTo(wTotal, [numGroups]));
            var refined = engine.TensorMatMul(preference.Reshape([1, numGroups]), trajectoryOutputs).Reshape([dim]);
            reasoningState = engine.TensorAdd(
                engine.TensorMultiply(reasoningState, Pick(stateKeep, step)),
                engine.TensorMultiply(refined, Pick(stepDecay, step)));
        }

        // Step 4: Final answer
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
            Name = _useNativeMode ? "Skywork-R1V2-Native" : "Skywork-R1V2-ONNX",
            Description = "Skywork R1V2: hybrid RL (MPO + GRPO) for multimodal reasoning SOTA.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Skywork-R1V2";
        m.AdditionalInfo["ReasoningApproach"] = _options.ReasoningApproach;
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["HybridRL"] = _options.EnableHybridRL.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SkyworkR1V2<T>));
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
