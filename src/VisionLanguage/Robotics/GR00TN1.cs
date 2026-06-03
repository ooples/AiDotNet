using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;
using AiDotNet.Extensions;

namespace AiDotNet.VisionLanguage.Robotics;

/// <summary>
/// GR00T N1: NVIDIA's open foundation model for generalist humanoid robots, combining a SigLIP +
/// Eagle-2 vision-language System-2 reasoner with a flow-matching DiT action head as System 1
/// (NVIDIA 2025, arXiv:2503.14734).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GR00T N1 is the first publicly released foundation model trained on the GR00T-1B humanoid
/// dataset. Like Helix, it uses a dual-system architecture, but the System-1 policy is a
/// <b>flow-matching</b> DiT (Lipman et al. 2023) rather than a direct regression head. At
/// inference the model:
/// </para>
/// <list type="number">
///   <item>Runs the Eagle-2 VLM (System 2) on the observation + instruction, emitting a 1536-dim semantic latent.</item>
///   <item>Samples Gaussian noise of shape <c>[PredictionHorizon × ActionDimension]</c>.</item>
///   <item>Euler-integrates the learned velocity field for <c>FlowMatchingSteps</c> steps (default 16), conditioning at every step on the System-2 latent.</item>
///   <item>Returns the de-noised continuous joint command tensor.</item>
/// </list>
/// <para><b>Paper-faithful pieces implemented here:</b></para>
/// <list type="bullet">
///   <item><see cref="GR00TFlowMatchingActionHead{T}"/>: paper §3.2 flow-matching inference path (Lipman et al. ICLR 2023, Black et al. π0 2024).</item>
///   <item>Dual-system coordination via <see cref="HelixDualSystemRunner{T}"/>: 50 Hz S1 default (paper §4.1) vs slower S2 invocations.</item>
///   <item>SigLIP-style vision encoder + Eagle-2 LLM composition through the existing layer factory.</item>
///   <item>System-2-conditioned velocity head: takes <c>[noisy_action, t, latent]</c> and emits the per-dim velocity field, matching the DiT-AdaLN conditioning pattern from the paper.</item>
///   <item>52-DOF whole-body action dimension default per paper §3.4 (arms 7×2 + hands 12×2 + torso 3 + legs 6×2 + neck 2).</item>
/// </list>
/// <para><b>What is NOT verified in-session:</b></para>
/// <list type="bullet">
///   <item>Numerical parity vs NVIDIA's public GR00T-N1-3B HuggingFace weights (loading requires the Eagle-2 tokenizer + SigLIP weight converter, beyond this PR's scope).</item>
///   <item>Hardware deployment on a real humanoid (requires NVIDIA's IsaacLab + sim2real pipeline).</item>
/// </list>
/// <para><b>For Beginners:</b> GR00T N1 is the first big-tech open humanoid brain. The fast policy
/// doesn't just predict joints directly — it predicts a noise-to-data flow, the same trick image
/// generators like Stable Diffusion use. This lets it produce smooth, physically-plausible joint
/// trajectories instead of jerky direct regressions.</para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 52);
/// var gr00t = new GR00TN1&lt;double&gt;(arch, new GR00TN1Options());
///
/// // Single inference chunk (one S2 pass + Horizon flow-matching samples).
/// var jointCommands = gr00t.PredictAction(image, "set the table");
///
/// // Streaming control at the paper's 50 Hz S1 rate.
/// var runner = gr00t.CreateDualSystemRunner();
/// while (running) {
///     var action = runner.Step(currentImage, currentInstruction);
///     SendToHumanoid(action);
/// }
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Robotics)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("GR00T N1: An Open Foundation Model for Generalist Humanoid Robots", "https://arxiv.org/abs/2503.14734", Year = 2025, Authors = "NVIDIA")]
public class GR00TN1<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly GR00TN1Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private readonly GR00TFlowMatchingActionHead<T> _actionHead;
    // Learned instruction-token embedding table — replaces the previous
    // deterministic sinusoidal fabrication in EmbedInstructionTokens. GR00T N1's
    // System-2 VLM consumes learned text embeddings (NVIDIA 2025, §3.1); synthetic
    // sin/cos vectors from token IDs weren't model-faithful and carried no training
    // signal. (SinusoidalTimeEmbedding below is unrelated — that is the flow-matching
    // timestep embedding, which is legitimately sinusoidal.) Embedding dim = decoder
    // dim so the embeddings concatenate with the visual-feature sequence. Mirrors the
    // learned EmbeddingLayer used by RT2<T>.
    private readonly EmbeddingLayer<T> _tokenEmbedding;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private int _system2EndLayerIndex;

    public override ModelOptions GetOptions() => _options;

    public GR00TN1(NeuralNetworkArchitecture<T> architecture, string modelPath, GR00TN1Options? options = null) : base(architecture)
    {
        _options = options ?? new GR00TN1Options();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _actionHead = BuildActionHead();
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        InitializeLayers();
    }

    public GR00TN1(NeuralNetworkArchitecture<T> architecture, GR00TN1Options? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new GR00TN1Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _actionHead = BuildActionHead();
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public string LanguageModelName => _options.LanguageModelName;
    public int ActionDimension => _options.ActionDimension;

    /// <summary>The flow-matching action head used by <see cref="PredictAction"/>. Exposed so callers can swap the integration-step count or provide a custom velocity callback.</summary>
    public GR00TFlowMatchingActionHead<T> ActionHead => _actionHead;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return L2Normalize(OnnxModel.Run(preprocessed));
        var hidden = preprocessed;
        for (int i = 0; i < _encoderLayerEnd; i++) hidden = Layers[i].Forward(hidden);
        return L2Normalize(hidden);
    }

    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(preprocessed);
        return System2Forward(preprocessed, prompt ?? string.Empty);
    }

    /// <summary>
    /// System-2 forward pass: SigLIP-style vision encoder + Eagle-2 LLM decoder + latent projection
    /// (paper §3.1). Returns the semantic latent that conditions the flow-matching action head.
    /// </summary>
    public Tensor<T> System2Forward(Tensor<T> preprocessedObservation, string instruction)
    {
        ThrowIfDisposed();
        var visual = preprocessedObservation;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visual = Layers[i].Forward(visual);

        var fused = string.IsNullOrEmpty(instruction)
            ? visual
            : visual.ConcatenateTensors(EmbedInstructionTokens(TokenizeText(instruction)));

        var hidden = fused;
        for (int i = _encoderLayerEnd; i < _system2EndLayerIndex; i++)
            hidden = Layers[i].Forward(hidden);
        return hidden;
    }

    /// <summary>
    /// System-1 velocity field: takes a noisy action tensor at flow time <paramref name="t"/> and the
    /// System-2 latent, runs the DiT-style velocity network, and returns the per-dim velocity. Public
    /// so callers can compose custom inference loops (e.g. with classifier-free guidance).
    /// </summary>
    public Tensor<T> System1Velocity(Tensor<T> noisyAction, double t, Tensor<T> system2Latent)
    {
        ThrowIfDisposed();
        if (noisyAction is null) throw new ArgumentNullException(nameof(noisyAction));
        if (system2Latent is null) throw new ArgumentNullException(nameof(system2Latent));

        // DiT-AdaLN conditioning: concatenate [noisy_action, time_embedding, latent] and run through
        // the System-1 transformer stack. Time is encoded as a sinusoidal frequency embedding so the
        // velocity field can be queried at any t in [0, 1].
        var timeEmbed = SinusoidalTimeEmbedding(t, _options.System1HiddenDim);
        var conditioned = noisyAction.ConcatenateTensors(timeEmbed).ConcatenateTensors(system2Latent);

        var hidden = conditioned;
        for (int i = _system2EndLayerIndex; i < Layers.Count; i++)
            hidden = Layers[i].Forward(hidden);

        // Action-head output is the velocity at the same shape as the noisy action.
        var velocity = new Tensor<T>([noisyAction.Length]);
        int copyLen = Math.Min(velocity.Length, hidden.Length);
        for (int d = 0; d < copyLen; d++) velocity[d] = hidden[d];
        return velocity;
    }

    /// <summary>
    /// Predicts a continuous joint-command horizon using the GR00T N1 dual-system protocol:
    /// one S2 pass (Eagle-2 VLM) + flow-matching Euler integration via <see cref="ActionHead"/>.
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = Math.Max(1, _options.PredictionHorizon);

        var preprocessed = PreprocessImage(observation);
        var s2Latent = System2Forward(preprocessed, instruction);
        return _actionHead.GenerateHorizon(actionDim, horizon, s2Latent);
    }

    /// <summary>
    /// Wires this model into a <see cref="HelixDualSystemRunner{T}"/> for streaming 50 Hz S1
    /// control with periodic S2 re-invocation. Reuses the Helix runner (same coordination logic
    /// applies to both models — paper §4.1).
    /// </summary>
    public HelixDualSystemRunner<T> CreateDualSystemRunner()
    {
        return new HelixDualSystemRunner<T>(
            system2Forward: (obs, instr) => System2Forward(PreprocessImage(obs), instr),
            system1Forward: (obs, latent) => _actionHead.GenerateHorizon(_options.ActionDimension, 1, latent),
            system2TicksValid: Math.Max(1, _options.System1ToSystem2Ratio));
    }

    private GR00TFlowMatchingActionHead<T> BuildActionHead()
    {
        return new GR00TFlowMatchingActionHead<T>(
            velocityNetwork: (xt, t, latent) => System1Velocity(xt, t, latent),
            numIntegrationSteps: _options.FlowMatchingSteps,
            seed: _options.Seed);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            int third = Math.Max(1, Layers.Count / 3);
            _encoderLayerEnd = third;
            _system2EndLayerIndex = Math.Min(Layers.Count, 2 * third);
            ValidateEncoderDecoderBoundary(_encoderLayerEnd);
            return;
        }

        // System 2: vision encoder + Eagle-2 LLM decoder via the standard robotics factory.
        Layers.AddRange(LayerHelper<T>.CreateDefaultRoboticsActionLayers(
            visionDim: _options.VisionDim,
            decoderDim: _options.DecoderDim,
            actionDim: _options.DecoderDim,
            numVisionLayers: _options.NumVisionLayers,
            numDecoderLayers: _options.NumDecoderLayers,
            numActionLayers: 2,
            numHeads: _options.NumHeads,
            dropoutRate: _options.DropoutRate));

        // S2 latent emission head.
        IActivationFunction<T> identity = new IdentityActivation<T>();
        IActivationFunction<T> gelu = new GELUActivation<T>();
        Layers.Add(new LayerNormalizationLayer<T>());
        Layers.Add(new DenseLayer<T>(_options.System2LatentDim, identity));
        ComputeEncoderDecoderBoundary();
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
        _system2EndLayerIndex = Layers.Count;

        // System 1: DiT velocity field. Input = [noisy_action, time_embedding, latent], output =
        // velocity at action shape. Paper §3.2: ~280M params via 12 transformer blocks @ 1024 dim.
        int s1Dim = _options.System1HiddenDim;
        int s1FfnDim = s1Dim * 4;
        for (int i = 0; i < _options.System1NumLayers; i++)
        {
            Layers.Add(new MultiHeadAttentionLayer<T>(_options.System1NumHeads, s1Dim / Math.Max(1, _options.System1NumHeads)));
            Layers.Add(new LayerNormalizationLayer<T>());
            Layers.Add(new DenseLayer<T>(s1FfnDim, gelu));
            Layers.Add(new DenseLayer<T>(s1Dim, identity));
            Layers.Add(new LayerNormalizationLayer<T>());
            if (_options.DropoutRate > 0) Layers.Add(new DropoutLayer<T>(_options.DropoutRate));
        }
        Layers.Add(new DenseLayer<T>(_options.ActionDimension, identity));
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int layersPerBlock = TransformerBlockLayerCount(_options.DropoutRate);
        _encoderLayerEnd = 1 + _options.NumVisionLayers * layersPerBlock + 2;
    }

    /// <summary>
    /// Looks up instruction-token embeddings through the learned
    /// <see cref="_tokenEmbedding"/> table (NVIDIA GR00T N1 2025, §3.1). Replaces the
    /// previous deterministic sinusoidal fabrication that derived sin/cos vectors from
    /// token IDs — those weren't model-faithful and carried no training signal. Returns
    /// an empty-safe <c>[DecoderDim]</c> tensor for a zero-length token sequence so
    /// downstream concatenation shapes stay valid. (Not to be confused with
    /// <see cref="SinusoidalTimeEmbedding"/>, the flow-matching timestep embedding.)
    /// </summary>
    private Tensor<T> EmbedInstructionTokens(Tensor<T> instructionTokens)
    {
        if (instructionTokens.Length == 0) return new Tensor<T>([_options.DecoderDim]);
        return _tokenEmbedding.Forward(instructionTokens);
    }

    private Tensor<T> SinusoidalTimeEmbedding(double t, int dim)
    {
        // Standard transformer sinusoidal embedding for the continuous flow-matching time variable
        // (Vaswani et al. 2017, adapted for continuous t per Lipman et al. 2023 eq. 5).
        var embed = new Tensor<T>([dim]);
        double scaledT = t * 1000.0;
        for (int d = 0; d < dim; d++)
        {
            double freq = 1.0 / Math.Pow(10000.0, 2.0 * (d / 2) / (double)dim);
            double val = (d % 2 == 0) ? Math.Sin(scaledT * freq) : Math.Cos(scaledT * freq);
            embed[d] = NumOps.FromDouble(val);
        }
        return embed;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null) throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++) tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        var hidden = input;
        foreach (var layer in Layers) hidden = layer.Forward(hidden);
        return hidden;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = (int)layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
        // _tokenEmbedding lives OUTSIDE Layers: it embeds token IDs on the dedicated
        // instruction path (EmbedInstructionTokens), so it cannot join the sequential
        // Layers walk that Predict runs image tensors through. Its parameters ride at
        // the TAIL of the flat vector — same layout as GetParameters/SetParameters —
        // so training updates reach the embedding table (same off-Layers contract as
        // PaLME._patchEmbed).
        int embedCount = (int)_tokenEmbedding.ParameterCount;
        if (embedCount > 0 && idx + embedCount <= parameters.Length)
            _tokenEmbedding.UpdateParameters(parameters.Slice(idx, embedCount));
    }

    /// <inheritdoc />
    /// <remarks>
    /// Includes the off-<see cref="NeuralNetworkBase{T}.Layers"/> instruction-token
    /// embedding table so the flat parameter APIs (<see cref="GetParameters"/> /
    /// <see cref="SetParameters"/> / <see cref="UpdateParameters"/>) agree on length.
    /// </remarks>
    public override long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var layer in Layers) total += layer.ParameterCount;
            return total + _tokenEmbedding.ParameterCount;
        }
    }

    /// <inheritdoc />
    /// <remarks>Layout: [layer params in Layers order ...] [token-embedding params].</remarks>
    public override Vector<T> GetParameters()
    {
        var baseParams = base.GetParameters();
        var embedParams = _tokenEmbedding.GetParameters();
        if (embedParams.Length == 0) return baseParams;
        var combined = new Vector<T>(baseParams.Length + embedParams.Length);
        for (int i = 0; i < baseParams.Length; i++) combined[i] = baseParams[i];
        for (int i = 0; i < embedParams.Length; i++) combined[baseParams.Length + i] = embedParams[i];
        return combined;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Accepts both the full layout produced by <see cref="GetParameters"/> (layers +
    /// embedding tail) and a layers-only vector (the embedding is left untouched), so
    /// older callers that sized their vector from the Layers sum keep working.
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        int embedCount = (int)_tokenEmbedding.ParameterCount;
        int baseCount = parameters.Length - embedCount;
        if (baseCount < 0) baseCount = parameters.Length;

        var baseSlice = new Vector<T>(baseCount);
        for (int i = 0; i < baseCount; i++) baseSlice[i] = parameters[i];
        base.SetParameters(baseSlice);

        if (embedCount > 0 && baseCount + embedCount == parameters.Length)
        {
            var embedSlice = new Vector<T>(embedCount);
            for (int i = 0; i < embedCount; i++) embedSlice[i] = parameters[baseCount + i];
            _tokenEmbedding.SetParameters(embedSlice);
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GR00T-N1-Native" : "GR00T-N1-ONNX",
            Description = "GR00T N1: dual-system VLA with SigLIP + Eagle-2 reasoning and flow-matching DiT action head (NVIDIA 2025, arXiv:2503.14734).",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers + _options.System1NumLayers,
        };
        meta.AdditionalInfo["Architecture"] = "GR00T-N1";
        meta.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        meta.AdditionalInfo["VisionEncoder"] = "SigLIP";
        meta.AdditionalInfo["ActionHead"] = "FlowMatching-DiT";
        meta.AdditionalInfo["FlowMatchingSteps"] = _options.FlowMatchingSteps.ToString();
        meta.AdditionalInfo["ActionDimension"] = _options.ActionDimension.ToString();
        meta.AdditionalInfo["System2LatentDim"] = _options.System2LatentDim.ToString();
        meta.AdditionalInfo["System1HiddenDim"] = _options.System1HiddenDim.ToString();
        meta.AdditionalInfo["S1S2Ratio"] = _options.System1ToSystem2Ratio.ToString();
        return meta;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.ActionDimension);
        writer.Write(_options.NumJoints);
        writer.Write(_options.System2LatentDim);
        writer.Write(_options.System1HiddenDim);
        writer.Write(_options.System1NumLayers);
        writer.Write(_options.System1NumHeads);
        writer.Write(_options.System1ToSystem2Ratio);
        writer.Write(_options.FlowMatchingSteps);

        // The instruction-token embedding lives outside Layers, so the base
        // per-layer serialization never persists it — without this block a trained
        // model's embedding table silently reverts to random init on load.
        var embedParams = _tokenEmbedding.GetParameters();
        writer.Write(embedParams.Length);
        for (int i = 0; i < embedParams.Length; i++)
            writer.Write(Convert.ToDouble(embedParams[i]));
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.ActionDimension = reader.ReadInt32();
        _options.NumJoints = reader.ReadInt32();
        _options.System2LatentDim = reader.ReadInt32();
        _options.System1HiddenDim = reader.ReadInt32();
        _options.System1NumLayers = reader.ReadInt32();
        _options.System1NumHeads = reader.ReadInt32();
        _options.System1ToSystem2Ratio = reader.ReadInt32();
        _options.FlowMatchingSteps = reader.ReadInt32();

        // Restore the trained instruction-token embedding written by
        // SerializeNetworkSpecificData (it lives outside Layers, so the base
        // per-layer restore never touches it).
        int embedCount = reader.ReadInt32();
        if (embedCount > 0)
        {
            if (embedCount != (int)_tokenEmbedding.ParameterCount)
                throw new InvalidOperationException(
                    $"Serialized GR00T-N1 token-embedding parameter count ({embedCount:N0}) does not match " +
                    $"this instance's embedding ({_tokenEmbedding.ParameterCount:N0}). The model was saved with " +
                    "a different VocabSize/DecoderDim configuration.");
            var embedParams = new Vector<T>(embedCount);
            for (int i = 0; i < embedCount; i++)
                embedParams[i] = NumOps.FromDouble(reader.ReadDouble());
            _tokenEmbedding.SetParameters(embedParams);
        }

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new GR00TN1<T>(Architecture, mp, _options);
        return new GR00TN1<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GR00TN1<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
