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
/// Helix: dual-system vision-language-action model for full-body humanoid control
/// (Figure AI, 2025, arXiv:2502.07092).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Helix is the first generalist VLA model deployed on a real humanoid for whole upper-body
/// dexterous manipulation. Its defining architectural choice is the <b>dual-system, dual-rate</b>
/// split between a slow VLM reasoner and a fast visuomotor policy:
/// </para>
/// <list type="bullet">
///   <item><b>System 2</b>: a 7B-parameter VLM (LLaMA-class) that runs at 7–9 Hz and produces a
///         semantic latent encoding "what the robot should be doing right now".</item>
///   <item><b>System 1</b>: an 80M-parameter visuomotor transformer that runs at 200 Hz, takes the
///         current observation and the most recent S2 latent, and emits continuous joint commands
///         for the 35-DOF humanoid upper body (torso, two 7-DOF arms, 2 hands × 4 finger DOFs).</item>
/// </list>
/// <para><b>Paper-faithful pieces implemented here:</b></para>
/// <list type="bullet">
///   <item><see cref="HelixSystem2Latent{T}"/>: typed conditioning signal with freshness tracking, so the runner can decide whether to reuse the cached S2 output or invoke a new pass.</item>
///   <item><see cref="HelixDualSystemRunner{T}"/>: explicit S1:S2 rate splitter (default 22:1, matching paper §4.1's 200 Hz : ~9 Hz).</item>
///   <item><see cref="PredictAction"/>: paper-faithful inference path that invokes S2 once per horizon and S1 every horizon-tick, so the returned action sequence respects the dual-rate constraint.</item>
///   <item><see cref="System2Forward"/> and <see cref="System1Forward"/>: callable directly for streaming control loops where the caller owns timing.</item>
/// </list>
/// <para><b>What is NOT verified in-session:</b></para>
/// <list type="bullet">
///   <item>Behaviour on a real Figure 02 humanoid (requires hardware + Figure's proprietary inference stack).</item>
///   <item>Numerical parity with Figure AI's public weights (not released publicly as of this PR's writing).</item>
///   <item>200 Hz inference throughput (target latency is hardware-dependent; CPU-only inference of an 80M policy at 5 ms requires fully-fused kernels — covered by other PRs in the Tensors repo).</item>
/// </list>
/// <para><b>For Beginners:</b> Helix is the first robot brain that's good enough to drive a real
/// humanoid through dexterous tasks like loading a dishwasher. The trick is two networks: a smart-but-
/// slow one decides intent ("pick up the red mug") and a fast-but-narrow one figures out the exact
/// motor commands needed to do it 200 times per second.</para>
/// </remarks>
/// <example>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 35);
/// var helix = new Helix&lt;double&gt;(arch, new HelixOptions());
///
/// // One-shot: predict a 16-tick horizon of joint commands.
/// var actions = helix.PredictAction(image, "stack the cups");
///
/// // Streaming: 200 Hz control loop with caller-owned timing.
/// var runner = helix.CreateDualSystemRunner();
/// while (running) {
///     var action = runner.Step(currentImage, currentInstruction);
///     SendToRobot(action);
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
[ResearchPaper("Helix: A Vision-Language-Action Model for Generalist Humanoid Control", "https://arxiv.org/abs/2502.07092", Year = 2025, Authors = "Figure AI")]
public class Helix<T> : VisionLanguageModelBase<T>, IVisionLanguageAction<T>
{
    private readonly HelixOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    // Learned instruction-token embedding table — replaces the previous
    // deterministic sinusoidal fabrication in EmbedInstructionTokens. Helix's
    // System-2 VLM consumes ordinary learned text embeddings (Figure AI 2025,
    // §3.2); synthetic sin/cos vectors derived from token IDs were not
    // model-faithful and decoupled the S2 context from any training signal.
    // Embedding dim = decoder dim so the embeddings concatenate with the
    // visual-feature sequence at the encoder boundary. Mirrors the learned
    // EmbeddingLayer used by RT2<T>.
    private readonly EmbeddingLayer<T> _tokenEmbedding;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private int _system2EndLayerIndex;

    public override ModelOptions GetOptions() => _options;

    public Helix(NeuralNetworkArchitecture<T> architecture, string modelPath, HelixOptions? options = null) : base(architecture)
    {
        _options = options ?? new HelixOptions();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        _tokenEmbedding = new EmbeddingLayer<T>(_options.VocabSize, _options.DecoderDim);
        InitializeLayers();
    }

    public Helix(NeuralNetworkArchitecture<T> architecture, HelixOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new HelixOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
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

    /// <summary>Default ratio of S1 to S2 invocations (paper §4.1: 200 Hz S1, 7–9 Hz S2 → ~22:1).</summary>
    public int System1ToSystem2Ratio => Math.Max(1, _options.System1ToSystem2Ratio);

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
    /// <b>System 2</b> forward pass: runs the full VLM (vision encoder + language decoder) and returns
    /// the semantic latent that conditions the fast System-1 controller. Paper §3.2 — the latent is the
    /// LLM's hidden state at the last position of the fused [visual_tokens, instruction_tokens] sequence.
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
    /// <b>System 1</b> forward pass: runs the fast 80M visuomotor policy at every control tick (200 Hz target).
    /// Consumes the current observation and the most recent S2 latent (paper §3.3) and emits a continuous
    /// joint-command tensor of length <see cref="ActionDimension"/>.
    /// </summary>
    public Tensor<T> System1Forward(Tensor<T> preprocessedObservation, Tensor<T> system2Latent)
    {
        ThrowIfDisposed();
        if (system2Latent is null) throw new ArgumentNullException(nameof(system2Latent));

        var visual = preprocessedObservation;
        for (int i = 0; i < _encoderLayerEnd; i++)
            visual = Layers[i].Forward(visual);

        // Fuse latent with visual features: S2 latent is the conditioning signal for S1's transformer.
        var fused = visual.ConcatenateTensors(system2Latent);

        var hidden = fused;
        for (int i = _system2EndLayerIndex; i < Layers.Count; i++)
            hidden = Layers[i].Forward(hidden);

        return TanhClampJointCommands(hidden, _options.ActionDimension);
    }

    /// <summary>
    /// Creates a <see cref="HelixDualSystemRunner{T}"/> wired to this model's S1/S2 callbacks for
    /// streaming-control use. The runner handles re-invoking S2 when its cached latent goes stale.
    /// </summary>
    public HelixDualSystemRunner<T> CreateDualSystemRunner()
    {
        return new HelixDualSystemRunner<T>(
            system2Forward: (obs, instr) => System2Forward(PreprocessImage(obs), instr),
            system1Forward: (obs, latent) => System1Forward(PreprocessImage(obs), latent),
            system2TicksValid: System1ToSystem2Ratio);
    }

    /// <summary>
    /// Predicts <see cref="ActionDimension"/> × <see cref="PredictionHorizon"/> continuous joint commands
    /// using Helix's dual-system inference path: one S2 invocation produces the semantic latent, then S1
    /// rolls out <c>PredictionHorizon</c> joint-command tensors conditioned on the same latent. Matches
    /// the paper §4.1 inference protocol (single S2 pass per chunk, multiple S1 ticks reusing the latent).
    /// </summary>
    public Tensor<T> PredictAction(Tensor<T> observation, string instruction)
    {
        ThrowIfDisposed();
        int actionDim = _options.ActionDimension;
        int horizon = Math.Max(1, _options.PredictionHorizon);

        var preprocessed = PreprocessImage(observation);
        var s2Latent = System2Forward(preprocessed, instruction);

        var horizonActions = new Tensor<T>([actionDim * horizon]);
        for (int step = 0; step < horizon; step++)
        {
            var s1Action = System1Forward(preprocessed, s2Latent);
            int writeBase = step * actionDim;
            int copyLen = Math.Min(actionDim, s1Action.Length);
            for (int d = 0; d < copyLen; d++)
                horizonActions[writeBase + d] = s1Action[d];
        }
        return horizonActions;
    }

    /// <summary>
    /// Squashes the action-head output into a per-joint tanh-bounded velocity command. The 35-DOF
    /// upper body in the paper is structured as: torso(3) + leftArm(7) + rightArm(7) + leftHand(8) +
    /// rightHand(8) + neck(2). Joint limits in the paper are enforced downstream by the inverse-
    /// kinematics controller; here we only ensure the network output stays in [-1, 1].
    /// </summary>
    private Tensor<T> TanhClampJointCommands(Tensor<T> raw, int actionDim)
    {
        int len = Math.Min(actionDim, raw.Length);
        var clamped = new Tensor<T>([actionDim]);
        for (int j = 0; j < len; j++)
        {
            double v = NumOps.ToDouble(raw[j]);
            clamped[j] = NumOps.FromDouble(Math.Tanh(v));
        }
        return clamped;
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

        // Vision encoder + S2 LLM decoder via the standard robotics factory.
        Layers.AddRange(LayerHelper<T>.CreateDefaultRoboticsActionLayers(
            visionDim: _options.VisionDim,
            decoderDim: _options.DecoderDim,
            actionDim: _options.DecoderDim,
            numVisionLayers: _options.NumVisionLayers,
            numDecoderLayers: _options.NumDecoderLayers,
            numActionLayers: 2,
            numHeads: _options.NumHeads,
            dropoutRate: _options.DropoutRate));

        // S2 latent emission head (LayerNorm + projection to S2_LatentDim).
        IActivationFunction<T> identity = new IdentityActivation<T>();
        Layers.Add(new LayerNormalizationLayer<T>());
        Layers.Add(new DenseLayer<T>(_options.System2LatentDim, identity));
        ComputeEncoderDecoderBoundary();
        ValidateEncoderDecoderBoundary(_encoderLayerEnd);
        _system2EndLayerIndex = Layers.Count;

        // System-1: fast 80M-class visuomotor transformer that consumes [visual_features, S2_latent]
        // and emits continuous joint commands. Composed from N transformer blocks at `System1HiddenDim`
        // followed by an action-head projection to `ActionDimension`.
        IActivationFunction<T> gelu = new GELUActivation<T>();
        int s1Dim = _options.System1HiddenDim;
        int s1FfnDim = s1Dim * 4;
        // The System-2 latent head emits System2LatentDim features, but System-1
        // runs at System1HiddenDim. Project the latent into S1's embedding width
        // so the first S1 attention sees a dimensionally-consistent input — paper
        // §3.3: S1 is conditioned on the S2 latent, which must be mapped into S1's
        // space. Without this projection the flat layer chain feeds a 512-d latent
        // into a 384-d attention and the forward throws on a shape mismatch.
        Layers.Add(new DenseLayer<T>(s1Dim, identity));
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
        // Vision stack: 1 leading LN + numVisionLayers × block + 2 projection layers (Dense + LN).
        _encoderLayerEnd = 1 + _options.NumVisionLayers * layersPerBlock + 2;
    }

    /// <summary>
    /// Looks up instruction-token embeddings through the learned
    /// <see cref="_tokenEmbedding"/> table (Figure AI 2025, §3.2). Replaces the
    /// previous deterministic sinusoidal fabrication that derived sin/cos vectors
    /// from token IDs — those weren't model-faithful and carried no training
    /// signal. Returns an empty-safe <c>[DecoderDim]</c> tensor for a zero-length
    /// token sequence so downstream concatenation shapes stay valid.
    /// </summary>
    private Tensor<T> EmbedInstructionTokens(Tensor<T> instructionTokens)
    {
        if (instructionTokens.Length == 0) return new Tensor<T>([_options.DecoderDim]);
        return _tokenEmbedding.Forward(instructionTokens);
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
        // PaLME._patchEmbed and GR00TN1).
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

        // Derive the layer-side size from the actual Layers walk, NOT from
        // parameters.Length − embedCount. With the subtraction form a legacy
        // layers-only vector silently sized baseCount to layerCount − embedCount,
        // dropping the tail of the regular layer weights before
        // base.SetParameters ran. Compute the true layer total once and pick
        // the matching layout explicitly.
        int layerCount = 0;
        foreach (var layer in Layers) layerCount += (int)layer.ParameterCount;

        if (parameters.Length != layerCount && parameters.Length != layerCount + embedCount)
            throw new ArgumentException(
                $"Expected {layerCount} (layers-only) or {layerCount + embedCount} (layers + embedding) parameters, got {parameters.Length}.",
                nameof(parameters));

        var baseSlice = new Vector<T>(layerCount);
        for (int i = 0; i < layerCount; i++) baseSlice[i] = parameters[i];
        base.SetParameters(baseSlice);

        if (embedCount > 0 && parameters.Length == layerCount + embedCount)
        {
            var embedSlice = new Vector<T>(embedCount);
            for (int i = 0; i < embedCount; i++) embedSlice[i] = parameters[layerCount + i];
            _tokenEmbedding.SetParameters(embedSlice);
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) => NormalizeImage(image, _options.ImageMean, _options.ImageStd);
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "Helix-Native" : "Helix-ONNX",
            Description = "Helix: dual-system VLA for full-body humanoid control (Figure AI 2025, arXiv:2502.07092).",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers + _options.System1NumLayers,
        };
        meta.AdditionalInfo["Architecture"] = "Helix";
        meta.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        meta.AdditionalInfo["ActionDimension"] = _options.ActionDimension.ToString();
        meta.AdditionalInfo["System2LatentDim"] = _options.System2LatentDim.ToString();
        meta.AdditionalInfo["System1HiddenDim"] = _options.System1HiddenDim.ToString();
        meta.AdditionalInfo["System1NumLayers"] = _options.System1NumLayers.ToString();
        meta.AdditionalInfo["System1NumHeads"] = _options.System1NumHeads.ToString();
        meta.AdditionalInfo["S1S2Ratio"] = System1ToSystem2Ratio.ToString();
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

        // Restore the trained instruction-token embedding written by
        // SerializeNetworkSpecificData (it lives outside Layers, so the base
        // per-layer restore never touches it).
        int embedCount = reader.ReadInt32();
        if (embedCount > 0)
        {
            if (embedCount != (int)_tokenEmbedding.ParameterCount)
                throw new InvalidOperationException(
                    $"Serialized Helix token-embedding parameter count ({embedCount:N0}) does not match " +
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
            return new Helix<T>(Architecture, mp, _options);
        return new Helix<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(Helix<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
