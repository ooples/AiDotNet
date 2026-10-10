using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>
/// F5-TTS: a fully non-autoregressive text-to-speech model that fills in speech with flow matching, using a Diffusion
/// Transformer and ConvNeXt V2 text refinement over characters padded to the speech length.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching" (Chen et
/// al., 2024) and its reference implementation (SWivid/F5-TTS) for what the paper leaves unstated.</para>
/// <para>
/// The characters, shifted so 0 is the filler token and padded to the frame count, are embedded, given an absolute
/// sinusoidal position and refined by ConvNeXt V2 blocks (§3.2). They are concatenated with the noisy mel and the masked
/// mel condition, projected and given a convolutional position embedding, then run through DiT blocks whose
/// adaLN-zero is conditioned on the flow step, with rotary self-attention; a final adaptive LayerNorm and a
/// zero-initialized projection give the flow. Training and sampling are <see cref="FlowMatchingTtsModelBase{T}"/>'s.
/// </para>
/// <para><b>For Beginners:</b> F5-TTS learns to fill in a masked part of a spectrogram given the text; to speak, it
/// fills in everything after an optional voice prompt.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching",
    "https://arxiv.org/abs/2410.06885",
    Year = 2024,
    Authors = "Chen et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 7.5e-5, WarmupSteps = 20000, ReferenceBatchSize = 307200,
                Source = "Chen et al. 2024, Sec. 4: AdamW with a peak learning rate of 7.5e-5, linearly warmed up "
                        + "for 20K updates and linearly decayed over the rest of training, max gradient norm 1, a "
                        + "batch of 307,200 audio frames, 1.2M updates.")]
public partial class F5TTS<T> : FlowMatchingTtsModelBase<T>, IAcousticModel<T>
{
    private readonly F5TTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private FlowTimestepEmbedding<T>? _time;
    private FlowTextEmbedding<T>? _text;
    private FlowInputEmbedding<T>? _input;
    private readonly List<FlowDiTBlock<T>> _blocks = new();
    private DenseLayer<T>? _finalModulation;
    private DenseLayer<T>? _projection;

    public override ModelOptions GetOptions() => _options;

    public F5TTS(NeuralNetworkArchitecture<T> architecture, string modelPath, F5TTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new F5TTSOptions();
        _useNativeMode = false;
        SeedTraining(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public F5TTS(
        NeuralNetworkArchitecture<T> architecture,
        F5TTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new F5TTSOptions();
        _useNativeMode = true;
        SeedTraining(_options.SamplingSeed);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (_options.MaxGradientNorm > 0)
            MaxGradNorm = NumOps.FromDouble(_options.MaxGradientNorm);
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (the paper pairs it with Vocos).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    protected override FlowMatchingSettings Settings => new(_options.VocabSize, _options.MelChannels, _options.MaskFractionMin,
        _options.MaskFractionMax, _options.AudioDropProbability, _options.ConditionDropProbability,
        _options.NumFunctionEvaluations, _options.CfgStrength, _options.SwayCoefficient, _options.FramesPerCharacter,
        _options.Speed, _options.SamplingSeed);

    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    protected override bool HasPaperBackbone => _time is not null;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var layers = new List<LayerBase<T>>();
        _time = new FlowTimestepEmbedding<T>(Engine, layers, o.HiddenDim);
        _text = new FlowTextEmbedding<T>(Engine, layers, o.VocabSize, o.TextDim, o.TextConvLayers, 2);
        _input = new FlowInputEmbedding<T>(Engine, layers, o.HiddenDim);
        for (int i = 0; i < o.NumLayers; i++)
            _blocks.Add(new FlowDiTBlock<T>(Engine, layers, o.HiddenDim, o.NumHeads, o.HeadDim, o.HiddenDim * o.FeedForwardMultiplier,
                o.DropoutRate, o.RotaryHeads));
        // AdaLayerNorm_Final and proj_out, both zero-initialized (DiT.initialize_weights).
        _finalModulation = FlowMatchingTts.Linear(layers, 2 * o.HiddenDim, zero: true);
        _projection = FlowMatchingTts.Linear(layers, o.MelChannels, zero: true);
        // The character table leads the layer stack, so the published input contract is the token domain.
        AddEncoderDecoderLayers(new ILayer<T>[] { _text.Embedding }, Array.Empty<ILayer<T>>());
        ComponentLayers.AddRange(layers.Where(layer => !ReferenceEquals(layer, _text.Embedding)));
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictFlow(Tensor<T> noisy, Tensor<T> condition, Tensor<T> tokens, double t, bool dropAudio, bool dropText)
    {
        int frames = noisy.Shape[0];
        var time = _time!.Forward(t);
        var text = _text!.Forward(tokens, frames, dropText);
        var cond = dropAudio ? new Tensor<T>(condition._shape) : condition;
        var x = _input!.Forward(noisy, cond, text);
        foreach (var block in _blocks) x = block.Forward(x, time);
        var m = _finalModulation!.Forward(Engine.Swish(time));
        x = FlowMatchingTts.Modulate(Engine, FlowMatchingTts.PlainLayerNorm(Engine, x),
            FlowMatchingTts.Chunk(Engine, m, 0, _options.HiddenDim), FlowMatchingTts.Chunk(Engine, m, 1, _options.HiddenDim));
        return _projection!.Forward(x);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        // Linear warmup to the peak over WarmupSteps, then linear decay to zero at TotalSteps (§4).
        int warmup = Math.Max(1, _options.WarmupSteps), total = Math.Max(warmup + 1, _options.TotalSteps);
        var scheduler = new AiDotNet.LearningRateSchedulers.LambdaLRScheduler(_options.LearningRate,
            step => step < warmup ? (step + 1.0) / warmup : Math.Max(0.0, (double)(total - step) / (total - warmup)));
        return PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            }));
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "F5TTS-Native" : "F5TTS-ONNX",
            Description = "F5-TTS: flow-matching speech infilling with a DiT and ConvNeXt V2 text (Chen et al., 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumLayers,
        };
        m.AdditionalInfo["Architecture"] = "F5TTS";
        return m;
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
