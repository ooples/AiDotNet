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
/// E2 TTS ("Embarrassingly Easy"): fully non-autoregressive text-to-speech by flow-matching speech infilling over
/// characters padded with filler tokens to the speech length, with a flat U-Net Transformer.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS" (Eskimez et al., SLT
/// 2024); the paper has no public code, so the details it leaves out follow the reproduction in F5-TTS (Chen et al. 2024,
/// SWivid/F5-TTS <c>UNetT</c>).</para>
/// <para>
/// The character sequence, padded with the filler token to the frame count, is embedded at the mel width and
/// concatenated with the noisy mel and the masked mel condition; a projection and a convolutional position embedding feed
/// a Transformer whose layers i and depth − 1 − i are joined by U-Net skip connections (concatenated and projected), with
/// the flow-step embedding prepended as a token, RMSNorm pre-normalization, rotary self-attention and GELU feed-forward.
/// Training and sampling are <see cref="FlowMatchingTtsModelBase{T}"/>'s.
/// </para>
/// <para><b>For Beginners:</b> E2 TTS learns to fill in masked parts of a spectrogram from the text alone — no phoneme
/// alignment, no duration model — and to speak it fills in everything after a voice prompt.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Diffusion)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS",
    "https://arxiv.org/abs/2406.18009",
    Year = 2024,
    Authors = "Eskimez et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 7.5e-5, WarmupSteps = 20000, ReferenceBatchSize = 307200,
                Provenance = RecipeProvenance.DerivedFromCitedWork,
                Source = "The reproduction in F5-TTS (Chen et al. 2024, Sec. 4), whose E2 TTS shares its recipe: AdamW at "
                        + "a peak 7.5e-5, linear warmup over 20K updates then linear decay, gradient clip 1.")]
public partial class E2TTS<T> : FlowMatchingTtsModelBase<T>, IAcousticModel<T>
{
    private readonly E2TTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private FlowTimestepEmbedding<T>? _time;
    private FlowTextEmbedding<T>? _text;
    private FlowInputEmbedding<T>? _input;
    private readonly List<(BiasFreeLinearLayer<T>? SkipProjection, RMSNormalizationLayer<T> AttentionNorm, FlowSelfAttention<T> Attention,
        RMSNormalizationLayer<T> FeedForwardNorm, FlowFeedForward<T> FeedForward)> _layers = new();
    private RMSNormalizationLayer<T>? _finalNorm;
    private DenseLayer<T>? _projection;

    public override ModelOptions GetOptions() => _options;

    public E2TTS(NeuralNetworkArchitecture<T> architecture, string modelPath, E2TTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new E2TTSOptions();
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

    public E2TTS(
        NeuralNetworkArchitecture<T> architecture,
        E2TTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new E2TTSOptions();
        if (_options.NumLayers % 2 != 0)
            throw new ArgumentException("The U-Net Transformer pairs its layers; NumLayers must be even.", nameof(options));
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

    /// <summary>Generates a mel spectrogram from text (paired with a Vocos vocoder in the reproduction).</summary>
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
        // E2 embeds the characters at the mel width with no text encoder.
        _text = new FlowTextEmbedding<T>(Engine, layers, o.VocabSize, o.MelChannels, 0, 2);
        _input = new FlowInputEmbedding<T>(Engine, layers, o.HiddenDim);
        for (int i = 0; i < o.NumLayers; i++)
        {
            bool laterHalf = i >= o.NumLayers / 2;
            // Later-half layers join the mirrored earlier layer: concatenate, then a bias-free projection (reference skip_proj).
            BiasFreeLinearLayer<T>? skip = laterHalf ? FlowMatchingTts.Own(layers, new BiasFreeLinearLayer<T>(2 * o.HiddenDim, o.HiddenDim)) : null;
            var attentionNorm = FlowMatchingTts.Own(layers, new RMSNormalizationLayer<T>(o.HiddenDim));
            var attention = new FlowSelfAttention<T>(Engine, layers, o.HiddenDim, o.NumHeads, o.HeadDim, o.DropoutRate, o.RotaryHeads);
            var feedForwardNorm = FlowMatchingTts.Own(layers, new RMSNormalizationLayer<T>(o.HiddenDim));
            var feedForward = new FlowFeedForward<T>(Engine, layers, o.HiddenDim, o.HiddenDim * o.FeedForwardMultiplier, o.DropoutRate);
            _layers.Add((skip, attentionNorm, attention, feedForwardNorm, feedForward));
        }
        _finalNorm = FlowMatchingTts.Own(layers, new RMSNormalizationLayer<T>(o.HiddenDim));
        _projection = FlowMatchingTts.Linear(layers, o.MelChannels);
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
        // The flow-step embedding is prepended as a token (reference UNetT: torch.cat([t, x])).
        var x = Engine.TensorConcatenate(new[] { time, _input!.Forward(noisy, cond, text) }, 0);
        var skips = new Stack<Tensor<T>>();
        for (int i = 0; i < _layers.Count; i++)
        {
            var (skipProjection, attentionNorm, attention, feedForwardNorm, feedForward) = _layers[i];
            if (skipProjection is null)
                skips.Push(x);
            else
                x = skipProjection.Forward(Engine.TensorConcatenate(new[] { x, skips.Pop() }, 1));
            x = Engine.TensorAdd(attention.Forward(attentionNorm.Forward(x)), x);
            x = Engine.TensorAdd(feedForward.Forward(feedForwardNorm.Forward(x)), x);
        }
        x = _finalNorm!.Forward(x);
        return _projection!.Forward(Engine.TensorSlice(x, new[] { 1, 0 }, new[] { frames, _options.HiddenDim }));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
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
            Name = _useNativeMode ? "E2TTS-Native" : "E2TTS-ONNX",
            Description = "E2 TTS: flow-matching speech infilling with a U-Net Transformer (Eskimez et al., 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumLayers,
        };
        m.AdditionalInfo["Architecture"] = "E2TTS";
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
