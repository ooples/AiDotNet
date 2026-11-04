using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>E3 TTS: BERT-conditioned, raw-waveform diffusion without acoustic intermediates or explicit alignment.</summary>
/// <remarks>
/// This implements Gao et al.'s published model rather than a generic flow stack: uncased BERT-base, the exact ten
/// DBlock/UBlock configuration in Table 1, adaptive softmax convolution kernels at high waveform resolutions, FiLM and
/// self/cross-attention at low resolutions, learned timestep loss variance, and the paper's 1000-step DDPM schedule.
/// The authors published a paper and samples but no code or weights, so non-BERT weights use an explicit AiDotNet
/// conversion schema documented by <see cref="LoadConvertedWeights"/>.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("E3 TTS: Easy End-to-End Diffusion-based Text to Speech", "https://arxiv.org/abs/2311.00945",
    Year = 2023, Authors = "Yuan Gao, Nobuyuki Morioka, Yu Zhang and Nanxin Chen")]
public partial class E3TTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly E3TTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly HashSet<LayerBase<T>> _frozen = new();
    private readonly WordPieceTokenizer _tokenizer = WordPieceTokenizer.BertBaseUncased();
    private Random _trainingRandom;
    private bool _useNativeMode;
    private bool _disposed;

    public E3TTS(NeuralNetworkArchitecture<T> architecture, string modelPath, E3TTSOptions? options = null) : base(architecture)
    {
        _options = options is null ? new E3TTSOptions() : new E3TTSOptions(options);
        _options.Validate();
        Options = _options;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 1);
        _useNativeMode = false;
        ApplyBaseOptions();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    public E3TTS(NeuralNetworkArchitecture<T> architecture, E3TTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options is null ? new E3TTSOptions() : new E3TTSOptions(options);
        _options.Validate();
        Options = _options;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 1);
        _useNativeMode = true;
        ApplyBaseOptions();
        InitializePaperModel();
        // E3 does not disclose an optimizer and has no official training release, so do not present a guessed recipe
        // as a paper setting. Callers can inject one; AdamW is only AiDotNet's native fallback.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
    }

    private void ApplyBaseOptions()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
    }

    private void InitializePaperModel()
    {
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var textLayers = new List<LayerBase<T>>();
        TextEncoder = new E3BertTextEncoder<T>(Engine, textLayers, _options, random);
        Layers.AddRange(textLayers);
        if (_options.FreezeBert) foreach (var layer in textLayers) _frozen.Add(layer);
        var unetLayers = new List<LayerBase<T>>();
        WaveformUnet = new E3WaveformUnet<T>(Engine, unetLayers, _options, random);
        Layers.AddRange(unetLayers);
    }

    internal E3BertTextEncoder<T>? TextEncoder { get; private set; }
    internal E3WaveformUnet<T>? WaveformUnet { get; private set; }
    internal bool HasPaperModel => TextEncoder is not null && WaveformUnet is not null;

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => Math.Min(_options.MaxTextLength, _options.BertMaxPositions);
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumDiffusionSteps;
    public override bool TrainsOnSampleObjective => true;
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Text | TtsSupervision.Recording;
    protected override bool SupportsParameterMutation => _useNativeMode;

    protected override ParameterSlotRole ResolveLayerParameterRole(int layerIndex, ParameterSlotRole declared) =>
        _useNativeMode && layerIndex >= 0 && layerIndex < Layers.Count && _frozen.Contains(Layers[layerIndex])
            ? ParameterSlotRole.Frozen : declared;

    protected override Tensor<T> PreprocessText(string text) => IndexTtsOps.Ids<T>(TokenizeText(text));

    /// <summary>Tokenizes plain text as BERT-base-uncased subwords, including [CLS] and [SEP].</summary>
    public int[] TokenizeText(string text)
    {
        Validation.Guard.NotNull(text);
        if (string.IsNullOrWhiteSpace(text)) throw new ArgumentException("Text must not be empty.", nameof(text));
        var ids = _tokenizer.Encode(text).TokenIds.ToArray();
        if (ids.Length <= MaxTextLength) return ids;
        var result = ids.Take(MaxTextLength).ToArray();
        result[^1] = 102; // bert-base-uncased [SEP]
        return result;
    }

    private int[] TextWithPrompt(string text)
    {
        var target = TokenizeText(text);
        if (Voice?.ReferenceTokens is not { } reference || reference.Length < 2) return target;
        var prompt = reference.ToArray().Select(v => (int)Math.Round(Ops.ToDouble(v))).ToArray();
        return prompt.Take(prompt.Length - 1).Concat(target.Skip(1)).Take(MaxTextLength).ToArray();
    }

    /// <summary>Builds the waveform prompt used by the paper's zero-shot prompt synthesis experiment.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string transcript)
    {
        Validation.Guard.NotNull(recording);
        if (string.IsNullOrWhiteSpace(transcript)) throw new ArgumentException("The waveform prompt needs its transcript.", nameof(transcript));
        var audio = sampleRate == _options.SampleRate ? Engine.Reshape(recording, new[] { recording.Length })
            : AudioHelper<T>.ResampleBandLimited(Engine, Engine.Reshape(recording, new[] { recording.Length }), sampleRate, _options.SampleRate);
        return new TtsVoice<T> { ReferenceAudio = audio, ReferenceTokens = IndexTtsOps.Ids<T>(TokenizeText(transcript)) };
    }

    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var ids = TextWithPrompt(text);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(IndexTtsOps.Ids<T>(ids));
        if (!HasPaperModel) return PostprocessAudio(Predict(IndexTtsOps.Ids<T>(ids)));
        var textFeatures = Encode(ids);
        Tensor<T>? known = null;
        Tensor<T>? knownMask = null;
        if (Voice?.ReferenceAudio is { } prompt)
        {
            int count = Math.Min(prompt.Length, _options.InferenceWaveformSamples - 1);
            known = new Tensor<T>(new[] { _options.InferenceWaveformSamples });
            knownMask = new Tensor<T>(new[] { _options.InferenceWaveformSamples });
            for (int i = 0; i < count; i++) { known[i] = prompt.GetFlat(i); knownMask[i] = Ops.One; }
        }
        return TrimTrailingSilence(Sample(textFeatures, _options.InferenceWaveformSamples, known, knownMask, _options.DefaultSpeakerId));
    }

    /// <summary>
    /// Re-synthesizes only <paramref name="startSample"/>..<paramref name="endSample"/> while the DDPM trajectory keeps
    /// the samples outside that span fixed, matching the paper's text-based speech editing procedure.
    /// </summary>
    public Tensor<T> Edit(string text, Tensor<T> waveform, int startSample, int endSample, int speakerId = -1)
    {
        ThrowIfDisposed();
        RequirePaperModel();
        Validation.Guard.NotNull(waveform);
        int samples = Math.Min(waveform.Length, _options.InferenceWaveformSamples);
        if (startSample < 0 || endSample <= startSample || endSample > samples)
            throw new ArgumentOutOfRangeException(nameof(startSample), "The editing span must be inside the waveform.");
        var known = new Tensor<T>(new[] { _options.InferenceWaveformSamples });
        var mask = new Tensor<T>(new[] { _options.InferenceWaveformSamples });
        for (int i = 0; i < samples; i++)
        {
            known[i] = waveform.GetFlat(i);
            if (i < startSample || i >= endSample) mask[i] = Ops.One;
        }
        return TrimTrailingSilence(Sample(Encode(TokenizeText(text)), _options.InferenceWaveformSamples, known, mask,
            speakerId < 0 ? _options.DefaultSpeakerId : speakerId));
    }

    private Tensor<T> Encode(IReadOnlyList<int> ids)
    {
        RequirePaperModel();
        if (!_options.FreezeBert) return TextEncoder!.Forward(ids);
        using var _ = new NoGradScope<T>();
        return TextEncoder!.Forward(ids);
    }

    internal Tensor<T> Sample(Tensor<T> text, int sampleCount, Tensor<T>? known, Tensor<T>? knownMask, int speakerId)
    {
        RequirePaperModel();
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        var waveform = Gaussian(sampleCount, random);
        for (int step = _options.NumDiffusionSteps; step > 0; step--)
        {
            double alphaBar = AlphaBar(step);
            double previousAlphaBar = AlphaBar(step - 1);
            double alpha = alphaBar / previousAlphaBar;
            double beta = 1 - alpha;
            var (predicted, _) = WaveformUnet!.Forward(waveform, text, Math.Sqrt(alphaBar), speakerId);
            if (known is not null && knownMask is not null)
            {
                var actual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(waveform,
                    Engine.TensorMultiplyScalar(known, Ops.FromDouble(Math.Sqrt(alphaBar)))),
                    Ops.FromDouble(1.0 / Math.Sqrt(Math.Max(1e-12, 1 - alphaBar))));
                predicted = Blend(predicted, actual, knownMask);
            }
            var mean = Engine.TensorMultiplyScalar(Engine.TensorSubtract(waveform,
                Engine.TensorMultiplyScalar(predicted, Ops.FromDouble(beta / Math.Sqrt(Math.Max(1e-12, 1 - alphaBar))))),
                Ops.FromDouble(1.0 / Math.Sqrt(alpha)));
            if (step == 1) waveform = mean;
            else
            {
                double variance = beta * (1 - previousAlphaBar) / Math.Max(1e-12, 1 - alphaBar);
                waveform = Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(Gaussian(sampleCount, random), Ops.FromDouble(Math.Sqrt(variance))));
            }
        }
        return waveform;
    }

    private Tensor<T> Blend(Tensor<T> generated, Tensor<T> known, Tensor<T> mask)
    {
        var inverse = Engine.TensorAddScalar(Engine.TensorNegate(mask), Ops.One);
        return Engine.TensorAdd(Engine.TensorMultiply(generated, inverse), Engine.TensorMultiply(known, mask));
    }

    /// <summary>Cumulative product of the linear training betas in Eq. 4, before interval interpolation.</summary>
    public double TrainingAlphaBar(int step)
    {
        if (step < 0 || step > _options.NumDiffusionSteps) throw new ArgumentOutOfRangeException(nameof(step));
        double product = 1;
        for (int i = 0; i < step; i++)
        {
            double fraction = _options.NumDiffusionSteps == 1 ? 0 : (double)i / (_options.NumDiffusionSteps - 1);
            double beta = _options.TrainingBetaStart + fraction * (_options.TrainingBetaEnd - _options.TrainingBetaStart);
            product *= 1 - beta;
        }
        return product;
    }

    /// <summary>Equation 7's cumulative inference noise schedule; step zero is the clean endpoint.</summary>
    public double AlphaBar(int step)
    {
        if (step < 0 || step > _options.NumDiffusionSteps) throw new ArgumentOutOfRangeException(nameof(step));
        double phase = step / (double)_options.NumDiffusionSteps * Math.PI / 2;
        return Math.Exp(Math.Log(_options.MinimumAlphaBar) * Math.Pow(1 - Math.Cos(phase), 1.5));
    }

    private Tensor<T> Gaussian(int count, Random random)
    {
        var result = new Tensor<T>(new[] { count });
        for (int i = 0; i < count; i++) result[i] = Ops.FromDouble(IndexTtsOps.Gaussian(random));
        return result;
    }

    internal Tensor<T> TrimTrailingSilence(Tensor<T> waveform)
    {
        int block = _options.SilenceBlockSize;
        int last = Math.Min(block, waveform.Length);
        for (int start = 0; start < waveform.Length; start += block)
        {
            int count = Math.Min(block, waveform.Length - start);
            double magnitude = 0;
            for (int i = 0; i < count; i++) magnitude += Math.Abs(Ops.ToDouble(waveform[start + i]));
            if (magnitude / count > _options.SilenceMagnitudeThreshold) last = start + count;
        }
        return Engine.TensorSlice(waveform, new[] { 0 }, new[] { MathPolyfill.Clamp(last, 1, waveform.Length) });
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperModel)
        {
            SetTrainingMode(false);
            var output = input;
            foreach (var layer in Layers) output = layer.Forward(output);
            return output;
        }
        var ids = FrontEnd.TtsTokenIds.Read(input, _options.BertVocabularySize,
            Math.Min(_options.MaxTextLength, _options.BertMaxPositions), "E3 BERT");
        return TrimTrailingSilence(Sample(Encode(ids), _options.InferenceWaveformSamples, null, null, _options.DefaultSpeakerId));
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is { Count: > 0 } && Layers.Count == 0) Layers.AddRange(Architecture.Layers);
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = _useNativeMode ? "E3-TTS-Native" : "E3-TTS-ONNX",
        Description = "E3 TTS: BERT-base conditioned raw-waveform 1D U-Net DDPM",
        FeatureCount = _options.HiddenDim,
        AdditionalInfo = new Dictionary<string, object>
        {
            ["ModelType"] = "E3-TTS", ["Mode"] = _useNativeMode ? "Native" : "ONNX",
            ["SampleRate"] = _options.SampleRate, ["WaveformSamples"] = _options.InferenceWaveformSamples,
            ["DiffusionSteps"] = _options.NumDiffusionSteps, ["BlockChannels"] = _options.BlockChannels,
            ["BlockStrides"] = _options.BlockStrides, ["TotalStride"] = _options.BlockStrides.Aggregate(1, (a, b) => a * b),
            ["TextEncoder"] = "bert-base-uncased", ["DirectWaveform"] = true,
        }
    };

    private void RequirePaperModel()
    {
        if (!HasPaperModel) throw new NotSupportedException("This operation needs E3's native BERT and waveform U-Net, not caller-supplied layers.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(E3TTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
