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

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperModel
        ? LayerInputDomain.Indices(_options.BertVocabularySize) : base.GetInputDomain(inputShape);

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => Math.Min(_options.MaxTextLength, _options.BertMaxPositions);
    public new int HiddenDim => _options.HiddenDim;
    /// <inheritdoc />
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
        var prompt = RoundedIds(reference);
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
            known = ZeroPadded(Engine.TensorSlice(Engine.Reshape(prompt, new[] { prompt.Length }), new[] { 0 }, new[] { count }),
                _options.InferenceWaveformSamples);
            knownMask = ZeroPadded(Tensor<T>.CreateDefault(new[] { count }, Ops.One), _options.InferenceWaveformSamples);
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
        int total = _options.InferenceWaveformSamples;
        var known = ZeroPadded(Engine.TensorSlice(Engine.Reshape(waveform, new[] { waveform.Length }), new[] { 0 }, new[] { samples }), total);
        var keep = new int[total];
        for (int i = 0; i < samples; i++) keep[i] = i < startSample || i >= endSample ? 1 : 0;
        var mask = new Tensor<int>(keep, new[] { total }).Cast<T>();
        return TrimTrailingSilence(Sample(Encode(TokenizeText(text)), _options.InferenceWaveformSamples, known, mask,
            speakerId < 0 ? _options.DefaultSpeakerId : speakerId));
    }

    /// <summary>A <c>[n]</c> tensor followed by zeros up to <paramref name="length"/>.</summary>
    private Tensor<T> ZeroPadded(Tensor<T> values, int length) => values.Length == length ? values
        : Engine.TensorConcatenate(new[] { values, new Tensor<T>(new[] { length - values.Length }) }, 0);

    /// <summary>
    /// The per-step reverse-process coefficients of Eq. 7's schedule as T, indexed by step − 1:
    /// √ᾱ_s, 1/√(1 − ᾱ_s), β_s/√(1 − ᾱ_s), 1/√α_s and the posterior std √(β_s (1 − ᾱ_{s−1}) / (1 − ᾱ_s)), with
    /// α_s = ᾱ_s/ᾱ_{s−1}, β_s = 1 − α_s and 1 − ᾱ floored at 1e-12. log ᾱ = log(ᾱ_min) (2 sin²(φ/2))^1.5 (= (1 − cos φ)^1.5),
    /// and 1 − exp(·) is taken as −expm1(·), so the small early-step β and 1 − ᾱ keep their precision in a narrow T.
    /// </summary>
    private (T[] SqrtAlphaBar, T[] InverseNoiseStd, T[] NoiseScale, T[] InverseSqrtAlpha, T[] PosteriorStd) ReverseCoefficients()
    {
        int n = _options.NumDiffusionSteps;
        var half = Engine.TensorSin(Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, n + 1), Ops.FromDouble(Math.PI / (4.0 * n))));
        var logAlphaBar = Engine.TensorMultiplyScalar(
            Engine.TensorPower(Engine.TensorMultiplyScalar(Engine.TensorSquare(half), Ops.FromDouble(2.0)), Ops.FromDouble(1.5)),
            Ops.FromDouble(Math.Log(_options.MinimumAlphaBar)));
        var current = Engine.TensorSlice(logAlphaBar, new[] { 1 }, new[] { n });
        var previous = Engine.TensorSlice(logAlphaBar, new[] { 0 }, new[] { n });
        var floor = Ops.FromDouble(1e-12);
        var noiseVariance = Engine.TensorMax(Engine.TensorNegate(Engine.TensorExpm1(current)), floor);              // 1 − ᾱ_s
        var previousNoiseVariance = Engine.TensorNegate(Engine.TensorExpm1(previous));                              // 1 − ᾱ_{s−1}
        var logAlpha = Engine.TensorSubtract(current, previous);
        var beta = Engine.TensorNegate(Engine.TensorExpm1(logAlpha));
        var inverseNoiseStd = Engine.TensorReciprocal(Engine.TensorSqrt(noiseVariance));
        var half1 = Ops.FromDouble(0.5);
        return (
            Engine.TensorExp(Engine.TensorMultiplyScalar(current, half1)).ToArray(),
            inverseNoiseStd.ToArray(),
            Engine.TensorMultiply(beta, inverseNoiseStd).ToArray(),
            Engine.TensorExp(Engine.TensorMultiplyScalar(logAlpha, Ops.FromDouble(-0.5))).ToArray(),
            Engine.TensorSqrt(Engine.TensorDivide(Engine.TensorMultiply(beta, previousNoiseVariance), noiseVariance)).ToArray());
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
        var (sqrtAlphaBar, inverseNoiseStd, noiseScale, inverseSqrtAlpha, posteriorStd) = ReverseCoefficients();
        for (int step = _options.NumDiffusionSteps; step > 0; step--)
        {
            int s = step - 1;
            var (predicted, _) = WaveformUnet!.Forward(waveform, text, Math.Sqrt(AlphaBar(step)), speakerId);
            if (known is not null && knownMask is not null)
            {
                var actual = Engine.TensorMultiplyScalar(Engine.TensorSubtract(waveform,
                    Engine.TensorMultiplyScalar(known, sqrtAlphaBar[s])), inverseNoiseStd[s]);
                predicted = Blend(predicted, actual, knownMask);
            }
            var mean = Engine.TensorMultiplyScalar(Engine.TensorSubtract(waveform,
                Engine.TensorMultiplyScalar(predicted, noiseScale[s])), inverseSqrtAlpha[s]);
            if (step == 1) waveform = mean;
            else
                waveform = Engine.TensorAdd(mean, Engine.TensorMultiplyScalar(Gaussian(sampleCount, random), posteriorStd[s]));
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
        int length = waveform.Length;
        int last = Math.Min(block, length);
        int blocks = (length + block - 1) / block;
        if (blocks > 0)
        {
            // The mean magnitude of each block (the last one may be short): zero-pad to whole blocks, sum each row,
            // divide by the block's real sample count.
            var magnitudes = Engine.TensorAbs(ZeroPadded(Engine.Reshape(waveform, new[] { length }), blocks * block));
            var sums = Engine.ReduceSum(Engine.Reshape(magnitudes, new[] { blocks, block }), new[] { 1 }, keepDims: false);
            var counts = new int[blocks];
            for (int b = 0; b < blocks; b++) counts[b] = Math.Min(block, length - b * block);
            var loud = Engine.TensorGreaterThan(Engine.TensorDivide(sums, new Tensor<int>(counts, new[] { blocks }).Cast<T>()),
                Ops.FromDouble(_options.SilenceMagnitudeThreshold)).Cast<int>().ToArray();
            for (int b = 0; b < blocks; b++)
                if (loud[b] != 0) last = b * block + counts[b];
        }
        return Engine.TensorSlice(waveform, new[] { 0 }, new[] { MathPolyfill.Clamp(last, 1, length) });

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

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
