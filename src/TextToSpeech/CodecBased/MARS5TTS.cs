using System.Runtime.CompilerServices;
using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// MARS5 (Camb.ai, 2024): two-stage voice cloning over EnCodec's 6 kbps codes. An autoregressive Mistral decoder writes
/// the first codebook as BPE-merged speech tokens after the text, conditioned on an implicit speaker embedding of the
/// reference's codes; a non-autoregressive encoder-decoder Transformer fills the other seven codebooks by multinomial
/// diffusion; Vocos decodes them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The pipeline is the release's <c>Mars5TTS.tts</c>: the text in minbpe tokens between start and end tokens (the
/// reference's transcript first for a deep clone), the reference's EnCodec codes for the speaker embedding (and, deep
/// cloned, its first codebook's speech tokens continued), AR sampling with frequency, presence and early-end penalties,
/// then 200 steps of guided diffusion with the first codebook held (revisable in the last 20 steps), Vocos at the 3 kbps
/// class, and the output's silences trimmed.</para>
/// <para>MARS5 has no paper, and the release ships no training code. Training follows its architecture notes (the AR's
/// cross-entropy with text tokens down-weighted, the NAR's multinomial diffusion with the first codebook given) and its
/// diffusion code (the variational loss), with the recipes its checkpoints' metadata records.</para>
/// <para><b>For Beginners:</b> MARS5 reads text in the voice of a short recording; with the recording's transcript it
/// copies the voice more closely.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new MARS5TTS&lt;float&gt;(architecture, MARS5TTSOptions.FromCheckpoint("mars5_ar.safetensors"));
/// model.LoadReferenceWeights(new Mars5Checkpoints { Autoregressive = "mars5_ar.safetensors", NonAutoregressive = "mars5_nar.safetensors",
///     Codec = "encodec_24khz-d7cc33bc.th", Vocoder = "vocos-encodec-24khz/pytorch_model.bin" });
/// model.Voice = model.CreateVoice(referenceAudio, 24000, "What the reference says.");
/// var audio = model.Synthesize("Text to speak in that voice.");   // 24 kHz
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("MARS5: A novel speech model for insane prosody", "https://github.com/Camb-ai/MARS5-TTS", Year = 2024, Authors = "Camb.ai")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 4e-4, WeightDecay = 0.1, MaxGradientNorm = 1.0, WarmupSteps = 10000, ReferenceBatchSize = 10,
                GradientAccumulationSteps = 4, Component = "ar",
                Source = "CAMB-AI/MARS5-TTS mars5_ar.safetensors metadata 'cfg': lr 4e-4 to end_lr 2e-5 at 700k steps after 10k warmup "
                        + "iterations, weight decay 0.1, grad clip 1.0, text_loss_scale 0.01, batch 10 with 4 accumulation steps.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.002, MaxGradientNorm = 1.0, WarmupSteps = 10000,
                ReferenceBatchSize = 16, GradientAccumulationSteps = 2, Component = "nar",
                Source = "CAMB-AI/MARS5-TTS mars5_nar.safetensors metadata 'cfg': lr 1e-4 to end_lr 5e-6 at 800k steps after 10k warmup "
                        + "iterations, betas (0.8, 0.99), weight decay 0.002, grad clip 1, T 200, batch 16 with 2 accumulation steps.")]
public partial class MARS5TTS<T> : TtsModelBase<T>
{
    private readonly MARS5TTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private Mars5CodecLm<T>? _ar;
    private Mars5ResidualTransformer<T>? _nar;
    private AudioCodecLayer<T>? _codec;
    private VocosEncodecDecoder<T>? _vocos;
    private MinBpeRegexTokenizer? _textTokenizer;
    private MinBpeCodebookTokenizer? _speechTokenizer;

    private readonly List<LayerBase<T>> _arLayers = new();
    private readonly List<LayerBase<T>> _narLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed MARS5 for inference.</summary>
    public MARS5TTS(NeuralNetworkArchitecture<T> architecture, string modelPath, MARS5TTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MARS5TTSOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable MARS5.</summary>
    public MARS5TTS(NeuralNetworkArchitecture<T> architecture, MARS5TTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MARS5TTSOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        if (o.HiddenDim % o.NumHeads != 0) throw new ArgumentException("HiddenDim must divide by NumHeads.");
        if (o.HiddenDim % o.NumCodebooks != 0 || o.NarDim % o.NumCodebooks != 0)
            throw new ArgumentException("The AR and NAR widths must divide by NumCodebooks (their chunked code embeddings).");
        if (o.NarDim % o.NarHeads != 0) throw new ArgumentException("NarDim must divide by NarHeads.");
        if (o.DiffusionSteps < 1) throw new ArgumentException("DiffusionSteps must be positive.");
        if (o.JumpLength < 1 || o.JumpSamples < 1) throw new ArgumentException("JumpLength and JumpSamples must be positive.");
        // The release's rotating cache spans min(window, prompt + max_len): generation never reaches past the window.
        if (o.MaxGenerationLength > o.SlidingWindow) throw new ArgumentException("MaxGenerationLength must not exceed SlidingWindow.");
        // Both checkpoints' configs clip the gradient norm at 1.
        MaxGradNorm = NumOps.FromDouble(o.GradientClipNorm);
    }

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>Synthesis imitates a recording of the speaker (<see cref="CreateVoice"/>).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    private bool HasPaperLayers => _ar is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal Mars5CodecLm<T> Autoregressive => _ar ?? throw NotNative();
    internal Mars5ResidualTransformer<T> NonAutoregressive => _nar ?? throw NotNative();

    /// <summary>The codec (EnCodec 24 kHz at 6 kbps), frozen.</summary>
    public EnCodec<T> Codec => (EnCodec<T>?)_codec?.Codec ?? throw NotNative();

    internal VocosEncodecDecoder<T> Vocos => _vocos ?? throw NotNative();

    /// <summary>The text tokenizer (the checkpoint's, or byte-level without merges).</summary>
    internal MinBpeRegexTokenizer TextTokenizer => _textTokenizer ??= new MinBpeRegexTokenizer(MinBpeModel.Parse(_options.TextTokenizerModel
        ?? "minbpe v1\n\n2\n<|startoftext|> 256\n<|endoftext|> 257\n"));

    /// <summary>The speech tokenizer (the checkpoint's, or the codes without merges).</summary>
    internal MinBpeCodebookTokenizer SpeechTokenizer => _speechTokenizer ??= new MinBpeCodebookTokenizer(MinBpeModel.Parse(_options.SpeechTokenizerModel
        ?? $"minbpe v1\n\n2\n<|endofspeech|> {_options.CodebookSize}\n<|pad|> {_options.CodebookSize + 1}\n"), _options.CodebookSize);

    /// <summary>The AR vocabulary: the text tokens, then the speech tokens.</summary>
    private int ArVocab => TextTokenizer.VocabSize + SpeechTokenizer.VocabSize;

    private int EndOfSpeech => TextTokenizer.VocabSize + SpeechTokenizer.Specials["<|endofspeech|>"];

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(TextTokenizer.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.TrainingSeed);
        _ar = new Mars5CodecLm<T>(Engine, _arLayers, new Mars5ArConfiguration(ArVocab, o.HiddenDim, o.NumHeads, o.ArLayers, o.ArSpeakerLayers,
            o.ArFeedForward, o.SlidingWindow, o.NumCodebooks, o.CodebookSize, o.ArHeadDim), random);
        _nar = new Mars5ResidualTransformer<T>(Engine, _narLayers, new Mars5NarConfiguration(TextTokenizer.VocabSize + 1, o.CodebookSize + 1, o.NarDim,
            o.NarHeads, o.NarEncoderLayers, o.NarDecoderLayers, o.NarSpeakerLayers, o.NumCodebooks, o.TimeDim, o.ConditionDropProbability), random);
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1)
            { RandomSeed = Architecture.RandomSeed }, codecOptions);
        int codecRate = ((IAudioCodec<T>)codec).SampleRate;
        if (codec.HopLength != o.HopSize || codecRate != o.SampleRate)
            throw new ArgumentException($"MARS5's {o.SampleRate} Hz audio and {o.HopSize}-sample frames must be its codec's ({codecRate} Hz, {codec.HopLength}-sample hop).");
        if (codec.CodebookSize != o.CodebookSize || codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps) != o.NumCodebooks)
            throw new ArgumentException($"MARS5 models {o.NumCodebooks} codebooks of {o.CodebookSize} codes; the codec's bandwidth must give them.");
        _codec = new AudioCodecLayer<T>(codec);
        // Vocos's bandwidth classes are 2, 4, 8 and 16 codebooks of the codec's frame rate.
        double perCodebook = (double)o.SampleRate / o.HopSize * Math.Log(o.CodebookSize, 2) / 1000.0;
        _vocos = new VocosEncodecDecoder<T>(Engine, random, new VocosEncodecConfiguration(new[] { 2, 4, 8, 16 }.Select(n => n * perCodebook).ToArray(),
            CodebookSize: o.CodebookSize, LatentDim: o.Codec.Dimension, FrameRate: o.SampleRate / o.HopSize, Dim: o.DecoderDim,
            IntermediateDim: o.DecoderIntermediateDim, Layers: o.DecoderLayers, FftSize: 4 * o.HopSize, HopSize: o.HopSize));
        AddUnsplitLayers(_arLayers.Concat(_narLayers).Cast<ILayer<T>>().ToList());
        ComponentLayers.Add(_codec);
        ComponentLayers.AddRange(_vocos.Layers);
    }

    // ---------------------------------------------------------------- codes

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(IReadOnlyList<double> samples) =>
        new(new[] { samples.Count }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <summary>The codec's codes <c>[frames, codebooks]</c> of audio at <see cref="TtsModelOptions.SampleRate"/>.</summary>
    internal int[,] Encode(Tensor<T> audio)
    {
        using var _ = new NoGradScope<T>();
        var codes = Codec.Encode(Engine.Reshape(audio, new[] { audio.Length }));
        int codebooks = codes.GetLength(0), frames = codes.GetLength(1);
        var transposed = new int[frames, codebooks];
        for (int q = 0; q < codebooks; q++)
            for (int f = 0; f < frames; f++) transposed[f, q] = codes[q, f];
        return transposed;
    }

    /// <summary>Vocos's waveform of codes <c>[frames, codebooks]</c> at <see cref="MARS5TTSOptions.VocosBandwidthId"/>.</summary>
    internal Tensor<T> Vocode(int[,] codes)
    {
        int frames = codes.GetLength(0), codebooks = codes.GetLength(1);
        var transposed = new int[codebooks, frames];
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < codebooks; q++) transposed[q, f] = codes[f, q];
        return Vocos.Decode(transposed, _options.VocosBandwidthId);
    }

    // ---------------------------------------------------------------- the voice

    private sealed class PromptInfo
    {
        public int[,] Codes = new int[0, 0];   // [frames, codebooks]
        public string? Transcript;
    }

    private readonly ConditionalWeakTable<TtsVoice<T>, PromptInfo> _prompts = new();

    /// <summary>
    /// A voice from a recording of the speaker at <paramref name="sampleRate"/> (resampled to 24 kHz, averaged to mono) and,
    /// for a deep clone, its transcript. The release suggests 2 to 12 seconds, about 6 being best.
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, int sampleRate, string? transcript = null)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (!HasPaperLayers) throw NotNative();
        var samples = Doubles(recording);
        var audio = sampleRate == _options.SampleRate ? Wave(samples) : AudioHelper<T>.ResampleBandLimited(Engine, Wave(samples), sampleRate, _options.SampleRate);
        var voice = new TtsVoice<T> { ReferenceAudio = audio };
        _prompts.Add(voice, PromptOf(audio, transcript));
        return voice;
    }

    private PromptInfo PromptOf(Tensor<T> audio, string? transcript)
    {
        // ref_audio_pad: silence before the reference.
        int pad = (int)(_options.SampleRate * _options.ReferencePadSeconds);
        if (pad > 0) audio = Engine.TensorConcatenate(new[] { new Tensor<T>(new[] { pad }), Engine.Reshape(audio, new[] { audio.Length }) }, 0);
        return new PromptInfo { Codes = Encode(audio), Transcript = transcript };
    }

    private PromptInfo Prompt(TtsVoice<T> voice)
    {
        if (_prompts.TryGetValue(voice, out var info)) return info;
        var audio = voice.ReferenceAudio ?? throw new InvalidOperationException($"{GetType().Name} imitates a recording; set ReferenceAudio on the voice.");
        info = PromptOf(audio, null);
        _prompts.Add(voice, info);
        return info;
    }

    // ---------------------------------------------------------------- synthesis

    /// <summary>The text's tokens between the start and end tokens.</summary>
    internal List<int> TextTokens(string text) => TextTokenizer.Encode("<|startoftext|>" + text + "<|endoftext|>");

    /// <inheritdoc />
    /// <remarks>The text's minbpe tokens between the start and end tokens.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return Wave(TextTokens(text.Trim()).Select(t => (double)t).ToArray());
    }

    /// <inheritdoc />
    /// <remarks>The release's <c>tts</c>: a deep clone when the voice has a transcript and <see cref="MARS5TTSOptions.DeepClone"/>
    /// is set, a shallow one otherwise.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(base.PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var prompt = Prompt(RequireVoice());
        bool deep = _options.DeepClone && prompt.Transcript is not null;
        var tokens = deep ? TextTokens(prompt.Transcript + " " + text.Trim()) : TextTokens(text.Trim());
        return Wave(Tts(tokens, text.Length, prompt, deep));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's tokens (<see cref="TextToTokens"/>), synthesized as a shallow clone
    /// (the tokens carry no transcript).</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        int vocabulary = TextTokenizer.VocabSize;
        var tokens = new List<int>();
        for (int i = 0; i < input.Length; i++)
        {
            int id = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (id < 0 || id >= vocabulary) throw new ArgumentOutOfRangeException(nameof(input), $"Text tokens lie in [0, {vocabulary}).");
            tokens.Add(id);
        }
        if (tokens.Count == 0) throw new ArgumentException("There are no text tokens.", nameof(input));
        // n_phones_gen is round(len(text)): the token count stands in for the characters it was encoded from.
        return Wave(Tts(tokens, tokens.Count, Prompt(RequireVoice()), deep: false));
    }

    /// <summary>The release's <c>tts</c> after tokenization.</summary>
    private double[] Tts(List<int> textTokens, int textLength, PromptInfo prompt, bool deep)
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var o = _options;
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
            int frames = prompt.Codes.GetLength(0), codebooks = o.NumCodebooks;
            var q0 = Enumerable.Range(0, frames).Select(f => prompt.Codes[f, 0]).ToList();
            int textVocab = TextTokenizer.VocabSize;
            var speech = deep ? SpeechTokenizer.Encode(q0).Select(t => t + textVocab).ToList() : new List<int>();
            var sequence = textTokens.Concat(speech).ToList();
            int firstCodec = sequence.Count - speech.Count + 1;
            var arTokens = Generate(sequence, prompt.Codes, firstCodec, (int)Math.Round(o.EosEstimatedLengthFactor * textLength), random);
            // Parse the AR output: the tokens from first_codec_idx on, less the text vocabulary, clamped at 0, decoded.
            var generated = arTokens.Skip(firstCodec).Select(t => Math.Max(0, t - textVocab)).ToList();
            var codes = SpeechTokenizer.DecodeCodes(generated).Where(c => c >= 0 && c < o.CodebookSize).ToArray();
            if (codes.Length == 0) return Array.Empty<double>();
            var dsh = new Mars5DiffusionSampling(o.JumpLength, o.JumpSamples, LastGreedy: false, o.X0Temperature, o.NarGuidance, ScaledInference: true,
                DeepClone: deep, o.Q0OverrideSteps);
            var diffusion = new Mars5Diffusion(o.CodebookSize + 1, o.DiffusionSteps);
            var nar = NonAutoregressive;
            double[] Predict(int[,] x, int t, bool drop)
            {
                var logits = nar.Forward(textTokens, prompt.Codes, x, t, drop);
                var flat = new double[logits.Length];
                for (int i = 0; i < flat.Length; i++) flat[i] = Ops.ToDouble(logits.GetFlat(i));
                return flat;
            }
            double[] Uniform(int n)
            {
                var u = new double[n];
                for (int i = 0; i < n; i++) u[i] = random.NextDouble();
                return u;
            }
            var full = diffusion.Infer(codes, codebooks, deep ? prompt.Codes : null, Predict, dsh, Uniform, () => random.Next(o.CodebookSize + 1));
            var audio = Doubles(Vocode(ClampPad(full)));
            return Mars5Trim.Trim(audio, o.TrimDb).Audio;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // The diffusion's classes include the pad class, which Vocos cannot decode; the release vocodes whatever came out, and
    // its embedding reads past the codebook. Pad codes are read as the last code.
    private int[,] ClampPad(int[,] codes)
    {
        int frames = codes.GetLength(0), codebooks = codes.GetLength(1);
        var clamped = new int[frames, codebooks];
        for (int f = 0; f < frames; f++)
            for (int q = 0; q < codebooks; q++) clamped[f, q] = Math.Min(codes[f, q], _options.CodebookSize - 1);
        return clamped;
    }

    /// <summary><c>ar_generate</c> with beam width 1: the sequence's continuation until the end-of-speech token or the
    /// length limit.</summary>
    private List<int> Generate(List<int> sequence, int[,] reference, int firstCodec, int estimatedLength, Random random)
    {
        var o = _options;
        var ar = Autoregressive;
        var speaker = ar.Speaker(reference);
        int textVocab = TextTokenizer.VocabSize, vocab = ArVocab, eos = EndOfSpeech;
        int validStart = textVocab, validEnd = textVocab + SpeechTokenizer.VocabSize + 1;
        var x = new List<int>(sequence);
        var previous = new List<int>();
        var cache = new LlamaCache<T>();
        while (x.Count < o.MaxGenerationLength)
        {
            var logitsTensor = ar.NextLogits(x, speaker, cache);
            var logits = new double[vocab];
            for (int i = 0; i < vocab; i++) logits[i] = Ops.ToDouble(logitsTensor.GetFlat(i));
            if (previous.Count > 1) Mars5Sampler.FrequencyPenalty(logits, previous, o.FrequencyPenalty, o.PresencePenalty, o.PenaltyWindow);
            Mask(logits, validStart, validEnd);
            Mars5Sampler.EosPenalty(logits, previous.Count, estimatedLength, o.EosPenaltyDecay, o.EosPenaltyFactor, eos);
            for (int i = 0; i < vocab; i++) logits[i] /= o.Temperature;
            Mars5Sampler.TopKTopP(logits, o.TopK, o.TopP);
            Mars5Sampler.TypicalP(logits, o.TypicalP);
            Mask(logits, validStart, validEnd);
            // multinomial over softmax(logits)
            double max = logits.Max(), sum = 0;
            var p = logits.Select(v => Math.Exp(v - max)).ToArray();
            foreach (var v in p) sum += v;
            double u = random.NextDouble() * sum, acc = 0;
            int next = vocab - 1;
            for (int i = 0; i < vocab; i++)
            {
                acc += p[i];
                if (u < acc) { next = i; break; }
            }
            if (next == eos) break;
            previous.Add(next);
            x.Add(next);
        }
        return x;
    }

    // filtered_logits[..., :valid_start-1] = -inf; filtered_logits[..., valid_end:] = -inf
    private static void Mask(double[] logits, int validStart, int validEnd)
    {
        for (int i = 0; i < Math.Min(validStart - 1, logits.Length); i++) logits[i] = double.NegativeInfinity;
        for (int i = validEnd; i < logits.Length; i++) logits[i] = double.NegativeInfinity;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MARS5TTS-Native" : "MARS5TTS-ONNX",
            Description = "MARS5: AR Mistral over BPE codec tokens plus multinomial-diffusion NAR over EnCodec codes (Camb.ai, 2024)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MARS5";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Stage"] = _options.CurrentStage.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MARS5TTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) (_codec?.Codec as IDisposable)?.Dispose();
        base.Dispose(disposing);
    }
}
