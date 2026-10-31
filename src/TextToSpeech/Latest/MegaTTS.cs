using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// Mega-TTS (Jiang et al. 2023): zero-shot speech from a disentangled mel-spectrogram model. A content encoder reads the
/// phonemes, a prosody encoder turns the low band of a mel spectrogram into one VQ code per phoneme, a timbre encoder
/// averages a reference recording into one vector, and a GAN-trained decoder makes the mel spectrogram from the three; a
/// prosody language model (P-LLM) continues the prompt's prosody codes for the new text, and HiFi-GAN V1 makes the audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The paper's model with its Table 7 sizes. Mega-TTS has no released code; what the paper leaves unstated comes from
/// the authors' NATSpeech code base, which it builds on: the text front end, the mel features, the encoder layers, the
/// duration predictor and its loss, and SyntaSpeech's multi-window discriminator and its training.</para>
/// <para><b>For Beginners:</b> give Mega-TTS a few seconds of someone speaking (<see cref="TtsModelBase{T}.Voice"/>'s
/// <see cref="TtsVoice{T}.ReferenceAudio"/>) with that recording's phonemes and their durations
/// (<see cref="TtsVoice{T}.ReferenceTokens"/>, <see cref="TtsVoice{T}.ReferenceDurations"/>), and it says new text in
/// that voice and with that person's way of speaking. Training goes in two stages: the acoustic model on recordings with
/// phoneme durations, then the prosody language model (<see cref="MegaTTSOptions.Stage"/>).</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new MegaTTS&lt;float&gt;(architecture, new MegaTTSOptions());
/// model.LoadVocoderWeights("generator_v1");   // a HiFi-GAN V1 generator trained on this model's mel spectrograms
/// model.Voice = new TtsVoice&lt;float&gt;
/// {
///     ReferenceAudio = prompt22kHz,
///     ReferenceTokens = model.TextToTokens(promptTranscript),
///     ReferenceDurations = promptDurations,   // frames per token, from a forced aligner
/// };
/// var audio = model.Synthesize("Hello, how are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Mega-TTS: Zero-Shot Text-to-Speech at Scale with Intrinsic Inductive Bias",
    "https://arxiv.org/abs/2306.03509",
    Year = 2023,
    Authors = "Jiang et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9, MaxGradientNorm = 1.0,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, Component = "acoustic",
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2023, Sec. 4.1: Adam with beta1 0.9, beta2 0.98 and epsilon 1e-9 on the learning rate schedule "
             + "of Vaswani et al. 2017 (its reference [56]); the warmup of 4000 steps is that paper's, and the gradient "
             + "norm clip of 1 is the authors' NATSpeech code base (clip_grad_norm). 320K steps at 30 sentences per GPU on 8 GPUs.")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9, MaxGradientNorm = 1.0,
    Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000, Component = "prosody",
    Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2023, Sec. 4.1: the P-LLM trains with the same optimizer and schedule for 100K steps.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-6, WeightDecay = 0.0,
    MaxGradientNorm = 1.0, Component = "discriminator", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Jiang et al. 2023 follows SyntaSpeech's discriminator (its reference [63]); SyntaSpeech's synta.yaml trains it "
             + "with AdamW at 1e-4 (betas 0.9, 0.98, eps 1e-6, weight decay 0), halved every 40000 steps.")]
public partial class MegaTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MegaTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly string[] _vocabulary;
    private readonly MegaTtsMel _mel;
    private bool _disposed;

    private MegaTtsNetwork<T>? _network;
    private HiFiGanGenerator<T>? _vocoder;

    private readonly List<LayerBase<T>> _acousticLayers = new();
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed Mega-TTS for inference.</summary>
    public MegaTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, MegaTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTSOptions();
        _useNativeMode = false;
        (_vocabulary, _mel) = ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Mega-TTS.</summary>
    public MegaTTS(NeuralNetworkArchitecture<T> architecture, MegaTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MegaTTSOptions();
        _useNativeMode = true;
        (_vocabulary, _mel) = ConfigureBase();
        InitializeLayers();
    }

    private (string[] Vocabulary, MegaTtsMel Mel) ConfigureBase()
    {
        var o = _options;
        if (o.Language == MegaTtsLanguage.Chinese && o.PhoneSet is null)
            throw new ArgumentException("The Chinese front end's phones come with the data (NATSpeech's ph_set.json): set PhoneSet.");
        if (o.VocoderRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException("The vocoder's upsampling rates must multiply to HopSize.");
        if (o.VocoderInitialChannels >> o.VocoderRates.Length < 1)
            throw new ArgumentException("Every vocoder stage halves the width: VocoderInitialChannels must keep at least one channel.");
        if (o.ProsodyBins > o.MelChannels) throw new ArgumentException("The prosody encoder reads at most MelChannels bins.");
        if (o.HiddenDim % o.ContentHeads != 0 || o.LmHidden % o.LmHeads != 0) throw new ArgumentException("A width must divide by its heads.");
        if (o.TopK < 1 || o.TopK > o.Codes) throw new ArgumentException("TopK lies in [1, Codes].");
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        return (o.Vocabulary, new MegaTtsMel(o.SampleRate, o.FftSize, o.HopSize, o.WindowSize, o.MelChannels, o.MelMinFrequency, o.MelMaxFrequency));
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.HiddenDim;

    /// <inheritdoc />
    /// <remarks>Mega-TTS decodes in one pass; it has no flow or diffusion steps.</remarks>
    public int NumFlowSteps => 0;

    /// <inheritdoc />
    /// <remarks>Mega-TTS is zero-shot: it reads the voice's recording (<see cref="TtsVoice{T}.ReferenceAudio"/>, or the
    /// mel spectrogram <see cref="TtsVoice{T}.Reference"/>) for the timbre, and with the recording's phonemes and their
    /// durations its prosody codes as the P-LLM's prompt.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;

    /// <inheritdoc />
    /// <remarks>The prosody encoder pools the mel spectrogram over each phoneme's frames, from a forced aligner (the paper
    /// uses the Montreal Forced Aligner). <see cref="TtsTrainingSample{T}.SpeakerReference"/> is the timbre encoder's
    /// reference, another recording of the same speaker (a waveform, or a mel spectrogram); without it the target
    /// itself is the reference.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Durations;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _network is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal MegaTtsNetwork<T> Network => _network ?? throw NotNative();
    internal HiFiGanGenerator<T> Vocoder => _vocoder ?? throw NotNative();
    internal MegaTtsMel MelSpectrogram => _mel;
    internal IReadOnlyList<string> Vocabulary => _vocabulary;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_vocabulary.Length) : base.GetInputDomain(inputShape);

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
        _network = new MegaTtsNetwork<T>(Engine, _acousticLayers, _lmLayers, _discriminatorLayers, new MegaTtsConfiguration
        {
            PhoneVocabulary = _vocabulary.Length, Hidden = o.HiddenDim, ContentLayers = o.ContentLayers, ContentHeads = o.ContentHeads,
            ProsodyBlocks = o.ConvBlocks, ProsodyBins = o.ProsodyBins, Codes = o.Codes, CodeDimension = o.CodeDimension,
            TimbreBlocks = o.ConvBlocks, TimbreKernel = o.TimbreKernel, DecoderBlocks = o.ConvBlocks, Mels = o.MelChannels,
            DurationLayers = o.DurationLayers, DurationKernel = o.DurationKernel, LmLayers = o.LmLayers, LmHidden = o.LmHidden,
            LmHeads = o.LmHeads, DiscriminatorWindows = o.DiscriminatorWindows, DiscriminatorHidden = o.DiscriminatorHidden,
        });
        _vocoder = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.VocoderInitialChannels, o.VocoderRates, o.VocoderKernels,
            o.VocoderResBlockKernels, o.VocoderResBlockDilations, resblockType1: true);
        _vocoderLayers.AddRange(_vocoder.Layers);
        AddUnsplitLayers(_acousticLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_lmLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
        ComponentLayers.AddRange(_vocoderLayers);
    }

    // ---------------------------------------------------------------- text

    /// <inheritdoc />
    /// <remarks>The text's phone ids: NATSpeech's English processor (or the Chinese one), encoded over the phone set.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        if (text is null) throw new ArgumentNullException(nameof(text));
        var phones = _options.Language == MegaTtsLanguage.English ? NatSpeechFrontEnd.EnglishPhonesOf(text) : NatSpeechFrontEnd.ChinesePhonesOf(text);
        var ids = NatSpeechFrontEnd.Encode(phones, _vocabulary).Take(_options.MaxTextLength).ToArray();
        if (ids.Length == 0) throw new ArgumentException("The text has no phonemes.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    internal int[] Ids(Tensor<T> tokens, string name)
    {
        var ids = tokens.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))).ToArray();
        if (ids.Length == 0) throw new ArgumentException("There are no phonemes.", name);
        if (ids.Any(i => i < 0 || i >= _vocabulary.Length))
            throw new ArgumentOutOfRangeException(name, $"Phone ids lie in [0, {_vocabulary.Length}).");
        return ids;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        return base.Synthesize(text);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's phone ids (<see cref="PreprocessText"/>); the acoustic model's mel
    /// spectrogram <c>[frames, mels]</c> of them in the voice, which <see cref="Synthesize"/> turns into audio.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var ids = Ids(input, nameof(input));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return SynthesizeMel(ids, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    /// <remarks>HiFi-GAN V1 on the mel spectrogram.</remarks>
    protected override Tensor<T> PostprocessAudio(Tensor<T> output)
    {
        if (!HasPaperLayers || IsOnnxMode) return output;
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Wave(output);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The paper's zero-shot decoding (Eq. 3–4): the prompt's timbre vector, and with its phonemes and durations its
    /// prosody codes and content; the P-LLM's codes for the text, sampled top-k after the prompt's; the durations the
    /// predictor gives the text's content and prosody, rounded; the decoder's mel spectrogram <c>[frames, mels]</c>.
    /// A voice without the prompt's phonemes gives the P-LLM no prompt codes.
    /// </summary>
    internal Tensor<T> SynthesizeMel(int[] phones, Random random)
    {
        var voice = RequireVoice();
        var promptMel = voice.ReferenceAudio is not null ? MelOf(Doubles(voice.ReferenceAudio)) : voice.Reference!;
        if (promptMel.Shape.Length != 2 || promptMel.Shape[1] != _options.MelChannels)
            throw new ArgumentException($"The voice's mel spectrogram is [frames, {_options.MelChannels}].");
        var timbre = Network.TimbreVector(promptMel);
        int[] promptCodes = Array.Empty<int>();
        Tensor<T>? promptContent = null;
        if (voice.ReferenceTokens is not null)
        {
            var promptPhones = Ids(voice.ReferenceTokens, nameof(voice.ReferenceTokens));
            var durations = voice.ReferenceDurations
                ?? throw new InvalidOperationException("The prompt's phonemes need their durations: set the voice's ReferenceDurations.");
            if (durations.Length != promptPhones.Length) throw new ArgumentException("The prompt has one duration per phoneme.");
            var mel2ph = FrameMap(durations, promptMel.Shape[0]);
            promptContent = Content(promptPhones);
            promptCodes = Row(Network.Quantizer.Encode(Network.ProsodyHidden(promptMel, mel2ph, promptPhones.Length)));
        }
        var content = Content(phones);
        var codes = SampleProsody(promptCodes, promptContent, content, timbre, random);
        var prosody = Network.ProsodyStates(Network.Quantizer.Lookup(Matrix(codes)));
        var predicted = Network.PredictDurations(content, prosody, phones).ToArray();
        // NATSpeech's LengthRegulator: torch.round (half to even), padding tokens kept at zero.
        var frames = predicted.Select((d, i) => phones[i] == 0 ? 0 : (int)Math.Round(NumOps.ToDouble(d), MidpointRounding.ToEven)).ToArray();
        var map = MegaTts3Aligner<T>.LengthRegulate(frames);
        if (map.Length == 0) return new Tensor<T>(new[] { 0, _options.MelChannels });
        return Network.DecodeMel(content, prosody, timbre, map);
    }

    /// <summary>The P-LLM's codes for the text, top-k sampled one at a time after the start token and the prompt's codes,
    /// each position reading its phoneme's content.</summary>
    internal int[] SampleProsody(int[] promptCodes, Tensor<T>? promptContent, Tensor<T> content, Tensor<T> timbre, Random random)
    {
        int l = content.Shape[1], h = content.Shape[2];
        var all = promptContent is null ? content : Engine.TensorConcatenate(new[] { promptContent, content }, 1);
        var previous = new List<int> { Network.StartCode };
        previous.AddRange(promptCodes);
        var result = new int[l];
        for (int i = 0; i < l; i++)
        {
            int n = previous.Count;
            var context = Engine.TensorSlice(all, new[] { 0, 0, 0 }, new[] { 1, n, h });
            var logits = Network.ProsodyLogits(previous.ToArray(), context, timbre);
            var last = new double[_options.Codes];
            for (int c = 0; c < last.Length; c++) last[c] = NumOps.ToDouble(logits[n - 1, c]);
            result[i] = TopK(last, _options.TopK, random);
            previous.Add(result[i]);
        }
        return result;
    }

    // Top-k random sampling (Fan et al. 2018): the k largest logits, softmax, one draw.
    private static int TopK(double[] logits, int k, Random random)
    {
        var top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ThenBy(i => i).Take(k).ToArray();
        double max = logits[top[0]];
        var weights = top.Select(i => Math.Exp(logits[i] - max)).ToArray();
        double u = random.NextDouble() * weights.Sum();
        for (int j = 0; j < top.Length; j++)
        {
            u -= weights[j];
            if (u < 0) return top[j];
        }
        return top[top.Length - 1];
    }

    internal Tensor<T> Content(int[] phones) => Network.Content.Forward(Matrix(phones));

    private static int[,] Matrix(int[] row)
    {
        var m = new int[1, row.Length];
        for (int i = 0; i < row.Length; i++) m[0, i] = row[i];
        return m;
    }

    private static int[] Row(int[,] m) => Enumerable.Range(0, m.GetLength(1)).Select(j => m[0, j]).ToArray();

    /// <summary>The 1-based frame-to-phoneme map of durations over <paramref name="frames"/> frames: frames past the
    /// durations read no phoneme.</summary>
    internal static int[] FrameMap(int[] durations, int frames)
    {
        if (durations.Any(d => d < 0)) throw new ArgumentException("Durations are not negative.");
        var map = MegaTts3Aligner<T>.LengthRegulate(durations);
        if (map.Length > frames) throw new ArgumentException($"The durations cover {map.Length} frames; the recording has {frames}.");
        Array.Resize(ref map, frames);
        return map;
    }

    /// <summary>NATSpeech's log10 mel spectrogram <c>[frames, mels]</c> of a recording at <see cref="TtsModelOptions.SampleRate"/>.</summary>
    internal Tensor<T> MelOf(double[] audio)
    {
        var mel = _mel.LogMel(audio);
        int frames = mel.GetLength(0), mels = mel.GetLength(1);
        var values = new T[frames * mels];
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < mels; m++) values[f * mels + m] = NumOps.FromDouble(mel[f, m]);
        return new Tensor<T>(new[] { frames, mels }, new Vector<T>(values));
    }

    /// <summary>HiFi-GAN's waveform <c>[samples]</c> of a mel spectrogram <c>[frames, mels]</c>.</summary>
    internal Tensor<T> Wave(Tensor<T> mel)
    {
        int frames = mel.Shape[0];
        if (frames == 0) return new Tensor<T>(new[] { 0 });
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(mel), new[] { 1, _options.MelChannels, frames }));
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => MathHelper.GetNumericOperations<T>().ToDouble(v)).ToArray();

    // ---------------------------------------------------------------- weights

    /// <summary>Loads a HiFi-GAN V1 generator checkpoint in the reference's format (jik876/hifi-gan's
    /// <c>{"generator": state_dict}</c>, or the bare state dict), trained on this model's mel spectrograms.</summary>
    public void LoadVocoderWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        string prefix = state.Keys.Any(k => k.StartsWith("generator.", StringComparison.Ordinal)) ? "generator." : "";
        Vocoder.LoadTorch(name => state.TryGetValue(name, out var v) ? v : null, prefix);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MegaTTS-Native" : "MegaTTS-ONNX",
            Description = "Mega-TTS: zero-shot speech with disentangled content, prosody and timbre and a prosody language model (Jiang et al., 2023)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MegaTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata.AdditionalInfo["Stage"] = _options.Stage.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MegaTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
