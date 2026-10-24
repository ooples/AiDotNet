using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// EmotiVoice: NetEase Youdao's multi-voice, prompt-controlled emotional text-to-speech (2023).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// EmotiVoice has no paper; this is its released code (netease-youdao/EmotiVoice: <c>models/prompt_tts_modified</c>,
/// <c>models/hifigan</c>, <c>train_am_vocoder_joint.py</c>). An emotion prompt ("开心", "very happy") and the text itself
/// are embedded by a SimBERT style encoder (its pooled output); a PromptTTS-style acoustic model — an ESPnet Transformer
/// encoder over the phonemes, the speaker, prompt and content embeddings concatenated and projected, JETS's alignment
/// module learning the durations, duration, pitch and energy predictors, Gaussian upsampling and a Transformer decoder —
/// produces an 80-dimensional latent per frame, and a HiFi-GAN V1 generator trained jointly with it (on 32-frame
/// segments) turns it into 16 kHz speech.
/// </para>
/// <para><b>For Beginners:</b> Tell EmotiVoice how to say something — "happy", "sad", "angry", "whispering" — and pick one
/// of its two thousand voices; it reads the text in that voice and mood.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "EmotiVoice: a Multi-Voice and Prompt-Controlled TTS Engine",
    "https://github.com/netease-youdao/EmotiVoice",
    Year = 2023,
    Authors = "NetEase Youdao"
)]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1.25e-5, Beta1 = 0.5, Beta2 = 0.9, Epsilon = 1e-9, WeightDecay = 0.0,
    Component = "generator", Provenance = RecipeProvenance.Stated,
    Source = "config/joint/config.yaml (optimizer: lr 1.25e-5, betas [0.5, 0.9], eps 1e-9, weight decay 0); "
             + "train_am_vocoder_joint.py builds torch.optim.Adam with them and decays the rate by 0.999875 per epoch.")]
[PaperOptimizer(OptimizerKind.Adam, LearningRate = 1.25e-5, Beta1 = 0.5, Beta2 = 0.9, Epsilon = 1e-9, WeightDecay = 0.0,
    Component = "discriminator", Provenance = RecipeProvenance.Stated,
    Source = "train_am_vocoder_joint.py: the discriminator's Adam has the generator's settings.")]
public partial class EmotiVoice<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly EmotiVoiceOptions _options;
    private readonly bool _useNativeMode;
    private readonly Random _trainingRandom;
    private bool _disposed;
    private WordPieceTokenizer? _styleTokenizer;

    private TiedEmbeddingLayer<T>? _wordEmbedding;
    private TiedEmbeddingLayer<T>? _speakerEmbedding;
    private EmotiVoiceEncoder<T>? _encoder;
    private EmotiVoiceEncoder<T>? _decoder;
    private DenseLayer<T>? _projection;
    private EmotiVoiceVariancePredictor<T>? _duration;
    private EmotiVoiceVariancePredictor<T>? _pitch;
    private EmotiVoiceVariancePredictor<T>? _energy;
    private NormedConv1DLayer<T>? _pitchEmbed;
    private NormedConv1DLayer<T>? _energyEmbed;
    private EmotiVoiceAlignment<T>? _alignment;
    private DenseLayer<T>? _toMel;
    private HiFiGanGenerator<T>? _vocoder;
    private HiFiGanDiscriminators<T>? _discriminators;
    private BertNetwork<T>? _styleBert;
    private DenseLayer<T>? _pooler;
    private DenseLayer<T>? _pitchHead;
    private DenseLayer<T>? _speedHead;
    private DenseLayer<T>? _energyHead;
    private DenseLayer<T>? _emotionHead;

    private readonly List<LayerBase<T>> _acousticLayers = new();
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private readonly List<LayerBase<T>> _styleLayers = new();

    /// <summary>Creates an ONNX-backed EmotiVoice for inference.</summary>
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, EmotiVoiceOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new EmotiVoiceOptions();
        Options = _options;
        _useNativeMode = false;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable EmotiVoice.</summary>
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, EmotiVoiceOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new EmotiVoiceOptions();
        Options = _options;
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.MelChannels = o.MelChannels;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        if (o.HiddenDim % o.Heads != 0) throw new ArgumentException("The width must divide by the heads.");
        if (o.UpsampleRates.Aggregate(1, (a, b) => a * b) != o.HopSize)
            throw new ArgumentException("HiFi-GAN's upsampling must equal the hop: it turns one latent frame into one hop of samples.");
        if (o.WindowLength > o.FftSize) throw new ArgumentException("The analysis window must fit the FFT.");
        // The released training clips no gradient.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public new int HiddenDim => _options.HiddenDim;

    /// <inheritdoc />
    /// <remarks>EmotiVoice has no flow; 0.</remarks>
    public int NumFlowSteps => 0;

    /// <inheritdoc />
    /// <remarks>The voice: a speaker id and the emotion prompt (<see cref="TtsVoice{T}.Description"/>, SimBERT token ids;
    /// <see cref="CreateVoice"/> builds one from text).</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.SpeakerId | TtsSupervision.Description;

    /// <inheritdoc />
    /// <remarks>EmotiVoice trains on the recording (the vocoder's segment and the alignment's mel spectrogram), the speaker
    /// and the emotion prompt.</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Recording | TtsSupervision.SpeakerId | TtsSupervision.Description;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    private bool HasPaperLayers => _encoder is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal EmotiVoiceEncoder<T> EncoderNetwork => _encoder ?? throw NotNative();
    internal EmotiVoiceEncoder<T> DecoderNetwork => _decoder ?? throw NotNative();
    internal EmotiVoiceAlignment<T> AlignmentNetwork => _alignment ?? throw NotNative();
    internal HiFiGanGenerator<T> Vocoder => _vocoder ?? throw NotNative();
    internal HiFiGanDiscriminators<T> Discriminators => _discriminators ?? throw NotNative();
    internal EmotiVoiceVariancePredictor<T> DurationPredictor => _duration ?? throw NotNative();
    internal EmotiVoiceVariancePredictor<T> PitchPredictor => _pitch ?? throw NotNative();
    internal EmotiVoiceVariancePredictor<T> EnergyPredictor => _energy ?? throw NotNative();

    /// <summary>SimBERT's tokenizer (its 13,685-token Chinese WordPiece vocabulary).</summary>
    public WordPieceTokenizer StyleTokenizer => _styleTokenizer ??= WordPieceTokenizer.SimBertChinese();

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

    /// <inheritdoc />
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
        int d = o.HiddenDim;
        TLayer Own<TLayer>(List<LayerBase<T>> list, TLayer layer) where TLayer : LayerBase<T> => StyleTtsOps.Own(list, layer);
        _wordEmbedding = Own(_acousticLayers, new TiedEmbeddingLayer<T>(o.VocabSize, d));
        _speakerEmbedding = Own(_acousticLayers, new TiedEmbeddingLayer<T>(o.SpeakerCount, d));
        _encoder = new EmotiVoiceEncoder<T>(Engine, _acousticLayers, d, o.Heads, 4 * d, o.EncoderLayers, o.FeedForwardKernel, o.DropoutRate);
        _projection = Own(_acousticLayers, StyleTtsOps.Linear<T>(d));
        _duration = new EmotiVoiceVariancePredictor<T>(Engine, _acousticLayers, d, o.DurationLayers, o.VarianceChannels, o.DurationKernel, o.DurationDropout);
        _pitch = new EmotiVoiceVariancePredictor<T>(Engine, _acousticLayers, d, o.PitchLayers, o.VarianceChannels, o.VarianceKernel, o.VarianceDropout);
        _energy = new EmotiVoiceVariancePredictor<T>(Engine, _acousticLayers, d, o.EnergyLayers, o.VarianceChannels, 3, o.VarianceDropout);
        _pitchEmbed = Own(_acousticLayers, new NormedConv1DLayer<T>(1, d, o.VarianceEmbedKernel, 1, 1, 1, (o.VarianceEmbedKernel - 1) / 2, false, ConvolutionNormalization.None));
        _energyEmbed = Own(_acousticLayers, new NormedConv1DLayer<T>(1, d, o.VarianceEmbedKernel, 1, 1, 1, (o.VarianceEmbedKernel - 1) / 2, false, ConvolutionNormalization.None));
        _alignment = new EmotiVoiceAlignment<T>(Engine, _acousticLayers, d, o.MelChannels);
        _decoder = new EmotiVoiceEncoder<T>(Engine, _decoderLayers, d, o.Heads, 4 * d, o.DecoderLayers, o.FeedForwardKernel, o.DropoutRate);
        _toMel = Own(_decoderLayers, StyleTtsOps.Linear<T>(o.MelChannels));
        _vocoder = new HiFiGanGenerator<T>(Engine, o.MelChannels, o.UpsampleInitialChannels, o.UpsampleRates, o.UpsampleKernels, o.ResBlockKernels,
            o.ResBlockDilations, resblockType1: true);
        _vocoderLayers.AddRange(_vocoder.Layers);
        _discriminators = new HiFiGanDiscriminators<T>(Engine, new[] { 2, 3, 5, 7, 11 }, 3, widthDivisor: o.DiscriminatorWidthDivisor);
        _discriminatorLayers.AddRange(_discriminators.Layers);
        _styleBert = new BertNetwork<T>(Engine, _styleLayers, o.StyleBert);
        int h = o.StyleBert.Hidden;
        _pooler = Own(_styleLayers, StyleTtsOps.Linear<T>(h));
        _pitchHead = Own(_styleLayers, StyleTtsOps.Linear<T>(o.PitchLabels));
        _speedHead = Own(_styleLayers, StyleTtsOps.Linear<T>(o.SpeedLabels));
        _energyHead = Own(_styleLayers, StyleTtsOps.Linear<T>(o.EnergyLabels));
        _emotionHead = Own(_styleLayers, StyleTtsOps.Linear<T>(o.EmotionLabels));
        InitializeLikeReference(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed));

        AddEncoderDecoderLayers(_acousticLayers.Cast<ILayer<T>>().ToList(), _decoderLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_vocoderLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
        ComponentLayers.AddRange(_styleLayers);
    }

    /// <summary>
    /// Runs every module once (sizing the lazily built layers) and initializes the acoustic model as the release does
    /// (<c>initialize(self, "xavier_uniform")</c>: Xavier-uniform matrices and kernels, zero biases, embeddings
    /// standard-normal, layer norms at 1 and 0).
    /// </summary>
    private void InitializeLikeReference(Random random)
    {
        var o = _options;
        var all = _acousticLayers.Concat(_decoderLayers).Concat(_vocoderLayers).Concat(_discriminatorLayers).Concat(_styleLayers).ToList();
        foreach (var layer in all) layer.SetTrainingMode(false);
        try
        {
            using (new NoGradScope<T>())
            {
                var rnd = new Random(0);
                var x = Encode(new[] { 1, 2 }, 0, new Tensor<T>(new[] { 1, o.StyleBert.Hidden }), new Tensor<T>(new[] { 1, o.StyleBert.Hidden }), false, rnd);
                _duration!.Forward(x, false, rnd);
                _pitch!.Forward(x, false, rnd);
                _energy!.Forward(x, false, rnd);
                _pitchEmbed!.Forward(new Tensor<T>(new[] { 1, 1, 2 }));
                _energyEmbed!.Forward(new Tensor<T>(new[] { 1, 1, 2 }));
                _alignment!.Forward(x, new Tensor<T>(new[] { 3, o.MelChannels }));
                var z = _toMel!.Forward(_decoder!.Forward(new Tensor<T>(new[] { 3, o.HiddenDim }), false, rnd));
                _vocoder!.Forward(Engine.Reshape(Engine.TensorTranspose(z), new[] { 1, o.MelChannels, 3 }));
                _discriminators!.Forward(new Tensor<T>(new[] { 4 * o.HopSize }));
                StyleEmbedding(new[] { 0, 0 });
                _pitchHead!.Forward(new Tensor<T>(new[] { 1, o.StyleBert.Hidden }));
                _speedHead!.Forward(new Tensor<T>(new[] { 1, o.StyleBert.Hidden }));
                _energyHead!.Forward(new Tensor<T>(new[] { 1, o.StyleBert.Hidden }));
                _emotionHead!.Forward(new Tensor<T>(new[] { 1, o.StyleBert.Hidden }));
            }
            foreach (var layer in _acousticLayers.Concat(_decoderLayers))
            {
                if (layer is TiedEmbeddingLayer<T> embedding && (embedding == _wordEmbedding || embedding == _speakerEmbedding))
                {
                    embedding.Reinitialize(() => StyleTts2StyleDiffusion<T>.Gaussian(random));
                    continue;
                }
                if (layer is DenseLayer<T> dense)
                {
                    int outputs = dense.GetOutputShape()[0], inputs = dense.GetParameters().Length / outputs - 1;
                    double bound = Math.Sqrt(6.0 / (inputs + outputs));
                    TorchParameters.Linear(Engine, dense, inputs, outputs,
                        Enumerable.Range(0, inputs * outputs).Select(_ => (2 * random.NextDouble() - 1) * bound).ToArray(), new double[outputs]);
                }
                else if (layer is NormedConv1DLayer<T> conv)
                {
                    var shape = conv.TorchWeightShape;
                    int fanIn = shape[1] * shape[2], fanOut = shape[0] * shape[2];
                    double bound = Math.Sqrt(6.0 / (fanIn + fanOut));
                    conv.Reinitialize(() => (2 * random.NextDouble() - 1) * bound);
                }
            }
        }
        finally
        {
            foreach (var layer in all) layer.SetTrainingMode(true);
        }
    }

    // ---------------------------------------------------------------- the acoustic model

    private Tensor<T> Ids(IReadOnlyList<int> ids)
    {
        var t = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) t[i] = NumOps.FromDouble(ids[i]);
        return t;
    }

    /// <summary>
    /// The phonemes' conditioned rows <c>[N, d]</c>: the encoder's output with the speaker, prompt and content embeddings
    /// concatenated to every row and projected (<c>embed_projection1</c>).
    /// </summary>
    internal Tensor<T> Encode(int[] phonemes, int speaker, Tensor<T> style, Tensor<T> content, bool training, Random random)
    {
        if (speaker < 0 || speaker >= _options.SpeakerCount)
            throw new ArgumentOutOfRangeException(nameof(speaker), $"Speaker ids lie in [0, {_options.SpeakerCount}).");
        int n = phonemes.Length;
        var x = EncoderNetwork.Forward(_wordEmbedding!.Forward(Ids(phonemes)), training, random);
        var speakerRow = _speakerEmbedding!.Forward(Ids(new[] { speaker }));
        var joined = Engine.TensorConcatenate(new[] { x, Engine.TensorTile(speakerRow, new[] { n, 1 }), Engine.TensorTile(style, new[] { n, 1 }),
            Engine.TensorTile(content, new[] { n, 1 }) }, 1);
        return _projection!.Forward(joined);
    }

    private Tensor<T> Embed(NormedConv1DLayer<T> conv, Tensor<T> values)
    {
        int n = values.Length;
        var y = conv.Forward(Engine.Reshape(values, new[] { 1, 1, n }));
        return Engine.TensorTranspose(Engine.Reshape(y, new[] { _options.HiddenDim, n }));
    }

    /// <summary>The decoder's latent <c>[frames, mels]</c> of conditioned rows plus the pitch and energy embeddings,
    /// upsampled by the durations.</summary>
    internal Tensor<T> DecodeLatent(Tensor<T> x, Tensor<T> pitch, Tensor<T> energy, Tensor<T> durations, int frames, bool training, Random random)
    {
        var h = Engine.TensorAdd(Engine.TensorAdd(x, Embed(_pitchEmbed!, pitch)), Embed(_energyEmbed!, energy));
        var upsampled = EmotiVoiceUpsampling.Upsample(Engine, h, durations, frames);
        return _toMel!.Forward(DecoderNetwork.Forward(upsampled, training, random));
    }

    /// <summary>The inference durations: <c>clamp(round(exp(log d) − 1), 0)</c>, all 1 when they sum to 0.</summary>
    internal int[] FramesOf(Tensor<T> logDurations)
    {
        var frames = new int[logDurations.Length];
        for (int i = 0; i < frames.Length; i++)
            frames[i] = Math.Max(0, (int)Math.Round(Math.Exp(NumOps.ToDouble(logDurations[i])) - 1.0, MidpointRounding.ToEven));
        if (frames.Sum() == 0) for (int i = 0; i < frames.Length; i++) frames[i] = 1;
        return frames;
    }

    /// <summary>The waveform of phonemes in a speaker's voice with the given prompt and content embeddings (the released
    /// inference: predicted durations, pitch and energy; HiFi-GAN on the whole latent).</summary>
    internal Tensor<T> SynthesizeWave(int[] phonemes, int speaker, Tensor<T> style, Tensor<T> content, Random random)
    {
        var x = Encode(phonemes, speaker, style, content, false, random);
        var pitch = PitchPredictor.Forward(x, false, random);
        var energy = EnergyPredictor.Forward(x, false, random);
        var frames = FramesOf(DurationPredictor.Forward(x, false, random));
        var durations = Ids(frames);
        var latent = DecodeLatent(x, pitch, energy, durations, frames.Sum(), false, random);
        var wave = Vocoder.Forward(Engine.Reshape(Engine.TensorTranspose(latent), new[] { 1, _options.MelChannels, latent.Shape[0] }));
        return Engine.Reshape(wave, new[] { wave.Length });
    }

    // ---------------------------------------------------------------- the style encoder

    /// <summary>SimBERT's pooled output <c>[1, 768]</c> (tanh of a linear map of the [CLS] state) of token ids — the
    /// prompt's or the content's embedding (<c>get_style_embedding</c>), computed with the encoder frozen.</summary>
    public Tensor<T> StyleEmbedding(IReadOnlyList<int> simBertIds)
    {
        if (simBertIds is null) throw new ArgumentNullException(nameof(simBertIds));
        using var _ = new NoGradScope<T>();
        var states = new List<LayerBase<T>>(_styleLayers);
        foreach (var layer in states) layer.SetTrainingMode(false);
        return Pooled(simBertIds);
    }

    internal Tensor<T> Pooled(IReadOnlyList<int> ids)
    {
        var grid = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) grid[0, i] = ids[i];
        var states = _styleBert!.Encode(grid);
        int h = _options.StyleBert.Hidden;
        var first = Engine.Reshape(Engine.TensorSlice(states, new[] { 0, 0, 0 }, new[] { 1, 1, h }), new[] { 1, h });
        return Engine.Tanh(_pooler!.Forward(first));
    }

    /// <summary>The SimBERT token ids ([CLS] … [SEP]) of a prompt or content text; an empty text is [CLS] [SEP], as
    /// Hugging Face's tokenizer gives it.</summary>
    public int[] StyleTokens(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (text.Length == 0)
            return new[] { StyleTokenizer.Vocabulary.GetTokenId(StyleTokenizer.SpecialTokens.ClsToken), StyleTokenizer.Vocabulary.GetTokenId(StyleTokenizer.SpecialTokens.SepToken) };
        return StyleTokenizer.Encode(text).TokenIds.ToArray();
    }

    /// <summary>A voice: a speaker of the released list and an emotion prompt such as "开心" or "非常开心".</summary>
    public TtsVoice<T> CreateVoice(int speaker, string prompt) =>
        new() { SpeakerId = speaker, Description = Ids(StyleTokens(prompt)) };

    /// <summary>A voice: a speaker of the released list by name (such as "8051") and an emotion prompt.</summary>
    public TtsVoice<T> CreateVoice(string speaker, string prompt) => CreateVoice(EmotiVoiceFrontEnd.Default.SpeakerId(speaker), prompt);

    /// <inheritdoc />
    /// <remarks>EmotiVoice's own front end (<see cref="EmotiVoiceFrontEnd"/>): Chinese and English text to the released
    /// token ids.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        return Ids(EmotiVoiceFrontEnd.Default.TokenIds(text));
    }

    // ---------------------------------------------------------------- synthesis entry points

    private Tensor<T> VoiceStyle(TtsVoice<T> voice) =>
        StyleEmbedding((voice.Description ?? throw new InvalidOperationException("An EmotiVoice voice needs its emotion prompt.")).ToVector()
            .Select(v => (int)Math.Round(NumOps.ToDouble(v))).ToArray());

    private int[] ToIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(NumOps.ToDouble(tensor[i]));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(name, $"{GetType().Name}'s phoneme ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks>The content embedding is the text's own SimBERT embedding, as the released inference computes it.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode || !HasPaperLayers) return base.Synthesize(text);
        var phonemes = ToIds(PreprocessText(text), nameof(text));
        return Run(phonemes, StyleTokens(text));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the phoneme ids. Without the text, the content embedding is that of an empty
    /// text; <see cref="Synthesize"/> reads the text's own.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        return Run(ToIds(input, nameof(input)), StyleTokens(string.Empty));
    }

    private Tensor<T> Run(int[] phonemes, int[] contentIds)
    {
        var voice = RequireVoice();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
            return SynthesizeWave(phonemes, voice.SpeakerId, VoiceStyle(voice), StyleEmbedding(contentIds), random);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads a released checkpoint flattened to named tensors: the acoustic model under <c>am.</c> and HiFi-GAN under
    /// <c>generator.</c> (the joint generator's state dict, <c>g_*</c>), with or without a <c>generator.</c> or
    /// <c>module.</c> prefix.
    /// </summary>
    public void LoadReferenceWeights(string path)
    {
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            var name = entry.Key.StartsWith("module.", StringComparison.Ordinal) ? entry.Key.Substring(7) : entry.Key;
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        LoadReferenceWeights((name, shape) => state.TryGetValue(name, out var v) ? v : throw new InvalidDataException($"The checkpoint has no tensor '{name}'."),
            name => state.TryGetValue(name, out var v) ? v : null);
    }

    internal void LoadReferenceWeights(TorchReader read, Func<string, double[]?> readOptional)
    {
        var o = _options;
        int d = o.HiddenDim;
        _wordEmbedding!.LoadTable(read("am.src_word_emb.weight", new[] { o.VocabSize, d }));
        _speakerEmbedding!.LoadTable(read("am.spk_tokenizer.weight", new[] { o.SpeakerCount, d }));
        EncoderNetwork.Load(read, "am.encoder");
        DecoderNetwork.Load(read, "am.decoder");
        StyleTtsOps.LoadLinear(Engine, _projection!, read, "am.embed_projection1", 2 * d + 2 * o.StyleBert.Hidden, d);
        DurationPredictor.Load(read, "am.duration_predictor", d);
        PitchPredictor.Load(read, "am.pitch_predictor", d);
        EnergyPredictor.Load(read, "am.energy_predictor", d);
        StyleTtsOps.LoadPlain(_pitchEmbed!, read, "am.pitch_embed.0");
        StyleTtsOps.LoadPlain(_energyEmbed!, read, "am.energy_embed.0");
        AlignmentNetwork.Load(read, "am.alignment_module");
        StyleTtsOps.LoadLinear(Engine, _toMel!, read, "am.to_mel", d, o.MelChannels);
        Vocoder.LoadTorch(readOptional, "generator.");
    }

    /// <summary>Loads the style encoder (SimBERT and the four prompt classifiers; the released <c>style_encoder</c>
    /// checkpoint's <c>model</c> entry: <c>bert.*</c>, <c>pitch_clf</c>, <c>speed_clf</c>, <c>energy_clf</c>,
    /// <c>emotion_clf</c>), or a plain SimBERT checkpoint when it has no classifiers.</summary>
    public void LoadStyleEncoderWeights(string path)
    {
        var all = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in all)
        {
            var name = entry.Key.StartsWith("module.", StringComparison.Ordinal) ? entry.Key.Substring(7) : entry.Key;
            state[name] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        }
        string prefix = state.Keys.Any(k => k.StartsWith("bert.", StringComparison.Ordinal)) ? "bert." : "";
        _styleBert!.LoadTorchWeights((name, shape) => state[name], prefix);
        int h = _options.StyleBert.Hidden;
        TorchParameters.Linear(Engine, _pooler!, h, h, state[prefix + "pooler.dense.weight"], state[prefix + "pooler.dense.bias"]);
        void Head(DenseLayer<T> head, string name, int labels)
        {
            if (state.TryGetValue(name + ".classifier.weight", out var w))
                TorchParameters.Linear(Engine, head, h, labels, w, state[name + ".classifier.bias"]);
        }
        Head(_pitchHead!, "pitch_clf", _options.PitchLabels);
        Head(_speedHead!, "speed_clf", _options.SpeedLabels);
        Head(_energyHead!, "energy_clf", _options.EnergyLabels);
        Head(_emotionHead!, "emotion_clf", _options.EmotionLabels);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "EmotiVoice-Native" : "EmotiVoice-ONNX",
            Description = "EmotiVoice: a multi-voice and prompt-controlled TTS engine (NetEase Youdao, 2023)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "EmotiVoice";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(EmotiVoice<T>));
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
