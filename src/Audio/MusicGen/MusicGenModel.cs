using AiDotNet.Attributes;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Specialized;

namespace AiDotNet.Audio.MusicGen;

/// <summary>
/// MusicGen (Copet et al. 2023): text-to-music generation with one transformer language model over the four codebooks
/// of a 32 kHz EnCodec, interleaved by the "delay" pattern and conditioned on frozen T5 text embeddings through
/// cross-attention, sampled with classifier-free guidance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The text is tokenized with t5-base's SentencePiece model and encoded by a T5 encoder (frozen, as the paper's
/// pretrained conditioner), projected to the model width. The language model (<see cref="MusicGenCore{T}"/>) reads the
/// delay-interleaved codes and is trained with each codebook's cross-entropy over its valid steps. Training applies the
/// paper's condition merging, description dropout, word dropout and condition dropout; the optimizer is AdamW (or
/// D-Adaptation for the 300M model) with clipping and a cosine schedule, and an EMA of the weights is what generation
/// uses once training has updated it. Generation fills the pattern step by step with
/// <c>uncond + 3 · (cond − uncond)</c> logits and top-k sampling, then decodes the codes with EnCodec.
/// </para>
/// <para>The T5 encoder and EnCodec are implemented natively; their published weights load with
/// <see cref="LoadTextEncoderWeights"/> and the codec's own loader, and the language model's with
/// <see cref="LoadLanguageModelWeights"/> (audiocraft state-dict names).</para>
/// <para><b>For Beginners:</b> Describe the music you want and MusicGen writes it: it predicts compressed audio a
/// frame at a time, steered toward the description, and a codec turns those frames into sound.</para>
/// </remarks>
/// <example>
/// <code>
/// var musicGen = new MusicGenModel&lt;float&gt;(architecture, new MusicGenOptions { ModelSize = MusicGenModelSize.Small });
/// musicGen.Train(new MusicGenTrainingSample&lt;float&gt; { Audio = clip, Description = "lofi hip hop with a mellow piano" });
/// var music = musicGen.GenerateMusic("an upbeat 80s synth-pop track", durationSeconds: 10);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Generative)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Simple and Controllable Music Generation", "https://arxiv.org/abs/2306.05284", Year = 2023,
    Authors = "Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez")]
[PaperOptimizer(OptimizerKind.DAdaptAdam, LearningRate = 1.0, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1,
    Variant = nameof(MusicGenModelSize.Small), Provenance = RecipeProvenance.Stated,
    Source = "Section 3.1: AdamW with beta1 = 0.9, beta2 = 0.95, decoupled weight decay 0.1, gradient clipping 1.0 and a "
             + "cosine schedule with 4000 warmup steps; D-Adaptation step sizes for the 300M model (audiocraft: lr 1).")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.1,
    Provenance = RecipeProvenance.Stated,
    Source = "Section 3.1: AdamW with beta1 = 0.9, beta2 = 0.95, decoupled weight decay 0.1, gradient clipping 1.0 and a "
             + "cosine schedule with 4000 warmup steps. The paper gives no learning rate; 1e-4 is audiocraft's.")]
public partial class MusicGenModel<T> : AudioNeuralNetworkBase<T>, IAudioGenerator<T>, IPaperOptimizerVariant,
    ITrainingObjectiveProvider<T>
{
    private readonly MusicGenOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _paperOptimizer;
    private readonly T5TextTokenizer _tokenizer = new();
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _textLayers = new();
    private MusicGenCore<T>? _core;
    private T5Seq2Seq<T>? _textEncoder;
    private DenseLayer<T>? _projection;
    private DenseLayer<T>? _melodyProjection;
    private AudioCodecLayer<T>? _codec;
    private int _hopLength;
    private Random _trainingRandom;
    private Dictionary<Tensor<T>, double[]>? _ema;
    private double _emaCount;
    private int _updates;
    private bool _disposed;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates a trainable MusicGen.</summary>
    public MusicGenModel(NeuralNetworkArchitecture<T> architecture, MusicGenOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new MusicGenOptions();
        Options = _options;
        _suppliedOptimizer = optimizer;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 0);
        base.SampleRate = _options.Codec.SampleRate;
        InitializeLayers();
    }

    // ------------------------------------------------------------------ configuration

    /// <summary>The width D, heads and layers of the configured size (preset unless overridden).</summary>
    public (int ModelDim, int Heads, int Layers) Dimensions
    {
        get
        {
            var (d, h, l) = _options.ModelSize switch
            {
                MusicGenModelSize.Small => (1024, 16, 24),
                MusicGenModelSize.Large => (2048, 32, 48),
                _ => (1536, 24, 48),
            };
            return (_options.ModelDim > 0 ? _options.ModelDim : d, _options.NumHeads > 0 ? _options.NumHeads : h,
                _options.NumLayers > 0 ? _options.NumLayers : l);
        }
    }

    /// <inheritdoc />
    public string? PaperOptimizerVariant => UsesDAdaptation ? nameof(MusicGenModelSize.Small) : null;

    private bool UsesDAdaptation => _options.UseDAdaptation ?? _options.ModelSize == MusicGenModelSize.Small;

    /// <inheritdoc />
    public new int SampleRate => _options.Codec.SampleRate;

    /// <summary>The codec's frames per second (50).</summary>
    public double FrameRate => (double)SampleRate / _hopLength;

    /// <inheritdoc />
    public double MaxDurationSeconds => _options.DurationSeconds;

    /// <inheritdoc />
    public bool SupportsTextToAudio => false;

    /// <inheritdoc />
    public bool SupportsTextToMusic => true;

    /// <inheritdoc />
    public bool SupportsAudioContinuation => true;

    /// <inheritdoc />
    public bool SupportsAudioInpainting => false;

    /// <inheritdoc />
    public new bool IsOnnxMode => false;

    private MusicGenCore<T> Core => _core ?? throw new InvalidOperationException("The language model is not built.");

    /// <summary>The language model, for tests.</summary>
    internal MusicGenCore<T> CoreForTests => Core;

    /// <summary>The description a training step reads, for tests.</summary>
    internal string? AugmentDescriptionForTests(MusicGenTrainingSample<T> sample, Random random) => AugmentDescription(sample, random);

    /// <summary>The words that survive word dropout, for tests.</summary>
    internal string? DropWordsForTests(string text, Random random) => DropWords(text, random);

    /// <summary>The code drawn from logits, for tests.</summary>
    internal int ChooseForTests(double[] logits, Random random) => Choose(logits, random);

    /// <summary>The optimizer training uses, for tests.</summary>
    internal IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerForTests => OptimizerFor();

    /// <summary>Whether training has produced EMA weights, for tests.</summary>
    internal bool HasEmaForTests => _ema is not null;

    /// <summary>The conditioning <c>[L, D]</c> of token ids, for tests.</summary>
    internal Tensor<T> ConditionForTests(Tensor<T>? tokens) => Condition(tokens, false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0));

    /// <summary>The prefix of token ids and a melody, for tests.</summary>
    internal Tensor<T> MelodyConditionForTests(Tensor<T>? tokens, Tensor<T>? melody) =>
        Condition(tokens, melody is null ? null : Chroma(melody), false, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0));
    private T5Seq2Seq<T> TextEncoder => _textEncoder ?? throw new InvalidOperationException("The text encoder is not built.");
    private DenseLayer<T> Projection => _projection ?? throw new InvalidOperationException("The text projection is not built.");
    private AudioCodecLayer<T> Codec => _codec ?? throw new InvalidOperationException("The codec is not built.");

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (_core is not null) return;
        var o = _options;
        var (d, heads, layers) = Dimensions;
        var initialization = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.Seed ?? 0);
        _core = new MusicGenCore<T>(Engine, _lmLayers, new MusicGenConfiguration(o.Codec.NumQuantizers, o.Codec.CodebookSize, d,
            heads, layers, o.FeedForwardScale * d, o.DropoutRate, o.Delays, CrossAttention: !o.MelodyConditioning));
        _core.InitializeLikeReference(initialization);
        _textEncoder = new T5Seq2Seq<T>(Engine, _textLayers, new T5Configuration(o.TextVocabularySize, o.TextModelDim,
            o.TextFeedForwardDim, o.TextKeyValueDim, o.TextHeads, o.TextLayers, DecoderLayers: 0, Dropout: 0.1));
        _textEncoder.InitializeLikeHuggingFace(initialization);
        // audiocraft T5Conditioner.output_proj: nn.Linear(768, D) with PyTorch's default initialization.
        _projection = new DenseLayer<T>(d, (IActivationFunction<T>)new AiDotNet.ActivationFunctions.IdentityActivation<T>());
        double bound = 1 / Math.Sqrt(o.TextModelDim);
        var weight = new double[d * o.TextModelDim];
        var bias = new double[d];
        for (int i = 0; i < weight.Length; i++) weight[i] = (2 * initialization.NextDouble() - 1) * bound;
        for (int i = 0; i < bias.Length; i++) bias[i] = (2 * initialization.NextDouble() - 1) * bound;
        TorchParameters.Linear(Engine, _projection, o.TextModelDim, d, weight, bias);
        if (o.MelodyConditioning)
        {
            // audiocraft ChromaStemConditioner.output_proj: nn.Linear(12, D), PyTorch's default initialization.
            _melodyProjection = new DenseLayer<T>(d, (IActivationFunction<T>)new AiDotNet.ActivationFunctions.IdentityActivation<T>());
            double chromaBound = 1 / Math.Sqrt(MusicGenChroma.Bins);
            var chromaWeight = new double[d * MusicGenChroma.Bins];
            var chromaBias = new double[d];
            for (int i = 0; i < chromaWeight.Length; i++) chromaWeight[i] = (2 * initialization.NextDouble() - 1) * chromaBound;
            for (int i = 0; i < chromaBias.Length; i++) chromaBias[i] = (2 * initialization.NextDouble() - 1) * chromaBound;
            TorchParameters.Linear(Engine, _melodyProjection, MusicGenChroma.Bins, d, chromaWeight, chromaBias);
        }
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(o.Codec);
        codecOptions.IncludeDiscriminators = false;
        var codec = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        int codebooks = codec.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps);
        if (codebooks != o.Codec.NumQuantizers || o.Delays.Length != codebooks)
            throw new ArgumentException($"MusicGen models {o.Delays.Length} delayed codebooks; the codec's bandwidth gives {codebooks}.");
        _hopLength = codec.HopLength;
        _codec = new AudioCodecLayer<T>(codec);
        // Every stage owns its parameters through the layer list, the text encoder first: its token embedding is
        // what the model's input (T5 ids) feeds.
        Layers.AddRange(_textLayers);
        Layers.Add(_projection);
        if (_melodyProjection is not null) Layers.Add(_melodyProjection);
        Layers.AddRange(_lmLayers);
        Layers.Add(_codec);
    }

    /// <inheritdoc />
    /// <remarks>The frozen T5 encoder and the codec are not trained.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var trainable = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(TrainableLayers().Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trainable.Contains).ToList();
    }

    private IEnumerable<LayerBase<T>> TrainableLayers()
    {
        foreach (var layer in _lmLayers) yield return layer;
        if (_projection is not null) yield return _projection;
        if (_melodyProjection is not null) yield return _melodyProjection;
        if (!_options.FreezeTextEncoder) foreach (var layer in _textLayers) yield return layer;
    }

    // ------------------------------------------------------------------ text conditioning

    /// <summary>The T5 token ids of <paramref name="text"/> (t5-base's tokenizer, ending with <c>&lt;/s&gt;</c>).</summary>
    public Tensor<T> TextToTokens(string text)
    {
        var ids = _tokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)));
        int length = Math.Min(ids.Length, _options.MaxTextLength);
        var tensor = new Tensor<T>(new[] { length });
        for (int i = 0; i < length; i++)
        {
            if (ids[i] >= _options.TextVocabularySize)
                throw new ArgumentException($"T5's tokenizer produced id {ids[i]}, beyond the text encoder's {_options.TextVocabularySize} " +
                    "embedding rows; set TextVocabularySize to at least 32100.", nameof(text));
            tensor[i] = NumOps.FromDouble(ids[i]);
        }
        return tensor;
    }

    /// <summary>
    /// The conditioning: for a text model the text's <c>[L, D]</c> (zeros <c>[1, D]</c> without text, as the reference
    /// masks an absent description); for a melody model the prefix <c>[text; chroma]</c>, the melody's one-hot
    /// chromagram projected to D (zeros when it is absent or dropped: the reference masks a nullified melody but keeps
    /// its frames, five for a missing one, the chromagram of a window of silence).
    /// </summary>
    private Tensor<T> Condition(Tensor<T>? tokens, double[,]? chroma, bool training, Random random)
    {
        var text = Condition(tokens, training, random);
        if (!_options.MelodyConditioning) return text;
        int d = Dimensions.ModelDim;
        int frames = chroma?.GetLength(0) ?? 1 + _options.ChromaWindow / _options.ChromaHop;
        Tensor<T> melody;
        if (chroma is null)
        {
            melody = new Tensor<T>(new[] { frames, d });
        }
        else
        {
            var oneHot = new Tensor<T>(new[] { frames, MusicGenChroma.Bins });
            for (int f = 0; f < frames; f++)
                for (int c = 0; c < MusicGenChroma.Bins; c++) oneHot[f, c] = NumOps.FromDouble(chroma[f, c]);
            melody = (_melodyProjection ?? throw new InvalidOperationException("The melody projection is not built.")).Forward(oneHot);
        }
        return Engine.TensorConcatenate(new[] { text, melody }, 0);
    }

    /// <summary>The one-hot chromagram of a waveform (§2.3).</summary>
    private double[,] Chroma(Tensor<T> audio)
    {
        var flat = audio.ToVector().ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
        return MusicGenChroma.Extract(flat, SampleRate, _options.ChromaWindow, _options.ChromaHop);
    }

    private Tensor<T> Condition(Tensor<T>? tokens, bool training, Random random)
    {
        if (tokens is null || tokens.Length == 0) return new Tensor<T>(new[] { 1, Dimensions.ModelDim });
        var ids = tokens.Rank == 1 ? tokens : Engine.Reshape(tokens, new[] { tokens.Length });
        Tensor<T> hidden;
        if (_options.FreezeTextEncoder)
        {
            using (new NoGradScope<T>()) hidden = TextEncoder.Encode(ids, false, random);
        }
        else hidden = TextEncoder.Encode(ids, training, random);
        return Projection.Forward(hidden);
    }

    /// <summary>
    /// The description a training step reads (audiocraft <c>augment_music_info_description</c>): with probability
    /// <see cref="MusicGenOptions.MergeMetadataProbability"/>, the metadata fields each kept with probability
    /// <see cref="MusicGenOptions.KeepMetadataFieldProbability"/>, shuffled and joined as "name: value", are appended
    /// to the description, which is itself dropped with probability <see cref="MusicGenOptions.DropDescriptionProbability"/>.
    /// </summary>
    private string? AugmentDescription(MusicGenTrainingSample<T> sample, Random random)
    {
        string? description = sample.Description;
        string metadata = "";
        if (random.NextDouble() < _options.MergeMetadataProbability)
        {
            var fields = new[] { "key", "bpm", "genre", "moods", "instrument", "keywords" };
            var pairs = new List<string>();
            foreach (var field in fields)
            {
                bool keep = random.NextDouble() < _options.KeepMetadataFieldProbability;
                if (keep && sample.Metadata is not null && sample.Metadata.TryGetValue(field, out var value) && value is not null)
                    pairs.Add($"{field}: {value}");
            }
            for (int i = pairs.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (pairs[i], pairs[j]) = (pairs[j], pairs[i]);
            }
            metadata = string.Join(". ", pairs);
            if (random.NextDouble() < _options.DropDescriptionProbability) description = null;
        }
        string? merged = description is null
            ? (metadata.Length > 1 ? metadata : null)
            : string.Join(". ", description.TrimEnd('.'), metadata);
        return merged is null || merged.Trim().Length == 0 ? null : merged.Trim();
    }

    // The words of `text` each kept with probability 1 − WordDropout (audiocraft T5Conditioner.tokenize); null when
    // none survive, which the reference masks like an absent description.
    private string? DropWords(string text, Random random)
    {
        var kept = text.Split(' ').Where(_ => random.NextDouble() >= _options.WordDropout).ToArray();
        string joined = string.Join(" ", kept);
        return joined.Length == 0 ? null : joined;
    }

    // ------------------------------------------------------------------ codes

    private int[,] CodesOf(Tensor<T>? codes, Tensor<T>? audio)
    {
        int k = _options.Delays.Length, n = _options.Codec.CodebookSize;
        if (codes is not null)
        {
            if (codes.Rank != 2 || codes.Shape[1] != k)
                throw new ArgumentException($"MusicGen codes are [frames, {k}], got [{string.Join(", ", codes.Shape)}].", nameof(codes));
            var byCodebook = new int[k, codes.Shape[0]];
            for (int f = 0; f < codes.Shape[0]; f++)
                for (int q = 0; q < k; q++)
                {
                    int code = (int)Math.Round(NumOps.ToDouble(codes[f, q]));
                    if (code < 0 || code >= n) throw new ArgumentOutOfRangeException(nameof(codes), $"Codes lie in [0, {n}).");
                    byCodebook[q, f] = code;
                }
            return byCodebook;
        }
        if (audio is null) throw new ArgumentException("MusicGen trains on codec codes or the recording.");
        using var _ = new NoGradScope<T>();
        return Codec.Encode(audio.Rank == 1 ? audio : Engine.Reshape(audio, new[] { audio.Length }));
    }

    // ------------------------------------------------------------------ training

    /// <summary>Trains one step on a clip and its description; returns the loss.</summary>
    public T Train(MusicGenTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        var codes = CodesOf(sample.Codes, sample.Audio);
        string? description = AugmentDescription(sample, _trainingRandom);
        var words = description is null ? null : DropWords(description, _trainingRandom);
        double[,]? chroma = null;
        if (_options.MelodyConditioning)
            chroma = Chroma(sample.Melody ?? sample.Audio ?? Decode(codes));
        return TrainOnCodes(codes, words is null ? null : TextToTokens(words), chroma);
    }

    /// <inheritdoc />
    /// <remarks>The input is the description's T5 token ids (<see cref="TextToTokens"/>; empty for none) and the
    /// target the clip's codes <c>[frames, codebooks]</c> or its waveform. The step drops the condition with
    /// probability <see cref="MusicGenOptions.ConditionDropout"/>.</remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfDisposed();
        Guard.NotNull(expectedOutput);
        bool codes = expectedOutput.Rank == 2 && expectedOutput.Shape[1] == _options.Delays.Length;
        var c = CodesOf(codes ? expectedOutput : null, codes ? null : expectedOutput);
        TrainOnCodes(c, input, _options.MelodyConditioning ? Chroma(codes ? Decode(c) : expectedOutput) : null);
    }

    private T TrainOnCodes(int[,] codes, Tensor<T>? tokens, double[,]? chroma)
    {
        ThrowIfDisposed();
        // Classifier-free guidance dropout (§3.1): the whole condition is dropped for the step.
        bool dropped = _trainingRandom.NextDouble() < _options.ConditionDropout;
        var condition = dropped ? null : tokens;
        // A melody model also drops the melody alone (attribute dropout); its frames stay, zeroed.
        double[,]? melody = chroma;
        if (melody is not null && (dropped || _trainingRandom.NextDouble() < _options.MelodyDropout))
            melody = new double[melody.GetLength(0), MusicGenChroma.Bins];
        bool zeroMelody = melody is not null && !ReferenceEquals(melody, chroma);
        int seed = _trainingRandom.Next();
        var placeholder = new Tensor<T>(new[] { 1 });
        SetTrainingMode(true);
        try
        {
            TrainWithCustomObjective(placeholder, placeholder, (_, _) =>
            {
                var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
                return Core.Loss(codes, ConditionWithMelody(condition, melody, zeroMelody, true, random), true, random);
            }, OptimizerFor());
        }
        finally
        {
            SetTrainingMode(false);
        }
        _updates++;
        if (_options.EmaEveryUpdates > 0 && _updates % _options.EmaEveryUpdates == 0) UpdateEma();
        return LastLoss ?? NumOps.Zero;
    }

    // The conditioning with a melody that is absent, present, or present but zeroed (dropped: its frames are kept).
    private Tensor<T> ConditionWithMelody(Tensor<T>? tokens, double[,]? chroma, bool zeroMelody, bool training, Random random)
    {
        if (!_options.MelodyConditioning) return Condition(tokens, training, random);
        if (!zeroMelody || chroma is null) return Condition(tokens, chroma, training, random);
        var text = Condition(tokens, training, random);
        return Engine.TensorConcatenate(new[] { text, new Tensor<T>(new[] { chroma.GetLength(0), Dimensions.ModelDim }) }, 0);
    }

    /// <summary>The training objective on codes and token ids, deterministic and without dropout.</summary>
    public T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        bool codes = target.Rank == 2 && target.Shape[1] == _options.Delays.Length;
        var c = CodesOf(codes ? target : null, codes ? null : target);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(0);
            var chroma = _options.MelodyConditioning ? Chroma(codes ? Decode(c) : target) : null;
            return Core.Loss(c, Condition(input, chroma, false, random), false, random)[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target) => EvaluateTrainingObjective(input, target);

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> OptimizerFor()
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        if (_paperOptimizer is not null) return _paperOptimizer;
        var o = _options;
        double rate = o.LearningRate ?? (UsesDAdaptation ? 1.0 : 1e-4);
        int warmup = o.WarmupSteps, total = Math.Max(o.TrainingSteps, warmup + 1);
        // audiocraft CosineLRScheduler: linear warmup, then a half cosine to zero at the last step.
        var schedule = new LambdaLRScheduler(rate, step =>
            step < warmup ? (double)step / Math.Max(1, warmup)
            : step <= total ? 0.5 * (1 + Math.Cos(Math.PI * (step - warmup) / (total - warmup)))
            : 0.0);
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer = UsesDAdaptation
            ? new DAdaptAdamOptimizer<T, Tensor<T>, Tensor<T>>(this, new DAdaptAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = rate, Beta1 = o.Beta1, Beta2 = o.Beta2, WeightDecay = o.WeightDecay,
                EnableGradientClipping = o.GradientClipNorm > 0, MaxGradientNorm = o.GradientClipNorm, LearningRateScheduler = schedule,
            })
            : new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = rate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
                EnableGradientClipping = o.GradientClipNorm > 0, MaxGradientNorm = o.GradientClipNorm, LearningRateScheduler = schedule,
                UseAdaptiveBetas = false,
            });
        return _paperOptimizer = PaperOptimizerFactory.VerifyHandBuilt(this, optimizer);
    }

    // The EMA of the trained weights (audiocraft ModuleDictEMA, unbiased): w = 1/count with count ← decay·count + 1.
    private void UpdateEma()
    {
        var parameters = Training.TapeTrainingStep<T>.CollectParameters(TrainableLayers().Cast<ILayer<T>>().ToList(), -1);
        if (_ema is null)
        {
            _ema = new Dictionary<Tensor<T>, double[]>(TensorReferenceComparer<Tensor<T>>.Instance);
            foreach (var p in parameters) _ema[p] = p.ToVector().ToArray().Select(v => NumOps.ToDouble(v)).ToArray();
            _emaCount = 0;
        }
        _emaCount = _emaCount * _options.EmaDecay + 1;
        double w = 1 / _emaCount;
        foreach (var p in parameters)
        {
            var shadow = _ema[p];
            var span = p.AsSpan();
            for (int i = 0; i < shadow.Length; i++) shadow[i] = shadow[i] * (1 - w) + NumOps.ToDouble(span[i]) * w;
        }
    }

    // Runs `action` with the EMA weights in place, if training has produced them.
    private TResult WithEmaWeights<TResult>(Func<TResult> action)
    {
        if (_ema is null) return action();
        var saved = new List<(Tensor<T> Parameter, T[] Values)>();
        foreach (var (p, shadow) in _ema)
        {
            var span = p.AsWritableSpan();
            saved.Add((p, span.ToArray()));
            for (int i = 0; i < span.Length; i++) span[i] = NumOps.FromDouble(shadow[i]);
            Engine.InvalidatePersistentTensor(p);
        }
        try
        {
            return action();
        }
        finally
        {
            foreach (var (p, values) in saved)
            {
                values.AsSpan().CopyTo(p.AsWritableSpan());
                Engine.InvalidatePersistentTensor(p);
            }
        }
    }

    // ------------------------------------------------------------------ generation

    /// <summary>Generates codes <c>[codebooks, frames]</c> for token ids (empty for unconditional generation),
    /// continuing an optional prompt of codes <c>[codebooks, prompt frames]</c>.</summary>
    public int[,] GenerateCodes(Tensor<T>? tokens, int frames, int seed, double? guidanceScale = null, int[,]? prompt = null)
        => GenerateCodes(tokens, null, frames, seed, guidanceScale, prompt);

    /// <summary>Generates codes conditioned on token ids and, for a melody model, a melody waveform.</summary>
    public int[,] GenerateCodes(Tensor<T>? tokens, Tensor<T>? melody, int frames, int seed, double? guidanceScale = null, int[,]? prompt = null)
    {
        if (melody is not null && !_options.MelodyConditioning)
            throw new InvalidOperationException("This MusicGen is a text model; set MelodyConditioning to condition on a melody.");
        ThrowIfDisposed();
        if (frames <= (prompt?.GetLength(1) ?? 0)) throw new ArgumentOutOfRangeException(nameof(frames), "Generate more frames than the prompt.");
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        return WithEmaWeights(() =>
        {
            bool wasTraining = IsTrainingMode;
            SetTrainingMode(false);
            try
            {
                using var _ = new NoGradScope<T>();
                var chroma = melody is null ? null : Chroma(melody);
                var condition = Condition(tokens, chroma, false, random);
                bool guided = tokens is { Length: > 0 } || chroma is not null;
                // The unconditional branch: the same prefix length, zeroed (the reference batches the two and pads).
                var unconditional = !guided ? null
                    : _options.MelodyConditioning ? new Tensor<T>(condition._shape) : Condition(null, false, random);
                return Core.Generate(condition, unconditional, guidanceScale ?? _options.GuidanceScale,
                    frames, logits => Choose(logits, random), prompt);
            }
            finally
            {
                SetTrainingMode(wasTraining);
            }
        });
    }

    // The next code from a codebook's logits (audiocraft LMModel._sample_next_token): argmax without sampling,
    // otherwise softmax(logits / T) restricted to the top p or top k and drawn.
    private int Choose(double[] logits, Random random)
    {
        int argmax = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[argmax]) argmax = i;
        if (!_options.UseSampling || _options.Temperature <= 0) return argmax;
        double max = logits[argmax];
        var probabilities = logits.Select(l => Math.Exp((l - max) / _options.Temperature)).ToArray();
        double sum = probabilities.Sum();
        for (int i = 0; i < probabilities.Length; i++) probabilities[i] /= sum;
        if (_options.TopP > 0)
        {
            // utils.sample_top_p: keep the smallest prefix of the descending order whose mass reaches p.
            var order = Enumerable.Range(0, probabilities.Length).OrderByDescending(i => probabilities[i]).ToArray();
            double cumulative = 0;
            var kept = new double[probabilities.Length];
            foreach (int i in order)
            {
                if (cumulative > _options.TopP) break;
                kept[i] = probabilities[i];
                cumulative += probabilities[i];
            }
            probabilities = kept;
        }
        else if (_options.TopK > 0 && _options.TopK < probabilities.Length)
        {
            // utils.sample_top_k: every probability at least the k-th largest survives (ties included).
            double threshold = probabilities.OrderByDescending(p => p).ElementAt(_options.TopK - 1);
            for (int i = 0; i < probabilities.Length; i++) if (probabilities[i] < threshold) probabilities[i] = 0;
        }
        double total = probabilities.Sum(), draw = random.NextDouble() * total, running = 0;
        for (int i = 0; i < probabilities.Length; i++)
        {
            running += probabilities[i];
            if (draw < running && probabilities[i] > 0) return i;
        }
        return Array.FindLastIndex(probabilities, p => p > 0);
    }

    private Tensor<T> Decode(int[,] codes)
    {
        using var _ = new NoGradScope<T>();
        var audio = Codec.Decode(codes);
        return audio.Reshape(new[] { audio.Length });
    }

    /// <inheritdoc />
    /// <remarks>Autoregressive generation runs one step per codec frame, so <paramref name="numInferenceSteps"/> does
    /// not apply; MusicGen has no negative prompts (the guidance contrasts the description with no condition).</remarks>
    public Tensor<T> GenerateMusic(string prompt, string? negativePrompt = null, double durationSeconds = 10.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
    {
        if (negativePrompt is not null)
            throw new NotSupportedException("MusicGen's guidance contrasts the description with no condition; it has no negative prompt.");
        int frames = Math.Max(1, (int)Math.Round(durationSeconds * FrameRate));
        return Decode(GenerateCodes(TextToTokens(prompt), frames, seed ?? _options.Seed ?? 0, guidanceScale));
    }

    /// <summary>Generates music following a melody and a description (the melody model, §2.3): the melody's
    /// chromagram, quantized to its dominant pitch class per frame, conditions the generation.</summary>
    public Tensor<T> GenerateWithMelody(string prompt, Tensor<T> melody, double durationSeconds = 10.0, double guidanceScale = 3.0,
        int? seed = null)
    {
        Guard.NotNull(melody);
        int frames = Math.Max(1, (int)Math.Round(durationSeconds * FrameRate));
        return Decode(GenerateCodes(TextToTokens(prompt), melody, frames, seed ?? _options.Seed ?? 0, guidanceScale));
    }

    /// <inheritdoc />
    public Tensor<T> GenerateAudio(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null)
        => GenerateMusic(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed);

    /// <inheritdoc />
    public Task<Tensor<T>> GenerateAudioAsync(string prompt, string? negativePrompt = null, double durationSeconds = 5.0,
        int numInferenceSteps = 100, double guidanceScale = 3.0, int? seed = null, CancellationToken cancellationToken = default)
        => Task.Run(() => GenerateAudio(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed), cancellationToken);

    /// <inheritdoc />
    /// <remarks>The input's codes are the prompt the generation continues (audiocraft <c>generate_continuation</c>);
    /// the result includes the prompt.</remarks>
    public Tensor<T> ContinueAudio(Tensor<T> inputAudio, string? prompt = null, double extensionSeconds = 5.0,
        int numInferenceSteps = 100, int? seed = null)
    {
        var promptCodes = CodesOf(null, inputAudio);
        int frames = promptCodes.GetLength(1) + Math.Max(1, (int)Math.Round(extensionSeconds * FrameRate));
        return Decode(GenerateCodes(prompt is null ? null : TextToTokens(prompt), frames, seed ?? _options.Seed ?? 0, null, promptCodes));
    }

    /// <inheritdoc />
    public Tensor<T> InpaintAudio(Tensor<T> audio, Tensor<T> mask, string? prompt = null, int numInferenceSteps = 100, int? seed = null)
        => throw new NotSupportedException("MusicGen generates left to right; it does not inpaint.");

    /// <inheritdoc />
    public AudioGenerationOptions<T> GetDefaultOptions() => new()
    {
        DurationSeconds = _options.DurationSeconds,
        GuidanceScale = _options.GuidanceScale,
        Seed = _options.Seed,
    };

    /// <inheritdoc />
    /// <remarks>The input is the description's token ids; the output is <see cref="MusicGenOptions.DurationSeconds"/>
    /// of generated audio, seeded by <see cref="ModelOptions.Seed"/>.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        int frames = Math.Max(1, (int)Math.Round(_options.DurationSeconds * FrameRate));
        return Decode(GenerateCodes(input, frames, _options.Seed ?? 0));
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    // ------------------------------------------------------------------ weights

    /// <summary>Loads an audiocraft MusicGen language model: the <c>LMModel</c> state dict and the text projection
    /// <c>condition_provider.conditioners.description.output_proj</c>.</summary>
    public void LoadLanguageModelWeights(Func<string, int[], double[]> read)
    {
        Guard.NotNull(read);
        Core.LoadTorchWeights(read);
        int d = Dimensions.ModelDim, text = _options.TextModelDim;
        TorchParameters.Linear(Engine, Projection, text, d,
            read("condition_provider.conditioners.description.output_proj.weight", new[] { d, text }),
            read("condition_provider.conditioners.description.output_proj.bias", new[] { d }));
        if (_melodyProjection is not null)
            TorchParameters.Linear(Engine, _melodyProjection, MusicGenChroma.Bins, d,
                read("condition_provider.conditioners.self_wav.output_proj.weight", new[] { d, MusicGenChroma.Bins }),
                read("condition_provider.conditioners.self_wav.output_proj.bias", new[] { d }));
        _ema = null;
    }

    /// <summary>Loads a Hugging Face T5 encoder state dict (<c>T5EncoderModel</c>, e.g. t5-base).</summary>
    public void LoadTextEncoderWeights(Func<string, int[], double[]> read) => TextEncoder.LoadHuggingFaceWeights(read);

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => true;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var (d, heads, layers) = Dimensions;
        var m = new ModelMetadata<T>
        {
            Name = "MusicGen",
            Description = "Simple and Controllable Music Generation (Copet et al., 2023)",
            FeatureCount = d,
            Complexity = layers,
        };
        m.AdditionalInfo["Size"] = _options.ModelSize.ToString();
        m.AdditionalInfo["Heads"] = heads.ToString();
        m.AdditionalInfo["SampleRate"] = SampleRate.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MusicGenModel<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
