using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// Orpheus (Canopy Labs, 2025): text-to-speech as plain language modeling — Llama 3.2 3B reads "<c>{voice}: {text}</c>"
/// and continues it with SNAC codes, seven tokens per frame, which SNAC's decoder turns into 24 kHz audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Orpheus has no paper; canopyai/Orpheus-TTS (release 0.1) is the reference: the prompt and the token layout, the
/// sampling and the streaming decoder come from its <c>orpheus_tts</c> package, the full decode and the training sequence
/// from the notebooks it links, and the recipe from <c>finetune/train.py</c>. SNAC (Siuzdak et al. 2024) is ported from
/// hubertsiuzdak/snac.</para>
/// <para>Each frame's seven tokens are SNAC's coarse code, the first medium code, two fine codes, the second medium code
/// and two more fine codes, token = 128266 + 4096 × slot + code. Emotive tags such as <c>&lt;laugh&gt;</c> and
/// <c>&lt;sigh&gt;</c> are written in the text.</para>
/// <para><b>For Beginners:</b> Orpheus treats speech as a language a chatbot-style model can write: after the sentence to
/// speak it writes "sound tokens" one at a time, and a neural codec turns those tokens into audio.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new OrpheusTTS&lt;float&gt;(architecture, new OrpheusTTSOptions { Voice = "tara" });
/// model.LoadReferenceWeights("orpheus-3b-0.1-ft");      // the model.safetensors shards (and tokenizer.json)
/// model.LoadCodecWeights("snac_24khz/pytorch_model.bin");
/// var audio = model.Synthesize("Hey there, I'm Tara. &lt;laugh&gt; How are you doing today?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Orpheus: Real-time Human-like Speech", "https://github.com/canopyai/Orpheus-TTS", Year = 2025, Authors = "Canopy Labs")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    Source = "canopyai/Orpheus-TTS finetune/train.py and config.yaml: Hugging Face's Trainer at learning rate 5e-5, batch size "
             + "1, one epoch, with the Trainer's defaults otherwise: AdamW (betas 0.9, 0.999, eps 1e-8, no weight decay), linear "
             + "decay without warm-up, gradient norm clipped to 1.0, and the loss on every token (labels = input_ids).")]
public partial class OrpheusTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly OrpheusTTSOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private LlamaNetwork<T>? _lm;
    private Snac<T>? _codec;
    private ByteLevelBpeTokenizer? _tokenizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Creates an ONNX-backed Orpheus for inference.</summary>
    public OrpheusTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, OrpheusTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OrpheusTTSOptions();
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable Orpheus.</summary>
    public OrpheusTTS(NeuralNetworkArchitecture<T> architecture, OrpheusTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new OrpheusTTSOptions();
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        base.SampleRate = o.SampleRate;
        base.HopSize = o.CodecEncoderRates.Aggregate(1, (a, b) => a * b);
        base.HiddenDim = o.HiddenDim;
        if (base.HopSize != o.CodecDecoderRates.Aggregate(1, (a, b) => a * b))
            throw new ArgumentException("The codec's decoder must upsample by its encoder's hop.");
        if (o.VocabSize < OrpheusTokens.VocabularySize)
            throw new ArgumentException($"Orpheus's tokens need a vocabulary of at least {OrpheusTokens.VocabularySize}.");
        // Hugging Face's Trainer clips the gradient norm to 1.0 (max_grad_norm), which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>SNAC's three codebooks.</remarks>
    public int NumCodebooks => 3;

    /// <inheritdoc />
    public int CodebookSize => OrpheusTokens.CodebookSize;

    /// <summary>Frames per second (one coarse code and six finer ones each: about 12 at 24 kHz).</summary>
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)(HopSize * 4));

    /// <inheritdoc />
    /// <remarks>Orpheus trains on recordings: the recording (24 kHz) or its frames' codes
    /// (<see cref="TtsTrainingSample{T}.CodecTokens"/>, <c>[frames, 7]</c> as <see cref="EncodeToTokens"/> gives them), and
    /// the transcript as the language model's token ids (<see cref="TtsTrainingSample{T}.Tokens"/>, from
    /// <see cref="TtsModelBase{T}.TextToTokens"/>).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    /// <remarks>A frame's seven codes (<see cref="EncodeToTokens"/>'s columns).</remarks>
    public override int CodecTokenCodebooks => OrpheusTokens.CodesPerFrame;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => OrpheusTokens.CodebookSize;

    private bool HasPaperLayers => _lm is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal LlamaNetwork<T> LanguageModel => _lm ?? throw NotNative();
    internal Snac<T> Codec => _codec ?? throw NotNative();
    internal IReadOnlyList<LayerBase<T>> LanguageModelLayers => _lmLayers;
    internal IReadOnlyList<LayerBase<T>> CodecLayers => _codecLayers;

    /// <summary>The language model's tokenizer: Llama 3's with Orpheus's audio tokens, or one read by
    /// <see cref="LoadTokenizer"/>.</summary>
    internal ByteLevelBpeTokenizer Tokenizer => _tokenizer ??= OrpheusTokens.Tokenizer;

    /// <summary>Reads the language model's tokenizer from a Hugging Face <c>tokenizer.json</c> (a fine-tune's own).</summary>
    public void LoadTokenizer(string path)
    {
        var tokenizer = ByteLevelBpeTokenizer.FromJson(File.ReadAllText(path));
        if (tokenizer.VocabularySize > _options.VocabSize)
            throw new InvalidDataException($"The tokenizer has {tokenizer.VocabularySize} tokens; the model's vocabulary is {_options.VocabSize}.");
        if (tokenizer.TokenId("<custom_token_0>") != OrpheusTokens.FirstCustomToken || tokenizer.TokenId("<|begin_of_text|>") != Llama3Tokenizers.BeginOfText)
            throw new InvalidDataException("The tokenizer does not place Orpheus's tokens where the model expects them.");
        _tokenizer = tokenizer;
    }

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(_options.VocabSize) : base.GetInputDomain(inputShape);

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
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        _lm = new LlamaNetwork<T>(Engine, _lmLayers, new LlamaConfiguration
        {
            VocabSize = o.VocabSize, Hidden = o.HiddenDim, Layers = o.Layers, Heads = o.Heads, KeyValueHeads = o.KeyValueHeads, HeadDim = o.HeadDim,
            Intermediate = o.Intermediate, RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, RopeScalingFactor = o.RopeScalingFactor,
            RopeLowFrequencyFactor = o.RopeLowFrequencyFactor, RopeHighFrequencyFactor = o.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = o.RopeOriginalMaxPositions, TieWordEmbeddings = true,
        });
        _lm.InitializeLikeReference(random);
        _codec = new Snac<T>(Engine, new SnacConfiguration
        {
            SampleRate = o.SampleRate, EncoderDim = o.CodecEncoderDim, EncoderRates = (int[])o.CodecEncoderRates.Clone(), DecoderDim = o.CodecDecoderDim,
            DecoderRates = (int[])o.CodecDecoderRates.Clone(), CodebookSize = OrpheusTokens.CodebookSize, CodebookDim = o.CodecCodebookDim,
            VqStrides = new[] { 4, 2, 1 }, Noise = o.CodecNoise, Depthwise = o.CodecDepthwise,
        }, _codecLayers);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
    }

    // ---------------------------------------------------------------- the codec

    private static double[] Doubles(Tensor<T> audio) => audio.ToArray().Select(v => Ops.ToDouble(v)).ToArray();

    private static Tensor<T> Wave(double[] samples) => new(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));

    /// <inheritdoc />
    /// <remarks>The frames of a 24 kHz recording as Orpheus writes them: <c>[frames, 7]</c>, each row SNAC's coarse code,
    /// first medium code, first and second fine codes, second medium code, and third and fourth fine codes (each in [0,
    /// 4096)).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var frames = OrpheusTokens.Frames(Codec.Encode(Doubles(audio)));
        var tokens = new Tensor<T>(new[] { frames.Count, OrpheusTokens.CodesPerFrame });
        for (int f = 0; f < frames.Count; f++)
            for (int s = 0; s < OrpheusTokens.CodesPerFrame; s++) tokens[f, s] = Ops.FromDouble(frames[f][s]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>The audio of frames <c>[frames, 7]</c> as <see cref="EncodeToTokens"/> gives them, decoded together.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        return Wave(DecodeFrames(FramesOf(tokens), NoiseSource()));
    }

    private List<int[]> FramesOf(Tensor<T> tokens)
    {
        if (tokens.Rank != 2 || tokens.Shape[1] != OrpheusTokens.CodesPerFrame)
            throw new ArgumentException($"Orpheus's frames are [frames, {OrpheusTokens.CodesPerFrame}].", nameof(tokens));
        var frames = new List<int[]>();
        for (int f = 0; f < tokens.Shape[0]; f++)
            frames.Add(Enumerable.Range(0, OrpheusTokens.CodesPerFrame).Select(s => (int)Math.Round(Ops.ToDouble(tokens[f, s]))).ToArray());
        return frames;
    }

    private Func<int, double[]> NoiseSource() =>
        Snac<T>.GaussianNoise(AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + 1));

    private double[] DecodeFrames(IReadOnlyList<int[]> frames, Func<int, double[]> noise) =>
        frames.Count == 0 ? Array.Empty<double>() : Codec.Decode(OrpheusTokens.Codes(frames), noise);

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>The text's token ids in the language model's tokenizer (synthesis builds the prompt from their text).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = Tokenizer.Encode(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    private string TextOf(Tensor<T> ids) =>
        Tokenizer.Decode(ids.ToArray().Select(v => (int)Math.Round(NumOps.ToDouble(v))), skipSpecial: true);

    /// <inheritdoc />
    /// <remarks>The package's generation: the prompt of the text (after the voice's name), sampled until the end of speech
    /// or <see cref="OrpheusTTSOptions.MaxNewTokens"/>, the codes decoded as <see cref="OrpheusTTSOptions.Decoding"/> says.
    /// Empty when the model writes no whole frame.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Wave(AudioOf(Generate(OrpheusTokens.Prompt(Tokenizer, text, _options.Voice))));
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a text's token ids (<see cref="PreprocessText"/>); the waveform
    /// <see cref="Synthesize"/> makes of their text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        if (input.ToArray().Any(v => NumOps.ToDouble(v) < 0 || NumOps.ToDouble(v) >= _options.VocabSize))
            throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        return Wave(AudioOf(Generate(OrpheusTokens.Prompt(Tokenizer, TextOf(input), _options.Voice))));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // vLLM's sampling as the package sets it: the repetition penalty over the prompt and the output, temperature, top-p,
    // and a draw, until the end of speech or the model's end token (128009, which vLLM stops at; neither kept) or the token
    // limit. The generated tokens. (The package's own default stop id, 49158, is an ordinary text token; its examples stop
    // at the end of speech.)
    internal List<int> Generate(IReadOnlyList<int> prompt)
    {
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var lm = LanguageModel;
            var ids = new List<int>(prompt);
            var generated = new List<int>();
            var cache = new LlamaCache<T>();
            var logits = lm.Forward(lm.Embed(Row(prompt)), null, null, cache, lastLogitsOnly: true).Logits!;
            while (generated.Count < o.MaxNewTokens)
            {
                var values = logits.ToArray();
                var scores = new double[values.Length];
                for (int i = 0; i < scores.Length; i++) scores[i] = NumOps.ToDouble(values[i]);
                if (o.ConstrainToAudioTokens) OrpheusTokens.MaskToSlot(scores, generated.Count % OrpheusTokens.CodesPerFrame);
                int next;
                if (o.Temperature > 0)
                    next = OuteTTSSampler.Sample(OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, int.MaxValue, o.Temperature, 0, o.TopP, 0), random);
                else
                {
                    var processed = OuteTTSSampler.Process(scores, ids, o.RepetitionPenalty, int.MaxValue, 1.0, 0, 1.0, 0);
                    next = Array.IndexOf(processed, processed.Max());
                }
                if (next == OrpheusTokens.EndOfSpeech || next == OrpheusTokens.EndOfText) break;
                ids.Add(next);
                generated.Add(next);
                logits = lm.Forward(lm.Embed(Row(new[] { next })), null, null, cache, lastLogitsOnly: true).Logits!;
            }
            return generated;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private static int[,] Row(IReadOnlyList<int> ids)
    {
        var row = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++) row[0, i] = ids[i];
        return row;
    }

    // The generated tokens' audio: the package's reading of the codes, then a full or a streaming decode.
    internal double[] AudioOf(IReadOnlyList<int> generated)
    {
        var codes = OrpheusTokens.ReadCodes(generated);
        var noise = NoiseSource();
        if (_options.Decoding == OrpheusDecoding.Full)
            return DecodeFrames(OrpheusTokens.WholeFrames(codes).Where(OrpheusTokens.IsValid).ToList(), noise);
        int frame = Codec.SamplesPerFrame;
        var audio = new List<double>();
        for (int count = 4 * OrpheusTokens.CodesPerFrame; count <= codes.Count; count += OrpheusTokens.CodesPerFrame)
        {
            var window = OrpheusTokens.WholeFrames(codes.GetRange(count - 4 * OrpheusTokens.CodesPerFrame, 4 * OrpheusTokens.CodesPerFrame));
            if (!window.All(OrpheusTokens.IsValid)) continue;
            var decoded = DecodeFrames(window, noise);
            for (int i = frame; i < 2 * frame; i++) audio.Add(decoded[i]);
        }
        return audio.ToArray();
    }

    // ---------------------------------------------------------------- weights

    /// <summary>
    /// Loads a released Orpheus model directory: its <c>*.safetensors</c> (the Hugging Face <c>LlamaForCausalLM</c> state
    /// dict) and, when present, its <c>tokenizer.json</c>.
    /// </summary>
    public void LoadReferenceWeights(string directory)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        string tokenizer = Path.Combine(directory, "tokenizer.json");
        if (File.Exists(tokenizer)) LoadTokenizer(tokenizer);
        var weights = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var file in Directory.GetFiles(directory, "*.safetensors"))
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file))
                weights[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        if (weights.Count == 0) throw new FileNotFoundException($"No .safetensors files in {directory}.");
        LanguageModel.LoadTorchWeights((name, shape) => weights.TryGetValue(name, out var v) ? v
            : throw new InvalidDataException($"The checkpoint has no tensor '{name}'."));
    }

    /// <summary>Loads SNAC: hubertsiuzdak/snac_24khz's <c>pytorch_model.bin</c>.</summary>
    public void LoadCodecWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        LoadCodec(name => state.TryGetValue(name, out var v) ? v : null, state.Keys);
    }

    internal void LoadCodec(Func<string, double[]?> read, IEnumerable<string> names)
    {
        var used = Codec.Load(read);
        var unused = names.Where(n => !used.Contains(n)).ToList();
        if (unused.Count > 0)
            throw new InvalidDataException($"The codec checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The release's training sequence of the transcript and the recording's frames (repeated frames dropped), and
    /// the causal language-model loss on every token, as Hugging Face's Trainer computes it with labels = input_ids.</remarks>
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        var ids = TrainingIds(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => Loss(ids), Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(sample);
        if (!HasPaperLayers) throw new NotSupportedException("The objective needs the release's networks; this model was built from caller-supplied layers.");
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return Loss(TrainingIds(sample))[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Recordings' frames (SNAC runs once per recording).
    private readonly Dictionary<Tensor<T>, List<int[]>> _frames = new(Helpers.TensorReferenceComparer<Tensor<T>>.Instance);

    internal int[] TrainingIds(TtsTrainingSample<T> sample)
    {
        Validation.Guard.NotNull(sample);
        List<int[]> frames;
        if (sample.CodecTokens is not null)
        {
            frames = FramesOf(sample.CodecTokens);
            if (!frames.All(OrpheusTokens.IsValid)) throw new ArgumentException($"Codes lie in [0, {OrpheusTokens.CodebookSize}).", nameof(sample));
        }
        else
        {
            var audio = sample.Audio ?? throw new ArgumentException($"{GetType().Name} trains on the recording or its codes; set Audio or CodecTokens.", nameof(sample));
            if (!_frames.TryGetValue(audio, out frames!))
            {
                frames = OrpheusTokens.Frames(Codec.Encode(Doubles(audio)));
                if (_frames.Count > 64) _frames.Clear();
                _frames[audio] = frames;
            }
        }
        if (_options.RemoveDuplicateFrames) frames = OrpheusTokens.RemoveDuplicateFrames(frames);
        return OrpheusTokens.TrainingSequence(Tokenizer, sample.Text ?? TextOf(sample.Tokens), _options.Voice, frames);
    }

    internal Tensor<T> Loss(int[] ids)
    {
        var lm = LanguageModel;
        var row = Row(ids);
        var (_, logits) = lm.ForwardWithLogits(lm.Embed(row));
        return lm.CausalLmLoss(logits, row);
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer()
    {
        if (_optimizer is not null) return _optimizer;
        var o = _options;
        var options = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = o.LearningRate, Beta1 = o.Beta1, Beta2 = o.Beta2, Epsilon = o.Epsilon, WeightDecay = o.WeightDecay,
            EnableGradientClipping = o.MaxGradNorm > 0, MaxGradientNorm = o.MaxGradNorm,
        };
        if (o.TrainingSteps > 0)
        {
            options.LearningRateScheduler = new LinearWarmupScheduler(o.LearningRate, 0, o.TrainingSteps, 0.0, LinearWarmupScheduler.DecayMode.Linear);
            options.SchedulerStepMode = SchedulerStepMode.StepPerBatch;
        }
        return _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, options));
    }

    /// <inheritdoc />
    /// <remarks>Training updates the language model; SNAC is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_lmLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "OrpheusTTS-Native" : "OrpheusTTS-ONNX",
            Description = "Orpheus: text-to-speech as Llama 3.2 language modeling over SNAC codes (Canopy Labs, 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "OrpheusTTS";
        metadata.AdditionalInfo["Voice"] = _options.Voice ?? "";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(OrpheusTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}

/// <summary>
/// Orpheus's token layout (the release's notebooks and <c>orpheus_tts</c> package): its markers after Llama 3's special
/// tokens, the prompt and training sequence, and SNAC codes as seven tokens per frame.
/// </summary>
internal static class OrpheusTokens
{
    public const int EndOfText = 128009;
    public const int FirstCustomToken = Llama3Tokenizers.FirstAddedId;
    public const int StartOfSpeech = FirstCustomToken + 1;
    public const int EndOfSpeech = FirstCustomToken + 2;
    public const int StartOfHuman = FirstCustomToken + 3;
    public const int EndOfHuman = FirstCustomToken + 4;
    public const int StartOfAi = FirstCustomToken + 5;
    public const int EndOfAi = FirstCustomToken + 6;
    public const int AudioStart = FirstCustomToken + 10;
    public const int CodesPerFrame = 7;
    public const int CodebookSize = 4096;

    /// <summary>The custom tokens <c>&lt;custom_token_0..28682&gt;</c> (7 × 4096 + 10 + 1 of them) and <c>&lt;|audio|&gt;</c>.</summary>
    public const int CustomTokens = CodesPerFrame * CodebookSize + 11;

    /// <summary>The tokenizer's size, 156,940.</summary>
    public const int VocabularySize = FirstCustomToken + CustomTokens + 1;

    private static readonly Lazy<ByteLevelBpeTokenizer> TokenizerInstance = new(() => Llama3Tokenizers.Create(
        Enumerable.Range(0, CustomTokens).Select(i => ($"<custom_token_{i}>", FirstCustomToken + i, false))
            .Append(("<|audio|>", FirstCustomToken + CustomTokens, true))), isThreadSafe: true);

    /// <summary>Orpheus's tokenizer (canopylabs/orpheus-3b-0.1-ft's).</summary>
    public static ByteLevelBpeTokenizer Tokenizer => TokenizerInstance.Value;

    private static IEnumerable<int> Human(ByteLevelBpeTokenizer tokenizer, string text, string? voice) =>
        new[] { StartOfHuman, Llama3Tokenizers.BeginOfText }
            .Concat(tokenizer.Encode(voice is null ? text : $"{voice}: {text}"))
            .Concat(new[] { EndOfText, EndOfHuman, StartOfAi, StartOfSpeech });

    /// <summary>The package's prompt as the model receives it: <c>_format_prompt</c>'s start of human, the text's ids
    /// (<c>&lt;|begin_of_text|&gt;</c> first), end of text, end of human, start of AI and start of speech, decoded to a string
    /// that vLLM tokenizes again with a <c>&lt;|begin_of_text|&gt;</c> in front.</summary>
    public static int[] Prompt(ByteLevelBpeTokenizer tokenizer, string text, string? voice) =>
        new[] { Llama3Tokenizers.BeginOfText }.Concat(Human(tokenizer, text, voice)).ToArray();

    /// <summary>The training sequence (<c>create_input_ids</c>): the prompt, the frames' tokens, end of speech, end of AI.</summary>
    public static int[] TrainingSequence(ByteLevelBpeTokenizer tokenizer, string text, string? voice, IReadOnlyList<int[]> frames) =>
        Human(tokenizer, text, voice)
            .Concat(frames.SelectMany(frame => frame.Select((code, slot) => AudioStart + slot * CodebookSize + code)))
            .Concat(new[] { EndOfSpeech, EndOfAi })
            .ToArray();

    /// <summary>SNAC's codes as frames (<c>tokenise_audio</c>'s order): coarse, medium 1, fine 1, fine 2, medium 2, fine 3,
    /// fine 4.</summary>
    public static List<int[]> Frames(int[][] codes)
    {
        var frames = new List<int[]>();
        for (int i = 0; i < codes[0].Length; i++)
            frames.Add(new[] { codes[0][i], codes[1][2 * i], codes[2][4 * i], codes[2][4 * i + 1], codes[1][2 * i + 1], codes[2][4 * i + 2], codes[2][4 * i + 3] });
        return frames;
    }

    /// <summary>Frames as SNAC's three rows (<c>redistribute_codes</c>).</summary>
    public static int[][] Codes(IReadOnlyList<int[]> frames) => new[]
    {
        frames.Select(f => f[0]).ToArray(),
        frames.SelectMany(f => new[] { f[1], f[4] }).ToArray(),
        frames.SelectMany(f => new[] { f[2], f[3], f[5], f[6] }).ToArray(),
    };

    /// <summary>The frames kept by <c>remove_duplicate_frames</c>: the first, then each whose coarse code differs from the
    /// last kept frame's.</summary>
    public static List<int[]> RemoveDuplicateFrames(IReadOnlyList<int[]> frames)
    {
        var kept = new List<int[]>();
        foreach (var frame in frames)
            if (kept.Count == 0 || frame[0] != kept[kept.Count - 1][0]) kept.Add(frame);
        return kept;
    }

    /// <summary>
    /// The codes of generated tokens as the package reads them (<c>turn_token_into_id</c>): each custom token gives
    /// <c>id − 128266 − 4096 × (n mod 7)</c>, n the codes read so far, and is read when that is not negative. A code of a
    /// later slot comes out at 4096 or more and spoils its frame.
    /// </summary>
    /// <remarks>The package keeps codes above 0, dropping a legitimate code 0; this keeps 0.</remarks>
    public static List<int> ReadCodes(IEnumerable<int> tokens)
    {
        var codes = new List<int>();
        foreach (int token in tokens)
        {
            if (token < FirstCustomToken || token >= FirstCustomToken + CustomTokens) continue;
            int code = token - AudioStart - codes.Count % CodesPerFrame * CodebookSize;
            if (code >= 0) codes.Add(code);
        }
        return codes;
    }

    /// <summary>The whole frames of codes (a trailing partial frame dropped).</summary>
    public static List<int[]> WholeFrames(List<int> codes) =>
        Enumerable.Range(0, codes.Count / CodesPerFrame).Select(f => codes.GetRange(f * CodesPerFrame, CodesPerFrame).ToArray()).ToList();

    /// <summary>Whether every code of a frame is in [0, 4096) (the package's check before decoding).</summary>
    public static bool IsValid(int[] frame) => frame.All(c => c >= 0 && c < CodebookSize);

    /// <summary>Removes every token but the codes of <paramref name="slot"/> (and, at a frame boundary, the end of
    /// speech).</summary>
    public static void MaskToSlot(double[] scores, int slot)
    {
        int first = AudioStart + slot * CodebookSize;
        for (int i = 0; i < scores.Length; i++)
            if ((i < first || i >= first + CodebookSize) && !(slot == 0 && i == EndOfSpeech)) scores[i] = double.NegativeInfinity;
    }
}
