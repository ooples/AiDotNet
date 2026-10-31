using AiDotNet.Audio.Codecs;
using AiDotNet.Audio.Foundations;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A word of a transcript with its start and end in seconds (Whisper's word timestamps, §3.4.4).</summary>
public readonly record struct MoshiWord(string Text, double Start, double End);

/// <summary>A word already in text tokens, with its start and end in seconds.</summary>
internal readonly record struct MoshiTokenSpan(int[] Tokens, double Start, double End);

/// <summary>A word Moshi wrote or recognized and the time in the audio it belongs to.</summary>
public readonly record struct MoshiTimedWord(string Text, double Time);

/// <summary>
/// What Moshi's text-to-speech model and its speech recognizer share: the RQ-Transformer, Mimi, Helium's tokenizer, the
/// inner monologue's aligned text stream (Eq. 5, the release's <c>Interleaver</c>), and App. C's streaming TTS and ASR
/// loops.
/// </summary>
internal sealed class MoshiCore<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private SentencePieceModel? _tokenizer;

    public MoshiCore(IEngine engine, MoshiOptions options, List<LayerBase<T>> lmLayers, List<LayerBase<T>> mimiLayers,
        List<LayerBase<T>> discriminatorLayers, List<LayerBase<T>> distillationLayers, List<LayerBase<T>> wavLmLayers, int seed)
    {
        _engine = engine;
        Options = options;
        Validate(options);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed);
        LanguageModel = new MoshiLanguageModel<T>(engine, lmLayers, LanguageModelConfiguration(options), random);
        Mimi = new Mimi<T>(engine, MimiConfiguration(options), mimiLayers);
        if (options.IncludeMimiTraining)
        {
            var wavLm = new WavLmConfiguration
            {
                HiddenSize = options.WavLmWidth, Layers = options.WavLmLayers, Heads = options.WavLmHeads, IntermediateSize = 4 * options.WavLmWidth,
                LayerNormFeatureEncoder = true, StableLayerNorm = true, ConvDims = Enumerable.Repeat(options.WavLmConvChannels, 7).ToArray(),
                PositionalConvGroups = Math.Min(16, options.WavLmWidth),
            };
            MimiTraining = new MimiTraining<T>(engine, options.SampleRate, options.MimiDimension, wavLm, options.MimiAdversarialWeight, options.MimiFeatureWeight,
                options.MimiDistillationWeight, options.MimiDiscriminatorWindows, options.MimiDiscriminatorFilters, discriminatorLayers,
                distillationLayers, wavLmLayers);
            MimiTraining.WavLm!.InitializeLikeReference(random);
        }
    }

    private static void Validate(MoshiOptions o)
    {
        if (o.HiddenDim <= 0 || o.NumHeads <= 0 || o.HiddenDim % o.NumHeads != 0 || o.DepthDim % o.DepthHeads != 0)
            throw new ArgumentException("Each Transformer's width must divide into its heads.");
        if (o.Codebooks < 2 || o.Codebooks > o.MimiCodebooks) throw new ArgumentException($"Moshi models 2 … {o.MimiCodebooks} Mimi codebooks.");
        if (o.AcousticDelay < 0 || o.TextAudioDelaySeconds < 0) throw new ArgumentException("Delays are not negative.");
        if (o.TextPadding < 0 || o.TextPadding >= o.VocabSize || o.TextEndPadding < 0 || o.TextEndPadding >= o.VocabSize || o.TextPadding == o.TextEndPadding)
            throw new ArgumentException("PAD and EPAD are two distinct text tokens.");
    }

    public MoshiOptions Options { get; }

    public MoshiLanguageModel<T> LanguageModel { get; }

    public Mimi<T> Mimi { get; }

    public MimiTraining<T>? MimiTraining { get; }

    public int Codebooks => Options.Codebooks;

    /// <summary>Codec frames a second (12.5).</summary>
    public double FrameRate => (double)Options.SampleRate / Mimi.Configuration.SamplesPerFrame;

    /// <summary>The text/audio delay of the TTS and ASR variants in frames (2 s: 25).</summary>
    public int DelayFrames => (int)Math.Round(Options.TextAudioDelaySeconds * FrameRate);

    internal static MoshiLanguageModelConfiguration LanguageModelConfiguration(MoshiOptions o)
    {
        int q = o.Codebooks, depth = o.DepthLayout == MoshiDepthLayout.AllStreams ? 2 * q : q;
        var speaker = new[] { 0 }.Concat(Enumerable.Repeat(o.AcousticDelay, q - 1));
        return new MoshiLanguageModelConfiguration
        {
            Codebooks = q, AudioStreams = 2 * q, DepthStreams = depth, AudioCardinality = o.CodebookSize, TextCardinality = o.VocabSize,
            Delays = new[] { 0 }.Concat(speaker).Concat(speaker).ToArray(), TextPadding = o.TextPadding, TextEndPadding = o.TextEndPadding,
            Temporal = new MoshiTransformerConfiguration
            {
                Dim = o.HiddenDim, Heads = o.NumHeads, Layers = o.Layers, FeedForward = o.FeedForward, Context = o.Context, Rotary = true,
                WeightsPerStep = 1,
            },
            Depth = new MoshiTransformerConfiguration
            {
                Dim = o.DepthDim, Heads = o.DepthHeads, Layers = o.DepthLayers, FeedForward = o.DepthFeedForward, Context = 0, Rotary = false,
                WeightsPerStep = depth,
            },
        };
    }

    internal static MimiConfiguration MimiConfiguration(MoshiOptions o) => new()
    {
        SampleRate = o.SampleRate, Dimension = o.MimiDimension, Filters = o.MimiFilters, Ratios = (int[])o.MimiRatios.Clone(),
        TransformerLayers = o.MimiTransformerLayers, Heads = o.MimiHeads, FeedForward = o.MimiFeedForward, Context = o.MimiContext,
        Codebooks = o.MimiCodebooks, CodebookSize = o.CodebookSize, CodebookDim = o.MimiCodebookDim,
    };

    // ---------------------------------------------------------------- text

    public SentencePieceModel Tokenizer
    {
        get
        {
            if (_tokenizer is not null) return _tokenizer;
            string path = Options.TokenizerPath ?? throw new InvalidOperationException("Text needs Helium's tokenizer: set MoshiOptions.TokenizerPath.");
            if (!File.Exists(path)) throw new FileNotFoundException($"Tokenizer not found: {path}", path);
            var tokenizer = new SentencePieceModel(File.ReadAllBytes(path));
            if (tokenizer.Count > Options.VocabSize)
                throw new InvalidOperationException($"The tokenizer has {tokenizer.Count} pieces; the text vocabulary is {Options.VocabSize}.");
            return _tokenizer = tokenizer;
        }
    }

    public void ResetTokenizer() => _tokenizer = null;

    /// <summary>A word's tokens as the release's interleaver writes them (<c>tokenize(word.strip(), bos=False)</c>).</summary>
    public int[] WordTokens(string word) => Tokenizer.Encode(word.Trim());

    private bool IsPadding(int token) => token == Options.TextPadding || token == Options.TextEndPadding;

    /// <summary>The text of tokens, padding and the zero token dropped.</summary>
    public string TextOf(IEnumerable<int> tokens) =>
        Tokenizer.Decode(tokens.Where(t => t >= 0 && !IsPadding(t) && t != Tokenizer.BosId && t != Tokenizer.EosId));

    /// <summary>
    /// The inner monologue's text stream over <paramref name="frames"/> frames (Eq. 5; the release's
    /// <c>Interleaver.build_token_stream</c>): PAD everywhere, each word's tokens from its start frame on, EPAD in the frame
    /// before them unless that frame holds a token. Words of zero or negative duration are dropped. A positive
    /// <paramref name="audioDelay"/> (seconds) puts the text ahead of the audio (TTS), a negative one behind it (ASR), whose
    /// first frames are then the zero token (−1, no input and no loss).
    /// </summary>
    public int[] TextStream(IReadOnlyList<MoshiWord> words, int frames, double audioDelay) =>
        TextStream(words.OrderBy(w => w.Start).Select(w => new MoshiTokenSpan(WordTokens(w.Text), w.Start, w.End)).ToList(), frames, audioDelay);

    /// <summary>The text stream of words already in tokens.</summary>
    public int[] TextStream(IReadOnlyList<MoshiTokenSpan> words, int frames, double audioDelay)
    {
        double rate = FrameRate;
        var aligned = words.OrderBy(w => w.Start)
            .Where(w => w.Start < w.End)
            .Select(w => (w.Tokens, Start: w.Start - audioDelay, End: w.End - audioDelay))
            .Where(w => audioDelay == 0 || w.End > 0)
            .ToList();
        var text = Enumerable.Repeat(Options.TextPadding, frames).ToArray();
        var pending = new Queue<int>();
        int i = 0, lastWordEnd = -1;
        for (int t = 0; t < frames; t++)
        {
            while (i < aligned.Count && aligned[i].Start * rate < t + 1)
            {
                pending = new Queue<int>(aligned[i].Tokens);
                lastWordEnd = (int)(aligned[i].End * rate);
                i++;
            }
            if (pending.Count > 0)
            {
                if (t > 0 && text[t - 1] == Options.TextPadding) text[t - 1] = Options.TextEndPadding;
                text[t] = pending.Dequeue();
            }
            else if (t <= lastWordEnd) text[t] = Options.TextPadding;
        }
        if (audioDelay < 0)
            for (int t = 0; t < Math.Min(frames, (int)(rate * -audioDelay)); t++) text[t] = -1;
        return text;
    }

    // ---------------------------------------------------------------- audio

    private static double[] Samples(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = Ops.ToDouble(audio.GetFlat(i));
        return samples;
    }

    /// <summary>Mimi's first Q codes of a waveform, <c>[Q, frames]</c>.</summary>
    public int[,] AudioCodes(double[] audio)
    {
        var codes = Mimi.Encode(audio, Codebooks);
        int frames = codes.GetLength(0);
        var rows = new int[Codebooks, frames];
        for (int q = 0; q < Codebooks; q++)
            for (int t = 0; t < frames; t++) rows[q, t] = codes[t, q];
        return rows;
    }

    public int[,] AudioCodes(Tensor<T> audio) => AudioCodes(Samples(audio));

    /// <summary>Mimi's waveform of codes <c>[Q, frames]</c>.</summary>
    public Tensor<T> Audio(int[,] codes)
    {
        int frames = codes.GetLength(1);
        if (frames == 0) return new Tensor<T>(new[] { 0 });
        var transposed = new int[frames, codes.GetLength(0)];
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int t = 0; t < frames; t++) transposed[t, q] = codes[q, t];
        var samples = Mimi.Decode(transposed);
        return new Tensor<T>(new[] { samples.Length }, new Vector<T>(samples.Select(v => Ops.FromDouble(v)).ToArray()));
    }

    /// <summary>The joint sequence <c>[1 + 2Q, T]</c>: text, Moshi's Q streams, the user's Q (−1 where absent), cut to the
    /// shortest.</summary>
    public int[,] Sequence(int[] text, int[,] moshi, int[,]? user)
    {
        int q = Codebooks, frames = Math.Min(text.Length, moshi.GetLength(1));
        if (user is not null) frames = Math.Min(frames, user.GetLength(1));
        var codes = new int[1 + 2 * q, frames];
        for (int t = 0; t < frames; t++)
        {
            codes[0, t] = text[t];
            for (int k = 0; k < q; k++)
            {
                codes[1 + k, t] = moshi[k, t];
                codes[1 + q + k, t] = user?[k, t] ?? -1;
            }
        }
        return codes;
    }

    // ---------------------------------------------------------------- streaming

    private MoshiStreamer<T> Streamer(MoshiSampling sampling, Random random) => new(LanguageModel, sampling, random);

    private bool UserIsInput => LanguageModel.Configuration.DepthStreams == Codebooks;

    // The step's inputs for the user's streams: their raw values when they are inputs (the release's layout), else none.
    private int[]? UserInput(int[,]? user, int t)
    {
        if (!UserIsInput) return null;
        var input = new int[Codebooks];
        for (int k = 0; k < Codebooks; k++) input[k] = user is null ? -1 : t < user.GetLength(1) ? user[k, t] : -1;
        return input;
    }

    // Forced values at step t for the predicted streams: the delayed value of a known stream (its frame t − delay), or
    // Ungenerated to sample. Moshi's streams are 0 … Q−1, the user's (when predicted) Q … 2Q−1.
    private int[] Forced(int[,]? moshi, int[,]? user, bool silentUser, int t)
    {
        var c = LanguageModel.Configuration;
        var forced = Enumerable.Repeat(MoshiStreamer<T>.Ungenerated, c.DepthStreams).ToArray();
        for (int s = 0; s < c.DepthStreams; s++)
        {
            var known = s < Codebooks ? moshi : user;
            int k = s % Codebooks, frame = t - c.Delays[1 + s];
            if (s >= Codebooks && known is null && silentUser) forced[s] = -1;
            else if (known is not null) forced[s] = frame < 0 ? 0 : frame < known.GetLength(1) ? known[k, frame] : -1;
        }
        return forced;
    }

    /// <summary>
    /// App. C's streaming TTS: the text runs ahead of the audio by the delay; the model samples PAD and EPAD freely, and as
    /// soon as it samples any other token the next word's tokens go in instead, one a step. While the running fraction of
    /// padding steps is under <see cref="MoshiOptions.TtsPaddingTarget"/> the padding logits get
    /// <see cref="MoshiOptions.TtsPaddingBonus"/>. After the last word only PAD is allowed, for the delay, so the audio
    /// catches up. Moshi's audio streams are sampled at §5.7's temperature; the user's are silent (−1). Returns the speech
    /// and when each word was consumed.
    /// </summary>
    public (Tensor<T> Audio, List<MoshiTimedWord> Words) Speak(string text, Random random)
    {
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0) throw new ArgumentException("There is no text.", nameof(text));
        var o = Options;
        var streamer = Streamer(new MoshiSampling
        {
            AudioTemperature = o.TtsTemperature, AudioTopK = o.AudioTopK, TextTemperature = o.TtsTemperature, TextTopK = o.TextTopK,
        }, random);
        int next = 0, padding = 0, steps = 0, tail = 0, tailLength = DelayFrames + streamer.MaxDelay;
        var pending = new Queue<int>();
        var timed = new List<MoshiTimedWord>();
        var frames = new List<int[]>();
        int StartWord(int step)
        {
            var tokens = WordTokens(words[next]);
            timed.Add(new MoshiTimedWord(words[next], Math.Max(0, step - DelayFrames) / FrameRate));
            next++;
            foreach (int token in tokens.Skip(1)) pending.Enqueue(token);
            return tokens.Length > 0 ? tokens[0] : o.TextEndPadding;
        }
        while (tail < tailLength && steps < o.Context)
        {
            int? forced = null;
            Action<double[]>? bias = null;
            Func<int, int>? filter = null;
            if (pending.Count > 0) forced = pending.Dequeue();
            else if (next >= words.Length)
            {
                forced = o.TextPadding;
                tail++;
            }
            else
            {
                int step = steps;
                double fraction = steps == 0 ? 1.0 : (double)padding / steps;
                if (fraction < o.TtsPaddingTarget)
                    bias = logits => { logits[o.TextPadding] += o.TtsPaddingBonus; logits[o.TextEndPadding] += o.TtsPaddingBonus; };
                filter = token => IsPadding(token) ? token : StartWord(step);
            }
            int textToken = MoshiStreamer<T>.Ungenerated;
            var frame = streamer.Step(UserInput(null, steps), forced, Forced(null, null, silentUser: true, steps), bias,
                token => textToken = filter is null ? token : filter(token));
            if (forced is not null) textToken = forced.Value;
            if (IsPadding(textToken)) padding++;
            steps++;
            if (frame is not null) frames.Add(frame);
        }
        return (Audio(MoshiAudio(frames)), timed);
    }

    // Moshi's Q audio streams of the undelayed frames, [Q, frames].
    private int[,] MoshiAudio(IReadOnlyList<int[]> frames)
    {
        var codes = new int[Codebooks, frames.Count];
        for (int t = 0; t < frames.Count; t++)
            for (int k = 0; k < Codebooks; k++) codes[k, t] = Math.Max(0, frames[t][1 + k]);
        return codes;
    }

    /// <summary>
    /// App. C's streaming ASR: the recording's codes are forced into Moshi's audio streams, followed by the delay of silence
    /// so the text catches up; the text stream, the zero token for the delay's first frames as in training, is decoded
    /// greedily (§5.7). Returns the transcript and each word with its time in the recording.
    /// </summary>
    public (string Text, List<MoshiTimedWord> Words) Transcribe(Tensor<T> audio)
    {
        var samples = Samples(audio);
        int spf = Mimi.Configuration.SamplesPerFrame;
        var padded = samples.Concat(new double[(DelayFrames + LanguageModel.Configuration.Delays.Max() + 1) * spf]).ToArray();
        var codes = AudioCodes(padded);
        int frames = codes.GetLength(1);
        var streamer = Streamer(new MoshiSampling { UseSampling = false }, new Random(0));
        var tokens = new List<(int Token, int Frame)>();
        for (int t = 0; t < frames; t++)
        {
            int? forced = t < DelayFrames ? -1 : null;
            var frame = streamer.Step(UserInput(null, t), forced, Forced(codes, null, silentUser: true, t));
            if (frame is not null && frame[0] >= 0 && !IsPadding(frame[0])) tokens.Add((frame[0], t - streamer.MaxDelay));
        }
        var words = new List<MoshiTimedWord>();
        foreach (var (token, frame) in tokens)
        {
            string piece = Tokenizer.IdToPiece(token);
            double time = Math.Max(0, frame - DelayFrames) / FrameRate;
            if (words.Count == 0 || piece.StartsWith("▁", StringComparison.Ordinal))
                words.Add(new MoshiTimedWord(Tokenizer.Decode(new[] { token }), time));
            else words[words.Count - 1] = words[words.Count - 1] with { Text = words[words.Count - 1].Text + Tokenizer.Decode(new[] { token }) };
        }
        return (TextOf(tokens.Select(p => p.Token)), words);
    }

    /// <summary>
    /// Full-duplex dialogue (§3.4.3, the release's <c>LMGen</c>): every 80 ms the user's frame goes in and Moshi writes a
    /// text token and its audio frame, sampled at the release's temperatures. Returns Moshi's speech over the user's
    /// recording and its inner monologue.
    /// </summary>
    public (Tensor<T> Audio, string Text) Chat(Tensor<T> userAudio, Random random)
    {
        var o = Options;
        var user = AudioCodes(userAudio);
        var streamer = Streamer(new MoshiSampling
        {
            AudioTemperature = o.AudioTemperature, AudioTopK = o.AudioTopK, TextTemperature = o.TextTemperature, TextTopK = o.TextTopK,
        }, random);
        var frames = new List<int[]>();
        int total = user.GetLength(1) + streamer.MaxDelay;
        for (int t = 0; t < total; t++)
        {
            var frame = streamer.Step(UserInput(user, t), null, Forced(null, user, silentUser: false, t));
            if (frame is not null) frames.Add(frame);
        }
        return (Audio(MoshiAudio(frames)), TextOf(frames.Select(f => f[0])));
    }

    // ---------------------------------------------------------------- augmentation

    /// <summary>
    /// §4.4's instruct augmentation of the user's stream: a gain of −24 … +15 dB half the time; 30% of the time noise from
    /// <paramref name="noise"/> (clips alternating with up to 30 s of silence, concatenated to cover the example) at −30 …
    /// +6 dB relative to the source; and 30% of the time Moshi's echo (scaled by U[0, 0.2], delayed 100 … 500 ms) with the
    /// reverb of Défossez et al. 2020 (denoiser's <c>RevEcho</c>: first echo U[0, 0.3], RT60 0.3 … 1.3 s, first delay 10 …
    /// 30 ms, three trains, jitter 10%). Without noise clips the noise step is skipped.
    /// </summary>
    public double[] AugmentUser(double[] user, double[] moshi, IReadOnlyList<double[]> noise, Random random)
    {
        var x = (double[])user.Clone();
        int rate = Options.SampleRate;
        double Uniform(double a, double b) => a + (b - a) * random.NextDouble();
        if (random.NextDouble() < 0.5)
        {
            double gain = Math.Pow(10, Uniform(-24, 15) / 20);
            for (int i = 0; i < x.Length; i++) x[i] *= gain;
        }
        if (random.NextDouble() < 0.3 && noise.Count > 0)
        {
            var track = new List<double>(x.Length);
            while (track.Count < x.Length)
            {
                if (random.NextDouble() < 0.5) track.AddRange(new double[(int)(Uniform(0, 30) * rate)]);
                else track.AddRange(noise[random.Next(noise.Count)]);
            }
            double source = Rms(x), level = Rms(track, x.Length);
            if (level > 0 && source > 0)
            {
                double scale = source / level * Math.Pow(10, Uniform(-30, 6) / 20);
                for (int i = 0; i < x.Length; i++) x[i] += scale * track[i];
            }
        }
        if (random.NextDouble() < 0.3)
        {
            double factor = Uniform(0, 0.2);
            int delay = (int)(Uniform(0.1, 0.5) * rate);
            for (int i = delay; i < x.Length && i - delay < moshi.Length; i++) x[i] += factor * moshi[i - delay];
            var reverb = Reverb(x, random.NextDouble() * 0.3, Uniform(0.01, 0.03), Uniform(0.3, 1.3), random);
            for (int i = 0; i < x.Length; i++) x[i] += reverb[i];
        }
        return x;
    }

    private static double Rms(IReadOnlyList<double> x, int? count = null)
    {
        int n = count ?? x.Count;
        double sum = 0;
        for (int i = 0; i < n; i++) sum += x[i] * x[i];
        return n == 0 ? 0 : Math.Sqrt(sum / n);
    }

    // denoiser's RevEcho._reverb: trains of echoes of the source, each delayed by the jittered first delay and attenuated so
    // that the train decays by 1e-3 over RT60.
    private double[] Reverb(double[] source, double initial, double firstDelay, double rt60, Random random)
    {
        const int repeat = 3;
        const double jitterRange = 0.1;
        int length = source.Length, rate = Options.SampleRate;
        var reverb = new double[length];
        for (int r = 0; r < repeat; r++)
        {
            double fraction = 1;
            var echo = source.Select(v => initial * v).ToArray();
            while (fraction > 1e-3)
            {
                double jitter = 1 + jitterRange * (2 * random.NextDouble() - 1);
                int delay = Math.Min(1 + (int)(jitter * firstDelay * rate), length);
                var shifted = new double[length];
                Array.Copy(echo, 0, shifted, delay, length - delay);
                echo = shifted;
                for (int i = 0; i < length; i++) reverb[i] += echo[i];
                jitter = 1 + jitterRange * (2 * random.NextDouble() - 1);
                double attenuation = Math.Pow(10, -3 * jitter * firstDelay / rt60);
                for (int i = 0; i < length; i++) echo[i] *= attenuation;
                fraction *= attenuation;
            }
        }
        return reverb;
    }

    // ---------------------------------------------------------------- the objective

    /// <summary>Eq. 7's loss of a joint sequence <c>[1 + 2Q, T]</c>.</summary>
    public Tensor<T> Loss(int[,] codes) => LanguageModel.Loss(codes, Options.SemanticWeight, Options.TextPaddingWeight);

    // Table 1's rates per stage: (Temporal, Depth).
    private (double Temporal, double Depth) StageRates() => Options.Stage switch
    {
        MoshiTrainingStage.PostTraining => (3e-6, 5e-5),
        MoshiTrainingStage.FisherFineTuning => (2e-6, 4e-6),
        MoshiTrainingStage.InstructFineTuning => (2e-6, 2e-6),
        _ => (Options.LearningRate, Options.DepthLearningRate),
    };

    /// <summary>
    /// The language model's AdamW at the current stage (§4.4, Table 1): β (0.9, 0.95), weight decay 0.1, clipping at 1,
    /// the Temporal Transformer's rate with pre-training's cosine schedule after a linear warmup (later stages fixed), the
    /// Depth Transformer at its own rate and, on audio batches, the text embedding and text linear at 0.75 of theirs.
    /// </summary>
    public AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateLanguageModelOptimizer(
        AiDotNet.Interfaces.IFullModel<T, Tensor<T>, Tensor<T>> model, IReadOnlyList<LayerBase<T>> lmLayers, bool textOnly)
    {
        var o = Options;
        var (temporal, depth) = StageRates();
        var optimizer = new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(model, new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = temporal, Beta1 = 0.9, Beta2 = 0.95, Epsilon = 1e-8, WeightDecay = o.WeightDecay,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
            EnableGradientClipping = o.MaxGradientNorm > 0, MaxGradientNorm = o.MaxGradientNorm > 0 ? o.MaxGradientNorm : 1.0,
            LearningRateScheduler = o.Stage == MoshiTrainingStage.PreTraining
                ? new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(temporal, o.WarmupSteps, o.TotalSteps, 0.0,
                    AiDotNet.LearningRateSchedulers.LinearWarmupScheduler.DecayMode.Cosine, 0.0)
                : null,
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        });
        var depthLayers = new HashSet<LayerBase<T>>(LanguageModel.DepthLayers());
        var textLayers = new HashSet<LayerBase<T>> { LanguageModel.TextEmbedding, LanguageModel.TextLinear };
        var parameters = new List<Tensor<T>>();
        var rates = new List<T>();
        foreach (var layer in lmLayers)
        {
            double rate = depthLayers.Contains(layer) ? depth / temporal : !textOnly && textLayers.Contains(layer) ? o.TextRateMultiplier : 1.0;
            foreach (var p in AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(new AiDotNet.Interfaces.ILayer<T>[] { layer }, -1))
            {
                parameters.Add(p);
                for (int i = 0; i < p.Length; i++) rates.Add(Ops.FromDouble(rate));
            }
        }
        var options = (AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.LearningRateMask = new Vector<T>(rates.ToArray());
        options.MaskParameters = parameters;
        return optimizer;
    }

    /// <summary>A text-only sequence (§4.4's text batches): Helium's tokens with &lt;s&gt; in the text stream, every audio
    /// stream the zero token.</summary>
    public int[,] TextOnlySequence(string text)
    {
        var ids = new[] { Tokenizer.BosId }.Concat(Tokenizer.Encode(text)).ToArray();
        var codes = new int[1 + 2 * Codebooks, ids.Length];
        for (int t = 0; t < ids.Length; t++)
        {
            codes[0, t] = ids[t];
            for (int k = 1; k <= 2 * Codebooks; k++) codes[k, t] = -1;
        }
        return codes;
    }
}
