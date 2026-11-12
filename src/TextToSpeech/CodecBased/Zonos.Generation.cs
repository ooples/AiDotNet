using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Zonos<T>
{
    // ---------------------------------------------------------------- the codec

    /// <inheritdoc />
    /// <remarks>DAC's codes of a 44.1 kHz recording, <c>[frames, 9]</c> (about 86 frames a second).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = Codec.Encode(audio ?? throw new ArgumentNullException(nameof(audio)));                        // [9, frames]
        var tokens = new Tensor<T>(new[] { codes.GetLength(1), codes.GetLength(0) });
        for (int f = 0; f < codes.GetLength(1); f++)
            for (int k = 0; k < codes.GetLength(0); k++) tokens[f, k] = Ops.FromDouble(codes[k, f]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>DAC's audio of codes <c>[frames, 9]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        return DecodeCodes(ByCodebook(CodesOf(tokens)));
    }

    // Codes [codebooks, frames] to audio.
    private Tensor<T> DecodeCodes(int[,] codes)
    {
        if (codes.GetLength(1) == 0) return new Tensor<T>(new[] { 0 });
        var audio = Codec.Decode(codes);
        return audio.Rank == 1 ? audio : new Tensor<T>(new[] { audio.Length }, new Vector<T>(audio.ToArray()));
    }

    private static int[,] ByCodebook(int[,] framesFirst)
    {
        var result = new int[framesFirst.GetLength(1), framesFirst.GetLength(0)];
        for (int f = 0; f < framesFirst.GetLength(0); f++)
            for (int k = 0; k < framesFirst.GetLength(1); k++) result[k, f] = framesFirst[f, k];
        return result;
    }

    private int[,] CodesOf(Tensor<T> tokens)
    {
        int books = _options.Codebooks, size = _options.Codec.CodebookSize;
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2 || tokens.Shape[1] != books) throw new ArgumentException($"Zonos's codes are [frames, {books}].", nameof(tokens));
        var codes = new int[tokens.Shape[0], books];
        for (int f = 0; f < tokens.Shape[0]; f++)
            for (int k = 0; k < books; k++)
            {
                codes[f, k] = (int)Math.Round(Ops.ToDouble(tokens[f, k]));
                if (codes[f, k] < 0 || codes[f, k] >= size) throw new ArgumentException($"Codes lie in [0, {size}).", nameof(tokens));
            }
        return codes;
    }

    // ---------------------------------------------------------------- conditioning

    /// <summary>
    /// The conditioners' inputs as <c>make_cond_dict</c> builds them from the options: the phonemes, the speaker embedding
    /// (none: unconditional), the emotion mix normalized to sum 1, bandwidth, pitch deviation, speaking rate, the language's
    /// index and the hybrid's quality targets, those in <see cref="ZonosOptions.UnconditionalInputs"/> left out.
    /// </summary>
    internal Dictionary<string, object> ConditionInputs(int[] phonemeIds, double[]? speaker)
    {
        var o = _options;
        double emotionSum = o.Emotion.Sum();
        var all = new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            ["espeak"] = phonemeIds,
            ["speaker"] = speaker,
            ["emotion"] = o.Emotion.Select(e => e / emotionSum).ToArray(),
            ["fmax"] = new[] { o.Fmax },
            ["pitch_std"] = new[] { o.PitchStd },
            ["speaking_rate"] = new[] { o.SpeakingRate },
            ["language_id"] = new[] { (double)Languages.ToList().IndexOf(o.Language) },
            ["vqscore_8"] = o.VqScore,
            ["ctc_loss"] = new[] { o.CtcLoss },
            ["dnsmos_ovrl"] = new[] { o.DnsmosOverall },
            ["speaker_noised"] = new[] { o.SpeakerNoised ? 1.0 : 0.0 },
        };
        foreach (var name in o.UnconditionalInputs) all.Remove(name);
        return all.Where(kv => kv.Value is not null).ToDictionary(kv => kv.Key, kv => kv.Value!, StringComparer.Ordinal);
    }

    /// <summary>The guided pair <c>[2, tokens, width]</c> (<c>prepare_conditioning</c>): the conditional prefix, then the
    /// unconditional one, which keeps only the required inputs (the phonemes).</summary>
    internal Tensor<T> PrepareConditioning(IReadOnlyDictionary<string, object> inputs)
    {
        var required = Conditioner.RequiredNames.ToHashSet(StringComparer.Ordinal);
        var unconditional = inputs.Where(kv => required.Contains(kv.Key)).ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
        return Engine.TensorConcatenate(new[] { Conditioner.Forward(inputs), Conditioner.Forward(unconditional) }, 0);
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Speaks the text in <see cref="ZonosOptions.Language"/>. A voice with a recording
    /// (<see cref="TtsVoice{T}.ReferenceAudio"/>, at the model's sample rate) speaks in that speaker's voice (the speaker
    /// model must be loaded, <see cref="LoadSpeakerModel"/>); a voice with <see cref="TtsVoice{T}.SpeakerEmbedding"/> (128
    /// values) uses it directly.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var voice = Voice;
        double[]? speaker = null;
        if (voice?.SpeakerEmbedding is not null) speaker = voice.SpeakerEmbedding.ToArray().Select(v => Ops.ToDouble(v)).ToArray();
        else if (voice?.ReferenceAudio is not null) speaker = SpeakerEmbedding(voice.ReferenceAudio, _options.SampleRate);
        return Generate(text, speaker);
    }

    /// <summary>
    /// Speaks <paramref name="text"/> (<c>Zonos.generate</c>) with the options' conditioning, as <paramref name="speaker"/>
    /// (a 128-value embedding from <see cref="SpeakerEmbedding"/>; null leaves it unconditional), continuing
    /// <paramref name="audioPrefix"/>'s codes <c>[frames, 9]</c> when given; returns the audio of the prefix and the new codes.
    /// </summary>
    public Tensor<T> Generate(string text, double[]? speaker, int[,]? audioPrefix = null)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (!HasPaperLayers) throw NotNative();
        var ids = ZonosPrefixConditioner<T>.PhonemeIds(Phonemes(text, _options.Language));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var prefix = PrepareConditioning(ConditionInputs(ids, speaker));
            return DecodeCodes(GenerateCodes(prefix, audioPrefix is null ? null : ByCodebook(audioPrefix), _options.MaxNewTokens));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>Sees each step's guided logits <c>[codebooks · 1025]</c> (for comparing against the release).</summary>
    internal Action<double[]>? GuidedLogits { get; set; }

    /// <summary>
    /// The release's generation loop on a guided prefix <c>[2, tokens, width]</c>: the codes (the prefix's, then unknown)
    /// are delayed (codebook k by k + 1 steps, mask tokens before and after); the backbone reads the prefix and the delayed
    /// prefix codes, then one step at a time; each step's guided logits are sampled (<see cref="SampleFrame"/>) into the
    /// unknown positions; an end token in codebook 0 leaves nine steps that close the codebooks one by one. Returns the codes
    /// <c>[codebooks, frames]</c> with the delay undone, tokens outside the codebook as 0.
    /// </summary>
    internal int[,] GenerateCodes(Tensor<T> prefix, int[,]? audioPrefix, int maxNewTokens)
    {
        var o = _options;
        int books = o.Codebooks, prefixFrames = audioPrefix?.GetLength(1) ?? 0, audioLength = prefixFrames + maxNewTokens, length = audioLength + books;
        const int Unknown = -1;
        if (audioPrefix is not null && audioPrefix.GetLength(0) != books) throw new ArgumentException($"The audio prefix has {books} codebooks.", nameof(audioPrefix));
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);

        // apply_delay_pattern: codebook k shifted right by k + 1, the shifted-out end (mask tokens) wrapping to the front.
        var delayed = new int[books, length];
        for (int k = 0; k < books; k++)
            for (int t = 0; t < length; t++)
            {
                int source = t - k - 1;
                delayed[k, t] = source < 0 || source >= audioLength ? o.MaskToken : source < prefixFrames ? audioPrefix![k, source] : Unknown;
            }

        var cache = new ZonosCache<T>();
        Tensor<T> Embed(int start, int count)
        {
            var codes = new int[2, books, count];
            for (int b = 0; b < 2; b++)
                for (int k = 0; k < books; k++)
                    for (int t = 0; t < count; t++) codes[b, k, t] = delayed[k, start + t];
            return Network.EmbedCodes(codes);
        }
        double[,] Guided(Tensor<T> inputs)
        {
            var hidden = Network.Forward(inputs, cache);
            var last = Engine.TensorSlice(hidden, new[] { 0, hidden.Shape[1] - 1, 0 }, new[] { 2, 1, hidden.Shape[2] });
            var logits = Network.ApplyHeads(last).ToArray();                                                          // [2, 1, 9, 1025]
            int v = o.Codec.CodebookSize + 1, per = books * v;
            var guided = new double[books, v];
            var flat = new double[per];
            for (int i = 0; i < per; i++)
            {
                double cond = Ops.ToDouble(logits[i]), uncond = Ops.ToDouble(logits[per + i]);
                flat[i] = guided[i / v, i % v] = uncond + (cond - uncond) * o.CfgScale;
            }
            GuidedLogits?.Invoke(flat);
            return guided;
        }
        // frame.masked_scatter_(frame == unknown, next_token): the unknown positions take the sampled codes in order, so
        // once the first codebooks already hold mask tokens (the delay pattern's end) the codes shift onto the later ones.
        void Fill(int column, int[] frame)
        {
            // The last step reads the final column and has no column left to write (the release's slice there is empty).
            if (column >= length) return;
            int next = 0;
            for (int k = 0; k < books; k++)
                if (delayed[k, column] == Unknown) delayed[k, column] = frame[next++];
        }

        // Prefill: the prefix and the delayed prefix codes up to the first unknown column.
        var first = Engine.TensorConcatenate(new[] { prefix, Embed(0, prefixFrames + 1) }, 1);
        Fill(prefixFrames + 1, SampleFrame(Guided(first), null, 0, random));
        int offset = prefixFrames + 1, remaining = length - offset;
        bool stopping = false;
        while (remaining > 0)
        {
            offset++;
            var logits = Guided(Embed(offset - 1, 1));
            for (int k = 1; k < books; k++) logits[k, o.EndToken] = double.NegativeInfinity;   // only codebook 0 ends the audio
            var frame = SampleFrame(logits, delayed, offset, random);
            if (frame[0] == o.EndToken)
            {
                remaining = Math.Min(remaining, books);
                stopping = true;
            }
            if (stopping)
            {
                int index = Math.Min(books - remaining, books - 1);
                for (int k = 0; k < index; k++) frame[k] = o.MaskToken;
                frame[index] = o.EndToken;
            }
            Fill(offset, frame);
            remaining--;
        }

        // revert_delay_pattern, then the first offset − 9 frames.
        int frames = Math.Max(0, offset - books);
        var result = new int[books, frames];
        for (int k = 0; k < books; k++)
            for (int t = 0; t < frames; t++)
            {
                int code = delayed[k, t + k + 1];
                result[k, t] = code >= o.Codec.CodebookSize ? 0 : code;
            }
        return result;
    }

    /// <summary>
    /// The release's <c>sample_from_logits</c> on one step's logits <c>[codebooks, 1025]</c>: the repetition penalty over the
    /// last <see cref="ZonosOptions.RepetitionWindow"/> generated columns (when there are any), then greedy at temperature 0,
    /// otherwise the softmax at the temperature through the unified sampler, top-p, top-k and min-p, and a draw (the argmax
    /// of probabilities over exponential draws, as the release's <c>multinomial</c> does it).
    /// </summary>
    internal int[] SampleFrame(double[,] logits, int[,]? generated, int generatedLength, Random random)
    {
        var o = _options;
        int books = logits.GetLength(0), v = logits.GetLength(1);
        var frame = new int[books];
        for (int k = 0; k < books; k++)
        {
            var row = new double[v];
            for (int i = 0; i < v; i++) row[i] = logits[k, i];
            if (o.RepetitionPenalty != 1.0 && generated is not null)
            {
                // Each token's factor is the penalty once per occurrence in the window (tokens past the vocabulary count as
                // its last one); negative logits are multiplied by it, positive ones divided.
                var factor = Enumerable.Repeat(1.0, v).ToArray();
                for (int t = Math.Max(0, generatedLength - o.RepetitionWindow); t < generatedLength; t++)
                    factor[Math.Min(generated[k, t], v - 1)] *= o.RepetitionPenalty;
                for (int i = 0; i < v; i++) row[i] = row[i] <= 0 ? row[i] * factor[i] : row[i] / factor[i];
            }
            frame[k] = o.Temperature > 0 ? Draw(row, random) : ArgMax(row);
        }
        return frame;
    }

    private static int ArgMax(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static double[] Normalized(double[] p)
    {
        double total = p.Sum();
        return p.Select(x => x / total).ToArray();
    }

    private int Draw(double[] logits, Random random)
    {
        var o = _options;
        double max = logits.Max();
        var p = Normalized(logits.Select(x => double.IsNegativeInfinity(x) ? 0.0 : Math.Exp((x - max) / o.Temperature)).ToArray());
        if (o.UnifiedLinear > 0)
        {
            var log = p.Select(x => Math.Log(Math.Max(x, 1e-20))).ToArray();
            double entropy = -p.Select((x, i) => x * log[i]).Sum();
            var raw = log.Select(l => l * (o.UnifiedLinear + entropy * o.UnifiedConfidence) - l * l * o.UnifiedQuadratic).ToArray();
            double rawMax = raw.Max();
            p = Normalized(raw.Select(x => Math.Exp(x - rawMax)).ToArray());
        }
        if (o.TopP > 0)
        {
            // Keep the tokens whose mass before them, in descending order, is at most top-p.
            var order = Enumerable.Range(0, p.Length).OrderByDescending(i => p[i]).ThenBy(i => i).ToArray();
            double before = 0;
            var kept = new double[p.Length];
            foreach (int i in order)
            {
                if (before <= o.TopP) kept[i] = p[i];
                before += p[i];
            }
            p = Normalized(kept);
        }
        if (o.TopK > 0)
        {
            double pivot = p.OrderByDescending(x => x).ElementAt(Math.Min(o.TopK, p.Length) - 1);
            p = Normalized(p.Select(x => x < pivot ? 0.0 : x).ToArray());
        }
        if (o.MinP > 0)
        {
            double floor = o.MinP * p.Max();
            p = Normalized(p.Select(x => x < floor ? 0.0 : x).ToArray());
        }
        int best = 0;
        double bestValue = double.NegativeInfinity;
        for (int i = 0; i < p.Length; i++)
        {
            double value = p[i] / -Math.Log(1.0 - random.NextDouble());
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }
        return best;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is phoneme ids (<see cref="PreprocessText"/>); the waveform spoken from them.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        var ids = input.ToArray().Select(v => (int)Math.Round(Ops.ToDouble(v))).ToArray();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var prefix = PrepareConditioning(ConditionInputs(ids, null));
            return DecodeCodes(GenerateCodes(prefix, null, _options.MaxNewTokens));
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>eSpeak's phonemes of (language, text); null calls the installed libespeak-ng (tests replay recorded output).</summary>
    internal Func<string, string, string>? EspeakOverride { get; set; }

    /// <summary>The phonemes Zonos conditions on (<c>phonemize</c>): Japanese read through Sudachi, other languages' numbers
    /// spelled out, then eSpeak with punctuation and stress kept.</summary>
    internal string Phonemes(string text, string language) =>
        ZonosFrontEnd.Phonemize(text, language, JapaneseReadings, EspeakOverride);

    /// <inheritdoc />
    /// <remarks>The text's phoneme ids, start and end included.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = ZonosPrefixConditioner<T>.PhonemeIds(Phonemes(text, _options.Language));
        return new Tensor<T>(new[] { ids.Length }, new Vector<T>(ids.Select(i => NumOps.FromDouble(i)).ToArray()));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
}
