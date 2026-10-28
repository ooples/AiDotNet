using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Dia<T>
{
    // ---------------------------------------------------------------- the codec

    private static Tensor<T> Wave(Tensor<T> audio) => audio.Rank == 1 ? audio : new Tensor<T>(new[] { audio.Length }, new Vector<T>(audio.ToArray()));

    /// <inheritdoc />
    /// <remarks>DAC's codes of a 44.1 kHz recording, <c>[frames, 9]</c> (about 86 frames a second).</remarks>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        var codes = FramesFirst(Codec.Encode(audio ?? throw new ArgumentNullException(nameof(audio))));
        var tokens = new Tensor<T>(new[] { codes.GetLength(0), codes.GetLength(1) });
        for (int f = 0; f < codes.GetLength(0); f++)
            for (int k = 0; k < codes.GetLength(1); k++) tokens[f, k] = Ops.FromDouble(codes[f, k]);
        return tokens;
    }

    /// <inheritdoc />
    /// <remarks>DAC's audio of codes <c>[frames, 9]</c>.</remarks>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        return DecodeCodes(CodesOf(tokens));
    }

    private Tensor<T> DecodeCodes(int[,] codes) =>
        codes.GetLength(0) == 0 ? new Tensor<T>(new[] { 0 }) : Wave(Codec.Decode(ByCodebook(codes)));

    // The codec's [codebooks, frames] as [frames, codebooks] and back.
    private static int[,] FramesFirst(int[,] codes)
    {
        var result = new int[codes.GetLength(1), codes.GetLength(0)];
        for (int q = 0; q < codes.GetLength(0); q++)
            for (int f = 0; f < codes.GetLength(1); f++) result[f, q] = codes[q, f];
        return result;
    }

    private static int[,] ByCodebook(int[,] codes) => FramesFirst(codes);

    private int[,] CodesOf(Tensor<T> tokens)
    {
        int books = _options.Channels, size = _options.Codec.CodebookSize;
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        if (tokens.Rank != 2 || tokens.Shape[1] != books) throw new ArgumentException($"Dia's codes are [frames, {books}].", nameof(tokens));
        var codes = new int[tokens.Shape[0], books];
        for (int f = 0; f < tokens.Shape[0]; f++)
            for (int k = 0; k < books; k++)
            {
                codes[f, k] = (int)Math.Round(Ops.ToDouble(tokens[f, k]));
                if (codes[f, k] < 0 || codes[f, k] >= size) throw new ArgumentException($"Codes lie in [0, {size}).", nameof(tokens));
            }
        return codes;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>Speaks the script (speaker tags [S1] and [S2]). A voice with a recording (<see cref="TtsVoice{T}.ReferenceAudio"/>
    /// at 44.1 kHz, or its codes <see cref="TtsVoice{T}.ReferenceCodecTokens"/>) and its transcript
    /// (<see cref="TtsVoice{T}.ReferenceTokens"/>, byte ids) is the audio prompt, its transcript put in front of the script
    /// as the release's example does.</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        var voice = Voice;
        if (voice?.ReferenceAudio is null && voice?.ReferenceCodecTokens is null) return Generate(text, null);
        if (voice.ReferenceTokens is null)
            throw new InvalidOperationException("A Dia voice is an audio prompt: give its transcript (ReferenceTokens) with the recording.");
        var prompt = voice.ReferenceCodecTokens is not null ? CodesOf(voice.ReferenceCodecTokens) : FramesFirst(Codec.Encode(voice.ReferenceAudio!));
        return Generate(TextOf(voice.ReferenceTokens) + text, prompt);
    }

    /// <summary>
    /// Speaks <paramref name="script"/> (<c>Dia.generate</c>), continuing <paramref name="audioPrompt"/>'s codes
    /// <c>[frames, 9]</c> when given (put the prompt's transcript at the front of the script); returns only the new audio.
    /// </summary>
    public Tensor<T> Generate(string script, int[,]? audioPrompt)
    {
        ThrowIfDisposed();
        if (script is null) throw new ArgumentNullException(nameof(script));
        if (!HasPaperLayers) throw NotNative();
        var ids = Tokenize(script, _options.MaxTextLength);
        if (ids.Length == 0) throw new ArgumentException("The script has no bytes.", nameof(script));
        return DecodeCodes(GenerateCodes(ids, audioPrompt, _options.MaxAudioTokens));
    }

    /// <summary>Sees each step's unconditional and conditional logits <c>[channels · vocabulary]</c> (for comparing against
    /// the release).</summary>
    internal Action<double[], double[]>? StepLogits { get; set; }

    /// <summary>
    /// The release's generation loop on byte ids: the encoder reads the script and an all-zero script of its length; the
    /// decoder is prefilled with the start frame and the prompt's codes, delayed; each step decodes both branches,
    /// samples a frame (<see cref="SampleFrame"/>), and keeps the prompt's delayed codes where they are still being read
    /// in; channel 0's end token (or the length limit) starts the countdown that closes the delayed channels. Returns the
    /// new codes <c>[frames, channels]</c>, the delay undone, tokens outside the codebook as 0.
    /// </summary>
    internal int[,] GenerateCodes(int[] text, int[,]? prompt, int maxTokens)
    {
        var o = _options;
        int channels = o.Channels, maxDelay = o.DelayPattern.Max(), promptFrames = prompt?.GetLength(0) ?? 0;
        if (prompt is not null && prompt.GetLength(1) != channels) throw new ArgumentException($"The prompt's codes are [frames, {channels}].", nameof(prompt));
        int prefillLength = promptFrames + maxDelay;
        if (prefillLength + 1 > maxTokens) throw new ArgumentException($"The prompt fills {prefillLength} of the {maxTokens} steps.", nameof(prompt));
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(o.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var both = new int[2, text.Length];
            for (int j = 0; j < text.Length; j++) both[1, j] = text[j];
            var cache = Network.StartDecoding(Network.Encode(both));

            // The prefill: the start frame, then the prompt, delayed (start tokens before each channel's delay, -1 past the
            // prompt, which the loop fills).
            var generated = new int[maxTokens + 1, channels];
            for (int t = 0; t < generated.GetLength(0); t++)
                for (int c = 0; c < channels; c++) generated[t, c] = -1;
            for (int t = 0; t < prefillLength; t++)
                for (int c = 0; c < channels; c++)
                {
                    int source = t - o.DelayPattern[c];
                    generated[t, c] = source < 0 ? o.BeginToken : source == 0 ? o.BeginToken : source <= promptFrames ? prompt![source - 1, c] : -1;
                }
            int prefillSteps = promptFrames + 1, step = prefillSteps - 1;
            if (step > 0) Network.Decode(Branches(generated, 0, step), cache);

            bool endDetected = false, startOver = false;
            int countdown = -1, finished = -1;
            while (step < maxTokens)
            {
                if (countdown == 0) break;
                int current = step + 1;
                var logits = Network.Decode(Branches(generated, step, 1), cache).ToArray();
                var frame = SampleFrame(logits, random);
                if (countdown != 0)
                {
                    bool trigger = (!endDetected && frame[0] == o.EndToken) || current >= maxTokens - maxDelay;
                    endDetected |= trigger;
                    if (trigger && countdown < 0)
                    {
                        countdown = maxDelay;
                        finished = current;
                    }
                }
                if (countdown > 0)
                {
                    int after = maxDelay - countdown;
                    for (int c = 0; c < channels; c++)
                    {
                        if (after == o.DelayPattern[c]) frame[c] = o.EndToken;
                        else if (after > o.DelayPattern[c]) frame[c] = o.PadToken;
                    }
                    countdown--;
                }
                if (!startOver) startOver = step - prefillSteps > maxDelay;
                for (int c = 0; c < channels; c++)
                    if (startOver || generated[current, c] == -1) generated[current, c] = frame[c];
                step++;
            }

            int finalStep = step + 1;
            if (finished == -1) finished = finalStep - maxDelay;
            int length = Math.Max(finished - prefillSteps, 0), span = length + maxDelay;
            var codes = new int[length, channels];
            for (int t = 0; t < length; t++)
                for (int c = 0; c < channels; c++)
                {
                    // The delay undone: frame t of channel c is step t + delay[c] of the span after the prefill.
                    int source = Math.Min(t + o.DelayPattern[c], span - 1), row = prefillSteps + source;
                    int code = t + o.DelayPattern[c] >= span || row >= generated.GetLength(0) ? o.PadToken : generated[row, c];
                    codes[t, c] = code < 0 || code >= o.Codec.CodebookSize ? 0 : code;
                }
            return codes;
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    // Rows [start, start + count) of the generated tokens for both branches, [2, count, channels].
    private int[,,] Branches(int[,] generated, int start, int count)
    {
        int channels = generated.GetLength(1);
        var tokens = new int[2, count, channels];
        for (int b = 0; b < 2; b++)
            for (int t = 0; t < count; t++)
                for (int c = 0; c < channels; c++)
                {
                    int token = generated[start + t, c];
                    if (token < 0) throw new InvalidOperationException($"Step {start + t} of channel {c} was read before it was written.");
                    tokens[b, t, c] = token;
                }
        return tokens;
    }

    /// <summary>
    /// The release's <c>_decoder_step</c> on one step's logits <c>[2, 1, channels, vocabulary]</c> (unconditional first):
    /// the guided logits <c>cond + s · (cond − uncond)</c> pick each channel's best k codes; the conditional logits outside
    /// them are removed, tokens past the end token are removed everywhere and the end token from channels 1 on; then
    /// <c>_sample_next_token</c>: greedy at temperature 0, otherwise the logits over the temperature, the end token removed
    /// unless it is the best (and everything below it removed when it is), top-k, top-p, and a draw from the softmax.
    /// </summary>
    internal int[] SampleFrame(T[] logits, Random random)
    {
        var o = _options;
        int channels = o.Channels, vocab = o.AudioVocabSize, end = o.EndToken;
        if (StepLogits is not null)
        {
            var all = logits.Select(v => Ops.ToDouble(v)).ToArray();
            StepLogits(all.Take(channels * vocab).ToArray(), all.Skip(channels * vocab).ToArray());
        }
        var frame = new int[channels];
        for (int c = 0; c < channels; c++)
        {
            var uncond = new double[vocab];
            var cond = new double[vocab];
            for (int v = 0; v < vocab; v++)
            {
                uncond[v] = Ops.ToDouble(logits[c * vocab + v]);
                cond[v] = Ops.ToDouble(logits[(channels + c) * vocab + v]);
            }
            var guided = cond.Select((x, v) => x + o.CfgScale * (x - uncond[v])).ToArray();
            var keep = TopK(guided, o.CfgFilterTopK);
            var scores = new double[vocab];
            for (int v = 0; v < vocab; v++)
                scores[v] = !keep[v] || v > end || (c > 0 && v >= end) ? double.NegativeInfinity : cond[v];
            frame[c] = SampleNext(scores, random);
        }
        return frame;
    }

    // The indices of the k largest values (torch.topk; the first of equal values first).
    private static bool[] TopK(double[] values, int k)
    {
        var order = Enumerable.Range(0, values.Length).OrderByDescending(i => values[i]).ThenBy(i => i).Take(Math.Min(k, values.Length));
        var keep = new bool[values.Length];
        foreach (int i in order) keep[i] = true;
        return keep;
    }

    private static int ArgMax(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    // _sample_next_token for one channel.
    private int SampleNext(double[] scores, Random random)
    {
        var o = _options;
        if (o.Temperature == 0) return ArgMax(scores);
        int end = o.EndToken;
        var x = scores.Select(v => v / o.Temperature).ToArray();
        if (ArgMax(x) != end) x[end] = double.NegativeInfinity;
        else for (int v = 0; v < end; v++) x[v] = double.NegativeInfinity;
        var keep = TopK(x, o.CfgFilterTopK);
        for (int v = 0; v < x.Length; v++)
            if (!keep[v]) x[v] = double.NegativeInfinity;
        var probabilities = Softmax(x);
        if (o.TopP < 1.0)
        {
            // Remove the codes past the nucleus: those whose cumulative probability, in descending order, exceeds top-p,
            // shifted by one so the code crossing it stays.
            var order = Enumerable.Range(0, x.Length).OrderByDescending(i => probabilities[i]).ThenBy(i => i).ToArray();
            double cumulative = 0;
            bool previousOver = false;
            for (int r = 0; r < order.Length; r++)
            {
                cumulative += probabilities[order[r]];
                if (r > 0 && previousOver) x[order[r]] = double.NegativeInfinity;
                previousOver = cumulative > o.TopP;
            }
            probabilities = Softmax(x);
        }
        double u = random.NextDouble(), sum = 0;
        int last = 0;
        for (int v = 0; v < probabilities.Length; v++)
        {
            if (probabilities[v] <= 0) continue;
            last = v;
            sum += probabilities[v];
            if (u < sum) return v;
        }
        return last;
    }

    private static double[] Softmax(double[] x)
    {
        double max = x.Max();
        var e = x.Select(v => double.IsNegativeInfinity(v) ? 0.0 : Math.Exp(v - max)).ToArray();
        double total = e.Sum();
        return e.Select(v => v / total).ToArray();
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a script's byte ids (<see cref="PreprocessText"/>); the waveform
    /// <see cref="Synthesize"/> makes of it.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        if (input.Length == 0) throw new ArgumentException("There is no text.", nameof(input));
        return Synthesize(TextOf(input));
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
}
