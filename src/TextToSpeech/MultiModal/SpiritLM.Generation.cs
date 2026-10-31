using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

public partial class SpiritLM<T>
{
    private TResult Inference<TResult>(Func<TResult> run)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        var outer = _samplingRandom;
        SetTrainingMode(false);
        // Each call samples from a generator seeded afresh, so a call depends on its input and the seed only.
        _samplingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);
        try
        {
            using var _ = new NoGradScope<T>();
            return run();
        }
        finally
        {
            _samplingRandom = outer;
            SetTrainingMode(wasTraining);
        }
    }

    /// <summary>
    /// The prompt of interleaved inputs (the release's <c>_build_prompt</c>): each run of a modality opened by its marker,
    /// speech as its token string, and the output modality's marker when the last input differs.
    /// </summary>
    public string Prompt(IReadOnlyList<SpiritLmContent<T>> inputs, SpiritLmModality output)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        var sb = new System.Text.StringBuilder();
        char previous = ' ';
        foreach (var input in inputs)
        {
            if (input.IsSpeech)
            {
                if (previous != 's') sb.Append(SpiritLmTokens.SpeechPrefix);
                sb.Append(input.SpeechTokens ?? SpeechString(input.Audio!));
                previous = 's';
            }
            else
            {
                if (previous != 't') sb.Append(SpiritLmTokens.TextPrefix);
                sb.Append(input.Text ?? throw new ArgumentException("An input has neither text nor speech."));
                previous = 't';
            }
        }
        if (output == SpiritLmModality.Text && previous != 't') sb.Append(SpiritLmTokens.TextPrefix);
        else if (output == SpiritLmModality.Speech && previous != 's') sb.Append(SpiritLmTokens.SpeechPrefix);
        return sb.ToString();
    }

    private double[] Row(Tensor<T> logits)
    {
        var values = new double[_options.VocabSize];
        int offset = logits.Length - values.Length;
        for (int i = 0; i < values.Length; i++) values[i] = Ops.ToDouble(logits.GetFlat(offset + i));
        return values;
    }

    private Tensor<T> Embed(IReadOnlyList<int> ids)
    {
        var table = new int[1, ids.Count];
        for (int i = 0; i < ids.Count; i++)
        {
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(nameof(ids), $"Token ids lie in [0, {_options.VocabSize}).");
            table[0, i] = ids[i];
        }
        return LanguageModel.Embed(table);
    }

    /// <summary>
    /// Continues prompt ids by up to <paramref name="maxNewTokens"/> (and the context of
    /// <see cref="TtsModelOptions.MaxTextLength"/>), until <c>&lt;/s&gt;</c>, with the forbidden ids at −∞ (transformers'
    /// <c>bad_words_ids</c>) and transformers' sampling (temperature, top-k, top-p; greedy at temperature 0).
    /// </summary>
    internal List<int> GenerateIds(int[] prompt, HashSet<int> forbidden, int maxNewTokens, Random random)
    {
        if (prompt.Length >= _options.MaxTextLength) throw new ArgumentException($"The prompt's {prompt.Length} tokens fill the {_options.MaxTextLength}-token context.");
        var cache = new LlamaCache<T>();
        var (_, logits) = LanguageModel.Forward(Embed(prompt), null, null, cache, lastLogitsOnly: true);
        int end = Tokenizer.EosId;
        var generated = new List<int>();
        while (generated.Count < maxNewTokens && cache.Length < _options.MaxTextLength)
        {
            var row = Row(logits!);
            foreach (int id in forbidden) if (id < row.Length) row[id] = double.NegativeInfinity;
            int next = _options.Temperature <= 0 ? NucleusSampling.Greedy(row) : NucleusSampling.Sample(row, _options.Temperature, _options.TopK, _options.TopP, null, random);
            generated.Add(next);
            if (next == end) break;
            (_, logits) = LanguageModel.Forward(Embed(new[] { next }), null, null, cache, lastLogitsOnly: true);
        }
        return generated;
    }

    /// <summary>The paper's limit (App. D) for a generation of <paramref name="output"/> after inputs ending in
    /// <paramref name="lastInputIsSpeech"/>.</summary>
    internal int MaxNewTokens(SpiritLmModality output, bool lastInputIsSpeech) => output switch
    {
        SpiritLmModality.Text => _options.MaxNewTextTokens,
        _ => lastInputIsSpeech ? _options.MaxNewSpeechTokensFromSpeech : _options.MaxNewSpeechTokensFromText,
    };

    /// <summary>
    /// Generates <paramref name="output"/> after interleaved text and speech (the release's <c>Spiritlm.generate</c>): the
    /// prompt built, the other modality's tokens forbidden unless the output is arbitrary, and the result decoded — text as
    /// it is, speech voiced. An arbitrary generation is split into its text and speech runs; a speech run continuing the
    /// prompt's last speech is voiced with that speech as context and the context's share of the audio cut, and speech runs
    /// of fewer than 25 units are dropped, as the release does.
    /// </summary>
    public IReadOnlyList<SpiritLmContent<T>> Generate(IReadOnlyList<SpiritLmContent<T>> inputs, SpiritLmModality output)
    {
        ThrowIfDisposed();
        if (inputs is null || inputs.Count == 0) throw new ArgumentException("Give at least one input.", nameof(inputs));
        string prompt = Prompt(inputs, output);
        return Inference(() => GenerateFrom(prompt, output, MaxNewTokens(output, inputs[inputs.Count - 1].IsSpeech)));
    }

    /// <summary>Generates after a prompt already written as a string (<c>[Text]…</c>, <c>[Speech][Hu…]…</c>).</summary>
    public IReadOnlyList<SpiritLmContent<T>> Generate(string prompt, SpiritLmModality output, int maxNewTokens)
    {
        ThrowIfDisposed();
        if (prompt is null) throw new ArgumentNullException(nameof(prompt));
        if (maxNewTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
        return Inference(() => GenerateFrom(prompt, output, maxNewTokens));
    }

    private IReadOnlyList<SpiritLmContent<T>> GenerateFrom(string prompt, SpiritLmModality output, int maxNewTokens)
    {
        var ids = GenerateIds(Encode(prompt), Forbidden(output), maxNewTokens, _samplingRandom!);
        string generated = Decode(ids);
        string fresh = generated.Trim();
        if (output == SpiritLmModality.Text) return new[] { SpiritLmContent<T>.FromText(fresh) };
        if (output == SpiritLmModality.Speech) return new[] { new SpiritLmContent<T> { SpeechTokens = fresh, Audio = Speak(fresh) } };

        bool continuesSpeech = SpiritLmTokens.EndsWithSpeechToken(prompt) && SpiritLmTokens.StartsWithSpeechToken(fresh);
        int contextStart = 0;
        string content = fresh;
        if (continuesSpeech)
        {
            contextStart = SpiritLmTokens.LastSpeechStart(prompt) ?? 0;
            content = prompt.Substring(contextStart) + generated;
        }
        var results = new List<SpiritLmContent<T>>();
        var chunks = SpiritLmTokens.SplitSpeechAndText(content);
        for (int i = 0; i < chunks.Count; i++)
        {
            var (chunk, isSpeech) = chunks[i];
            if (!isSpeech)
            {
                results.Add(SpiritLmContent<T>.FromText(chunk));
                continue;
            }
            // The release counts split("[Hu") pieces: one more than the units.
            int pieces = chunk.Split(new[] { "[Hu" }, StringSplitOptions.None).Length;
            if (i == 0 && continuesSpeech)
            {
                int promptUnits = prompt.Substring(contextStart).Split(new[] { "[Hu" }, StringSplitOptions.None).Length - 1;
                if (pieces - promptUnits < 25) continue;
                var audio = Speak(chunk);
                int cut = (int)Math.Ceiling(audio.Length * ((double)promptUnits / pieces));
                var kept = new Tensor<T>(new[] { audio.Length - cut });
                for (int s = 0; s < kept.Length; s++) kept[s] = audio[cut + s];
                results.Add(new SpiritLmContent<T> { SpeechTokens = chunk, Audio = kept });
            }
            else
            {
                if (i > 0 && pieces < 25) continue;
                results.Add(new SpiritLmContent<T> { SpeechTokens = chunk, Audio = Speak(chunk) });
            }
        }
        return results;
    }

    // ---------------------------------------------------------------- TTS

    /// <summary>The speech SpiritLM writes after <c>[Text]{text}[Speech]</c> (text-to-speech continuation, App. D's 200
    /// tokens).</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
    }

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    private int[] TokenIds(Tensor<T> tensor, string name)
    {
        var ids = new int[tensor.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(tensor.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(name, $"Token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is the text's token ids; the result is the 16 kHz speech SpiritLM writes after
    /// it.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        string text = Decode(TokenIds(input, nameof(input)));
        var result = Generate(new[] { SpiritLmContent<T>.FromText(text) }, SpiritLmModality.Speech);
        return result[0].Audio ?? new Tensor<T>(new[] { 0 });
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
}
