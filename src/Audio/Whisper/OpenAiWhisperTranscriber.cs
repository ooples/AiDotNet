using System.Globalization;
using System.IO.Compression;
using System.Text;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.Audio.Whisper;

/// <summary>A word with its time span and probability (openai-whisper's <c>WordTiming</c> and segment words).</summary>
internal sealed record WhisperWordTiming(string Word, int[] Tokens, double Start, double End, double Probability);

/// <summary>A transcription segment (openai-whisper's segment dictionary).</summary>
internal sealed class WhisperTranscriptSegment
{
    public int Id;
    public int Seek;
    public double Start;
    public double End;
    public string Text = "";
    public List<int> Tokens = new();
    public double Temperature;
    public double AvgLogprob;
    public double CompressionRatio;
    public double NoSpeechProb;
    public List<WhisperWordTiming> Words = new();
}

/// <summary>A transcription (openai-whisper's <c>transcribe</c> result).</summary>
internal sealed record WhisperTranscription(string Text, string Language, IReadOnlyList<WhisperTranscriptSegment> Segments);

/// <summary>A decoding's result (openai-whisper's <c>DecodingResult</c>).</summary>
internal sealed record WhisperDecodingResult(List<int> Tokens, string Text, string Language, double AvgLogprob, double NoSpeechProb, double Temperature,
    double CompressionRatio);

/// <summary>
/// openai-whisper 20240930's transcription for a <see cref="OpenAiWhisperNetwork{T}"/>: the log-mel spectrogram
/// (<c>audio.py</c>), its tokenizer conventions (<c>tokenizer.py</c>), decoding with the blank, token and timestamp
/// rules and temperature fallback (<c>decoding.py</c>), word timing by dynamic time warping over the alignment heads'
/// cross-attention (<c>timing.py</c>) and the 30-second window loop (<c>transcribe.py</c>, without the
/// hallucination-silence and clip-timestamp options, which keep their defaults).
/// </summary>
internal sealed class OpenAiWhisperTranscriber<T>
{
    public const int SampleRate = 16000;
    public const int FftSize = 400;
    public const int HopLength = 160;
    public const int ChunkLength = 30;
    public const int ChunkSamples = ChunkLength * SampleRate;
    public const int ChunkFrames = ChunkSamples / HopLength;
    public const int FramesPerSecond = SampleRate / HopLength;
    public const int TokensPerSecond = SampleRate / (HopLength * 2);

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly OpenAiWhisperNetwork<T> _model;
    private readonly IEngine _engine;
    private readonly TiktokenEncoding _encoding;
    private readonly int _languages;

    public OpenAiWhisperTranscriber(IEngine engine, OpenAiWhisperNetwork<T> model)
    {
        _engine = engine;
        _model = model;
        if (!model.Dimensions.IsMultilingual) throw new NotSupportedException("Only multilingual Whisper models are supported.");
        _languages = model.Dimensions.NumLanguages;
        _encoding = WhisperTiktoken.Multilingual(_languages);
        Eot = Special("<|endoftext|>");
        Sot = Special("<|startoftranscript|>");
        Translate = Special("<|translate|>");
        Transcribe = Special("<|transcribe|>");
        SotLm = Special("<|startoflm|>");
        SotPrev = Special("<|startofprev|>");
        NoSpeech = Special("<|nospeech|>");
        NoTimestamps = Special("<|notimestamps|>");
        TimestampBegin = Special("<|0.00|>");
    }

    private int Special(string token) => _encoding.SpecialTokenId(token);

    public int Eot { get; }
    public int Sot { get; }
    public int Translate { get; }
    public int Transcribe { get; }
    public int SotLm { get; }
    public int SotPrev { get; }
    public int NoSpeech { get; }
    public int NoTimestamps { get; }
    public int TimestampBegin { get; }

    public IReadOnlyList<string> LanguageCodes => WhisperTiktoken.LanguageCodes.Take(_languages).ToList();

    public int LanguageToken(string code) => Special($"<|{code}|>");

    public int[] SotSequence(string language) => new[] { Sot, LanguageToken(language), Transcribe };

    public int[] Encode(string text) => _encoding.Encode(text, allowSpecial: false);

    /// <summary>The text of tokens below the timestamps (<c>Tokenizer.decode</c>).</summary>
    public string Decode(IEnumerable<int> tokens) => _encoding.Decode(tokens.Where(t => t < TimestampBegin));

    public string DecodeWithTimestamps(IEnumerable<int> tokens) => _encoding.Decode(tokens);

    private int[]? _nonSpeech;

    /// <summary><c>Tokenizer.non_speech_tokens</c>.</summary>
    public int[] NonSpeechTokens()
    {
        if (_nonSpeech is not null) return _nonSpeech;
        var symbols = "\"#()*+/:;<=>@[\\]^_`{|}~\u300c\u300d\u300e\u300f".Select(c => c.ToString()).ToList();
        symbols.AddRange("<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} \u266a\u266a \u266a\u266a\u266a".Split(' '));
        var miscellaneous = "\u2669\u266a\u266b\u266c\u266d\u266e\u266f".Select(c => c.ToString()).ToList();
        var result = new SortedSet<int> { _encoding.Encode(" -", false)[0], _encoding.Encode(" '", false)[0] };
        foreach (var symbol in symbols.Concat(miscellaneous))
            foreach (var tokens in new[] { _encoding.Encode(symbol, false), _encoding.Encode(" " + symbol, false) })
                if (tokens.Length == 1 || miscellaneous.Contains(symbol)) result.Add(tokens[0]);
        return _nonSpeech = result.ToArray();
    }

    private const string AsciiPunctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

    /// <summary><c>Tokenizer.split_to_word_tokens</c>: per character for zh, ja, th, lo, my and yue, else at spaces and
    /// punctuation.</summary>
    public (List<string> Words, List<List<int>> Tokens) SplitToWordTokens(IReadOnlyList<int> tokens, string language)
    {
        if (language is "zh" or "ja" or "th" or "lo" or "my" or "yue") return SplitOnUnicode(tokens);
        var (subwords, subwordTokens) = SplitOnUnicode(tokens);
        var words = new List<string>();
        var wordTokens = new List<List<int>>();
        for (int i = 0; i < subwords.Count; i++)
        {
            string subword = subwords[i];
            bool special = subwordTokens[i][0] >= Eot;
            bool withSpace = subword.StartsWith(" ", StringComparison.Ordinal);
            bool punctuation = AsciiPunctuation.Contains(PythonStrip(subword));
            if (special || withSpace || punctuation || words.Count == 0)
            {
                words.Add(subword);
                wordTokens.Add(new List<int>(subwordTokens[i]));
            }
            else
            {
                words[words.Count - 1] += subword;
                wordTokens[wordTokens.Count - 1].AddRange(subwordTokens[i]);
            }
        }
        return (words, wordTokens);
    }

    private (List<string> Words, List<List<int>> Tokens) SplitOnUnicode(IReadOnlyList<int> tokens)
    {
        string full = DecodeWithTimestamps(tokens);
        const char replacement = '\uFFFD';
        var words = new List<string>();
        var wordTokens = new List<List<int>>();
        var current = new List<int>();
        int offset = 0;
        foreach (int token in tokens)
        {
            current.Add(token);
            string decoded = DecodeWithTimestamps(current);
            int at = decoded.IndexOf(replacement);
            if (at < 0 || (offset + at < full.Length && full[offset + at] == replacement))
            {
                words.Add(decoded);
                wordTokens.Add(current);
                current = new List<int>();
                offset += decoded.Length;
            }
        }
        return (words, wordTokens);
    }

    private static string PythonStrip(string s) => s.Trim(' ', '\t', '\n', '\r', '\v', '\f');

    // ---------------------------------------------------------------- audio

    private static readonly Lazy<(float[] Mel80, float[] Mel128)> Filters = new(() =>
    {
        using var stream = typeof(OpenAiWhisperTranscriber<T>).Assembly.GetManifestResourceStream("AiDotNet.Audio.Whisper.whisper_mel_filters.gz")
            ?? throw new InvalidDataException("The embedded resource whisper_mel_filters.gz is missing.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new BinaryReader(gzip);
        var mel80 = new float[80 * 201];
        var mel128 = new float[128 * 201];
        for (int i = 0; i < mel80.Length; i++) mel80[i] = reader.ReadSingle();
        for (int i = 0; i < mel128.Length; i++) mel128[i] = reader.ReadSingle();
        return (mel80, mel128);
    }, isThreadSafe: true);

    /// <summary><c>log_mel_spectrogram(audio, n_mels, padding)</c>: a centred, reflect-padded 400-point STFT (hop 160,
    /// periodic Hann), its power without the last frame, the mel filter bank, log10 above 1e-10, floored 8 below the
    /// maximum, then (x + 4) / 4: <c>[mels, frames]</c>.</summary>
    public static double[,] LogMelSpectrogram(double[] audio, int mels, int padding = 0)
    {
        if (mels != 80 && mels != 128) throw new ArgumentException($"Unsupported n_mels: {mels}", nameof(mels));
        var x = new double[audio.Length + padding];
        Array.Copy(audio, x, audio.Length);
        int half = FftSize / 2, length = x.Length;
        if (length <= half) throw new ArgumentException("The audio is shorter than the STFT's reflect padding.", nameof(audio));
        var padded = new double[length + 2 * half];
        for (int i = 0; i < length; i++) padded[half + i] = x[i];
        for (int i = 0; i < half; i++)
        {
            padded[half - 1 - i] = x[i + 1];
            padded[half + length + i] = x[length - 2 - i];
        }
        int frames = 1 + length / HopLength, bins = FftSize / 2 + 1;
        var window = new double[FftSize];
        for (int n = 0; n < FftSize; n++) window[n] = 0.5 - 0.5 * Math.Cos(2 * Math.PI * n / FftSize);
        var cos = new double[bins * FftSize];
        var sin = new double[bins * FftSize];
        for (int k = 0; k < bins; k++)
            for (int n = 0; n < FftSize; n++)
            {
                double angle = 2 * Math.PI * ((long)k * n % FftSize) / FftSize;
                cos[k * FftSize + n] = Math.Cos(angle);
                sin[k * FftSize + n] = Math.Sin(angle);
            }
        var filters = mels == 80 ? Filters.Value.Mel80 : Filters.Value.Mel128;
        int used = frames - 1;
        var power = new double[bins];
        var frame = new double[FftSize];
        var output = new double[mels, used];
        double max = double.NegativeInfinity;
        for (int f = 0; f < used; f++)
        {
            bool silent = true;
            for (int n = 0; n < FftSize; n++)
            {
                frame[n] = padded[f * HopLength + n] * window[n];
                if (frame[n] != 0.0) silent = false;
            }
            for (int k = 0; k < bins; k++)
            {
                // A frame of zeros (the padding of a short window) has zero power at every bin.
                if (silent) { power[k] = 0.0; continue; }
                double re = 0, im = 0;
                for (int n = 0; n < FftSize; n++)
                {
                    re += frame[n] * cos[k * FftSize + n];
                    im -= frame[n] * sin[k * FftSize + n];
                }
                power[k] = re * re + im * im;
            }
            for (int m = 0; m < mels; m++)
            {
                double sum = 0;
                for (int k = 0; k < bins; k++) sum += filters[m * bins + k] * power[k];
                double log = Math.Log10(Math.Max(sum, 1e-10));
                output[m, f] = log;
                if (log > max) max = log;
            }
        }
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < used; f++) output[m, f] = (Math.Max(output[m, f], max - 8.0) + 4.0) / 4.0;
        return output;
    }

    // pad_or_trim(mel[:, start:start + size], N_FRAMES)
    private static Tensor<T> Window(double[,] mel, int start, int size)
    {
        int mels = mel.GetLength(0);
        var window = new Tensor<T>(new[] { mels, ChunkFrames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < Math.Min(size, ChunkFrames); f++)
                if (start + f < mel.GetLength(1)) window[m, f] = NumOps.FromDouble(mel[m, start + f]);
        return window;
    }

    // ---------------------------------------------------------------- language detection and decoding

    /// <summary><c>detect_language</c>: the language tokens' probabilities after the start token.</summary>
    public Dictionary<string, double> DetectLanguage(Tensor<T> audioFeatures)
    {
        var logits = _model.Logits(new[] { Sot }, audioFeatures);
        var codes = LanguageCodes;
        var allowed = codes.Select(LanguageToken).ToArray();
        var values = allowed.Select(t => NumOps.ToDouble(logits[0, t])).ToArray();
        double max = values.Max();
        double sum = values.Sum(v => Math.Exp(v - max));
        var probs = new Dictionary<string, double>(StringComparer.Ordinal);
        for (int i = 0; i < codes.Count; i++) probs[codes[i]] = Math.Exp(values[i] - max) / sum;
        return probs;
    }

    private static string ArgmaxLanguage(Dictionary<string, double> probs)
    {
        string best = "";
        double value = double.NegativeInfinity;
        foreach (var pair in probs)
            if (pair.Value > value)
            {
                value = pair.Value;
                best = pair.Key;
            }
        return best;
    }

    /// <summary>The logit filters of a decoding task: blank suppression at the first step, the suppressed tokens, and the
    /// timestamp rules.</summary>
    internal void ApplyFilters(double[] logits, IReadOnlyList<int> tokens, int sampleBegin, int? maxInitialTimestampIndex)
    {
        // SuppressBlank
        if (tokens.Count == sampleBegin)
            foreach (int t in Encode(" ").Concat(new[] { Eot })) logits[t] = double.NegativeInfinity;
        // SuppressTokens
        foreach (int t in SuppressTokens()) logits[t] = double.NegativeInfinity;
        // ApplyTimestampRules
        logits[NoTimestamps] = double.NegativeInfinity;
        var seq = tokens.Skip(sampleBegin).ToList();
        bool lastWasTimestamp = seq.Count >= 1 && seq[seq.Count - 1] >= TimestampBegin;
        bool penultimateWasTimestamp = seq.Count < 2 || seq[seq.Count - 2] >= TimestampBegin;
        if (lastWasTimestamp)
        {
            if (penultimateWasTimestamp)
                for (int i = TimestampBegin; i < logits.Length; i++) logits[i] = double.NegativeInfinity;
            else
                for (int i = 0; i < Eot; i++) logits[i] = double.NegativeInfinity;
        }
        var timestamps = seq.Where(t => t >= TimestampBegin).ToList();
        if (timestamps.Count > 0)
        {
            int last = lastWasTimestamp && !penultimateWasTimestamp ? timestamps[timestamps.Count - 1] : timestamps[timestamps.Count - 1] + 1;
            for (int i = TimestampBegin; i < Math.Min(last, logits.Length); i++) logits[i] = double.NegativeInfinity;
        }
        if (tokens.Count == sampleBegin)
        {
            for (int i = 0; i < TimestampBegin; i++) logits[i] = double.NegativeInfinity;
            if (maxInitialTimestampIndex is int index)
                for (int i = TimestampBegin + index + 1; i < logits.Length; i++) logits[i] = double.NegativeInfinity;
        }
        var logprobs = LogSoftmax(logits);
        double timestampLogprob = LogSumExp(logprobs, TimestampBegin, logprobs.Length);
        double maxText = double.NegativeInfinity;
        for (int i = 0; i < TimestampBegin; i++) maxText = Math.Max(maxText, logprobs[i]);
        if (timestampLogprob > maxText)
            for (int i = 0; i < TimestampBegin; i++) logits[i] = double.NegativeInfinity;
    }

    private int[]? _suppress;

    // DecodingTask._get_suppress_tokens with suppress_tokens "-1".
    private int[] SuppressTokens()
    {
        if (_suppress is not null) return _suppress;
        var set = new SortedSet<int>(NonSpeechTokens()) { Transcribe, Translate, Sot, SotPrev, SotLm, NoSpeech };
        return _suppress = set.ToArray();
    }

    internal static double[] LogSoftmax(double[] logits)
    {
        double max = logits.Max();
        if (double.IsNegativeInfinity(max)) return logits.Select(_ => double.NaN).ToArray();
        double sum = 0;
        foreach (double v in logits) sum += Math.Exp(v - max);
        double log = max + Math.Log(sum);
        return logits.Select(v => v - log).ToArray();
    }

    private static double LogSumExp(double[] values, int from, int to)
    {
        double max = double.NegativeInfinity;
        for (int i = from; i < to; i++) max = Math.Max(max, values[i]);
        if (double.IsNegativeInfinity(max)) return double.NegativeInfinity;
        double sum = 0;
        for (int i = from; i < to; i++) sum += Math.Exp(values[i] - max);
        return max + Math.Log(sum);
    }

    /// <summary><c>DecodingTask.run</c> for one window with <c>beam_size</c> and <c>best_of</c> unset: greedy at
    /// temperature 0, a draw from logits / temperature otherwise.</summary>
    public WhisperDecodingResult Decode(Tensor<T> window, double temperature, string? language, IReadOnlyList<int>? prompt, Random random)
    {
        using var _ = new NoGradScope<T>();
        var d = _model.Dimensions;
        var features = window.Shape.Length == 3 ? window : _model.Encode(window);
        int sampleLength = d.TextContext / 2;
        var initial = new List<int>();
        if (prompt is not null && prompt.Count > 0)
        {
            initial.Add(SotPrev);
            int keep = d.TextContext / 2 - 1;
            initial.AddRange(prompt.Skip(Math.Max(0, prompt.Count - keep)));
        }
        int sotIndex = initial.Count;
        initial.AddRange(SotSequence(language ?? "en"));
        string chosen = language ?? "en";
        if (language is null)
        {
            var probs = DetectLanguage(features);
            chosen = ArgmaxLanguage(probs);
            initial[sotIndex + 1] = LanguageToken(chosen);
        }
        int sampleBegin = initial.Count;
        int maxInitial = (int)Math.Round(1.0 / (ChunkLength / (double)d.AudioContext), MidpointRounding.ToEven);
        var tokens = new List<int>(initial);
        var cache = new OpenAiWhisperNetwork<T>.Cache();
        double sumLogprob = 0, noSpeech = double.NaN;
        for (int i = 0; i < sampleLength; i++)
        {
            var input = i == 0 ? (IReadOnlyList<int>)tokens : new[] { tokens[tokens.Count - 1] };
            var logits = _model.Logits(input, features, cache);
            if (i == 0)
            {
                var atSot = Row(logits, sotIndex);
                noSpeech = Softmax(atSot)[NoSpeech];
            }
            var last = Row(logits, logits.Shape[0] - 1);
            ApplyFilters(last, tokens, sampleBegin, maxInitial);
            int next = temperature == 0 ? Argmax(last) : SampleFrom(last, temperature, random);
            var logprobs = LogSoftmax(last);
            if (tokens[tokens.Count - 1] != Eot) sumLogprob += logprobs[next];
            if (tokens[tokens.Count - 1] == Eot) next = Eot;
            tokens.Add(next);
            if (next == Eot || tokens.Count > d.TextContext) break;
        }
        tokens.Add(Eot);
        var sampled = tokens.Skip(sampleBegin).ToList();
        int end = sampled.IndexOf(Eot);
        var result = sampled.Take(end).ToList();
        string text = PythonStrip(Decode(result));
        return new WhisperDecodingResult(result, text, chosen, sumLogprob / (result.Count + 1), noSpeech, temperature, CompressionRatio(text));
    }

    private static double[] Row(Tensor<T> logits, int row)
    {
        int v = logits.Shape[1];
        var values = new double[v];
        for (int i = 0; i < v; i++) values[i] = NumOps.ToDouble(logits[row, i]);
        return values;
    }

    private static double[] Softmax(double[] logits)
    {
        double max = logits.Max();
        var e = logits.Select(x => Math.Exp(x - max)).ToArray();
        double sum = e.Sum();
        return e.Select(x => x / sum).ToArray();
    }

    private static int Argmax(double[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static int SampleFrom(double[] logits, double temperature, Random random)
    {
        var probs = Softmax(logits.Select(v => v / temperature).ToArray());
        double u = random.NextDouble(), cumulative = 0;
        int last = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            if (probs[i] <= 0) continue;
            last = i;
            cumulative += probs[i];
            if (u < cumulative) return i;
        }
        return last;
    }

    /// <summary><c>compression_ratio</c>: the text's UTF-8 length over its length after <c>zlib.compress</c>.</summary>
    internal static double CompressionRatio(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        return bytes.Length / (double)AiDotNet.Helpers.ZlibDeflate.Compress(bytes).Length;
    }

    // ---------------------------------------------------------------- word timing (timing.py)

    private static double[] MedianFilter(double[] x, int width)
    {
        int pad = width / 2, n = x.Length;
        if (n <= pad) return x;
        var padded = new double[n + 2 * pad];
        for (int i = 0; i < n; i++) padded[pad + i] = x[i];
        for (int i = 0; i < pad; i++)
        {
            padded[pad - 1 - i] = x[i + 1];
            padded[pad + n + i] = x[n - 2 - i];
        }
        var result = new double[n];
        var window = new double[width];
        for (int i = 0; i < n; i++)
        {
            Array.Copy(padded, i, window, 0, width);
            Array.Sort(window);
            result[i] = window[pad];
        }
        return result;
    }

    // dtw_cpu on -matrix: a float32 cost matrix, ties to the left move, then the backtrace.
    internal static (int[] TextIndices, int[] TimeIndices) Dtw(double[,] x)
    {
        int n = x.GetLength(0), m = x.GetLength(1);
        var cost = new float[n + 1, m + 1];
        var trace = new float[n + 1, m + 1];
        for (int i = 0; i <= n; i++)
            for (int j = 0; j <= m; j++)
            {
                cost[i, j] = float.PositiveInfinity;
                trace[i, j] = -1;
            }
        cost[0, 0] = 0;
        for (int j = 1; j <= m; j++)
            for (int i = 1; i <= n; i++)
            {
                float c0 = cost[i - 1, j - 1], c1 = cost[i - 1, j], c2 = cost[i, j - 1];
                float c;
                int t;
                if (c0 < c1 && c0 < c2) { c = c0; t = 0; }
                else if (c1 < c0 && c1 < c2) { c = c1; t = 1; }
                else { c = c2; t = 2; }
                cost[i, j] = (float)(x[i - 1, j - 1] + c);
                trace[i, j] = t;
            }
        for (int j = 0; j <= m; j++) trace[0, j] = 2;
        for (int i = 0; i <= n; i++) trace[i, 0] = 1;
        var path = new List<(int, int)>();
        int a = n, b = m;
        while (a > 0 || b > 0)
        {
            path.Add((a - 1, b - 1));
            float t = trace[a, b];
            if (t == 0) { a--; b--; }
            else if (t == 1) a--;
            else if (t == 2) b--;
            else throw new InvalidOperationException("Unexpected trace[i, j]");
        }
        path.Reverse();
        return (path.Select(p => p.Item1).ToArray(), path.Select(p => p.Item2).ToArray());
    }

    /// <summary><c>find_alignment</c>: each word's span from the alignment heads' cross-attention, normalized per frame
    /// across tokens, median-filtered (width 7) and warped against the tokens.</summary>
    public List<WhisperWordTiming> FindAlignment(string language, IReadOnlyList<int> textTokens, Tensor<T> window, int numFrames)
    {
        if (textTokens.Count == 0) return new List<WhisperWordTiming>();
        using var _ = new NoGradScope<T>();
        var sot = SotSequence(language);
        var tokens = sot.Concat(new[] { NoTimestamps }).Concat(textTokens).Concat(new[] { Eot }).ToList();
        var features = _model.Encode(window);
        var scores = new List<Tensor<T>>();
        var logits = _model.Logits(tokens, features, null, scores);
        var probabilities = new double[textTokens.Count];
        for (int i = 0; i < textTokens.Count; i++)
        {
            var row = Row(logits, sot.Length + i).Take(Eot).ToArray();
            probabilities[i] = Softmax(row)[textTokens[i]];
        }
        int rows = tokens.Count, frames = numFrames / 2;
        var matrix = new double[rows, frames];
        foreach (var (layer, head) in _model.AlignmentHeads)
        {
            var qk = scores[layer];
            // softmax over the frames kept, then per frame the z-score across tokens, then the median filter.
            var weights = new double[rows, frames];
            for (int r = 0; r < rows; r++)
            {
                double max = double.NegativeInfinity;
                for (int f = 0; f < frames; f++) max = Math.Max(max, NumOps.ToDouble(qk[head, r, f]));
                double sum = 0;
                for (int f = 0; f < frames; f++) sum += Math.Exp(NumOps.ToDouble(qk[head, r, f]) - max);
                for (int f = 0; f < frames; f++) weights[r, f] = Math.Exp(NumOps.ToDouble(qk[head, r, f]) - max) / sum;
            }
            for (int f = 0; f < frames; f++)
            {
                double mean = 0;
                for (int r = 0; r < rows; r++) mean += weights[r, f];
                mean /= rows;
                double variance = 0;
                for (int r = 0; r < rows; r++) variance += (weights[r, f] - mean) * (weights[r, f] - mean);
                double std = Math.Sqrt(variance / rows);
                for (int r = 0; r < rows; r++) weights[r, f] = (weights[r, f] - mean) / std;
            }
            for (int r = 0; r < rows; r++)
            {
                var line = new double[frames];
                for (int f = 0; f < frames; f++) line[f] = weights[r, f];
                var filtered = MedianFilter(line, 7);
                for (int f = 0; f < frames; f++) matrix[r, f] += filtered[f];
            }
        }
        int heads = _model.AlignmentHeads.Count;
        int first = sot.Length, count = rows - 1 - first;
        var cost = new double[count, frames];
        for (int r = 0; r < count; r++)
            for (int f = 0; f < frames; f++) cost[r, f] = -(matrix[first + r, f] / heads);
        var (textIndices, timeIndices) = Dtw(cost);
        var (words, wordTokens) = SplitToWordTokens(textTokens.Concat(new[] { Eot }).ToList(), language);
        if (wordTokens.Count <= 1) return new List<WhisperWordTiming>();
        var boundaries = new List<int> { 0 };
        int cumulative = 0;
        for (int i = 0; i < wordTokens.Count - 1; i++) boundaries.Add(cumulative += wordTokens[i].Count);
        var jumpTimes = new List<double>();
        for (int i = 0; i < textIndices.Length; i++)
            if (i == 0 || textIndices[i] != textIndices[i - 1]) jumpTimes.Add(timeIndices[i] / (double)TokensPerSecond);
        var result = new List<WhisperWordTiming>();
        for (int w = 0; w < boundaries.Count - 1; w++)
        {
            double start = jumpTimes[boundaries[w]], end = jumpTimes[boundaries[w + 1]];
            int from = boundaries[w], to = boundaries[w + 1];
            double probability = probabilities.Skip(from).Take(Math.Max(0, Math.Min(to, probabilities.Length) - from)).DefaultIfEmpty(double.NaN).Average();
            result.Add(new WhisperWordTiming(words[w], wordTokens[w].ToArray(), start, end, probability));
        }
        return result;
    }

    // merge_punctuations
    private static void MergePunctuations(List<(string Word, List<int> Tokens, double Start, double End, double Probability)> alignment,
        string prepended, string appended)
    {
        int i = alignment.Count - 2, j = alignment.Count - 1;
        while (i >= 0)
        {
            var previous = alignment[i];
            var following = alignment[j];
            if (previous.Word.StartsWith(" ", StringComparison.Ordinal) && prepended.Contains(PythonStrip(previous.Word)))
            {
                alignment[j] = (previous.Word + following.Word, previous.Tokens.Concat(following.Tokens).ToList(), following.Start, following.End, following.Probability);
                alignment[i] = ("", new List<int>(), previous.Start, previous.End, previous.Probability);
            }
            else
            {
                j = i;
            }
            i--;
        }
        i = 0;
        j = 1;
        while (j < alignment.Count)
        {
            var previous = alignment[i];
            var following = alignment[j];
            if (!previous.Word.EndsWith(" ", StringComparison.Ordinal) && appended.Contains(following.Word))
            {
                alignment[i] = (previous.Word + following.Word, previous.Tokens.Concat(following.Tokens).ToList(), previous.Start, previous.End, previous.Probability);
                alignment[j] = ("", new List<int>(), following.Start, following.End, following.Probability);
            }
            else
            {
                i = j;
            }
            j++;
        }
    }

    internal const string PrependPunctuations = "\"'\u201c\u00bf([{-";
    internal const string AppendPunctuations = "\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001";

    /// <summary><c>add_word_timestamps</c>: the window's words timed, punctuation merged, long words at sentence and
    /// segment boundaries shortened, and each segment's bounds reconciled with its words.</summary>
    public void AddWordTimestamps(List<WhisperTranscriptSegment> segments, string language, Tensor<T> window, int numFrames, double lastSpeechTimestamp)
    {
        if (segments.Count == 0) return;
        var perSegment = segments.Select(s => s.Tokens.Where(t => t < Eot).ToList()).ToList();
        var textTokens = perSegment.SelectMany(t => t).ToList();
        var timed = FindAlignment(language, textTokens, window, numFrames);
        var alignment = timed.Select(w => (w.Word, w.Tokens.ToList(), w.Start, w.End, w.Probability)).ToList();
        var durations = alignment.Select(t => t.End - t.Start).Where(v => v != 0).OrderBy(v => v).ToList();
        double median = durations.Count == 0 ? 0.0 : durations.Count % 2 == 1 ? durations[durations.Count / 2]
            : (durations[durations.Count / 2 - 1] + durations[durations.Count / 2]) / 2;
        median = Math.Min(0.7, median);
        double maxDuration = median * 2;
        if (durations.Count > 0)
        {
            const string sentenceEnds = ".\u3002!\uff01?\uff1f";
            for (int i = 1; i < alignment.Count; i++)
                if (alignment[i].End - alignment[i].Start > maxDuration)
                {
                    if (sentenceEnds.Contains(alignment[i].Word))
                        alignment[i] = (alignment[i].Word, alignment[i].Item2, alignment[i].Start, alignment[i].Start + maxDuration, alignment[i].Probability);
                    else if (sentenceEnds.Contains(alignment[i - 1].Word))
                        alignment[i] = (alignment[i].Word, alignment[i].Item2, alignment[i].End - maxDuration, alignment[i].End, alignment[i].Probability);
                }
        }
        MergePunctuations(alignment, PrependPunctuations, AppendPunctuations);
        double timeOffset = segments[0].Seek * HopLength / (double)SampleRate;
        int wordIndex = 0;
        for (int s = 0; s < segments.Count; s++)
        {
            var segment = segments[s];
            int saved = 0;
            var words = new List<WhisperWordTiming>();
            while (wordIndex < alignment.Count && saved < perSegment[s].Count)
            {
                var timing = alignment[wordIndex];
                if (timing.Word.Length > 0)
                    words.Add(new WhisperWordTiming(timing.Word, timing.Item2.ToArray(), PythonRound(timeOffset + timing.Start, 2), PythonRound(timeOffset + timing.End, 2),
                        timing.Probability));
                saved += timing.Item2.Count;
                wordIndex++;
            }
            if (words.Count > 0)
            {
                if (words[0].End - lastSpeechTimestamp > median * 4 && (words[0].End - words[0].Start > maxDuration
                    || (words.Count > 1 && words[1].End - words[0].Start > maxDuration * 2)))
                {
                    if (words.Count > 1 && words[1].End - words[1].Start > maxDuration)
                    {
                        double boundary = Math.Max(words[1].End / 2, words[1].End - maxDuration);
                        words[0] = words[0] with { End = boundary };
                        words[1] = words[1] with { Start = boundary };
                    }
                    words[0] = words[0] with { Start = Math.Max(0, words[0].End - maxDuration) };
                }
                if (segment.Start < words[0].End && segment.Start - 0.5 > words[0].Start)
                    words[0] = words[0] with { Start = Math.Max(0, Math.Min(words[0].End - median, segment.Start)) };
                else
                    segment.Start = words[0].Start;
                if (segment.End > words[words.Count - 1].Start && segment.End + 0.5 < words[words.Count - 1].End)
                    words[words.Count - 1] = words[words.Count - 1] with { End = Math.Max(words[words.Count - 1].Start + median, segment.End) };
                else
                    segment.End = words[words.Count - 1].End;
                lastSpeechTimestamp = segment.End;
            }
            segment.Words = words;
        }
    }

    /// <summary>Python's <c>round(value, digits)</c>: the exact binary value rounded half to even.</summary>
    internal static double PythonRound(double value, int digits) =>
        double.Parse(AiDotNet.TextToSpeech.Latest.OuteTTSPrompts.FormatFixed(value, digits), CultureInfo.InvariantCulture);

    // ---------------------------------------------------------------- transcribe

    /// <summary>
    /// <c>transcribe(audio, temperature, word_timestamps, language)</c> with the release's other defaults (compression
    /// ratio 2.4, log-probability −1, no-speech 0.6, conditioning on previous text, no initial prompt): 30-second windows
    /// decoded with temperature fallback, segmented at timestamp tokens, and, with word timestamps, each word timed and the
    /// next window started after the last word.
    /// </summary>
    /// <param name="audio">16 kHz mono audio.</param>
    public WhisperTranscription TranscribeAudio(double[] audio, IReadOnlyList<double> temperatures, bool wordTimestamps, string? language, Random random)
    {
        var d = _model.Dimensions;
        var mel = LogMelSpectrogram(audio, d.Mels, ChunkSamples);
        int contentFrames = mel.GetLength(1) - ChunkFrames;
        if (language is null)
        {
            using var _ = new NoGradScope<T>();
            language = ArgmaxLanguage(DetectLanguage(_model.Encode(Window(mel, 0, ChunkFrames))));
        }
        int inputStride = ChunkFrames / d.AudioContext;
        double timePrecision = inputStride * HopLength / (double)SampleRate;
        var allTokens = new List<int>();
        var allSegments = new List<WhisperTranscriptSegment>();
        int promptResetSince = 0, seek = 0;
        double lastSpeech = 0.0;
        while (seek < contentFrames)
        {
            double timeOffset = seek * HopLength / (double)SampleRate;
            int segmentSize = Math.Min(ChunkFrames, contentFrames - seek);
            var window = Window(mel, seek, segmentSize);
            double segmentDuration = segmentSize * HopLength / (double)SampleRate;
            var prompt = allTokens.Skip(promptResetSince).ToList();
            WhisperDecodingResult result = null!;
            foreach (double t in temperatures)
            {
                result = Decode(window, t, language, prompt, random);
                bool fallback = result.CompressionRatio > 2.4 || result.AvgLogprob < -1.0;
                if (result.NoSpeechProb > 0.6) fallback = false;
                if (!fallback) break;
            }
            var tokens = result.Tokens;
            bool skip = result.NoSpeechProb > 0.6;
            if (result.AvgLogprob > -1.0) skip = false;
            if (skip)
            {
                seek += segmentSize;
                continue;
            }
            var current = new List<WhisperTranscriptSegment>();
            WhisperTranscriptSegment NewSegment(double start, double end, List<int> sliced) => new()
            {
                Seek = seek, Start = start, End = end, Text = Decode(sliced.Where(x => x < Eot)), Tokens = sliced, Temperature = result.Temperature,
                AvgLogprob = result.AvgLogprob, CompressionRatio = result.CompressionRatio, NoSpeechProb = result.NoSpeechProb,
            };
            var isTimestamp = tokens.Select(x => x >= TimestampBegin).ToList();
            bool singleEnding = isTimestamp.Count >= 2 && !isTimestamp[isTimestamp.Count - 2] && isTimestamp[isTimestamp.Count - 1];
            var consecutive = new List<int>();
            for (int i = 0; i + 1 < isTimestamp.Count; i++) if (isTimestamp[i] && isTimestamp[i + 1]) consecutive.Add(i + 1);
            int previousSeek = seek;
            if (consecutive.Count > 0)
            {
                var slices = new List<int>(consecutive);
                if (singleEnding) slices.Add(tokens.Count);
                int lastSlice = 0;
                foreach (int slice in slices)
                {
                    var sliced = tokens.GetRange(lastSlice, slice - lastSlice);
                    int startPos = sliced[0] - TimestampBegin, endPos = sliced[sliced.Count - 1] - TimestampBegin;
                    current.Add(NewSegment(timeOffset + startPos * timePrecision, timeOffset + endPos * timePrecision, sliced));
                    lastSlice = slice;
                }
                if (singleEnding) seek += segmentSize;
                else seek += (tokens[lastSlice - 1] - TimestampBegin) * inputStride;
            }
            else
            {
                double duration = segmentDuration;
                var timestamps = tokens.Where(x => x >= TimestampBegin).ToList();
                if (timestamps.Count > 0 && timestamps[timestamps.Count - 1] != TimestampBegin)
                    duration = (timestamps[timestamps.Count - 1] - TimestampBegin) * timePrecision;
                current.Add(NewSegment(timeOffset, timeOffset + duration, new List<int>(tokens)));
                seek += segmentSize;
            }
            if (wordTimestamps)
            {
                AddWordTimestamps(current, language, window, segmentSize, lastSpeech);
                if (!singleEnding)
                {
                    double? lastWordEnd = GetEnd(current);
                    if (lastWordEnd is double end && end > timeOffset) seek = (int)Math.Round(end * FramesPerSecond, MidpointRounding.ToEven);
                }
                if (GetEnd(current) is double speechEnd) lastSpeech = speechEnd;
            }
            foreach (var segment in current)
                if (segment.Start == segment.End || PythonStrip(segment.Text).Length == 0)
                {
                    segment.Text = "";
                    segment.Tokens = new List<int>();
                    segment.Words = new List<WhisperWordTiming>();
                }
            foreach (var segment in current)
            {
                segment.Id = allSegments.Count;
                allSegments.Add(segment);
            }
            allTokens.AddRange(current.SelectMany(s => s.Tokens));
            if (result.Temperature > 0.5) promptResetSince = allTokens.Count;
        }
        return new WhisperTranscription(Decode(allTokens), language, allSegments);
    }

    private static double? GetEnd(List<WhisperTranscriptSegment> segments)
    {
        for (int s = segments.Count - 1; s >= 0; s--)
            if (segments[s].Words.Count > 0) return segments[s].Words[segments[s].Words.Count - 1].End;
        return segments.Count > 0 ? segments[segments.Count - 1].End : null;
    }
}
