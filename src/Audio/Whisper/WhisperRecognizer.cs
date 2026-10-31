using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Whisper;

/// <summary>
/// One Whisper checkpoint as a recognizer: the openai-whisper network, its transcription loop, the 30-second log-mel
/// input every forward reads (<c>pad_or_trim</c>), and teacher-forced training on transcript tokens (Radford et al. 2022,
/// section 2.3). The Whisper-family recognizers and the services built on Whisper weights share it.
/// </summary>
internal sealed class WhisperRecognizer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    public WhisperRecognizer(IEngine engine, List<LayerBase<T>> layers, WhisperDimensions dimensions, int seed)
    {
        _engine = engine;
        Network = new OpenAiWhisperNetwork<T>(engine, layers, dimensions, Tensors.Helpers.RandomHelper.CreateSeededRandom(seed));
        Transcriber = new OpenAiWhisperTranscriber<T>(engine, Network);
        Seed = seed;
    }

    public OpenAiWhisperNetwork<T> Network { get; }

    public OpenAiWhisperTranscriber<T> Transcriber { get; }

    public int Seed { get; }

    /// <summary>The log-mel window <c>[mels, 3000]</c> of a 16 kHz waveform padded or trimmed to 30 seconds.</summary>
    public Tensor<T> Window(double[] audio)
    {
        int mels = Network.Dimensions.Mels, frames = OpenAiWhisperTranscriber<T>.ChunkFrames;
        var samples = new double[OpenAiWhisperTranscriber<T>.ChunkSamples];
        Array.Copy(audio, samples, Math.Min(audio.Length, samples.Length));
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(samples, mels);
        var window = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < Math.Min(frames, mel.GetLength(1)); f++) window[m, f] = NumOps.FromDouble(mel[m, f]);
        return window;
    }

    /// <summary>The decoder's start: start of transcript, language, transcribe, no timestamps.</summary>
    public int[] Prompt(string language) =>
        Transcriber.SotSequence(language).Concat(new[] { Transcriber.NoTimestamps }).ToArray();

    /// <summary>The decoder's logits <c>[prompt, vocabulary]</c> over the start sequence; the last row is the first
    /// token's distribution.</summary>
    public Tensor<T> PromptLogits(Tensor<T> window, string language) => Network.Logits(Prompt(language), Network.Encode(window));

    /// <summary>
    /// The teacher-forced cross-entropy of a transcript: the decoder reads the start sequence and the transcript, and
    /// predicts each transcript token and the closing end of text (section 2.3: training predicts the transcript tokens
    /// after the special tokens).
    /// </summary>
    public Tensor<T> TranscriptLoss(Tensor<T> window, IReadOnlyList<int> transcript, string language)
    {
        var prompt = Prompt(language);
        var input = prompt.Concat(transcript).ToArray();
        var logits = Network.Logits(input, Network.Encode(window));
        int predicted = transcript.Count + 1, vocabulary = Network.Dimensions.Vocabulary;
        var targets = _engine.TensorSlice(logits, new[] { prompt.Length - 1, 0 }, new[] { predicted, vocabulary });
        var weights = new Tensor<T>(new[] { predicted, vocabulary });
        for (int i = 0; i < predicted; i++) weights[i, i < transcript.Count ? transcript[i] : Transcriber.Eot] = NumOps.FromDouble(-1.0 / predicted);
        return _engine.ReduceSum(_engine.TensorMultiply(_engine.TensorLogSoftmax(targets, axis: -1), weights), new[] { 0, 1 }, keepDims: false);
    }

    /// <summary>
    /// The transcript tokens a training target names: token ids <c>[length]</c>, or one distribution per token
    /// <c>[length, vocabulary]</c> (the shape <see cref="PromptLogits"/> returns), read as its most likely tokens.
    /// </summary>
    public int[] TranscriptTokens(Tensor<T> target)
    {
        int vocabulary = Network.Dimensions.Vocabulary;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            int rows = target.Length / vocabulary;
            var span = target.Data.Span;
            var tokens = new int[rows];
            for (int r = 0; r < rows; r++)
            {
                int best = 0;
                for (int v = 1; v < vocabulary; v++)
                    if (NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best])) best = v;
                tokens[r] = Math.Min(best, Transcriber.Eot - 1);
            }
            return tokens;
        }
        var ids = new int[target.Length];
        var values = target.Data.Span;
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(values[i]));
            if (value < 0 || value >= Transcriber.Eot)
                throw new ArgumentException($"Transcript token {value} is not a text token (0 to {Transcriber.Eot - 1}).", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    /// <summary>openai-whisper's <c>transcribe</c> of a 16 kHz waveform.</summary>
    public TranscriptionResult<T> Transcribe(double[] audio, string? language, bool wordTimestamps, IReadOnlyList<double> temperatures)
    {
        using var _ = new NoGradScope<T>();
        var result = Transcriber.TranscribeAudio(audio, temperatures, wordTimestamps, language, Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        double confidence = result.Segments.Count == 0 ? 0.0 : result.Segments.Average(s => Math.Exp(s.AvgLogprob));
        return new TranscriptionResult<T>
        {
            Text = result.Text.Trim(),
            Language = result.Language,
            Confidence = NumOps.FromDouble(confidence),
            DurationSeconds = audio.Length / (double)OpenAiWhisperTranscriber<T>.SampleRate,
            Segments = result.Segments
                .SelectMany(s => wordTimestamps && s.Words.Count > 0
                    ? s.Words.Select(w => new TranscriptionSegment<T> { Text = w.Word.Trim(), StartTime = w.Start, EndTime = w.End, Confidence = NumOps.FromDouble(w.Probability) })
                    : new[] { new TranscriptionSegment<T> { Text = s.Text.Trim(), StartTime = s.Start, EndTime = s.End, Confidence = NumOps.FromDouble(Math.Exp(s.AvgLogprob)) } })
                .ToList(),
        };
    }

    /// <summary>The language probabilities of a waveform's first 30 seconds (<c>detect_language</c>).</summary>
    public Dictionary<string, double> DetectLanguage(double[] audio)
    {
        using var _ = new NoGradScope<T>();
        return Transcriber.DetectLanguage(Network.Encode(Window(audio)));
    }
}
