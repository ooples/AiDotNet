using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Whisper;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.SpeechRecognition.AlibabaASR;

/// <summary>
/// SenseVoice-Large (FunAudioLLM, 2024): an autoregressive encoder-decoder for multilingual recognition, language
/// identification, speech emotion recognition and audio event detection, its tasks selected by decoder input tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SenseVoice-Large listens to up to 30 seconds of audio and writes the transcript one token at
/// a time. Before the transcript you can ask it to name the language, the speaker's emotion and the audio events, by
/// putting the matching task tokens at the start of what the decoder reads.</para>
/// <para>
/// The report (section 2.1) describes SenseVoice-Large as an autoregressive encoder-decoder that, "similar to Whisper",
/// specifies tasks by input tokens to the decoder (LID, SER and AED), with 1,587M parameters; the CosyVoice report builds
/// its S3 tokenizer from the first six layers of this encoder, which the released tokenizer shows to be Whisper-large's
/// (128 mel bands, width 1,280, 20 heads). The network is therefore openai-whisper's
/// (<see cref="OpenAiWhisperNetwork{T}"/>) at the large geometry: a 30-second log-mel window, two convolutions, 32
/// pre-norm encoder layers, and 32 decoder layers with cross-attention, tied to the token embedding. The decoder reads the
/// start of transcript followed by the selected task tokens and predicts the task outputs and the transcript, closed by
/// end of text; training minimizes their teacher-forced cross-entropy. The vocabulary is unpublished: its size is
/// configurable, with the five special tokens at its top (see <see cref="SenseVoiceLargeOptions.VocabSize"/>). The report
/// states no optimizer; AdamW at 1e-4 is the default.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.SpeechRecognition,
///     inputSize: 480000, outputSize: 51866);
/// var model = new SenseVoiceLarge&lt;float&gt;(architecture);
/// var text = model.Transcribe(waveform).Text;
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs", "https://arxiv.org/abs/2407.04051", Year = 2024, Authors = "An et al.")]
public partial class SenseVoiceLarge<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>, ITrainingObjectiveProvider<T>
{
    /// <inheritdoc />
    /// <remarks>The vocabulary, special tokens included.</remarks>
    protected override int OutputFeatureWidth => _options.VocabSize;

    private readonly SenseVoiceLargeOptions _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    // The encoder-decoder on the native path; null in ONNX mode or with custom layers.
    private OpenAiWhisperNetwork<T>? _network;
    internal OpenAiWhisperNetwork<T>? Network => _network;
    public IReadOnlyList<string> SupportedLanguages { get; }
    /// <inheritdoc />
    public bool SupportsStreaming => false;
    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>The end-of-text token: the vocabulary's last.</summary>
    internal int EndOfText => _options.VocabSize - 1;
    /// <summary>The start-of-transcript token.</summary>
    internal int StartOfTranscript => _options.VocabSize - 2;
    /// <summary>The language identification task token.</summary>
    internal int LanguageTask => _options.VocabSize - 3;
    /// <summary>The speech emotion recognition task token.</summary>
    internal int EmotionTask => _options.VocabSize - 4;
    /// <summary>The audio event detection task token.</summary>
    internal int EventTask => _options.VocabSize - 5;

    public SenseVoiceLarge(NeuralNetworkArchitecture<T> architecture, string modelPath, SenseVoiceLargeOptions? options = null) : base(architecture)
    {
        _options = options ?? new SenseVoiceLargeOptions(); Options = _options; _useNativeMode = false;
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath; OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    public SenseVoiceLarge(NeuralNetworkArchitecture<T> architecture, SenseVoiceLargeOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new SenseVoiceLargeOptions(); Options = _options; _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay });
        base.SampleRate = _options.SampleRate; base.NumMels = _options.NumMels;
        SupportedLanguages = new[] { _options.Language }; InitializeLayersCore();
    }

    /// <inheritdoc />
    /// <remarks>The model's optimizer, unless a base training optimizer was set explicitly.</remarks>
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>The decoder's start: start of transcript and the task tokens the options select.</summary>
    internal int[] Prompt()
    {
        var prompt = new List<int> { StartOfTranscript };
        if (_options.PredictLanguage) prompt.Add(LanguageTask);
        if (_options.PredictEmotion) prompt.Add(EmotionTask);
        if (_options.PredictEvents) prompt.Add(EventTask);
        return prompt.ToArray();
    }

    /// <summary>Transcribes the first 30 seconds of audio by greedy decoding after the prompt. The task outputs (language,
    /// emotion, events) are part of the decoded tokens and are rendered through <see cref="SenseVoiceLargeOptions.Vocabulary"/>.</summary>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (ResolveReturnTimestamps(includeTimestamps)) throw new NotSupportedException("Word-level timestamps are not supported for SenseVoice-Large.");
        var network = _network ?? throw new NotSupportedException("Transcription runs on the native network.");
        var tokens = new List<int>();
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var encoded = network.Encode(Window(audio));
            var sequence = new List<int>(Prompt());
            int limit = Math.Min(_options.MaxTextLength, network.Dimensions.TextContext - sequence.Count);
            for (int step = 0; step < limit; step++)
            {
                var logits = network.Logits(sequence, encoded);
                int last = logits.Shape[0] - 1, best = 0;
                for (int v = 1; v < logits.Shape[1]; v++) if (NumOps.GreaterThan(logits[last, v], logits[last, best])) best = v;
                if (best == EndOfText) break;
                sequence.Add(best);
                if (best < EventTask) tokens.Add(best);
            }
        }
        return new TranscriptionResult<T>
        {
            Text = Text(tokens), Language = language ?? _options.Language, Confidence = NumOps.FromDouble(tokens.Count > 0 ? 1.0 : 0.0),
            DurationSeconds = audio.Length / (double)SampleRate, Segments = Array.Empty<TranscriptionSegment<T>>(),
        };
    }

    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null, CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);
    public string DetectLanguage(Tensor<T> audio) => _options.Language;
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) => new Dictionary<string, T> { [_options.Language] = NumOps.One };
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) => throw new NotSupportedException("SenseVoice-Large does not support streaming.");

    // Each token is its vocabulary entry when a vocabulary is configured, otherwise its Unicode code point.
    private string Text(List<int> tokens)
    {
        var vocabulary = _options.Vocabulary;
        var sb = new System.Text.StringBuilder();
        foreach (int t in tokens)
        {
            if (vocabulary is not null && t < vocabulary.Length) sb.Append(vocabulary[t]);
            else if (t <= 0x10FFFF && (t < 0xD800 || t > 0xDFFF)) sb.Append(char.ConvertFromUtf32(t));
        }
        return sb.ToString().Trim();
    }

    protected override void InitializeLayers() => InitializeLayersCore();

    private void InitializeLayersCore()
    {
        if (!_useNativeMode) return;
        _network = null;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) { Layers.AddRange(Architecture.Layers); return; }
        // The encoder-decoder (see the class remarks). It used to be a generic Paraformer template.
        var o = _options;
        if (o.NumMels != 80 && o.NumMels != 128) throw new ArgumentException("The log-mel front end has 80 or 128 bands.", nameof(Options));
        if (o.VocabSize <= 5) throw new ArgumentException("The vocabulary needs tokens below its five special tokens.", nameof(Options));
        var layers = new List<LayerBase<T>>();
        _network = new OpenAiWhisperNetwork<T>(Engine, layers, new WhisperDimensions(o.NumMels, 1500, o.EncoderDim, o.NumAttentionHeads, o.NumEncoderLayers,
            o.VocabSize, 448, o.EncoderDim, o.NumAttentionHeads, o.NumDecoderLayers), Tensors.Helpers.RandomHelper.CreateSeededRandom(o.Seed ?? 42));
        Layers.AddRange(layers);
    }

    // The log-mel window [mels, 3000] of a 16 kHz waveform padded or trimmed to 30 seconds.
    private Tensor<T> Window(Tensor<T> waveform)
    {
        int mels = _options.NumMels, frames = OpenAiWhisperTranscriber<T>.ChunkFrames;
        var samples = new double[OpenAiWhisperTranscriber<T>.ChunkSamples];
        var span = waveform.Data.Span;
        for (int i = 0; i < Math.Min(span.Length, samples.Length); i++) samples[i] = NumOps.ToDouble(span[i]);
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(samples, mels);
        var window = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int f = 0; f < Math.Min(frames, mel.GetLength(1)); f++) window[m, f] = NumOps.FromDouble(mel[m, f]);
        return window;
    }

    private Tensor<T> Row(Tensor<T> input, int row, int rows)
    {
        if (rows == 1) return input;
        int samples = input.Length / rows;
        return Engine.Reshape(Engine.TensorSlice(Engine.Reshape(input, new[] { rows, samples }), new[] { row, 0 }, new[] { 1, samples }), new[] { samples });
    }

    // The decoder's logits [prompt, vocabulary] over the prompt for a waveform [S] (the last row is the first output's
    // distribution), or [batch, prompt, vocabulary] for [batch, S].
    private Tensor<T> PromptLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        if (_network is null) { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        var outputs = new Tensor<T>[rows];
        for (int r = 0; r < rows; r++) outputs[r] = _network.Logits(Prompt(), _network.Encode(Window(Row(input, r, rows))));
        return rows == 1 && input.Rank == 1 ? outputs[0] : Engine.TensorStack(outputs, 0);
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio) => rawAudio;

    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); return PromptLogits(input); }

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => _network is null ? base.ForwardForTraining(input) : PromptLogits(input);

    // The output tokens a training target names: ids [length] below the special tokens, or one distribution per token
    // [length, vocabulary], read as its most likely tokens.
    private int[] Outputs(Tensor<T> target)
    {
        int vocabulary = _options.VocabSize;
        var span = target.Data.Span;
        if (target.Rank >= 2 && target.Shape[^1] == vocabulary)
        {
            var tokens = new int[target.Length / vocabulary];
            for (int r = 0; r < tokens.Length; r++)
            {
                int best = 0;
                for (int v = 1; v < vocabulary; v++) if (NumOps.GreaterThan(span[r * vocabulary + v], span[r * vocabulary + best])) best = v;
                tokens[r] = Math.Min(best, EventTask - 1);
            }
            return tokens;
        }
        var ids = new int[target.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            double value = Math.Round(NumOps.ToDouble(span[i]));
            if (value < 0 || value >= EventTask) throw new ArgumentException($"Token {value} is a special token or outside the vocabulary.", nameof(target));
            ids[i] = (int)value;
        }
        return ids;
    }

    // The teacher-forced cross-entropy [1] of the outputs and end of text after the prompt.
    private Tensor<T> OutputLoss(OpenAiWhisperNetwork<T> network, Tensor<T> audio, int[] outputs)
    {
        var prompt = Prompt();
        var logits = network.Logits(prompt.Concat(outputs).ToArray(), network.Encode(Window(audio)));
        int predicted = outputs.Length + 1, vocabulary = _options.VocabSize;
        var rows = Engine.TensorSlice(logits, new[] { prompt.Length - 1, 0 }, new[] { predicted, vocabulary });
        var weights = new Tensor<T>(new[] { predicted, vocabulary });
        for (int i = 0; i < predicted; i++) weights[i, i < outputs.Length ? outputs[i] : EndOfText] = NumOps.FromDouble(-1.0 / predicted);
        return Engine.Reshape(Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorLogSoftmax(rows, axis: -1), weights), new[] { 0, 1 }, keepDims: false), new[] { 1 });
    }

    private Tensor<T> BatchLoss(OpenAiWhisperNetwork<T> network, Tensor<T> input, Tensor<T> target)
    {
        int rows = input.Rank == 1 ? 1 : input.Shape[0];
        Tensor<T>? total = null;
        for (int r = 0; r < rows; r++)
        {
            var loss = OutputLoss(network, Row(input, r, rows), Outputs(Row(target, r, rows)));
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return Engine.TensorMultiplyScalar(total ?? throw new ArgumentException("No audio.", nameof(input)), NumOps.FromDouble(1.0 / rows));
    }

    /// <inheritdoc />
    /// <remarks>The teacher-forced cross-entropy of the target outputs after the prompt, averaged over the batch.</remarks>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        var network = _network;
        if (network is null) return false;
        TrainWithCustomLoss(input, _ => BatchLoss(network, input, target));
        return true;
    }

    TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.Supervised;

    Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget) => proposedTarget;

    /// <inheritdoc />
    /// <remarks>The objective <see cref="TryApplyLearningRule"/> minimizes.</remarks>
    T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
    {
        var network = _network ?? throw new NotSupportedException("The objective is defined on the native network.");
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return BatchLoss(network, input, target)[0];
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;
    public override ModelMetadata<T> GetModelMetadata() => new() { Name = _useNativeMode ? "SenseVoiceLarge-Native" : "SenseVoiceLarge-ONNX", Description = "SenseVoice-Large: an autoregressive encoder-decoder with task tokens (FunAudioLLM 2024)", FeatureCount = _options.NumMels, Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers, AdditionalInfo = BaseAudioMetadataInfo() };

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SenseVoiceLarge<T>)); }
    /// <inheritdoc />
    protected override void Dispose(bool disposing) { if (_disposed) return; if (disposing) OnnxEncoder?.Dispose(); _disposed = true; base.Dispose(disposing); }
}
