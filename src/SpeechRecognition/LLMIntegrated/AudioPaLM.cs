using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.TextToSpeech.MultiModal;

namespace AiDotNet.SpeechRecognition.LLMIntegrated;

/// <summary>
/// AudioPaLM's speech recognition (Rubenstein et al. 2023): the same decoder over text and audio tokens as
/// <see cref="AiDotNet.TextToSpeech.MultiModal.AudioPaLM{T}"/>, given <c>[ASR language]</c> and a recording's audio tokens,
/// writing the transcript.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>The recording becomes 25 Hz audio tokens (w2v-BERT's k-means codes, §3.1), offset past the text vocabulary
/// (§3.2), after the task tag in the text tokenizer's tokens (§3.4); the decoder continues with the transcript's text
/// tokens and an end token. Training is AudioPaLM's: the cross-entropy of the transcript with the input's positions
/// masked, Adafactor at a constant 5e-5 (§3.6).</para>
/// <para><b>For Beginners:</b> AudioPaLM is a language model that also understands "audio words". To transcribe, it
/// turns speech into audio words and writes the text that follows the instruction "[ASR English]".</para>
/// </remarks>
/// <example>
/// <code>
/// var options = AudioPaLMOptions.Gemma2B();
/// var model = new AudioPaLM&lt;float&gt;(new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.SpeechRecognition, inputSize: 1, outputSize: 1), options);
/// model.LoadGemmaWeights("gemma-2b");
/// model.FitSpeechTokenizer(recordings);
/// model.Train(recording, transcriptIds);
/// var result = model.Transcribe(recording, "en");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.SpeechRecognition)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AudioPaLM: A Large Language Model That Can Speak and Listen", "https://arxiv.org/abs/2306.12925", Year = 2023, Authors = "Rubenstein et al.")]
[PaperOptimizer(OptimizerKind.Adafactor, LearningRate = 5e-5,
                Source = "Rubenstein et al. 2023 §3.6: the finetuning setup of Chowdhery et al. 2022 §6.1.2, Adafactor with a "
                        + "constant learning rate of 5e-5, dropout 0.1, loss masking on the inputs.")]
[TaskVariantOf(typeof(AiDotNet.TextToSpeech.MultiModal.AudioPaLM<>), "speech recognition")]
public partial class AudioPaLM<T> : AudioNeuralNetworkBase<T>, ISpeechRecognizer<T>
{
    private readonly AudioPaLMOptions _options;
    private readonly AudioPalmCore<T> _core;
    private readonly List<LayerBase<T>> _decoderLayers = new();
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private Random? _trainingRandom;
    private bool _disposed;

    /// <summary>Creates AudioPaLM's speech recognizer.</summary>
    public AudioPaLM(NeuralNetworkArchitecture<T> architecture, AudioPaLMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new AudioPaLMOptions();
        Options = _options;
        AudioPalmCore<T>.Validate(_options);
        base.SampleRate = _options.SampleRate;
        _core = new AudioPalmCore<T>(Engine, _options, _decoderLayers, architecture.RandomSeed ?? _options.TrainingSeed, withDetokenizer: false);
        Layers.AddRange(_decoderLayers);
        // The speech tokenizer (its encoder and fitted codebook), so clones and saved models keep it; it does not train.
        Layers.AddRange(_core.ComponentLayers());
    }

    internal AudioPalmCore<T> Core => _core;

    /// <inheritdoc />
    /// <remarks>Languages are named in the task tag by their English names (§3.4); any language the tokens were trained
    /// on can be asked for, so the list is the one <see cref="AudioPaLMOptions.Language"/> names.</remarks>
    public IReadOnlyList<string> SupportedLanguages => new[] { _options.Language };

    /// <inheritdoc />
    public bool SupportsStreaming => false;

    /// <inheritdoc />
    public bool SupportsWordTimestamps => false;

    /// <summary>The English name the task tag uses for a language given as a name or an ISO code ("en" → "English").</summary>
    private string LanguageName(string? language)
    {
        if (string.IsNullOrWhiteSpace(language)) return _options.Language;
        if (language!.Length > 3) return language;
        try
        {
            return System.Globalization.CultureInfo.GetCultureInfo(language).EnglishName;
        }
        catch (System.Globalization.CultureNotFoundException)
        {
            return language;
        }
    }

    // ---------------------------------------------------------------- tokens and weights

    /// <summary>Fits the audio tokens' k-means codebook on 16 kHz recordings (§3.1).</summary>
    public void FitSpeechTokenizer(IEnumerable<Tensor<T>> recordings) => _core.FitSpeechTokenizer(recordings);

    /// <summary>Sets the w2v-BERT 2.0 tokenizer's k-means centroids.</summary>
    public void LoadSpeechTokenizerCodebook(double[,] centroids) => _core.LoadSpeechTokenizerCodebook(centroids);

    /// <summary>The text token ids of <paramref name="text"/>.</summary>
    public int[] TokenizeText(string text) => _core.TokenizeText(text);

    /// <summary>Loads a Hugging Face Gemma checkpoint directory (use <see cref="AudioPaLMOptions.Gemma2B"/>).</summary>
    public void LoadGemmaWeights(string directory) => _core.LoadGemma(directory);

    /// <summary>Loads a lucidrains/PaLM-pytorch state dict.</summary>
    public void LoadPalmPytorchWeights(string path) => _core.LoadPalmPytorch(path);

    /// <summary>Loads w2v-BERT 2.0 for the w2v-BERT 2.0 tokenizer.</summary>
    public void LoadW2vBert2Weights(string path) => _core.LoadW2vBert2(path);

    // ---------------------------------------------------------------- recognition

    private int[] Prefix(Tensor<T> audio, string? language)
    {
        var tokens = _core.AudioTokensOf(audio.Rank == 1 ? audio : Engine.Reshape(audio, new[] { audio.Length }));
        return _core.TaskPrefix($"[ASR {LanguageName(language)}]", _core.AudioIds(tokens));
    }

    /// <inheritdoc />
    /// <remarks>The transcript AudioPaLM writes after <c>[ASR language]</c> and the recording's audio tokens.</remarks>
    public TranscriptionResult<T> Transcribe(Tensor<T> audio, string? language = null, bool? includeTimestamps = null)
    {
        ThrowIfDisposed();
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var ids = _core.Generate(Prefix(audio, language), audioOutput: false, out var probabilities);
            string text = _core.DecodeText(ids);
            double duration = (double)audio.Length / SampleRate;
            // The mean probability the decoder gave the tokens it chose.
            var confidence = NumOps.FromDouble(probabilities.Length == 0 ? 0.0 : probabilities.Average());
            return new TranscriptionResult<T>
            {
                Text = text,
                Language = language ?? _options.Language,
                Confidence = confidence,
                DurationSeconds = duration,
                Segments = ResolveReturnTimestamps(includeTimestamps) && text.Length > 0
                    ? new[] { new TranscriptionSegment<T> { Text = text, StartTime = 0.0, EndTime = duration, Confidence = confidence } }
                    : Array.Empty<TranscriptionSegment<T>>(),
            };
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    public Task<TranscriptionResult<T>> TranscribeAsync(Tensor<T> audio, string? language = null, bool? includeTimestamps = null,
        CancellationToken cancellationToken = default) => Task.Run(() => Transcribe(audio, language, includeTimestamps), cancellationToken);

    /// <inheritdoc />
    /// <remarks>AudioPaLM is told the language in its task tag; the paper defines no language-identification task.</remarks>
    public string DetectLanguage(Tensor<T> audio) =>
        throw new NotSupportedException("AudioPaLM is told the language in its task tag (§3.4); it has no language-identification task.");

    /// <inheritdoc />
    public IReadOnlyDictionary<string, T> DetectLanguageProbabilities(Tensor<T> audio) =>
        throw new NotSupportedException("AudioPaLM is told the language in its task tag (§3.4); it has no language-identification task.");

    /// <inheritdoc />
    public IStreamingTranscriptionSession<T> StartStreamingSession(string? language = null) =>
        throw new NotSupportedException("AudioPaLM decodes a whole recording.");

    // ---------------------------------------------------------------- the network

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
    }

    /// <inheritdoc />
    /// <remarks>The decoder's layers are not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is a 16 kHz recording; the result is the transcript's text token ids.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            var ids = _core.Generate(Prefix(input, null), audioOutput: false);
            return IndexTtsOps.Ids<T>(ids.Length == 0 ? new[] { _options.EndId } : ids);
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <summary>
    /// One update on a 16 kHz recording <paramref name="input"/> and its transcript's text token ids
    /// <paramref name="expected"/> (<see cref="TokenizeText"/>): the cross-entropy of the transcript and the end token after
    /// <c>[ASR language]</c> and the audio tokens, the input masked (§3.6), plus PaLM's z-loss.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
    {
        ThrowIfDisposed();
        var numOps = MathHelper.GetNumericOperations<T>();
        var transcript = new int[expected.Length];
        for (int i = 0; i < transcript.Length; i++)
        {
            transcript[i] = (int)Math.Round(numOps.ToDouble(expected.GetFlat(i)));
            if (transcript[i] < 0 || transcript[i] >= _options.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(expected), $"Transcript ids are text tokens in [0, {_options.VocabSize}).");
        }
        var prefix = Prefix(input, null);
        var random = _trainingRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.TrainingSeed);
        TrainWithCustomObjective(input, expected, (_, _) => _core.Loss(prefix, transcript, true, random), Optimizer());
        return true;
    }

    /// <inheritdoc />
    /// <remarks>The decoder trains; the speech tokenizer is frozen.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        var trained = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_decoderLayers.Cast<ILayer<T>>().ToList(), -1),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(trained.Contains).ToList();
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer() => _optimizer ??=
        PaperOptimizerFactory.VerifyHandBuilt(this, new AdafactorOptimizer<T, Tensor<T>, Tensor<T>>(this, new AdafactorOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate, UseRelativeStepSize = false, MultiplyByParameterScale = true, Factored = false,
            Beta1 = _options.Beta1, EnableGradientClipping = _options.MaxGradientNorm > 0,
            MaxGradientNorm = _options.MaxGradientNorm > 0 ? _options.MaxGradientNorm : 1.0,
        }));

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => true;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        Name = "AudioPaLM-ASR",
        Description = "AudioPaLM speech recognition (Rubenstein et al., 2023)",
        FeatureCount = _options.HiddenDim,
        AdditionalInfo = BaseAudioMetadataInfo(),
    };

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AudioPaLM<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) _core.SpeechEncoder?.Dispose();
        base.Dispose(disposing);
    }
}
