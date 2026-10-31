using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>
/// MinMo (Chen et al. 2025): a multimodal LLM for voice interaction. A voice encoder and projector feed speech into
/// Qwen2.5-7B-Instruct, whose answer a streaming voice decoder (CosyVoice 2's language model and token2wav) speaks five
/// text tokens at a time, and a full duplex predictor decides when to answer and when to stop.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>There is no official code; the same team's FunASR LLM-ASR and CosyVoice 2 fill what the paper leaves out (see
/// <see cref="MinMoOptions"/>). Speech in and out is at <see cref="MinMoOptions.SampleRate"/> (24 kHz); the encoder reads it
/// resampled to 16 kHz.</para>
/// <para>Tasks: <see cref="Synthesize"/> is LLM-driven text-to-speech (the instruction and the text go through the LLM,
/// whose tokens and hidden states drive the voice decoder); <see cref="Understand"/> answers an instruction about a
/// recording (recognition, translation, emotion, …); <see cref="Chat"/> answers speech with text and speech;
/// <see cref="DuplexActions"/> runs the full duplex predictor. Training follows §3.4's stages through
/// <see cref="MinMoOptions.Stage"/>.</para>
/// <para><b>For Beginners:</b> MinMo is a voice assistant made from a text chatbot: it listens, writes an answer, and reads
/// the answer aloud while it is still writing it.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "MinMo: A Multimodal Large Language Model for Seamless Voice Interaction",
    "https://arxiv.org/abs/2501.06282",
    Year = 2025,
    Authors = "Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.0, MaxGradientNorm = 1.0,
                Provenance = RecipeProvenance.ReleasedCode,
                Source = "Chen et al. 2025 §3.4 names the stages and which modules each updates, not the optimizer. The same team's "
                        + "FunASR LLM-ASR trains with AdamW; the rate (5e-5) and clipping (1.0) are not stated.")]
public partial class MinMo<T> : TtsModelBase<T>
{
    private readonly MinMoOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private MinMoCore<T>? _core;
    private readonly List<LayerBase<T>> _encoderLayers = new();
    private readonly List<LayerBase<T>> _inputProjectorLayers = new();
    private readonly List<LayerBase<T>> _llmLayers = new();
    private readonly List<LayerBase<T>> _outputProjectorLayers = new();
    private readonly List<LayerBase<T>> _duplexLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates MinMo running an ONNX export.</summary>
    public MinMo(NeuralNetworkArchitecture<T> architecture, string modelPath, MinMoOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MinMoOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates MinMo with its networks (randomly initialized).</summary>
    public MinMo(NeuralNetworkArchitecture<T> architecture, MinMoOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new MinMoOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    /// <summary>The longest answer, in tokens.</summary>
    public int MaxTextLength => _options.MaxTextLength;

    private bool HasPaperLayers => _core is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal MinMoCore<T> Core => _core ?? throw NotNative();

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
        _core = new MinMoCore<T>(Engine, _options, _encoderLayers, _inputProjectorLayers, _llmLayers, _outputProjectorLayers, _duplexLayers,
            Architecture.RandomSeed ?? SeedValue);
        AddUnsplitLayers(_llmLayers.Cast<ILayer<T>>().ToList());
        // The encoder, projectors, voice decoder and duplex predictor, so clones and saved models keep them.
        ComponentLayers.AddRange(_encoderLayers);
        ComponentLayers.AddRange(_inputProjectorLayers);
        ComponentLayers.AddRange(_outputProjectorLayers);
        ComponentLayers.AddRange(VoiceLayers);
        ComponentLayers.AddRange(_duplexLayers);
    }

    private IEnumerable<LayerBase<T>> VoiceLayers =>
        Core.Voice.LanguageModelLayers.Concat(Core.Voice.FlowLayers).Concat(Core.Voice.FrozenLayers).Concat(Core.Voice.DiscriminatorLayers);

    /// <summary>The voice speech is spoken in (CosyVoice 2's: a reference recording's speaker embedding, or a zero-shot
    /// prompt; see CosyVoice 2's <c>CreateVoice</c>).</summary>
    public TtsVoice<T>? SpeakerVoice
    {
        get => Core.Voice.Voice;
        set => Core.Voice.Voice = value;
    }

    /// <summary>A voice from a reference recording, made by the voice decoder.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, string? transcript = null) =>
        Core.Voice.CreateVoice(recording ?? throw new ArgumentNullException(nameof(recording)), _options.SampleRate, transcript);

    // ---------------------------------------------------------------- tasks

    /// <summary>Qwen2.5's token ids of text.</summary>
    public int[] TokenizeText(string text)
    {
        ThrowIfDisposed();
        return Core.Tokenizer.Encode(text ?? throw new ArgumentNullException(nameof(text)));
    }

    /// <summary>The LLM's answer to an instruction about a recording (speech-to-text: "Speech Transcription",
    /// "Translate {source} into {target}", an emotion or event question, …).</summary>
    public string Understand(Tensor<T> speech, string instruction)
    {
        ThrowIfDisposed();
        if (speech is null) throw new ArgumentNullException(nameof(speech));
        if (instruction is null) throw new ArgumentNullException(nameof(instruction));
        using var _ = Inference();
        var core = Core;
        var (answer, _, _) = core.Generate(core.BuildPrompt(instruction, speech, _options.SampleRate), SamplingRandom());
        return core.TextOf(answer);
    }

    /// <summary>The recording's transcript (<see cref="MinMoOptions.TranscriptionInstruction"/>).</summary>
    public string Transcribe(Tensor<T> speech) => Understand(speech, _options.TranscriptionInstruction);

    /// <summary>
    /// Speaks text the way MinMo's text-to-speech alignment trains it (Table 3): the user asks "{instruction}: {text}", the
    /// LLM answers, and the voice decoder speaks the answer from its tokens and hidden states. An instruction can set the
    /// emotion, speaking rate, dialect or role.
    /// </summary>
    public (Tensor<T> Audio, string Text) Speak(string text, string? instruction = null)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        using var _ = Inference();
        return Respond(Core.BuildPrompt($"{instruction ?? _options.SynthesisInstruction}: {text}", null, _options.SampleRate));
    }

    /// <summary>A spoken answer to a spoken question (speech-to-speech): the answer's text and its speech.</summary>
    public (Tensor<T> Audio, string Text) Chat(Tensor<T> speech, string? instruction = null)
    {
        ThrowIfDisposed();
        if (speech is null) throw new ArgumentNullException(nameof(speech));
        using var _ = Inference();
        return Respond(Core.BuildPrompt(instruction ?? string.Empty, speech, _options.SampleRate));
    }

    private (Tensor<T> Audio, string Text) Respond(MinMoCore<T>.Prompt prompt)
    {
        var core = Core;
        var random = SamplingRandom();
        var (answer, hidden, embeddings) = core.Generate(prompt, random);
        if (answer.Length == 0) return (new Tensor<T>(new[] { 0 }), string.Empty);
        var semantic = core.SemanticVectors(embeddings, hidden);
        var query = Engine.TensorSlice(semantic, new[] { prompt.UserStart, 0 }, new[] { prompt.UserEnd - prompt.UserStart, semantic.Shape[1] });
        var vectors = Engine.TensorSlice(semantic, new[] { prompt.Ids.Length, 0 }, new[] { answer.Length, semantic.Shape[1] });
        var tokens = core.SpeechTokens(query, vectors, random);
        return (core.Waveform(tokens, random), core.TextOf(answer));
    }

    /// <summary>The full duplex predictor's action at each speech position of a recording of the user (the input
    /// projector's rate: 25 a second with SenseVoice-Large).</summary>
    public MinMoDuplexAction[] DuplexActions(Tensor<T> speech)
    {
        ThrowIfDisposed();
        if (speech is null) throw new ArgumentNullException(nameof(speech));
        using var _ = Inference();
        return Core.DuplexActions(Core.BuildPrompt(string.Empty, speech, _options.SampleRate));
    }

    /// <inheritdoc />
    /// <remarks>LLM-driven text-to-speech (<see cref="Speak"/>).</remarks>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return Speak(text).Audio;
    }

    /// <inheritdoc />
    /// <remarks>Qwen2.5's token ids of the text.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    /// <inheritdoc />
    /// <remarks><paramref name="input"/> is Qwen2.5 token ids; the result is the speech MinMo makes of their text.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        string text = Core.TextOf(IdsOf(input));
        return string.IsNullOrWhiteSpace(text) ? new Tensor<T>(new[] { 0 }) : Speak(text).Audio;
    }

    private int[] IdsOf(Tensor<T> input)
    {
        var ids = new int[input.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = (int)Math.Round(Ops.ToDouble(input.GetFlat(i)));
            if (ids[i] < 0 || ids[i] >= _options.VocabSize) throw new ArgumentOutOfRangeException(nameof(input), $"Token ids lie in [0, {_options.VocabSize}).");
        }
        return ids;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // A fresh generator per call: the same input gives the same output.
    private Random SamplingRandom() => AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(SeedValue);

    // Evaluation mode without gradients, restored afterwards.
    private IDisposable Inference()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        var scope = new NoGradScope<T>();
        return new Restore(() =>
        {
            scope.Dispose();
            SetTrainingMode(wasTraining);
        });
    }

    private sealed class Restore : IDisposable
    {
        private Action? _action;
        public Restore(Action action) => _action = action;
        public void Dispose()
        {
            _action?.Invoke();
            _action = null;
        }
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MinMo-Native" : "MinMo-ONNX",
            Description = "MinMo: a multimodal large language model for seamless voice interaction (Chen et al., 2025)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "MinMo";
        metadata.AdditionalInfo["Encoder"] = _options.Encoder.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(MinMo<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) _core?.Voice.Dispose();
        base.Dispose(disposing);
    }
}
