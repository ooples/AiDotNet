using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Latest;

namespace AiDotNet.TextToSpeech.MultiModal;

/// <summary>A SpeechGPT response: its text and its units, and the speech voiced from them.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class SpeechGptResponse<T>
{
    /// <summary>The text of the response (for chain-of-modality responses, everything outside the units).</summary>
    public string Text { get; init; } = string.Empty;

    /// <summary>The units of the speech response.</summary>
    public int[] Units { get; init; } = Array.Empty<int>();

    /// <summary>The speech, empty when the response has no units.</summary>
    public Tensor<T> Audio { get; init; } = new(new[] { 0 });
}

/// <summary>
/// SpeechGPT (Zhang et al., Findings of EMNLP 2023): a large language model with intrinsic cross-modal conversational
/// abilities. Speech becomes discrete mHuBERT units that join LLaMA's vocabulary, so one model reads and writes both
/// speech and text; a unit vocoder voices the units it writes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>Units</b> (§3.1, §4.1): mHuBERT's layer-11 features in 1000 k-means clusters, adjacent repeats removed,
/// written <c>&lt;sosp&gt;&lt;u₁&gt;…&lt;eosp&gt;</c> as tokens added to the vocabulary (eq. 1–2).</para>
/// <para><b>Tasks</b> are instructions in the release's template, <c>[Human]: {instruction} &lt;eoh&gt;. [SpeechGPT]:</c>
/// after the meta instruction: TTS and ASR (§3.1, "{description} This is input: {input}") and the chain-of-modality
/// dialogues of App. C (<c>[tq] … ; [ta] … ; [ua] …</c>).</para>
/// <para><b>Training</b> (§4.2): stage 1 predicts every token of unlabelled speech's units (eq. 3); stage 2 fine-tunes on
/// cross-modal instructions with the loss on the response (eq. 4); stage 3 trains LoRA adapters on chain-of-modality
/// data. The vocoder is the paper's multi-speaker unit HiFi-GAN, trained with HiFi-GAN's discriminators
/// (<see cref="TrainVocoder"/>), or the release's single-speaker one.</para>
/// <para><b>For Beginners:</b> SpeechGPT is a chatbot that has learned speech as if it were another language. It reads
/// "speech units" (symbols a speech model extracts from audio) and writes them back, and a vocoder turns the units it
/// writes into sound.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var options = SpeechGPTOptions.Release7B();
/// options.Vocoder = SpeechGptVocoder.ReleaseLJ;
/// var model = new SpeechGPT&lt;float&gt;(architecture, options);
/// model.LoadReleasedWeights("SpeechGPT-7B-cm");
/// model.LoadUnitExtractor("mhubert_base_vp_en_es_fr_it3.pt", "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin");
/// model.LoadVocoder("vocoder/g_00500000");
/// var audio = model.Synthesize("Today is a sunny day.");
/// string text = model.Transcribe(recording16kHz);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities",
    "https://arxiv.org/abs/2305.11000",
    Year = 2023,
    Authors = "Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, Xipeng Qiu"
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 2e-4, Beta1 = 0.9, Beta2 = 0.999, WeightDecay = 0.0, MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.CosineAnnealing, WarmupFraction = 0.03,
                Source = "Zhang et al. 2023 App. D (Table 3): a peak learning rate of 2e-4 in every stage. The optimizer and "
                        + "schedule are the release's Hugging Face Trainer scripts: AdamW, no weight decay, a cosine schedule "
                        + "with 3% warmup, clipping at 1.0.")]
[PaperOptimizer(OptimizerKind.AdamW, Component = "vocoder", LearningRate = 2e-4, Beta1 = 0.8, Beta2 = 0.99, WeightDecay = 0.01,
                Provenance = RecipeProvenance.ReleasedCode,
                Source = "The unit HiFi-GAN of Polyak et al. 2021 as speech-resynthesis trains it: AdamW (0.8, 0.99) at 2e-4 for "
                        + "the generator and the discriminators, decayed by 0.999 per epoch.")]
public partial class SpeechGPT<T> : TtsModelBase<T>
{
    private readonly SpeechGPTOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;
    private SpeechGptCore<T>? _core;
    private readonly List<LayerBase<T>> _lmLayers = new();
    private readonly List<LayerBase<T>> _hubertLayers = new();
    private readonly List<LayerBase<T>> _vocoderLayers = new();
    private readonly List<LayerBase<T>> _discriminatorLayers = new();
    private Random? _samplingRandom;

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Creates an ONNX-backed SpeechGPT for inference.</summary>
    public SpeechGPT(NeuralNetworkArchitecture<T> architecture, string modelPath, SpeechGPTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeechGPTOptions();
        Options = _options;
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        SpeechGptCore<T>.Validate(_options);
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable SpeechGPT.</summary>
    public SpeechGPT(NeuralNetworkArchitecture<T> architecture, SpeechGPTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SpeechGPTOptions();
        Options = _options;
        _useNativeMode = true;
        base.SampleRate = _options.SampleRate;
        base.HiddenDim = _options.HiddenDim;
        SpeechGptCore<T>.Validate(_options);
        InitializeLayers();
    }

    /// <summary>The longest sequence the language model takes (2048).</summary>
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    /// <remarks>The multi-speaker vocoder speaks as <see cref="SpeechGPTOptions.Speaker"/>; no recording is needed.</remarks>
    protected override TtsSupervision RequiredVoice => TtsSupervision.None;

    private bool HasPaperLayers => _core is not null;
    private int SeedValue => _options.Seed ?? 42;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");
    internal SpeechGptCore<T> Core => _core ?? throw NotNative();

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
        _core = new SpeechGptCore<T>(Engine, _options, _lmLayers, _hubertLayers, _vocoderLayers, _discriminatorLayers,
            Architecture.RandomSeed ?? SeedValue, withVocoder: true);
        AddUnsplitLayers(_lmLayers.Cast<ILayer<T>>().ToList());
        // The unit extractor, the vocoder and its discriminators, so clones and saved models keep them.
        ComponentLayers.AddRange(_hubertLayers);
        ComponentLayers.AddRange(_vocoderLayers);
        ComponentLayers.AddRange(_discriminatorLayers);
    }

    // ---------------------------------------------------------------- tokens and units

    /// <summary>The token ids of <paramref name="text"/> (no <c>&lt;s&gt;</c>).</summary>
    public int[] TokenizeText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        return Core.Tokenizer.Encode(text, addBos: false);
    }

    /// <summary>The units of 16 kHz speech, repeats merged (§3.1).</summary>
    public int[] UnitsOf(Tensor<T> recording)
    {
        ThrowIfDisposed();
        if (recording is null) throw new ArgumentNullException(nameof(recording));
        return Core.UnitsOf(recording);
    }

    // ---------------------------------------------------------------- tasks

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

    private List<int> Respond(string question) => Core.Generate(Core.Prompt(question), _samplingRandom!);

    /// <summary>Units voiced by the vocoder (as <see cref="SpeechGPTOptions.Speaker"/> with the multi-speaker one).</summary>
    public Tensor<T> Speak(IReadOnlyList<int> units) => units.Count == 0 ? new Tensor<T>(new[] { 0 }) : Inference(() =>
        (Core.Vocoder ?? throw new InvalidOperationException("This SpeechGPT has no vocoder."))
            .Forward(units, true, _options.Vocoder == SpeechGptVocoder.MultiSpeaker ? _options.Speaker : null));

    /// <summary>The TTS question (§3.1's template, the release's inference wording): <c>{TTS description} This is input: {text}</c>.</summary>
    internal string TtsQuestion(string text) => $"{_options.TtsInstruction} This is input: {text}";

    /// <summary>The ASR question: <c>{ASR description} This is input: &lt;sosp&gt;…&lt;eosp&gt;</c>.</summary>
    internal string AsrQuestion(IReadOnlyList<int> units) => $"{_options.AsrInstruction} This is input: {SpeechGptCore<T>.UnitString(units)}";

    /// <summary>The transcript SpeechGPT writes for 16 kHz speech (its ASR task).</summary>
    public string Transcribe(Tensor<T> recording)
    {
        var units = UnitsOf(recording);
        return Inference(() => Core.TextOf(Respond(AsrQuestion(units))));
    }

    /// <summary>
    /// A chain-of-modality reply (App. C) to 16 kHz speech or to text: the speech instruction transcribed (<c>[tq]</c>), a
    /// text response (<c>[ta]</c>) and, with <paramref name="speechResponse"/>, its speech (<c>[ua]</c>), voiced.
    /// </summary>
    public SpeechGptResponse<T> Chat(Tensor<T>? recording = null, string? text = null, bool speechResponse = true)
    {
        if ((recording is null) == (text is null)) throw new ArgumentException("Give either a recording or a text.");
        string question = recording is not null
            ? ChainOfModalityQuestion(SpeechGptCore<T>.UnitString(UnitsOf(recording)), speechInstruction: true, speechResponse)
            : ChainOfModalityQuestion(text!, speechInstruction: false, speechResponse);
        var ids = Inference(() => Respond(question));
        var units = Core.UnitsIn(ids);
        return new SpeechGptResponse<T> { Text = Core.TextOf(ids), Units = units, Audio = speechResponse ? Speak(units) : new Tensor<T>(new[] { 0 }) };
    }

    /// <summary>App. C's chain-of-modality instruction for a speech (its unit string) or text instruction.</summary>
    internal static string ChainOfModalityQuestion(string instruction, bool speechInstruction, bool speechResponse) => (speechInstruction, speechResponse) switch
    {
        (true, true) => $"This is a speech instruction: {instruction}. And your response should be speech. You can do it step by step. You can first transcribe the instruction and get the text Instruction. Then you can think about the instruction and get the text response. Last, you should speak the response aloud",
        (true, false) => $"This is a speech instruction: {instruction}. And your response should be text. You can do it step by step. You can first transcribe the instruction and get the text instruction. Then you can think about the instruction and get the text response.",
        (false, true) => $"This is a text instruction: {instruction}. And your response should be speech. You can do it step by step. You can think about the instruction and get the text response. Then you should speak the response aloud",
        _ => $"This is a text instruction: {instruction}. And your response should be text. You can think about the instruction and get the text response.",
    };

    // ---------------------------------------------------------------- TTS

    /// <inheritdoc />
    /// <remarks>The text's token ids (<see cref="TokenizeText"/>).</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (!HasPaperLayers) return base.PreprocessText(text);
        var ids = TokenizeText(text);
        if (ids.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        return IndexTtsOps.Ids<T>(ids);
    }

    /// <inheritdoc />
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        if (!HasPaperLayers) return base.Synthesize(text);
        return PredictCore(PreprocessText(text));
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
    /// <remarks><paramref name="input"/> is the text's token ids; the result is the 16 kHz speech SpeechGPT writes for its
    /// TTS task, voiced.</remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (!HasPaperLayers) return base.PredictCore(input);
        var text = Core.Tokenizer.Decode(TokenIds(input, nameof(input)));
        var units = Inference(() => Core.UnitsIn(Respond(TtsQuestion(text))));
        return Speak(units);
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- weights

    private static Dictionary<string, Tensor<float>> ReadAll(IEnumerable<string> paths)
    {
        var tensors = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var path in paths)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"Checkpoint not found: {path}", path);
            foreach (var pair in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path)) tensors[pair.Key] = pair.Value;
        }
        return tensors;
    }

    private static double[] Values(Tensor<float> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
        return values;
    }

    private static IEnumerable<string> Shards(string directory)
    {
        foreach (var index in new[] { "model.safetensors.index.json", "pytorch_model.bin.index.json" })
        {
            string path = Path.Combine(directory, index);
            if (File.Exists(path))
                return Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(path))["weight_map"]!.Values<string>().Distinct().Select(f => Path.Combine(directory, f!)).ToList();
        }
        foreach (var single in new[] { "model.safetensors", "pytorch_model.bin" })
            if (File.Exists(Path.Combine(directory, single))) return new[] { Path.Combine(directory, single) };
        throw new FileNotFoundException($"No checkpoint in {directory}.");
    }

    /// <summary>
    /// Loads fnlp/SpeechGPT-7B-cm (use <see cref="SpeechGPTOptions.Release7B"/>) and, with <paramref name="loraDirectory"/>
    /// (fnlp/SpeechGPT-7B-com's <c>adapter_model.bin</c>, with <see cref="SpeechGptTrainingStage.ChainOfModality"/>), its
    /// LoRA adapters; unless one is set, the tokenizer of <paramref name="directory"/>.
    /// </summary>
    public void LoadReleasedWeights(string directory, string? loraDirectory = null)
    {
        ThrowIfDisposed();
        if (directory is null) throw new ArgumentNullException(nameof(directory));
        var tensors = ReadAll(Shards(directory));
        if (loraDirectory is not null)
        {
            if (_options.Stage != SpeechGptTrainingStage.ChainOfModality) throw new InvalidOperationException("LoRA adapters need Stage = ChainOfModality.");
            foreach (var pair in ReadAll(new[] { Path.Combine(loraDirectory, "adapter_model.bin") }))
                tensors[pair.Key.StartsWith("base_model.model.", StringComparison.Ordinal) ? pair.Key.Substring("base_model.model.".Length) : pair.Key] = pair.Value;
        }
        var used = new HashSet<string>(StringComparer.Ordinal);
        Core.LanguageModel.LoadTorchWeights((name, _) =>
        {
            if (!tensors.TryGetValue(name, out var tensor)) throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
            used.Add(name);
            return Values(tensor);
        }, "model.", "lm_head.weight");
        var left = tensors.Keys.Where(k => !used.Contains(k) && !k.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal)).ToList();
        if (left.Count > 0) throw new InvalidDataException($"The checkpoint has {left.Count} tensors this model has no place for (first: '{left[0]}').");
        if (_options.TokenizerDirectory is null && File.Exists(Path.Combine(directory, "tokenizer.model")))
        {
            _options.TokenizerDirectory = directory;
            Core.ResetTokenizer();
        }
    }

    /// <summary>Loads the release's vocoder, fairseq's single-speaker unit HiFi-GAN (<c>g_00500000</c>); needs
    /// <see cref="SpeechGptVocoder.ReleaseLJ"/>.</summary>
    public void LoadVocoder(string path)
    {
        ThrowIfDisposed();
        if (_options.Vocoder != SpeechGptVocoder.ReleaseLJ) throw new InvalidOperationException("The released vocoder is single-speaker; set Vocoder = ReleaseLJ.");
        var tensors = ReadAll(new[] { path });
        (Core.Vocoder ?? throw NotNative()).LoadTorchWeights(name => tensors.TryGetValue(name, out var t) ? Values(t) : null, "generator.");
    }

    /// <summary>Loads mHuBERT and its k-means model, the unit extractor (<c>mhubert_base_vp_en_es_fr_it3.pt</c>,
    /// <c>…_L11_km1000.bin</c>).</summary>
    public void LoadUnitExtractor(string hubertCheckpoint, string kmeansModel)
    {
        ThrowIfDisposed();
        var tensors = ReadAll(new[] { hubertCheckpoint });
        Core.Units.LoadFairseqWeights(name => tensors.TryGetValue("model." + name, out var t) ? Values(t) : null);
        Core.Units.Centroids = AiDotNet.Audio.Foundations.JoblibKMeans.ReadClusterCenters(kmeansModel);
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    /// <remarks>The unit extractor's k-means centroids are fitted, not trained.</remarks>
    protected override void RegisterState(AiDotNet.Models.ModelStateRegistry<T> state)
    {
        base.RegisterState(state);
        _core?.Units.DeclareCentroids(state, "UnitCentroids");
    }

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpeechGPT-Native" : "SpeechGPT-ONNX",
            Description = "SpeechGPT: empowering large language models with intrinsic cross-modal conversational abilities (Zhang et al., 2023)",
            FeatureCount = _options.HiddenDim,
        };
        metadata.AdditionalInfo["Architecture"] = "SpeechGPT";
        metadata.AdditionalInfo["Stage"] = _options.Stage.ToString();
        metadata.AdditionalInfo["Vocoder"] = _options.Vocoder.ToString();
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SpeechGPT<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
