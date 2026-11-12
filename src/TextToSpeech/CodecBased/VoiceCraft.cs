using AiDotNet.Attributes;
using AiDotNet.Audio;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>VoiceCraft: causal-masked, delayed-stacked neural-codec speech infilling and zero-shot TTS.</summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// The native graph implements Peng et al. (2024) and the authors' release: the frozen 16 kHz GigaSpeech EnCodec
/// (four 2048-entry RVQ codebooks at 50 Hz), the released IPA front end, causal masking, delayed stacking, learned mask
/// embeddings, a pre-norm decoder-only Transformer, four independent two-layer codebook heads and the weighted
/// likelihood objective. Text-to-speech is the paper's insertion-at-the-end case; <see cref="EditSpeech"/> performs
/// general infilling while preserving unmasked codec frames exactly.
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild",
    "https://arxiv.org/abs/2403.16973",
    Year = 2024,
    Authors = "Peng et al.")]
[PaperOptimizer(OptimizerKind.ScaledAdam, LearningRate = 0.05, Beta1 = 0.9, Beta2 = 0.98,
    ReferenceBatchSize = 400000, Provenance = RecipeProvenance.Stated,
    Source = "Peng et al. 2024 Sec. 5.1 and App. C: ScaledAdam at base LR .05, 400k codec frames per update, "
        + "50k updates; Eden uses step 3000, pseudo-epoch 3000, epoch 4 and a 500-step warmup from factor .5.")]
public partial class VoiceCraft<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly VoiceCraftOptions _options;
    private IReadOnlyDictionary<string, int>? _phoneVocabulary;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ScaledAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>? _scaledAdamOptions;
    private readonly Random _trainingRandom;
    private bool _useNativeMode;
    private bool _disposed;
    private int _trainingStep;

    private EnCodec<T>? _codecNetwork;
    private AudioCodecLayer<T>? _codec;
    private VoiceCraftNetwork<T>? _network;

    /// <summary>Creates an ONNX inference wrapper.</summary>
    public VoiceCraft(NeuralNetworkArchitecture<T> architecture, string modelPath, VoiceCraftOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VoiceCraftOptions();
        ValidateOptions(_options);
        Options = _options;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBaseProperties();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
    }

    /// <summary>Creates the native EnCodec and VoiceCraft Transformer.</summary>
    public VoiceCraft(
        NeuralNetworkArchitecture<T> architecture,
        VoiceCraftOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VoiceCraftOptions();
        ValidateOptions(_options);
        Options = _options;
        _useNativeMode = true;
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        if (optimizer is not null) _optimizer = optimizer;
        else if (_options.UseScaledAdam)
        {
            _scaledAdamOptions = new ScaledAdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = 0.9,
                Beta2 = 0.98,
                ClippingUpdatePeriod = 1000
            };
            _optimizer = new ScaledAdamOptimizer<T, Tensor<T>, Tensor<T>>(this, _scaledAdamOptions);
        }
        else
        {
            _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = _options.LearningRate,
                    WeightDecay = 0
                });
        }
        ConfigureBaseProperties();
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;
    /// <inheritdoc />
    public int NumCodebooks => _options.NumCodebooks;
    /// <inheritdoc />
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;
    public override int CodecTokenCodebooks => _options.NumCodebooks;
    public override int CodecTokenVocabulary => _options.CodebookSize;
    protected override int OutputFeatureWidth => _options.AudioVocabularySize;
    protected override TtsSupervision RequiredVoice => TtsSupervision.ReferenceRecording;
    protected override TtsSupervision RequiredSupervision => TtsSupervision.Text;
    public override bool TrainsOnSampleObjective => true;

    public bool HasPaperPipeline => _codec is not null && _network is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) => HasPaperPipeline
        ? LayerInputDomain.Indices(_options.TextVocabularySize) : base.GetInputDomain(inputShape);
    public int TransformerLayerCount => _network?.LayerCount ?? 0;
    public int PhoneVocabularySize => _phoneVocabulary?.Count ?? VoiceCraftPhoneTokenizer.Count;

    /// <summary>Sets the original checkpoint's phn2num mapping without changing its token IDs.</summary>
    public void SetPhoneVocabulary(IReadOnlyDictionary<string, int> vocabulary)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(vocabulary);
        if (vocabulary.Count == 0 || vocabulary.Any(kv => string.IsNullOrEmpty(kv.Key)
            || kv.Value < 0 || kv.Value >= _options.TextVocabularySize)
            || vocabulary.Values.Distinct().Count() != vocabulary.Count)
            throw new ArgumentException("Phone vocabulary needs unique IDs within the text embedding table.", nameof(vocabulary));
        _phoneVocabulary = vocabulary.ToDictionary(kv => kv.Key, kv => kv.Value, StringComparer.Ordinal);
    }
    public bool CodecCodebooksInitialized => _codecNetwork?.CodebooksInitialized ?? false;

    /// <summary>Runs the release's eSpeak-style IPA front end and exact checkpoint <c>phn2num</c> table.</summary>
    public Tensor<T> TokenizeText(string text)
    {
        ThrowIfDisposed();
        return VoiceCraftOps<T>.Ids(VoiceCraftPhoneTokenizer.Tokenize(text, _options.MaxTextLength, _phoneVocabulary, _options.TextPhonemizer));
    }

    /// <summary>Enrolls the prompt recording and its transcript, both required by VoiceCraft continuation.</summary>
    public TtsVoice<T> CreateVoice(Tensor<T> recording, string transcript, int sampleRate = 16000)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(recording);
        return new TtsVoice<T>
        {
            ReferenceAudio = Resample(recording, sampleRate, _options.SampleRate),
            ReferenceTokens = TokenizeText(transcript)
        };
    }

    /// <summary>Synthesizes with a prompt without changing the configured voice.</summary>
    public Tensor<T> SynthesizeWithVoice(string text, Tensor<T> recording, string transcript, int sampleRate = 16000)
    {
        var previous = Voice;
        try
        {
            Voice = CreateVoice(recording, transcript, sampleRate);
            return Synthesize(text);
        }
        finally { Voice = previous; }
    }

    /// <summary>Performs the paper's zero-shot TTS case: an insertion after the enrolled prompt.</summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(PreprocessText(text));
        RequirePipeline();
        var voice = RequireVoice();
        if (voice.ReferenceTokens is null)
            throw new InvalidOperationException("VoiceCraft requires the transcript of its prompt in Voice.ReferenceTokens; use CreateVoice.");
        var target = VoiceCraftPhoneTokenizer.Tokenize(text, _options.MaxTextLength, _phoneVocabulary, _options.TextPhonemizer);
        var promptText = VoiceCraftOps<T>.IdsOf(voice.ReferenceTokens);
        var phones = promptText.Concat(target).Take(_options.MaxTextLength).ToArray();
        if (phones.Length == 0) throw new ArgumentException("The text produced no VoiceCraft phones.", nameof(text));
        using var _ = new NoGradScope<T>();
        int[,] prompt = _codec!.Encode(voice.ReferenceAudio!);
        var prefix = DelayRows(prompt, trimTail: true);
        int maximum = Math.Min(_options.MaxCodecFrames,
            Math.Max(_options.MinimumGeneratedFrames, target.Length * _options.MaximumFramesPerPhone));
        return _codec.Decode(GenerateShortest(phones, prefix, maximum));
    }

    /// <summary>
    /// Infills the specified codec-frame spans and returns a decoded waveform. A caller obtains the spans from a forced
    /// word alignment and may include the paper's small co-articulation margin. Frames outside the spans are copied.
    /// </summary>
    public Tensor<T> EditSpeech(string targetTranscript, Tensor<T> originalAudio,
        IReadOnlyList<(int StartFrame, int EndFrame)> maskSpans, int sampleRate = 16000)
    {
        ThrowIfDisposed();
        RequirePipeline();
        Validation.Guard.NotNull(originalAudio);
        Validation.Guard.NotNull(maskSpans);
        if (maskSpans.Count == 0) return Resample(originalAudio, sampleRate, _options.SampleRate);
        int[] phones = VoiceCraftPhoneTokenizer.Tokenize(targetTranscript, _options.MaxTextLength, _phoneVocabulary, _options.TextPhonemizer);
        using var _ = new NoGradScope<T>();
        int[,] original = _codec!.Encode(Resample(originalAudio, sampleRate, _options.SampleRate));
        var spans = NormalizeSpans(maskSpans, original.GetLength(1));
        var pieces = new List<int[,]>();
        int cursor = 0;
        foreach (var span in spans)
        {
            pieces.Add(SliceCodes(original, cursor, span.StartFrame));
            var condition = BuildEditingCondition(original, spans);
            int maximum = Math.Min(_options.MaxCodecFrames,
                Math.Max(_options.MinimumGeneratedFrames, (span.EndFrame - span.StartFrame) * 2));
            pieces.Add(GenerateShortest(phones, condition.Rows, maximum, condition.Masks, endOfUtterance: false));
            cursor = span.EndFrame;
        }
        pieces.Add(SliceCodes(original, cursor, original.GetLength(1)));
        return _codec!.Decode(ConcatenateCodes(pieces));
    }

    /// <summary>Encodes audio as <c>[frames, four codebooks]</c> VoiceCraft tokens.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(audio);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return CodesToTensor(_codec!.Encode(audio));
    }

    public Tensor<T> EncodeToTokens(Tensor<T> audio, int sampleRate)
        => EncodeToTokens(Resample(audio, sampleRate, _options.SampleRate));

    /// <summary>Decodes <c>[frames, four codebooks]</c> or flattened frame-major codes through EnCodec.</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        Validation.Guard.NotNull(tokens);
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);
        RequirePipeline();
        using var _ = new NoGradScope<T>();
        return _codec!.Decode(TensorToCodes(tokens));
    }

    /// <summary>Exposes the paper's delayed stacking for diagnostics: codebook q is shifted by q rows.</summary>
    public Tensor<T> DelayCodecTokens(Tensor<T> tokens)
    {
        int[,] codes = TensorToCodes(tokens);
        var rows = DelayRows(codes, trimTail: false);
        int nq = _options.NumCodebooks;
        var flat = new int[rows.Count * nq];
        for (int r = 0; r < rows.Count; r++) Array.Copy(rows[r], 0, flat, r * nq, nq);
        return new Tensor<int>(flat, new[] { rows.Count, nq }).Cast<T>();
    }

    protected override Tensor<T> PreprocessText(string text) => TokenizeText(text);

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        if (_optimizer is null) throw new NotSupportedException("Native optimizer is unavailable.");
        if (_scaledAdamOptions is not null && _optimizer is ScaledAdamOptimizer<T, Tensor<T>, Tensor<T>> scaled)
            scaled.SetScheduledLearningRate(EdenLearningRate(_trainingStep));
        T loss = TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => BuildObjective(sample), _optimizer);
        _trainingStep++;
        return loss;
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        RequirePipeline();
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return BuildObjective(sample)[0];
        }
        finally { SetTrainingMode(wasTraining); }
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is { Count: > 0 })
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        var codecOptions = (EnCodecOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(_options.Codec);
        codecOptions.IncludeDiscriminators = false;
        var parentInitialization = LayerInitializationSeedScope.CaptureScope();
        try
        {
            _codecNetwork = new EnCodec<T>(new NeuralNetworkArchitecture<T>(InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = Architecture.RandomSeed }, codecOptions);
        }
        finally { LayerInitializationSeedScope.RestoreScope(parentInitialization); }
        if (_codecNetwork.HopLength != _options.HopSize || codecOptions.SampleRate != _options.SampleRate ||
            codecOptions.CodebookSize != _options.CodebookSize ||
            _codecNetwork.QuantizersForBandwidth(codecOptions.TargetBandwidthKbps) != _options.NumCodebooks)
            throw new ArgumentException("VoiceCraft options must match the nested codec's sample rate, hop, codebook size and quantizer bandwidth.");
        _codec = new AudioCodecLayer<T>(_codecNetwork);
        ComponentLayers.Add(_codec);
        _network = new VoiceCraftNetwork<T>(Engine, ComponentLayers, _options);
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        if (HasPaperPipeline)
        {
            var voice = RequireVoice();
            if (voice.ReferenceTokens is null) throw new InvalidOperationException("VoiceCraft's prompt transcript is missing.");
            int[] target = VoiceCraftOps<T>.IdsOf(input);
            int[] phones = VoiceCraftOps<T>.IdsOf(voice.ReferenceTokens).Concat(target).Take(_options.MaxTextLength).ToArray();
            using var _ = new NoGradScope<T>();
            var prefix = DelayRows(_codec!.Encode(voice.ReferenceAudio!), trimTail: true);
            int maximum = Math.Min(_options.MaxCodecFrames,
                Math.Max(_options.MinimumGeneratedFrames, target.Length * _options.MaximumFramesPerPhone));
            return _codec.Decode(GenerateShortest(phones, prefix, maximum));
        }
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VoiceCraft-Native" : "VoiceCraft-ONNX",
            Description = "VoiceCraft causal-masked and delayed-stacked neural codec language model",
            FeatureCount = _options.LLMDim
        };
        metadata.AdditionalInfo["Architecture"] = "GigaSpeech EnCodec + causal decoder Transformer + four MLP heads";
        metadata.AdditionalInfo["TransformerLayers"] = _options.NumLLMLayers;
        metadata.AdditionalInfo["AttentionHeads"] = _options.NumHeads;
        metadata.AdditionalInfo["Codec"] = $"{_options.SampleRate} Hz, {_options.CodecFrameRate} Hz, {_options.NumCodebooks}x{_options.CodebookSize}";
        metadata.AdditionalInfo["Sampling"] = _options.TopK > 0 ? $"top-{_options.TopK}" : $"top-p {_options.TopP}";
        metadata.AdditionalInfo["OfficialWeightsAvailable"] = true;
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        return metadata;
    }

    private Tensor<T> BuildObjective(TtsTrainingSample<T> sample)
    {
        int[] text = NormalizePhones(VoiceCraftOps<T>.IdsOf(sample.Tokens));
        int[,] codes = sample.CodecTokens is not null
            ? TensorToCodes(sample.CodecTokens)
            : sample.Audio is not null
                ? _codec!.Encode(sample.Audio)
                : throw new ArgumentException("VoiceCraft training requires Audio or CodecTokens.", nameof(sample));
        var layout = BuildTrainingLayout(codes);
        var logits = _network!.Forward(text, layout.Rows, layout.Masks);
        Tensor<T>? total = null;
        for (int q = 0; q < _options.NumCodebooks; q++)
        {
            var positions = Enumerable.Range(0, layout.Labels[q].Length)
                .Where(i => layout.Labels[q][i] >= 0).ToArray();
            if (positions.Length == 0) continue;
            var indices = new Tensor<int>(positions, new[] { positions.Length });
            var selected = Engine.TensorIndexSelect(logits[q], indices, 0);
            var labels = positions.Select(i => layout.Labels[q][i]).ToArray();
            var loss = UniAudioOps<T>.CrossEntropy(Engine, selected, labels);
            loss = Engine.TensorMultiplyScalar(loss, NumOps.FromDouble(_options.CodebookLossWeights[q]));
            total = total is null ? loss : Engine.TensorAdd(total, loss);
        }
        return total ?? new Tensor<T>(new[] { 1 });
    }

    private (List<int[]> Rows, List<int> Masks, int[][] Labels) BuildTrainingLayout(int[,] codes)
    {
        int frames = codes.GetLength(1), spans = Math.Min(SamplePoissonOne(), Math.Min(_options.MaxMaskSpans, Math.Max(1, frames / 2)));
        var starts = new SortedSet<int>();
        int attempts = 0;
        while (starts.Count < spans && attempts++ < 100)
        {
            int candidate = frames <= 2 ? 0 : _trainingRandom.Next(1, Math.Max(2, frames - 1));
            if (starts.All(x => Math.Abs(x - candidate) >= _options.MinimumGapFrames)) starts.Add(candidate);
        }
        if (starts.Count == 0) starts.Add(frames <= 2 ? 0 : 1);
        var intervals = new List<(int Start, int End)>();
        foreach (int start in starts)
        {
            int maximum = Math.Min(_options.MaximumMaskFrames, Math.Max(1, frames - start));
            int length = Math.Min(maximum, Math.Max(_options.MinimumMaskFrames,
                maximum == 1 ? 1 : _trainingRandom.Next(_options.MinimumMaskFrames, maximum + 1)));
            int end = Math.Min(frames, start + length);
            if (intervals.Count == 0 || start >= intervals[^1].End) intervals.Add((start, end));
        }
        var segments = new List<int[,]>();
        int cursor = 0;
        foreach (var span in intervals) { segments.Add(SliceCodes(codes, cursor, span.Start)); cursor = span.End; }
        segments.Add(AppendSpecial(SliceCodes(codes, cursor, frames), _options.EndOfUtteranceToken));
        foreach (var span in intervals)
            segments.Add(AppendSpecial(SliceCodes(codes, span.Start, span.End), _options.EndOfGenerationToken));

        var rows = new List<int[]>();
        var masks = new List<int>();
        var labelsByRow = new List<int[]>();
        for (int segment = 0; segment < segments.Count; segment++)
        {
            if (segment > 0)
            {
                rows.Add(Enumerable.Repeat(_options.EndOfGenerationToken, _options.NumCodebooks).ToArray());
                masks.Add((segment - 1) % intervals.Count);
                labelsByRow.Add(Enumerable.Repeat(-1, _options.NumCodebooks).ToArray());
            }
            AddTeacherForcedDelayedSegment(segments[segment], rows, masks, labelsByRow);
        }
        var labels = Enumerable.Range(0, _options.NumCodebooks)
            .Select(q => labelsByRow.Select(row => row[q]).ToArray()).ToArray();
        return (rows, masks, labels);
    }

    private void AddTeacherForcedDelayedSegment(int[,] segment, List<int[]> rows, List<int> masks, List<int[]> labels)
    {
        int frames = segment.GetLength(1), delayedLength = frames + _options.NumCodebooks - 1;
        var delayed = Enumerable.Range(0, delayedLength)
            .Select(_ => Enumerable.Repeat(_options.EmptyToken, _options.NumCodebooks).ToArray()).ToArray();
        var valid = Enumerable.Range(0, delayedLength)
            .Select(_ => new bool[_options.NumCodebooks]).ToArray();
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++)
            {
                delayed[frame + q][q] = segment[q, frame];
                valid[frame + q][q] = true;
            }
        for (int row = 0; row < delayedLength; row++)
        {
            rows.Add(row == 0 ? Enumerable.Repeat(_options.EmptyToken, _options.NumCodebooks).ToArray() : delayed[row - 1]);
            masks.Add(-1);
            var target = new int[_options.NumCodebooks];
            for (int q = 0; q < _options.NumCodebooks; q++) target[q] = valid[row][q] ? delayed[row][q] : -1;
            labels.Add(target);
        }
    }

    private int[,] GenerateShortest(int[] phones, IReadOnlyList<int[]> prefix, int maximum,
        IReadOnlyList<int>? masks = null, bool endOfUtterance = true)
    {
        int[,]? shortest = null;
        int candidates = endOfUtterance ? Math.Max(1, _options.SampleBatchSize) : 1;
        for (int candidate = 0; candidate < candidates; candidate++)
        {
            var generated = _network!.Generate(phones, prefix, masks,
                endOfUtterance ? Math.Min(_options.MinimumGeneratedFrames, maximum) : 0, maximum,
                AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed + candidate), endOfUtterance);
            if (shortest is null || generated.GetLength(1) < shortest.GetLength(1)) shortest = generated;
        }
        return shortest!;
    }

    private List<int[]> DelayRows(int[,] codes, bool trimTail)
    {
        int frames = codes.GetLength(1), count = Math.Max(1, frames + _options.NumCodebooks - 1);
        var rows = Enumerable.Range(0, count)
            .Select(_ => Enumerable.Repeat(_options.EmptyToken, _options.NumCodebooks).ToArray()).ToList();
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++) rows[frame + q][q] = codes[q, frame];
        if (trimTail && rows.Count > frames) rows.RemoveRange(frames, rows.Count - frames);
        return rows;
    }

    private (List<int[]> Rows, List<int> Masks) BuildEditingCondition(int[,] codes,
        IReadOnlyList<(int StartFrame, int EndFrame)> spans)
    {
        var rows = new List<int[]>();
        var masks = new List<int>();
        int cursor = 0;
        for (int i = 0; i <= spans.Count; i++)
        {
            int end = i < spans.Count ? spans[i].StartFrame : codes.GetLength(1);
            foreach (var row in DelayRows(SliceCodes(codes, cursor, end), trimTail: false))
            {
                rows.Add(row); masks.Add(-1);
            }
            if (i < spans.Count)
            {
                rows.Add(Enumerable.Repeat(_options.EndOfGenerationToken, _options.NumCodebooks).ToArray());
                masks.Add(i);
                cursor = spans[i].EndFrame;
            }
        }
        return (rows, masks);
    }

    private int[] NormalizePhones(int[] phones)
    {
        var normalized = phones.Take(_options.MaxTextLength)
            .Where(id => id >= 0 && id < _options.TextVocabularySize).ToArray();
        if (normalized.Length == 0) throw new ArgumentException("VoiceCraft training needs at least one valid phone token.");
        return normalized;
    }

    private IReadOnlyList<(int StartFrame, int EndFrame)> NormalizeSpans(
        IReadOnlyList<(int StartFrame, int EndFrame)> spans, int frames)
    {
        var result = spans.OrderBy(span => span.StartFrame).ToArray();
        int previous = 0;
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i].StartFrame < previous || result[i].StartFrame < 0 || result[i].EndFrame <= result[i].StartFrame || result[i].EndFrame > frames)
                throw new ArgumentException("Mask spans must be ordered, non-overlapping half-open codec-frame intervals.", nameof(spans));
            previous = result[i].EndFrame;
        }
        if (result.Length > _options.MaxMaskSpans)
            throw new ArgumentException($"VoiceCraft supports at most {_options.MaxMaskSpans} edit spans.", nameof(spans));
        return result;
    }

    private Tensor<T> CodesToTensor(int[,] codes)
    {
        return FrameMajorCodes(codes, _options.NumCodebooks);
    }

    private int[,] TensorToCodes(Tensor<T> tokens)
    {
        int frames;
        var codes = tokens.Rank == 2 && tokens.Shape[1] == _options.NumCodebooks
            ? new int[_options.NumCodebooks, frames = tokens.Shape[0]]
            : tokens.Rank == 2 && tokens.Shape[0] == _options.NumCodebooks
                ? new int[_options.NumCodebooks, frames = tokens.Shape[1]]
                : tokens.Rank == 1 && tokens.Length % _options.NumCodebooks == 0
                    ? new int[_options.NumCodebooks, frames = tokens.Length / _options.NumCodebooks]
                    : throw new ArgumentException($"VoiceCraft codes must be [frames, {_options.NumCodebooks}], [{_options.NumCodebooks}, frames], or flattened frame-major.", nameof(tokens));
        var rounded = RoundedIds(tokens);
        bool frameMajor = tokens.Rank == 1 || tokens.Shape[1] == _options.NumCodebooks;
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < _options.NumCodebooks; q++)
            {
                int value = frameMajor ? rounded[frame * _options.NumCodebooks + q] : rounded[q * frames + frame];
                codes[q, frame] = MathPolyfill.Clamp(value, 0, _options.CodebookSize - 1);
            }
        return codes;
    }

    private static int[,] SliceCodes(int[,] codes, int start, int end)
    {
        int nq = codes.GetLength(0), length = Math.Max(0, end - start);
        var result = new int[nq, length];
        for (int q = 0; q < nq; q++) for (int f = 0; f < length; f++) result[q, f] = codes[q, start + f];
        return result;
    }

    private int[,] AppendSpecial(int[,] codes, int special)
    {
        int frames = codes.GetLength(1);
        var result = new int[_options.NumCodebooks, frames + 1];
        for (int q = 0; q < _options.NumCodebooks; q++)
        {
            for (int f = 0; f < frames; f++) result[q, f] = codes[q, f];
            result[q, frames] = special;
        }
        return result;
    }

    private int[,] ConcatenateCodes(IReadOnlyList<int[,]> pieces)
    {
        int frames = pieces.Sum(piece => piece.GetLength(1)), cursor = 0;
        var result = new int[_options.NumCodebooks, frames];
        foreach (var piece in pieces)
        {
            for (int q = 0; q < _options.NumCodebooks; q++)
                for (int f = 0; f < piece.GetLength(1); f++) result[q, cursor + f] = piece[q, f];
            cursor += piece.GetLength(1);
        }
        return result;
    }

    private int SamplePoissonOne()
    {
        double limit = Math.Exp(-1), product = 1;
        int k = 0;
        do { k++; product *= _trainingRandom.NextDouble(); } while (product > limit);
        return MathPolyfill.Clamp(k - 1, 1, _options.MaxMaskSpans);
    }

    private double EdenLearningRate(int step)
    {
        double t = step, epoch = step / (double)Math.Max(1, _options.PseudoEpochSteps);
        double stepFactor = Math.Pow((t * t + (double)_options.EdenStep * _options.EdenStep) /
            (_options.EdenStep * (double)_options.EdenStep), -0.25);
        double epochFactor = Math.Pow((epoch * epoch + (double)_options.EdenEpoch * _options.EdenEpoch) /
            (_options.EdenEpoch * (double)_options.EdenEpoch), -0.25);
        double warmup = _options.EdenWarmupSteps <= 0 ? 1 : 0.5 + 0.5 * Math.Min(1, t / _options.EdenWarmupSteps);
        return _options.LearningRate * stepFactor * epochFactor * warmup;
    }

    private void ConfigureBaseProperties()
    {
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
    }

    private Tensor<T> Resample(Tensor<T> audio, int from, int to)
    {
        var flat = Engine.Reshape(audio, new[] { audio.Length });
        return from == to ? flat : AudioHelper<T>.ResampleBandLimited(Engine, flat, from, to);
    }

    private static void ValidateOptions(VoiceCraftOptions options)
    {
        if (options.SampleRate != 16000 || options.HopSize != 320 || options.CodecFrameRate != 50)
            throw new ArgumentException("VoiceCraft uses 16 kHz audio and a 320-sample/50 Hz codec hop.", nameof(options));
        if (options.NumCodebooks <= 0 || options.CodebookSize <= 1 || options.LLMDim <= 0 || options.NumLLMLayers <= 0 ||
            options.NumHeads <= 0 || options.LLMDim % options.NumHeads != 0 || options.FeedForwardDim <= 0)
            throw new ArgumentException("VoiceCraft Transformer dimensions are invalid.", nameof(options));
        if (options.CodebookLossWeights.Length != options.NumCodebooks || options.CodebookLossWeights.Any(weight => weight <= 0))
            throw new ArgumentException("VoiceCraft needs one positive loss weight per codebook.", nameof(options));
        if (options.TopP <= 0 || options.TopP > 1 || options.Temperature <= 0 || options.SampleBatchSize <= 0)
            throw new ArgumentException("VoiceCraft sampling settings are invalid.", nameof(options));
    }

    private void RequirePipeline()
    {
        if (!HasPaperPipeline) throw new NotSupportedException("This VoiceCraft instance does not own the native EnCodec/Transformer graph.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VoiceCraft<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) _codecNetwork?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }
}
