using System.Text.RegularExpressions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.TextToSpeech.Vocoders;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// ChatTTS (2noise, 2024): conversational text-to-speech as a GPT over text and DVAE audio codes, with prosody control
/// tokens, a speaker embedding, and Vocos rendering.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>GPT</b>: a Llama decoder (<see cref="LlamaNetwork{T}"/> without its own token table or head) reads one embedding
/// per position: a text id's from the text table, or an audio frame's as the sum of its 4 codes' embeddings. Two kinds
/// of weight-normalized heads read the last hidden state: the text head (refining text) and 4 code heads (one per
/// code of the next frame).
/// </para>
/// <para>
/// <b>Synthesis</b> (<c>Chat.infer</c>): the text is first refined (<c>[Sbreak]</c> text <c>[Pbreak]</c>, sampled with
/// the text head at temperature 0.7, top-p 0.7, top-k 20, until <c>[Ebreak]</c>), which adds <c>[uv_break]</c>,
/// <c>[laugh]</c> and similar tokens; then <c>[Stts]</c>, <c>[spk_emb]</c> (or <c>[empty_spk]</c>), the text and
/// <c>[Ptts]</c> prompt code generation: 4 codes per frame sampled at temperature 0.3 after a repetition penalty of 1.05
/// over the last 16 frames, top-p 0.7 and top-k 20 (each keeping at least 3), until any code is the end id (625).
/// The speaker embedding, L2-normalized, replaces the <c>[spk_emb]</c> position's embedding; a recording's DVAE codes
/// can follow the text as audio positions. The GPT's hidden states that produced the frames go through the
/// hidden-state "Decoder" to 100-band mels (or the codes through the DVAE), and Vocos renders 24 kHz audio.
/// </para>
/// <para>
/// <b>Training</b>: 2noise released no training code. <see cref="TtsModelBase{T}.Train(TtsTrainingSample{T})"/>
/// trains the GPT, its embeddings and heads on the code-generation sequence: the decorated text, then the utterance's
/// DVAE codes and the end frame, with each frame's 4 codes predicted from the hidden state before it (the mean
/// cross-entropy of the 4 heads). The DVAE, the Decoder and Vocos stay as loaded.
/// </para>
/// <para>
/// Text is bert-base-chinese's WordPiece (shipped with the library) plus ChatTTS's 50 control tokens, as the
/// released tokenizer is. Load the released networks with <see cref="LoadPretrainedWeights"/>.
/// </para>
/// <para><b>For Beginners:</b> ChatTTS reads text aloud the way people talk, with pauses and laughs, in a random voice
/// or one you give it.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
/// var model = new ChatTTS&lt;float&gt;(architecture, new ChatTTSOptions());
/// model.LoadPretrainedWeights("ChatTTS/asset");
/// Tensor&lt;float&gt; audio = model.Synthesize("Hello there [uv_break] how are you?");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "ChatTTS: A generative speech model for daily dialogue",
    "https://github.com/2noise/ChatTTS",
    Year = 2024,
    Authors = "2noise"
)]
[PaperOptimizer(OptimizerKind.Unspecified,
    Source = "2noise released ChatTTS's networks and inference code but no training code or paper, so no optimizer or "
             + "schedule is stated; the model trains with AdamW at the options' learning rate, a library default.")]
public partial class ChatTTS<T> : TtsModelBase<T>, ICodecTts<T>
{
    /// <summary>ChatTTS's control tokens, in the order its tokenizer adds them after bert-base-chinese's 21,128.</summary>
    internal static readonly string[] ControlTokens =
    {
        "[Sasr]", "[Pasr]", "[Easr]", "[Stts]", "[Ptts]", "[Etts]", "[Sbreak]", "[Pbreak]", "[Ebreak]", "[uv_break]",
        "[v_break]", "[lbreak]", "[llbreak]", "[undefine]", "[laugh]", "[spk_emb]", "[empty_spk]", "[music]", "[pure]",
        "[break_0]", "[break_1]", "[break_2]", "[break_3]", "[break_4]", "[break_5]", "[break_6]", "[break_7]",
        "[laugh_0]", "[laugh_1]", "[laugh_2]", "[oral_0]", "[oral_1]", "[oral_2]", "[oral_3]", "[oral_4]", "[oral_5]",
        "[oral_6]", "[oral_7]", "[oral_8]", "[oral_9]", "[speed_0]", "[speed_1]", "[speed_2]", "[speed_3]", "[speed_4]",
        "[speed_5]", "[speed_6]", "[speed_7]", "[speed_8]", "[speed_9]",
    };

    private static readonly Regex ControlTokenPattern = new(
        "(" + string.Join("|", ControlTokens.Select(Regex.Escape)) + ")", RegexOptions.CultureInvariant);

    private readonly ChatTTSOptions _options;
    private readonly bool _useNativeMode;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _suppliedOptimizer;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly Random _trainingRandom;
    private bool _disposed;

    private readonly List<LayerBase<T>> _gptLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();
    private LlamaNetwork<T>? _gpt;
    private TiedEmbeddingLayer<T>? _textTable;
    private readonly List<TiedEmbeddingLayer<T>> _codeTables = new();
    private NormedConv1DLayer<T>? _textHead;
    private readonly List<NormedConv1DLayer<T>> _codeHeads = new();
    private ChatTtsGroupedFsq<T>? _quantizer;
    private ChatTtsDvaeEncoder<T>? _dvaeEncoder;
    private ChatTtsMelDecoder<T>? _dvaeDecoder;
    private ChatTtsMelDecoder<T>? _hiddenDecoder;
    private VocosGenerator<T>? _vocos;
    private double[]? _speakerStd, _speakerMean;

    /// <summary>Creates an ONNX-backed ChatTTS for inference.</summary>
    public ChatTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, ChatTTSOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new ChatTTSOptions();
        Options = _options;
        _useNativeMode = false;
        _tokenizer = WordPieceTokenizer.BertBaseChinese(ControlTokens);
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable ChatTTS.</summary>
    public ChatTTS(NeuralNetworkArchitecture<T> architecture, ChatTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new ChatTTSOptions();
        Options = _options;
        _useNativeMode = true;
        _suppliedOptimizer = optimizer;
        _tokenizer = WordPieceTokenizer.BertBaseChinese(ControlTokens);
        _trainingRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.FsqGroups * o.FsqResiduals != o.NumVq)
            throw new ArgumentException($"The DVAE's {o.FsqGroups} groups × {o.FsqResiduals} stages must give the GPT's {o.NumVq} codes per frame.");
        if (o.HiddenDim != 2 * o.DecoderDim)
            throw new ArgumentException($"The hidden-state decoder reads the GPT's {o.HiddenDim} channels as two halves of {o.DecoderDim}.");
        base.SampleRate = o.SampleRate;
        base.HopSize = o.HopSize;
        base.HiddenDim = o.HiddenDim;
        base.MelChannels = o.MelChannels;
        o.VocabSize = TextVocabulary;
    }

    /// <summary>The text vocabulary: bert-base-chinese's 21,128 and the 50 control tokens.</summary>
    public int TextVocabulary => 21128 + ControlTokens.Length;

    private int CodebookSize => _options.FsqLevels.Aggregate(1, (a, b) => a * b);

    /// <summary>The end id of the code heads (625: one past the DVAE's codes).</summary>
    public int EndCode => CodebookSize;

    private int TokenId(string token) => 21128 + Array.IndexOf(ControlTokens, token);

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.NumVq;

    /// <inheritdoc />
    int ICodecTts<T>.CodebookSize => CodebookSize;

    /// <inheritdoc />
    /// <remarks>DVAE frames: the mel frame rate halved (46.875 a second at 24 kHz, hop 256).</remarks>
    public int CodecFrameRate => _options.SampleRate / (2 * _options.HopSize);

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.NumVq;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => CodebookSize;

    private bool HasPaperLayers => _gpt is not null;

    /// <inheritdoc />
    public override LayerInputDomain GetInputDomain(int[]? inputShape) =>
        HasPaperLayers ? LayerInputDomain.Indices(TextVocabulary) : base.GetInputDomain(inputShape);

    private LlamaNetwork<T> Gpt => _gpt ?? throw new InvalidOperationException("ChatTTS's GPT exists in native mode only.");

    // Networks, for tests.
    internal LlamaNetwork<T>? GptNetwork => _gpt;
    internal IReadOnlyList<LayerBase<T>> GptLayers => _gptLayers;
    internal IReadOnlyList<LayerBase<T>> CodecLayers => _codecLayers;
    internal ChatTtsGroupedFsq<T>? Quantizer => _quantizer;
    internal ChatTtsDvaeEncoder<T>? DvaeEncoder => _dvaeEncoder;
    internal ChatTtsMelDecoder<T>? DvaeDecoder => _dvaeDecoder;
    internal ChatTtsMelDecoder<T>? HiddenStateDecoder => _hiddenDecoder;
    internal VocosGenerator<T>? Vocos => _vocos;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        // config.GPT: a LlamaModel, whose embeddings and heads ChatTTS's Embed module supplies.
        _gpt = new LlamaNetwork<T>(Engine, _gptLayers, new LlamaConfiguration
        {
            VocabSize = 0, Hidden = o.HiddenDim, Layers = o.NumDecoderLayers, Heads = o.NumHeads, KeyValueHeads = o.NumHeads,
            Intermediate = o.IntermediateSize, RmsEpsilon = o.RmsNormEpsilon,
        });
        _gpt.InitializeLikeReference(random);
        Func<double> normal = () => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        _textTable = new TiedEmbeddingLayer<T>(TextVocabulary, o.HiddenDim);
        _textTable.Reinitialize(normal);
        _gptLayers.Add(_textTable);
        for (int q = 0; q < o.NumVq; q++)
        {
            var table = new TiedEmbeddingLayer<T>(CodebookSize + 1, o.HiddenDim);
            table.Reinitialize(normal);
            _gptLayers.Add(table);
            _codeTables.Add(table);
        }
        // weight_norm(nn.Linear(..., bias=False)): a 1-wide weight-normalized convolution over [1, hidden, positions],
        // initialized as nn.Linear is (U(±1/√fan_in)).
        double bound = 1.0 / Math.Sqrt(o.HiddenDim);
        NormedConv1DLayer<T> NewHead(int outputs)
        {
            var head = new NormedConv1DLayer<T>(o.HiddenDim, outputs, 1, 1, 1, 1, 0, false, ConvolutionNormalization.Weight, false);
            head.Reinitialize(() => bound * (2 * random.NextDouble() - 1));
            _gptLayers.Add(head);
            return head;
        }
        _textHead = NewHead(TextVocabulary);
        for (int q = 0; q < o.NumVq; q++) _codeHeads.Add(NewHead(CodebookSize + 1));

        // DVAE (config.DVAE), the hidden-state Decoder (config.Decoder) and Vocos (config.Vocos).
        _quantizer = new ChatTtsGroupedFsq<T>(Engine, _codecLayers, 2 * o.DvaeDim, o.FsqLevels, o.FsqGroups, o.FsqResiduals, random);
        _dvaeEncoder = new ChatTtsDvaeEncoder<T>(Engine, _codecLayers,
            new ChatTtsStackConfiguration(o.DvaeDim, 2 * o.DvaeDim, o.DvaeEncoderHidden, o.DvaeLayers, o.DvaeBottleneck),
            o.DvaeDim, o.SampleRate, o.FftSize, o.HopSize, o.MelChannels, _quantizer);
        _dvaeDecoder = new ChatTtsMelDecoder<T>(Engine, _codecLayers,
            new ChatTtsStackConfiguration(o.DvaeDim, o.DvaeDim, o.DvaeDecoderHidden, o.DvaeLayers, o.DvaeBottleneck), o.DvaeDim, o.MelChannels);
        _hiddenDecoder = new ChatTtsMelDecoder<T>(Engine, _codecLayers,
            new ChatTtsStackConfiguration(o.DecoderDim, o.DecoderDim, o.DecoderHidden, o.DecoderLayers, o.DecoderBottleneck), o.DecoderDim, o.MelChannels);
        _vocos = new VocosGenerator<T>(Engine, random, o.MelChannels, o.VocosDim, o.VocosIntermediate, o.VocosLayers, o.FftSize, o.HopSize);
        _codecLayers.AddRange(_vocos.Layers);

        AddEncoderDecoderLayers(Array.Empty<ILayer<T>>(), _gptLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
    }

    // ---------------------------------------------------------------- text

    /// <summary>The token ids of text: ChatTTS's control tokens as themselves, everything else through bert-base-chinese's
    /// WordPiece (each CJK character a word, lowercased), as its <c>BertTokenizerFast</c> with added tokens does.</summary>
    public int[] Tokenize(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var ids = new List<int>();
        foreach (var part in ControlTokenPattern.Split(text))
        {
            if (part.Length == 0) continue;
            int control = Array.IndexOf(ControlTokens, part);
            if (control >= 0) { ids.Add(21128 + control); continue; }
            ids.AddRange(_tokenizer.ConvertTokensToIds(_tokenizer.Tokenize(part)));
        }
        return ids.ToArray();
    }

    /// <inheritdoc />
    /// <remarks>The code-generation prompt (<c>Speaker.decorate_code_prompts</c>): <c>[Stts]</c>, then
    /// <c>[spk_emb]</c> when <see cref="TtsModelBase{T}.Voice"/> carries a speaker embedding (else
    /// <c>[empty_spk]</c>), the voice's transcript, the text, and <c>[Ptts]</c>.</remarks>
    protected override Tensor<T> PreprocessText(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        var body = Tokenize(text.Replace("[Stts]", "").Replace("[spk_emb]", "").Replace("[empty_spk]", "").Trim());
        if (body.Length == 0) throw new ArgumentException("The text has no tokens.", nameof(text));
        var ids = new List<int> { TokenId("[Stts]"), TokenId(Voice?.Reference is not null ? "[spk_emb]" : "[empty_spk]") };
        if (Voice?.ReferenceTokens is { } transcript)
            for (int i = 0; i < transcript.Length; i++) ids.Add((int)Math.Round(NumOps.ToDouble(transcript[i])));
        ids.AddRange(body.Take(_options.MaxTextLength));
        ids.Add(TokenId("[Ptts]"));
        var tensor = new Tensor<T>(new[] { ids.Count });
        for (int i = 0; i < ids.Count; i++) tensor[i] = NumOps.FromDouble(ids[i]);
        return tensor;
    }

    /// <summary>
    /// A voice for <see cref="TtsModelBase{T}.Voice"/>: a speaker embedding (<c>[HiddenDim]</c>, e.g. from
    /// <see cref="SampleRandomSpeaker"/>), and optionally a recording of the speaker (its DVAE codes follow the text as
    /// audio, <c>spk_smp</c>) with its transcript (<c>txt_smp</c>).
    /// </summary>
    public TtsVoice<T> CreateVoice(Tensor<T>? speakerEmbedding, Tensor<T>? recording = null, string? transcript = null)
    {
        if (speakerEmbedding is not null && speakerEmbedding.Length != _options.HiddenDim)
            throw new ArgumentException($"A speaker embedding has {_options.HiddenDim} values.", nameof(speakerEmbedding));
        Tensor<T>? tokens = null;
        if (transcript is not null)
        {
            var ids = Tokenize(transcript);
            tokens = new Tensor<T>(new[] { ids.Length });
            for (int i = 0; i < ids.Length; i++) tokens[i] = NumOps.FromDouble(ids[i]);
        }
        return new TtsVoice<T> { Reference = speakerEmbedding, ReferenceAudio = recording, ReferenceTokens = tokens };
    }

    /// <summary>A random speaker embedding (<c>Speaker.sample_random</c>): N(0, 1) per dimension, scaled by the speaker
    /// statistics' std and shifted by their mean once <see cref="LoadSpeakerStatistics"/> has loaded them.</summary>
    public Tensor<T> SampleRandomSpeaker(Random random)
    {
        Guard.NotNull(random);
        var speaker = new Tensor<T>(new[] { _options.HiddenDim });
        for (int i = 0; i < speaker.Length; i++)
        {
            double z = Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
            speaker[i] = NumOps.FromDouble(_speakerStd is null || _speakerMean is null ? z : z * _speakerStd[i] + _speakerMean[i]);
        }
        return speaker;
    }

    /// <summary>Sets the speaker statistics (<c>spk_stat</c>: the std, then the mean, <c>[2 · HiddenDim]</c>).</summary>
    public void LoadSpeakerStatistics(double[] statistics)
    {
        Guard.NotNull(statistics);
        int d = _options.HiddenDim;
        if (statistics.Length != 2 * d) throw new ArgumentException($"Expected {2 * d} values: the std, then the mean.", nameof(statistics));
        _speakerStd = statistics.Take(d).ToArray();
        _speakerMean = statistics.Skip(d).ToArray();
    }

    // ---------------------------------------------------------------- the GPT

    /// <summary>
    /// The input embeddings <c>[1, L, HiddenDim]</c> (<c>Embed.forward</c>): a text position's from the text table, an
    /// audio position's as the sum of its codes' tables; then the L2-normalized speaker embedding at every
    /// <c>[spk_emb]</c> text position (<c>Speaker.apply</c>).
    /// </summary>
    internal Tensor<T> Embed(IReadOnlyList<int[]> rows, IReadOnlyList<bool> textMask, Tensor<T>? speaker)
    {
        int length = rows.Count, d = _options.HiddenDim, vq = _options.NumVq;
        var textIds = new Tensor<T>(new[] { length });
        var mask = new Tensor<T>(new[] { length, d });
        for (int i = 0; i < length; i++)
        {
            textIds[i] = textMask[i] ? NumOps.FromDouble(rows[i][0]) : NumOps.Zero;
            if (textMask[i]) for (int k = 0; k < d; k++) mask[i, k] = NumOps.One;
        }
        var text = (_textTable ?? throw new InvalidOperationException("native model")).Forward(textIds);
        Tensor<T> CodeEmbedding(int q)
        {
            var ids = new Tensor<T>(new[] { length });
            for (int i = 0; i < length; i++) ids[i] = textMask[i] ? NumOps.Zero : NumOps.FromDouble(rows[i][q]);
            return _codeTables[q].Forward(ids);
        }
        var code = CodeEmbedding(0);
        for (int q = 1; q < vq; q++) code = Engine.TensorAdd(code, CodeEmbedding(q));
        var inverse = Engine.TensorSubtract(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(mask, NumOps.Zero), NumOps.One), mask);
        var embeddings = Engine.TensorAdd(Engine.TensorMultiply(text, mask), Engine.TensorMultiply(code, inverse));
        if (speaker is not null)
        {
            int speakerId = TokenId("[spk_emb]");
            double norm = 0;
            for (int k = 0; k < d; k++) norm += Math.Pow(NumOps.ToDouble(speaker[k]), 2);
            norm = Math.Max(Math.Sqrt(norm), 1e-12);
            var replace = new Tensor<T>(new[] { length, d });
            var values = new Tensor<T>(new[] { length, d });
            for (int i = 0; i < length; i++)
            {
                if (!textMask[i] || rows[i][0] != speakerId) continue;
                for (int k = 0; k < d; k++)
                {
                    replace[i, k] = NumOps.One;
                    values[i, k] = NumOps.FromDouble(NumOps.ToDouble(speaker[k]) / norm);
                }
            }
            var keep = Engine.TensorSubtract(Engine.TensorAddScalar(Engine.TensorMultiplyScalar(replace, NumOps.Zero), NumOps.One), replace);
            embeddings = Engine.TensorAdd(Engine.TensorMultiply(embeddings, keep), values);
        }
        return Engine.Reshape(embeddings, new[] { 1, length, d });
    }

    /// <summary>The GPT's final hidden states <c>[L, HiddenDim]</c>.</summary>
    internal Tensor<T> Hidden(IReadOnlyList<int[]> rows, IReadOnlyList<bool> textMask, Tensor<T>? speaker)
    {
        var (hidden, _) = Gpt.Forward(Embed(rows, textMask, speaker));
        return Engine.Reshape(hidden, new[] { rows.Count, _options.HiddenDim });
    }

    // A head on hidden states [N, HiddenDim]: logits [N, outputs].
    private Tensor<T> Head(NormedConv1DLayer<T> head, Tensor<T> hidden)
    {
        int n = hidden.Shape[0];
        var channels = Engine.Reshape(Engine.TensorTranspose(hidden), new[] { 1, _options.HiddenDim, n });
        var logits = head.Forward(channels);
        return Engine.TensorTranspose(Engine.Reshape(logits, new[] { logits.Shape[1], n }));
    }

    internal Tensor<T> TextLogits(Tensor<T> hidden) => Head(_textHead ?? throw new InvalidOperationException("native model"), hidden);

    internal Tensor<T> CodeLogits(int q, Tensor<T> hidden) => Head(_codeHeads[q], hidden);

    // ---------------------------------------------------------------- training

    /// <inheritdoc />
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        ThrowIfTokenMelTrainingUnsupported();
        return false;
    }

    /// <inheritdoc />
    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var objective = BuildObjective(sample);
        return TrainWithCustomObjective(sample.Tokens, sample.Tokens, (_, _) => objective(), Optimizer());
    }

    /// <inheritdoc />
    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        ThrowIfDisposed();
        var objective = BuildObjective(sample);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            return objective()[0];
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? Optimizer()
    {
        if (_suppliedOptimizer is not null) return _suppliedOptimizer;
        return _optimizer ??= PaperOptimizerFactory.VerifyHandBuilt(this, new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = _options.LearningRate, WeightDecay = _options.WeightDecay }));
    }

    // The utterance's codes [frames, NumVq], each in [0, 625).
    private int[][] CodesOf(TtsTrainingSample<T> sample)
    {
        int vq = _options.NumVq;
        if (sample.CodecTokens is { } tokens)
        {
            if (tokens.Rank != 2 || tokens.Shape[1] != vq)
                throw new ArgumentException($"Expected DVAE codes [frames, {vq}].", nameof(sample));
            var rows = new int[tokens.Shape[0]][];
            for (int f = 0; f < rows.Length; f++)
            {
                rows[f] = new int[vq];
                for (int q = 0; q < vq; q++)
                {
                    rows[f][q] = (int)Math.Round(NumOps.ToDouble(tokens[f, q]));
                    if (rows[f][q] < 0 || rows[f][q] >= CodebookSize)
                        throw new ArgumentOutOfRangeException(nameof(sample), $"DVAE codes lie in [0, {CodebookSize}).");
                }
            }
            return rows;
        }
        var audio = sample.Audio ?? throw new ArgumentException("ChatTTS trains on DVAE codes [frames, 4], or on the recording.", nameof(sample));
        return EncodeCodes(audio);
    }

    /// <summary>
    /// The training objective: the decorated text (as <see cref="PreprocessText"/> builds it, text positions), the
    /// utterance's code frames and the end frame (audio positions); each audio frame's 4 codes are predicted from the
    /// hidden state of the position before it, and the loss is the mean of the 4 heads' mean cross-entropies.
    /// </summary>
    internal Func<Tensor<T>> BuildObjective(TtsTrainingSample<T> sample)
    {
        Guard.NotNull(sample);
        if (!HasPaperLayers)
            throw new NotSupportedException("The objective needs ChatTTS's layers; this model was built from caller-supplied layers.");
        int vq = _options.NumVq;
        var rows = new List<int[]>();
        var textMask = new List<bool>();
        for (int i = 0; i < sample.Tokens.Length; i++)
        {
            int id = (int)Math.Round(NumOps.ToDouble(sample.Tokens[i]));
            if (id < 0 || id >= TextVocabulary) throw new ArgumentOutOfRangeException(nameof(sample), $"Text ids lie in [0, {TextVocabulary}).");
            rows.Add(Enumerable.Repeat(id, vq).ToArray());
            textMask.Add(true);
        }
        int firstFrame = rows.Count;
        foreach (var frame in CodesOf(sample))
        {
            rows.Add(frame);
            textMask.Add(false);
        }
        rows.Add(Enumerable.Repeat(EndCode, vq).ToArray());
        textMask.Add(false);
        int targets = rows.Count - firstFrame;
        var speaker = sample.SpeakerReference;
        return () =>
        {
            var hidden = Hidden(rows, textMask, speaker);                                                    // [L, d]
            var predicting = Engine.TensorSlice(hidden, new[] { firstFrame - 1, 0 }, new[] { targets, _options.HiddenDim });
            Tensor<T> Term(int q)
            {
                var logProbabilities = Engine.TensorLogSoftmax(CodeLogits(q, predicting), axis: -1);           // [targets, V]
                var picks = new Tensor<T>(logProbabilities._shape);
                for (int t = 0; t < targets; t++) picks[t, rows[firstFrame + t][q]] = NumOps.FromDouble(-1.0 / (targets * vq));
                return Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, picks), new[] { 0, 1 }, keepDims: false);
            }
            var total = Term(0);
            for (int q = 1; q < vq; q++) total = Engine.TensorAdd(total, Term(q));
            return total;
        };
    }

    /// <inheritdoc />
    /// <remarks>Training updates the GPT, its embeddings and heads; the DVAE, the Decoder and Vocos stay as loaded.</remarks>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
    {
        if (!HasPaperLayers) return parameters;
        var gpt = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(_gptLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(gpt.Contains).ToList();
    }

    // ---------------------------------------------------------------- sampling (ChatTTS/model/processors.py)

    /// <summary>
    /// One draw from logits as ChatTTS's generation does: divide by the temperature; apply the repetition penalty (each
    /// token seen n times in <paramref name="history"/> has its logit multiplied by penalty^n when negative, divided
    /// when positive); Hugging Face's top-p then top-k warpers, each keeping at least 3 tokens; then a multinomial draw
    /// from the softmax. <paramref name="forbidden"/> removes a token first (the end, before the first frame).
    /// </summary>
    internal static int Sample(double[] logits, double temperature, double penalty, IEnumerable<int> history, double topP, int topK,
        int forbidden, Random random)
    {
        int vocabulary = logits.Length;
        var scores = logits.Select(l => l / temperature).ToArray();
        if (penalty != 1.0)
        {
            var counts = new int[vocabulary];
            foreach (int token in history) if (token >= 0 && token < vocabulary) counts[token]++;
            for (int v = 0; v < vocabulary; v++)
            {
                if (counts[v] == 0) continue;
                double alpha = Math.Pow(penalty, counts[v]);
                scores[v] = scores[v] < 0 ? scores[v] * alpha : scores[v] / alpha;
            }
        }
        if (forbidden >= 0) scores[forbidden] = double.NegativeInfinity;
        const int minKeep = 3;
        // TopPLogitsWarper: sort ascending, drop tokens whose cumulative probability stays at or below 1 − p, keep the
        // last min_tokens_to_keep.
        var ascending = Enumerable.Range(0, vocabulary).OrderBy(v => scores[v]).ThenBy(v => v).ToArray();
        double max = scores.Max();
        var probabilities = scores.Select(s => double.IsNegativeInfinity(s) ? 0 : Math.Exp(s - max)).ToArray();
        double total = probabilities.Sum();
        double cumulative = 0;
        for (int i = 0; i < vocabulary; i++)
        {
            int v = ascending[i];
            cumulative += probabilities[v] / total;
            if (cumulative <= 1 - topP && i < vocabulary - minKeep) scores[v] = double.NegativeInfinity;
        }
        // TopKLogitsWarper: keep the scores at least the k-th largest, k = max(top_k, min_tokens_to_keep).
        int k = Math.Min(Math.Max(topK, minKeep), vocabulary);
        double threshold = scores.OrderByDescending(s => s).ElementAt(k - 1);
        for (int v = 0; v < vocabulary; v++) if (scores[v] < threshold) scores[v] = double.NegativeInfinity;
        max = scores.Max();
        var weights = scores.Select(s => double.IsNegativeInfinity(s) ? 0 : Math.Exp(s - max)).ToArray();
        double draw = random.NextDouble() * weights.Sum();
        for (int v = 0; v < vocabulary; v++)
        {
            if (weights[v] <= 0) continue;
            draw -= weights[v];
            if (draw <= 0) return v;
        }
        return Array.FindLastIndex(weights, w => w > 0);
    }

    private double[] Row(Tensor<T> logits, int row)
    {
        var values = new double[logits.Shape[1]];
        for (int v = 0; v < values.Length; v++) values[v] = NumOps.ToDouble(logits[row, v]);
        return values;
    }

    /// <summary>Refines text ids (<c>Chat._refine_text</c>): the text head continues <c>[Sbreak]</c> text
    /// <c>[Pbreak]</c> until <c>[Ebreak]</c>; the ids below <c>[break_0]</c> are kept, as <c>_infer</c> keeps them.</summary>
    internal int[] RefineTextIds(IReadOnlyList<int> textIds, Random random)
    {
        var o = _options;
        int vq = o.NumVq, end = TokenId("[Ebreak]"), firstBreak = TokenId("[break_0]");
        var rows = new List<int[]>();
        var textMask = new List<bool>();
        void Add(int id) { rows.Add(Enumerable.Repeat(id, vq).ToArray()); textMask.Add(true); }
        Add(TokenId("[Sbreak]"));
        foreach (int id in textIds) Add(id);
        Add(TokenId("[Pbreak]"));
        int start = rows.Count;
        var generated = new List<int>();
        for (int step = 0; step < o.MaxNewRefineTokens; step++)
        {
            var hidden = Hidden(rows, textMask, null);
            var last = Engine.TensorSlice(hidden, new[] { rows.Count - 1, 0 }, new[] { 1, o.HiddenDim });
            int next = Sample(Row(TextLogits(last), 0), o.RefineTemperature, o.RefineRepetitionPenalty,
                generated.Skip(Math.Max(0, generated.Count - 16)), o.RefineTopP, o.RefineTopK, step == 0 ? end : -1, random);
            Add(next);
            if (next == end) break;
            generated.Add(next);
        }
        return generated.Where(id => id < firstBreak).ToArray();
    }

    /// <summary>
    /// Generates code frames (<c>Chat._infer_code</c>) from the decorated prompt: returns the frames <c>[n][NumVq]</c>
    /// and the hidden states that produced them <c>[n, HiddenDim]</c>. Each step samples the 4 codes independently; the
    /// first frame may not end the utterance (the release regenerates an empty result).
    /// </summary>
    internal (List<int[]> Frames, Tensor<T> Hidden) GenerateCodes(IReadOnlyList<int> promptIds, IReadOnlyList<int[]> promptFrames,
        Tensor<T>? speaker, Random random)
    {
        var o = _options;
        int vq = o.NumVq;
        var rows = promptIds.Select(id => Enumerable.Repeat(id, vq).ToArray()).ToList();
        var textMask = Enumerable.Repeat(true, rows.Count).ToList();
        foreach (var frame in promptFrames) { rows.Add(frame); textMask.Add(false); }
        var frames = new List<int[]>();
        var hiddens = new List<Tensor<T>>();
        for (int step = 0; step < o.MaxNewCodeTokens; step++)
        {
            var hidden = Hidden(rows, textMask, speaker);
            var last = Engine.TensorSlice(hidden, new[] { rows.Count - 1, 0 }, new[] { 1, o.HiddenDim });
            var frame = new int[vq];
            bool end = false;
            for (int q = 0; q < vq; q++)
            {
                var history = frames.Skip(Math.Max(0, frames.Count - 16)).Select(f => f[q]);
                frame[q] = Sample(Row(CodeLogits(q, last), 0), o.CodeTemperature, o.CodeRepetitionPenalty, history,
                    o.CodeTopP, o.CodeTopK, step == 0 ? EndCode : -1, random);
                end |= frame[q] == EndCode;
            }
            if (end) break;
            frames.Add(frame);
            hiddens.Add(last);
            rows.Add(frame);
            textMask.Add(false);
        }
        if (frames.Count == 0) throw new InvalidOperationException("The GPT generated no audio frames.");
        return (frames, Engine.TensorConcatenate(hiddens.ToArray(), 0));
    }

    // ---------------------------------------------------------------- codec

    /// <summary>The DVAE codes <c>[frames][NumVq]</c> of a 24 kHz recording (<c>DVAE.sample_audio</c>).</summary>
    public int[][] EncodeCodes(Tensor<T> audio)
    {
        Guard.NotNull(audio);
        var encoder = _dvaeEncoder ?? throw new InvalidOperationException("ChatTTS's DVAE exists in native mode only.");
        var decoder = _dvaeDecoder ?? throw new InvalidOperationException("ChatTTS's DVAE exists in native mode only.");
        using var _ = new NoGradScope<T>();
        var codes = encoder.Encode(audio, decoder.Coefficients.Scale);
        int frames = codes.GetLength(0), vq = codes.GetLength(1);
        var rows = new int[frames][];
        for (int f = 0; f < frames; f++)
        {
            rows[f] = new int[vq];
            for (int q = 0; q < vq; q++) rows[f][q] = codes[f, q];
        }
        return rows;
    }

    /// <summary>The 100-band mel <c>[1, mels, 2n]</c> of code frames through the DVAE decoder.</summary>
    internal Tensor<T> MelFromCodes(IReadOnlyList<int[]> frames)
    {
        var quantizer = _quantizer ?? throw new InvalidOperationException("native model");
        var codes = new int[frames.Count, _options.NumVq];
        for (int f = 0; f < frames.Count; f++)
            for (int q = 0; q < _options.NumVq; q++) codes[f, q] = frames[f][q];
        var features = quantizer.Decode(codes);                                                            // [n, 2·dim]
        var channels = Engine.Reshape(Engine.TensorTranspose(features), new[] { 1, features.Shape[1], frames.Count });
        return (_dvaeDecoder ?? throw new InvalidOperationException("native model")).Forward(channels);
    }

    /// <summary>The 100-band mel <c>[1, mels, 2n]</c> of the GPT's hidden states <c>[n, HiddenDim]</c> through the
    /// Decoder.</summary>
    internal Tensor<T> MelFromHidden(Tensor<T> hidden)
    {
        int n = hidden.Shape[0];
        var channels = Engine.Reshape(Engine.TensorTranspose(hidden), new[] { 1, _options.HiddenDim, n });
        return (_hiddenDecoder ?? throw new InvalidOperationException("native model")).Forward(channels);
    }

    /// <summary>The waveform of code frames <c>[frames, NumVq]</c> (DVAE decoder, then Vocos).</summary>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(tokens);
        Guard.NotNull(tokens);
        if (tokens.Rank != 2 || tokens.Shape[1] != _options.NumVq) throw new ArgumentException($"Expected codes [frames, {_options.NumVq}].", nameof(tokens));
        var frames = new List<int[]>();
        for (int f = 0; f < tokens.Shape[0]; f++)
            frames.Add(Enumerable.Range(0, _options.NumVq).Select(q => (int)Math.Round(NumOps.ToDouble(tokens[f, q]))).ToArray());
        using var _ = new NoGradScope<T>();
        var audio = (_vocos ?? throw new InvalidOperationException("native model")).Forward(MelFromCodes(frames));
        return audio.Reshape(new[] { audio.Length });
    }

    /// <summary>The DVAE codes <c>[frames, NumVq]</c> of a recording.</summary>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(audio);
        var rows = EncodeCodes(audio);
        var tokens = new Tensor<T>(new[] { rows.Length, _options.NumVq });
        for (int f = 0; f < rows.Length; f++)
            for (int q = 0; q < _options.NumVq; q++) tokens[f, q] = NumOps.FromDouble(rows[f][q]);
        return tokens;
    }

    // ---------------------------------------------------------------- synthesis

    /// <inheritdoc />
    /// <remarks>
    /// <paramref name="input"/> is the decorated prompt (<see cref="PreprocessText"/>). The text between its
    /// decorations is refined first when <see cref="ChatTTSOptions.RefineText"/> is set (the refined ids replace it, as
    /// re-tokenizing the release's decoded text gives the same ids). The voice's recording, if any, follows as audio
    /// positions. The waveform <c>[samples]</c> comes from the hidden-state Decoder (or the DVAE) and Vocos.
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (!HasPaperLayers) return base.PredictCore(input);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed);
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            using var _ = new NoGradScope<T>();
            var ids = Enumerable.Range(0, input.Length).Select(i => (int)Math.Round(NumOps.ToDouble(input[i]))).ToList();
            if (ids.Any(id => id < 0 || id >= TextVocabulary))
                throw new ArgumentOutOfRangeException(nameof(input), $"Text ids lie in [0, {TextVocabulary}).");
            int stts = TokenId("[Stts]"), ptts = TokenId("[Ptts]");
            int head = ids.Count >= 2 && ids[0] == stts ? 2 : 0;
            int tail = ids.Count > head && ids[^1] == ptts ? 1 : 0;
            if (_options.RefineText && ids.Count - head - tail > 0)
            {
                var refined = RefineTextIds(ids.Skip(head).Take(ids.Count - head - tail).ToList(), random);
                if (refined.Length > 0) ids = ids.Take(head).Concat(refined).Concat(ids.Skip(ids.Count - tail)).ToList();
            }
            var promptFrames = Voice?.ReferenceAudio is { } recording ? EncodeCodes(recording) : Array.Empty<int[]>();
            var (frames, hidden) = GenerateCodes(ids, promptFrames, Voice?.Reference, random);
            var mel = _options.UseHiddenStateDecoder ? MelFromHidden(hidden) : MelFromCodes(frames);
            var audio = (_vocos ?? throw new InvalidOperationException("native model")).Forward(mel);
            return audio.Reshape(new[] { audio.Length });
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // ---------------------------------------------------------------- pretrained weights

    /// <summary>
    /// Loads the released networks from ChatTTS's <c>asset</c> directory: <c>gpt/model.safetensors</c> (the
    /// <c>LlamaModel</c>), <c>Embed.safetensors</c> (tables and weight-normalized heads), <c>DVAE.safetensors</c>,
    /// <c>Decoder.safetensors</c> and <c>Vocos.safetensors</c>. The speaker statistics (<c>spk_stat</c>) ship in the
    /// release's configuration; pass them to <see cref="LoadSpeakerStatistics"/>.
    /// </summary>
    public void LoadPretrainedWeights(string assetDirectory)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(assetDirectory)) throw new ArgumentException("The asset directory is required.", nameof(assetDirectory));
        var o = _options;
        Gpt.LoadTorchWeights(Reader(Path.Combine(assetDirectory, "gpt", "model.safetensors")), prefix: "");
        LoadEmbedWeights(Reader(Path.Combine(assetDirectory, "Embed.safetensors")));
        var dvae = Reader(Path.Combine(assetDirectory, "DVAE.safetensors"));
        (_dvaeEncoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(dvae, "");
        (_dvaeDecoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(dvae, "");
        (_hiddenDecoder ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader(Path.Combine(assetDirectory, "Decoder.safetensors")), "");
        (_vocos ?? throw new InvalidOperationException("native model")).LoadTorchWeights(Reader(Path.Combine(assetDirectory, "Vocos.safetensors")),
            o.MelChannels, o.VocosIntermediate);
    }

    /// <summary>Loads <c>Embed</c>'s state: <c>emb_text</c>, <c>emb_code.{q}</c>, and the heads' weight-norm
    /// parametrizations (<c>original0</c> the gain, <c>original1</c> the direction).</summary>
    internal void LoadEmbedWeights(Func<string, int[], double[]> read)
    {
        var o = _options;
        int d = o.HiddenDim, codes = CodebookSize + 1;
        (_textTable ?? throw new InvalidOperationException("native model")).LoadTable(read("emb_text.weight", new[] { TextVocabulary, d }));
        for (int q = 0; q < o.NumVq; q++) _codeTables[q].LoadTable(read($"emb_code.{q}.weight", new[] { codes, d }));
        void Head(NormedConv1DLayer<T> head, string name, int outputs) =>
            head.LoadTorchWeights(read(name + ".parametrizations.weight.original1", new[] { outputs, d }),
                read(name + ".parametrizations.weight.original0", new[] { outputs, 1 }), new double[outputs]);
        Head(_textHead ?? throw new InvalidOperationException("native model"), "head_text", TextVocabulary);
        for (int q = 0; q < o.NumVq; q++) Head(_codeHeads[q], $"head_code.{q}", codes);
    }

    private static Func<string, int[], double[]> Reader(string path)
    {
        var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
        return (name, shape) =>
        {
            if (!file.TryGetValue(name, out var tensor))
                throw new InvalidDataException($"The checkpoint {path} has no tensor '{name}'.");
            var actual = tensor.Shape.ToArray();
            if (actual.Aggregate(1, (a, b) => a * b) != shape.Aggregate(1, (a, b) => a * b))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}] in the checkpoint but [{string.Join(", ", shape)}] " +
                    "in this model; build the model with the checkpoint's configuration.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        };
    }

    // ---------------------------------------------------------------- housekeeping

    /// <inheritdoc />
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ChatTTS-Native" : "ChatTTS-ONNX",
            Description = "ChatTTS: a generative speech model for daily dialogue (2noise, 2024)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "ChatTTS";
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(ChatTTS<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
