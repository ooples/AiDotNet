using AiDotNet.Attributes;
using AiDotNet.Audio.Codecs;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Safety.Watermarking;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization.Algorithms;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// A turn of conversation CSM conditions on: who spoke (<see cref="Speaker"/>), what they said (<see cref="Text"/>), and
/// the recording of it (<see cref="Audio"/>, at the model's 24 kHz).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class CsmSegment<T>
{
    /// <summary>The speaker's index (CSM writes it as "[n]" in front of the text).</summary>
    public int Speaker { get; init; }

    /// <summary>The words spoken.</summary>
    public required string Text { get; init; }

    /// <summary>The recording, <c>[samples]</c> at the model's sample rate.</summary>
    public required Tensor<T> Audio { get; init; }
}

/// <summary>A position of CSM's backbone: a text token (<see cref="Codes"/> null) or an audio frame's codes.</summary>
internal readonly record struct CsmFrame(int Text, int[]? Codes);

/// <summary>
/// CSM, Sesame's Conversational Speech Model (Sesame 2025, "Crossing the uncanny valley of conversational voice"):
/// a Llama backbone reads the conversation — text and Mimi's audio codes interleaved, one frame per position — and predicts
/// each next audio frame's first codebook; a small Llama audio decoder then predicts the frame's other 31 codebooks one by
/// one from the backbone's state; Mimi decodes the frames to 24 kHz audio, which SilentCipher watermarks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>Sesame published a blog post, not a paper, and released inference code (SesameAILabs/csm <c>models.py</c>,
/// <c>generator.py</c>, <c>watermarking.py</c>). The blog fixes the design — two autoregressive Transformers over Mimi's
/// split-RVQ tokens (one semantic and 31 acoustic codebooks at 12.5 Hz), a Llama tokenizer for text, the zeroth codebook
/// predicted by the backbone and the rest by the decoder with a separate head for each — and its training's compute
/// amortization: the decoder learns on a random 1/16 of the frames, the zeroth codebook on all of them. The release fills
/// the rest. Each position of the backbone is a frame of 33 tokens (32 audio codes and one text token) of which a mask
/// keeps either the text token or the codes; the embeddings it keeps are summed. A turn is its text — "[speaker]words"
/// between Llama 3's begin and end of text, one token a frame — then its audio's frames and a frame of zeros. Generation
/// samples every codebook with top-k 50 at temperature 0.9 until a frame of zeros, the decoder running afresh from the
/// backbone's last state for each frame.</para>
/// <para>Training follows the blog's split and Hugging Face's CSM port (written with Sesame), which is the only training
/// code: the backbone's next-frame cross-entropy on codebook 0 over the audio frames, plus the decoder's cross-entropy on
/// codebooks 1–31 over the frames chosen for it, with the Trainer's default AdamW.</para>
/// <para><b>For Beginners:</b> CSM speaks one turn of a conversation at a time. Give it the earlier turns (who said what,
/// and how it sounded) and it says the next line in a voice and manner that fit them.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var model = new CSM&lt;float&gt;(architecture, new CSMOptions { WatermarkModelPath = "silentcipher/44_1_khz/73999_iteration" });
/// model.LoadReferenceWeights("csm-1b/ckpt.pt");
/// model.LoadCodecWeights("moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors");
/// var audio = model.Generate("Hello from Sesame.", speaker: 0, context: Array.Empty&lt;CsmSegment&lt;float&gt;&gt;());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Crossing the uncanny valley of conversational voice", "https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice",
    Year = 2025, Authors = "Sesame")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-5, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.0,
    Source = "Sesame published no training recipe: the blog gives the loss split (codebook 0 on every frame, the decoder on a "
             + "random 1/16 of them) and the release has no training code. Hugging Face's CSM port, written with Sesame, trains "
             + "through the Trainer, whose defaults are these: AdamW at 5e-5 with betas (0.9, 0.999), eps 1e-8, no weight "
             + "decay, a linear decay to 0 without warm-up, gradient norm clipped to 1.0.")]
public partial class CSM<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly CSMOptions _options;
    private readonly bool _useNativeMode;
    private bool _disposed;

    private LlamaNetwork<T>? _backbone;
    private LlamaNetwork<T>? _decoder;
    private BiasFreeLinearLayer<T>? _textEmbeddings;
    private BiasFreeLinearLayer<T>? _audioEmbeddings;
    private BiasFreeLinearLayer<T>? _projection;
    private BiasFreeLinearLayer<T>? _codebook0Head;
    private BiasFreeLinearLayer<T>[] _audioHeads = Array.Empty<BiasFreeLinearLayer<T>>();
    private Mimi<T>? _mimi;
    private SilentCipherNetwork<T>? _watermarker;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    private readonly List<LayerBase<T>> _modelLayers = new();
    private readonly List<LayerBase<T>> _codecLayers = new();

    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();

    /// <summary>Llama 3's begin and end of text.</summary>
    internal const int BeginOfText = 128000, EndOfText = 128001;

    /// <summary>Creates an ONNX-backed CSM for inference.</summary>
    public CSM(NeuralNetworkArchitecture<T> architecture, string modelPath, CSMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CSMOptions();
        Options = _options;
        _useNativeMode = false;
        ConfigureBase();
        if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a native, trainable CSM.</summary>
    public CSM(NeuralNetworkArchitecture<T> architecture, CSMOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new CSMOptions();
        Options = _options;
        _useNativeMode = true;
        ConfigureBase();
        InitializeLayers();
    }

    private void ConfigureBase()
    {
        var o = _options;
        if (o.AudioCodebooks < 2) throw new ArgumentOutOfRangeException(nameof(o.AudioCodebooks), o.AudioCodebooks, "CSM splits each frame between the backbone's codebook and the decoder's; it needs at least two.");
        if (o.MimiCodebookSize > o.AudioVocabSize) throw new ArgumentException("The model's audio tables must cover Mimi's codebooks.");
        if (o.VocabSize <= EndOfText) throw new ArgumentException($"CSM reads Llama 3's tokens; the text vocabulary must exceed {EndOfText}.");
        if (o.TopK <= 0) throw new ArgumentOutOfRangeException(nameof(o.TopK), o.TopK, "Top-k sampling keeps at least one code.");
        if (o.Temperature <= 0) throw new ArgumentOutOfRangeException(nameof(o.Temperature), o.Temperature, "The release always samples; the temperature is positive.");
        if (o.DecoderFrameFraction <= 0 || o.DecoderFrameFraction > 1)
            throw new ArgumentOutOfRangeException(nameof(o.DecoderFrameFraction), o.DecoderFrameFraction, "The decoder's share of frames lies in (0, 1].");
        if (o.Watermark && (o.WatermarkKey is null || o.WatermarkKey.Length != new SilentCipherConfiguration().KeyBytes))
            throw new ArgumentException("The watermark key has five bytes.");
        base.SampleRate = o.SampleRate;
        base.HopSize = MimiConfiguration().SamplesPerFrame;
        base.HiddenDim = o.HiddenDim;
        // Hugging Face's Trainer clips the gradient norm (max_grad_norm), which the optimizer does here.
        MaxGradNorm = NumOps.Zero;
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;

    /// <inheritdoc />
    public int MaxTextLength => _options.MaxTextLength;

    /// <inheritdoc />
    public int NumCodebooks => _options.AudioCodebooks;

    /// <inheritdoc />
    /// <remarks>Mimi's codebook size; the model's tables have <see cref="CSMOptions.AudioVocabSize"/> entries.</remarks>
    public int CodebookSize => _options.MimiCodebookSize;

    /// <inheritdoc />
    public int CodecFrameRate => (int)Math.Round(_options.SampleRate / (double)HopSize);

    /// <inheritdoc />
    /// <remarks>CSM trains on a recording (24 kHz) or its Mimi codes (<see cref="TtsTrainingSample{T}.CodecTokens"/>,
    /// <c>[frames, 32]</c> as <see cref="EncodeToTokens"/> gives them) with its transcript as text token ids
    /// (<see cref="TtsTrainingSample{T}.Tokens"/>) or <see cref="TtsTrainingSample{T}.Text"/>; the speaker is
    /// <see cref="TtsTrainingSample{T}.SpeakerId"/> (0 when unset).</remarks>
    protected override TtsSupervision RequiredSupervision => TtsSupervision.CodecTokens;

    /// <inheritdoc />
    public override bool TrainsOnSampleObjective => true;

    /// <inheritdoc />
    public override int CodecTokenCodebooks => _options.AudioCodebooks;

    /// <inheritdoc />
    public override int CodecTokenVocabulary => _options.MimiCodebookSize;

    private bool HasPaperLayers => _backbone is not null;
    private InvalidOperationException NotNative() => new($"{GetType().Name}'s networks exist in native mode only.");

    internal LlamaNetwork<T> Backbone => _backbone ?? throw NotNative();
    internal LlamaNetwork<T> AudioDecoder => _decoder ?? throw NotNative();
    internal Mimi<T> Codec => _mimi ?? throw NotNative();
    internal IReadOnlyList<LayerBase<T>> ModelLayers => _modelLayers;
    internal IReadOnlyList<LayerBase<T>> CodecLayers => _codecLayers;

    /// <summary>Llama 3's tokenizer, as the release loads it (<c>meta-llama/Llama-3.2-1B</c>).</summary>
    internal static ByteLevelBpeTokenizer Tokenizer => Llama3Tokenizers.Base;

    internal MimiConfiguration MimiConfiguration() => new()
    {
        SampleRate = _options.SampleRate, Dimension = _options.MimiDimension, Filters = _options.MimiFilters,
        Ratios = (int[])_options.MimiRatios.Clone(), TransformerLayers = _options.MimiTransformerLayers, Heads = _options.MimiHeads,
        FeedForward = _options.MimiFeedForward, Context = _options.MimiContext, Codebooks = _options.AudioCodebooks,
        CodebookSize = _options.MimiCodebookSize, CodebookDim = _options.MimiCodebookDim,
    };

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
        var o = _options;
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Architecture.RandomSeed ?? o.SamplingSeed);
        LlamaConfiguration Flavor(int hidden, int layers, int heads, int kvHeads, int intermediate) => new()
        {
            VocabSize = 0, Hidden = hidden, Layers = layers, Heads = heads, KeyValueHeads = kvHeads, Intermediate = intermediate,
            RmsEpsilon = o.RmsEpsilon, RopeTheta = o.RopeTheta, RopeScalingFactor = o.RopeScalingFactor,
            RopeLowFrequencyFactor = o.RopeLowFrequencyFactor, RopeHighFrequencyFactor = o.RopeHighFrequencyFactor,
            RopeOriginalMaxPositions = o.RopeOriginalMaxPositions, IncludeHead = false,
        };
        _backbone = new LlamaNetwork<T>(Engine, _modelLayers, Flavor(o.HiddenDim, o.BackboneLayers, o.BackboneHeads, o.BackboneKeyValueHeads, o.BackboneIntermediate));
        _decoder = new LlamaNetwork<T>(Engine, _modelLayers, Flavor(o.DecoderHidden, o.DecoderLayers, o.DecoderHeads, o.DecoderKeyValueHeads, o.DecoderIntermediate));
        TLayer Add<TLayer>(TLayer layer) where TLayer : LayerBase<T>
        {
            _modelLayers.Add(layer);
            return layer;
        }
        _textEmbeddings = Add(new BiasFreeLinearLayer<T>(1, o.VocabSize * o.HiddenDim));
        _audioEmbeddings = Add(new BiasFreeLinearLayer<T>(1, o.AudioVocabSize * o.AudioCodebooks * o.HiddenDim));
        _projection = Add(new BiasFreeLinearLayer<T>(o.HiddenDim, o.DecoderHidden));
        _codebook0Head = Add(new BiasFreeLinearLayer<T>(o.HiddenDim, o.AudioVocabSize));
        _audioHeads = Enumerable.Range(1, o.AudioCodebooks - 1).Select(_ => Add(new BiasFreeLinearLayer<T>(o.DecoderHidden, o.AudioVocabSize))).ToArray();
        InitializeLikeReference(random);
        _mimi = new Mimi<T>(Engine, MimiConfiguration(), _codecLayers);
        AddUnsplitLayers(_modelLayers.Cast<ILayer<T>>().ToList());
        ComponentLayers.AddRange(_codecLayers);
    }

    // Hugging Face's CsmPreTrainedModel._init_weights (the release's audio_head is torch.empty, never initialized):
    // linear weights, embeddings and each codebook head N(0, 0.02²), RMSNorm weights 1.
    private void InitializeLikeReference(Random random)
    {
        Backbone.InitializeLikeReference(random);
        AudioDecoder.InitializeLikeReference(random);
        double Normal() => 0.02 * Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        foreach (var layer in new[] { _textEmbeddings!, _audioEmbeddings!, _projection!, _codebook0Head! }.Concat(_audioHeads))
        {
            var weights = layer.Weights;
            for (int i = 0; i < weights.Length; i++) weights[i] = Ops.FromDouble(Normal());
            Engine.InvalidatePersistentTensor(weights);
        }
    }

    // ---------------------------------------------------------------- frames

    /// <summary>The text frames of a turn: "[speaker]text" between Llama 3's begin and end of text (the release's
    /// <c>TemplateProcessing</c>), one token a frame.</summary>
    internal static List<CsmFrame> TextFrames(string text, int speaker) =>
        new[] { BeginOfText }.Concat(Tokenizer.Encode($"[{speaker}]{text}")).Append(EndOfText).Select(id => new CsmFrame(id, null)).ToList();

    /// <summary>The audio frames of codes <c>[frames, codebooks]</c> and the frame of zeros that ends them.</summary>
    internal static List<CsmFrame> AudioFrames(int[,] codes)
    {
        int frames = codes.GetLength(0), books = codes.GetLength(1);
        var result = new List<CsmFrame>(frames + 1);
        for (int f = 0; f < frames; f++) result.Add(new CsmFrame(0, Enumerable.Range(0, books).Select(k => codes[f, k]).ToArray()));
        result.Add(new CsmFrame(0, new int[books]));
        return result;
    }

    // The table's rows at the given indices [n, hidden].
    private Tensor<T> Rows(BiasFreeLinearLayer<T> table, int width, IReadOnlyList<int> indices)
    {
        int count = table.Weights.Length / width;
        var index = new Tensor<int>(new[] { indices.Count });
        for (int i = 0; i < indices.Count; i++)
        {
            if (indices[i] < 0 || indices[i] >= count) throw new ArgumentOutOfRangeException(nameof(indices), indices[i], $"Ids lie in [0, {count}).");
            index[i] = indices[i];
        }
        return Engine.TensorIndexSelect(Engine.Reshape(LlamaNetwork<T>.Paged(table.Weights), new[] { count, width }), index, 0);
    }

    /// <summary>The audio table's rows for codebook <paramref name="codebook"/>'s codes (<c>_embed_audio</c>).</summary>
    private Tensor<T> AudioEmbeddings(int codebook, IReadOnlyList<int> codes) =>
        Rows(_audioEmbeddings!, _options.HiddenDim, codes.Select(c => codebook * _options.AudioVocabSize + c).ToArray());

    /// <summary>
    /// The backbone's inputs <c>[1, frames, hidden]</c> (<c>_embed_tokens</c> with the mask): a text frame's token
    /// embedding, or the sum of an audio frame's codes' embeddings, each codebook offset into the audio table.
    /// </summary>
    internal Tensor<T> EmbedFrames(IReadOnlyList<CsmFrame> frames)
    {
        int h = _options.HiddenDim, books = _options.AudioCodebooks;
        var textPositions = new List<int>();
        var audioPositions = new List<int>();
        var textIds = new List<int>();
        var audioIds = new List<int>();
        for (int p = 0; p < frames.Count; p++)
        {
            var frame = frames[p];
            if (frame.Codes is null)
            {
                textPositions.Add(p);
                textIds.Add(frame.Text);
                continue;
            }
            if (frame.Codes.Length != books) throw new ArgumentException($"An audio frame has {books} codes.", nameof(frames));
            audioPositions.Add(p);
            for (int k = 0; k < books; k++)
            {
                int code = frame.Codes[k];
                if (code < 0 || code >= _options.AudioVocabSize) throw new ArgumentOutOfRangeException(nameof(frames), code, $"Codes lie in [0, {_options.AudioVocabSize}).");
                audioIds.Add(k * _options.AudioVocabSize + code);
            }
        }
        var parts = new List<Tensor<T>>();
        if (textIds.Count > 0) parts.Add(Rows(_textEmbeddings!, h, textIds));
        if (audioIds.Count > 0)
            parts.Add(Engine.ReduceSum(Engine.Reshape(Rows(_audioEmbeddings!, h, audioIds), new[] { audioPositions.Count, books, h }), new[] { 1 }, keepDims: false));
        var stacked = parts.Count == 1 ? parts[0] : Engine.TensorConcatenate(parts.ToArray(), 0);
        var order = new int[frames.Count];
        for (int i = 0; i < textPositions.Count; i++) order[textPositions[i]] = i;
        for (int i = 0; i < audioPositions.Count; i++) order[audioPositions[i]] = textPositions.Count + i;
        var index = new Tensor<int>(new[] { order.Length }, new Vector<int>(order));
        return Engine.Reshape(Engine.TensorIndexSelect(stacked, index, 0), new[] { 1, frames.Count, h });
    }

    private Tensor<T> Linear(BiasFreeLinearLayer<T> layer, Tensor<T> x)
    {
        var shape = x.Shape.ToArray();
        int last = shape[^1], rows = x.Length / last;
        var y = layer.Forward(Engine.Reshape(x, new[] { rows, last }));
        shape[^1] = y.Shape[1];
        return Engine.Reshape(y, shape);
    }

    /// <summary>The audio decoder's hidden states <c>[B, n, decoder hidden]</c> of inputs in the backbone's width
    /// <c>[B, n, hidden]</c> (projected first), continuing <paramref name="cache"/>.</summary>
    internal Tensor<T> DecoderHidden(Tensor<T> inputs, LlamaCache<T>? cache) =>
        AudioDecoder.Forward(Linear(_projection!, inputs), null, null, cache, lastLogitsOnly: false).Hidden;

    /// <summary>The logits of codebook <paramref name="codebook"/> (1 or more) from decoder states <c>[n, decoder hidden]</c>.</summary>
    internal Tensor<T> CodebookLogits(int codebook, Tensor<T> hidden) => _audioHeads[codebook - 1].Forward(hidden);

    /// <summary>Codebook 0's logits from backbone states <c>[..., hidden]</c>.</summary>
    internal Tensor<T> Codebook0Logits(Tensor<T> hidden) => Linear(_codebook0Head!, hidden);
}
