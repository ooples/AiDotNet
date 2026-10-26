using AiDotNet.ComputerVision.Segmentation;
using AiDotNet.ComputerVision.Segmentation.Foundation;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>One turn of a GLaMM conversation.</summary>
/// <param name="User">The user's message; it may name regions with <c>&lt;bbox&gt;</c> tokens.</param>
/// <param name="Assistant">The answer; each <c>[SEG]</c> in it has a mask.</param>
public sealed record GlammTurn(string User, string Assistant);

/// <summary>A GLaMM training sample: an image, a conversation about it, the regions its <c>&lt;bbox&gt;</c> tokens name,
/// and a mask for every <c>[SEG]</c> in the answers.</summary>
/// <param name="Image">The RGB image <c>[3, H, W]</c> with values in 0..1.</param>
/// <param name="Turns">The conversation.</param>
/// <param name="Regions">Per <c>&lt;bbox&gt;</c>, in order, its box (x1, y1, x2, y2) as fractions of the image, or null.</param>
/// <param name="Masks">Per <c>[SEG]</c> in the answers, in order, its mask <c>[H, W]</c> (1 inside, 0 outside), or null.</param>
public sealed record GlammSample<T>(Tensor<T> Image, IReadOnlyList<GlammTurn> Turns, IReadOnlyList<double[]>? Regions = null,
    IReadOnlyList<Tensor<T>>? Masks = null);

/// <summary>GLaMM's answer: the text and a mask for every <c>[SEG]</c> in it.</summary>
/// <param name="Text">The answer, as the official app shows it.</param>
/// <param name="Masks">Per <c>[SEG]</c>, in order, its mask <c>[H, W]</c> at the image's size (1 inside, 0 outside).</param>
public sealed record GlammAnswer<T>(string Text, IReadOnlyList<Tensor<T>> Masks);

/// <summary>
/// GLaMM: Pixel Grounding Large Multimodal Model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GLaMM (Rasheed et al., 2024) is LLaVA-1.5 (a CLIP ViT-L/14-336 image encoder, a GELU projector and Vicuna-7B) with a
/// region encoder, so a question can point at boxes (<c>&lt;bbox&gt;</c>), and SAM ViT-H, so the answer can point at
/// pixels: every <c>[SEG]</c> the language model writes is decoded into a mask from that token's hidden state. It is
/// built as the official code builds it (mbzuai-oryx/groundingLMM), and one training sample matches the official
/// computation to 1e-9.
/// </para>
/// <para>
/// <see cref="Chat"/> runs the official app: the instruction after "The &lt;image&gt; provides an overview of the
/// picture.", greedy decoding (the released checkpoints set no sampling), and the masks of the [SEG] tokens.
/// <see cref="SegmentExpression"/>, <see cref="DescribeGrounded"/> and <see cref="CaptionRegion"/> use the official
/// referring-segmentation, grounded-caption and region-caption prompts. Training takes conversations with masks
/// (<see cref="TrainConversation"/>): the language-model loss plus 2 x mask cross-entropy plus 0.5 x DICE, updating the
/// modules the official training updates (the token embeddings and head, LoRA on q_proj and v_proj, the region encoder,
/// the [SEG] projection and SAM's mask decoder).
/// </para>
/// <para>
/// Images are read in 0..1 at any size. As the official processors do, CLIP sees the shorter side resized to 336
/// (bicubic) and centre-cropped, and SAM the longer side resized to 1024 (bilinear) and zero-padded. Masks come back at
/// the image's size.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "GLaMM: Pixel Grounding Large Multimodal Model" (Rasheed et al., CVPR 2024)</item></list></para>
/// <para><b>For Beginners:</b> GLaMM chats about an image and draws a mask on it for each object it mentions.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.TextGeneration)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[PreprocessesInput("GlobalInput resizes, centre-crops and normalizes the image as CLIPImageProcessor does, and "
    + "ClipVisionNetwork unfolds it into patches before the bias-free patch embedding, its first layer.")]
[StackInputLayout(TensorAxis.Batch, TensorAxis.Length, TensorAxis.Features)]
[ResearchPaper(
    "GLaMM: Pixel Grounding Large Multimodal Model",
    "https://arxiv.org/abs/2311.03356",
    Year = 2024,
    Authors = "Rasheed et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 3e-4, Beta1 = 0.9, Beta2 = 0.95, WeightDecay = 0.0,
                MaxGradientNorm = 1.0, Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 100,
                PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
                Source = "The official train.py: AdamW with a learning rate of 3e-4, betas (0.9, 0.95) and no weight decay, "
                        + "DeepSpeed's WarmupDecayLR (100 linear warmup steps, then linear decay to zero over the run), "
                        + "and gradient clipping at 1.0.")]
public partial class GLaMM<T> : VisionLanguageModelBase<T>, IGroundedConversationTrainable<T>
{
    private const string BeginText = "The <im_start><image><im_end> provides an overview of the picture.\n";
    private static readonly double[] ClipMean = { 0.48145466, 0.4578275, 0.40821073 };
    private static readonly double[] ClipStd = { 0.26862954, 0.26130258, 0.27577711 };

    private readonly GLaMMOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;
    private GlammNetwork<T>? _network;
    private GlammTokenizer? _tokenizer;
    private HashSet<Tensor<T>>? _trainable;


    internal GlammNetwork<T> Network => _network ?? throw new InvalidOperationException("The network is not initialized.");

    /// <summary>The tokenizer, read from the options on first use when the network was built without it.</summary>
    internal GlammTokenizer Tokenizer => _tokenizer ??= LoadTokenizer();

    private GlammTokenizer LoadTokenizer()
    {
        var model = TokenizerBytes() ?? throw new InvalidOperationException(
            "GLaMM reads Vicuna's SentencePiece model: set GLaMMOptions.TokenizerPath (tokenizer.model) or TokenizerModel.");
        var tokenizer = new GlammTokenizer(new SentencePieceModel(model));
        int expected = _options.TokenizerPieces + GlammTokenizer.AddedTokens.Length;
        if (tokenizer.VocabularySize != expected)
            throw new InvalidOperationException($"The tokenizer has {tokenizer.VocabularySize} tokens but the network embeds {expected} "
                + $"(GLaMMOptions.TokenizerPieces {_options.TokenizerPieces} plus {GlammTokenizer.AddedTokens.Length} added).");
        return tokenizer;
    }

    private byte[]? TokenizerBytes() => _options.TokenizerModel
        ?? (_options.TokenizerPath is { Length: > 0 } path ? File.ReadAllBytes(path) : null);

    public GLaMM(
        NeuralNetworkArchitecture<T> architecture,
        GLaMMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        // The defaults describe the released model; its tokenizer must still be supplied (TokenizerPath or TokenizerModel).
        _options = options ?? new GLaMMOptions(); Options = _options;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? throw new InvalidOperationException("GLaMM declares its paper recipe.");
        base.ImageSize = _options.ClipImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.LlmHidden;
        InitializeLayers();
    }

    /// <summary>The configuration the options describe.</summary>
    internal static GlammConfiguration Configuration(GLaMMOptions o, int vocabulary) => new()
    {
        Clip = new ClipVisionConfiguration
        {
            ImageSize = o.ClipImageSize, PatchSize = o.ClipPatchSize, Hidden = o.ClipHidden, Layers = o.ClipLayers, Heads = o.ClipHeads,
            Intermediate = o.ClipIntermediate,
        },
        Llm = new LlamaConfiguration
        {
            VocabSize = vocabulary, Hidden = o.LlmHidden, Layers = o.LlmLayers, Heads = o.LlmHeads, KeyValueHeads = o.LlmKeyValueHeads,
            Intermediate = o.LlmIntermediate, LoraRank = o.LoraRank, LoraAlpha = o.LoraAlpha, LoraDropout = o.LoraDropout,
        },
        Region = new GlammRegionConfiguration
        {
            Embed = o.ClipHidden, Grid = o.ClipImageSize / o.ClipPatchSize, RegionWidth = o.RegionWidth, OutDims = o.LlmHidden,
            GroupNormGroups = o.RegionGroupNormGroups,
        },
        Sam = new SamConfiguration
        {
            ImageSize = o.SamImageSize, PatchSize = o.SamPatchSize, EncoderDim = o.SamEncoderDim, EncoderDepth = o.SamEncoderDepth,
            EncoderHeads = o.SamEncoderHeads, WindowSize = o.SamWindowSize, GlobalAttention = (int[])o.SamGlobalAttention.Clone(),
            OutChannels = o.SamPromptDim, MaskInChannels = o.SamMaskInChannels, DecoderHeads = o.SamDecoderHeads,
            DecoderMlp = o.SamDecoderMlpDim, IouHidden = o.SamIouHiddenDim,
        },
        OutDim = o.SamPromptDim,
    };

    protected override void InitializeLayers()
    {
        if (_options is null || _network is not null) return;
        if (Architecture.Layers is { Count: > 0 })
            throw new NotSupportedException("GLaMM builds the paper's network; custom layers are not supported.");
        // With the tokenizer supplied, the network embeds its vocabulary. Without it the network is sized from
        // TokenizerPieces, so a model can be built (and its released weights loaded) before the tokenizer is at hand;
        // the tokenizer is then read on first use and must match.
        if (TokenizerBytes() is { } model)
            _tokenizer = new GlammTokenizer(new SentencePieceModel(model));
        int vocabulary = _tokenizer?.VocabularySize ?? _options.TokenizerPieces + GlammTokenizer.AddedTokens.Length;
        var layers = new List<LayerBase<T>>();
        _network = new GlammNetwork<T>(Engine, layers, Configuration(_options, vocabulary));
        _network.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.Seed ?? 42));
        Layers.AddRange(layers);
    }

    /// <summary>Loads a released GLaMM checkpoint's (merged) state dict.</summary>
    internal void LoadTorchWeights(Func<string, int[], double[]> read) => Network.LoadTorchWeights(read);

    /// <inheritdoc />
    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        if (_network is not null) _network.Training = isTraining;
    }

    /// <summary>Only the modules the official training updates.</summary>
    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> candidates)
    {
        _trainable ??= new HashSet<Tensor<T>>(AiDotNet.Training.TapeTrainingStep<T>.CollectParameters(Network.TrainableLayers().ToList()),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        return candidates.Where(_trainable.Contains).ToList();
    }

    private static Tensor<T> Single(Tensor<T> image)
    {
        if (image.Rank == 4 && image.Shape[0] == 1) return image;
        if (image.Rank == 3 && image.Shape[0] == 3) return new Tensor<T>(new[] { 1, 3, image.Shape[1], image.Shape[2] }, image.ToVector());
        throw new ArgumentException($"GLaMM reads one RGB image [3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
    }

    /// <summary>CLIPImageProcessor: the shorter side resized to the CLIP side (bicubic), centre-cropped, normalized.</summary>
    private Tensor<T> GlobalInput(Tensor<T> image)
    {
        int h = image.Shape[2], w = image.Shape[3], side = _options.ClipImageSize;
        int newH = h <= w ? side : (int)(side * (double)h / w), newW = h <= w ? (int)(side * (double)w / h) : side;
        var resized = SpatialResampling.ResizeBicubicAntialias(Engine, image, newH, newW);
        int top = (int)((newH - side) / 2.0), left = (int)((newW - side) / 2.0);
        var cropped = Engine.TensorSlice(resized, new[] { 0, 0, top, left }, new[] { image.Shape[0], 3, side, side });
        var mean = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(ClipMean.Select(NumOps.FromDouble).ToArray()));
        var scale = new Tensor<T>(new[] { 1, 3, 1, 1 }, new Vector<T>(ClipStd.Select(s => NumOps.FromDouble(1 / s)).ToArray()));
        var shape = cropped._shape;
        return Engine.TensorMultiply(Engine.TensorSubtract(cropped, Engine.TensorBroadcastTo(mean, shape)), Engine.TensorBroadcastTo(scale, shape));
    }

    /// <summary>ResizeLongestSide then Sam.preprocess: the longer side resized to the SAM side (bilinear), the pixels on
    /// SAM's 0..255 scale normalized and zero-padded. Returns the input and its unpadded size.</summary>
    private (Tensor<T> Input, int Height, int Width) GroundingInput(Tensor<T> image)
    {
        int h = image.Shape[2], w = image.Shape[3], side = _options.SamImageSize;
        double s = (double)side / Math.Max(h, w);
        int newH = (int)(h * s + 0.5), newW = (int)(w * s + 0.5);
        var resized = SpatialResampling.ResizeAntialias(Engine, image, newH, newW);
        return (Network.Sam.Preprocess(Engine.TensorMultiplyScalar(resized, NumOps.FromDouble(255))), newH, newW);
    }

    /// <summary>The projected CLIP patch features, LLaVA's visual tokens, of images in 0..1: <c>[Patches, LlmHidden]</c> for
    /// one image <c>[3, H, W]</c>, <c>[B, Patches, LlmHidden]</c> for a batch <c>[B, 3, H, W]</c>.</summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        bool single = image.Rank == 3;
        var batch = single ? Single(image) : image;
        if (batch.Rank != 4 || batch.Shape[1] != 3)
            throw new ArgumentException($"GLaMM reads RGB images [3, H, W] or [B, 3, H, W]; got [{string.Join(", ", image.Shape.ToArray())}].", nameof(image));
        var (features, _) = Network.EncodeGlobal(GlobalInput(batch));
        return single ? Engine.Reshape(features, new[] { features.Shape[1], features.Shape[2] }) : features;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input) => EncodeImage(input);

    public override Tensor<T> ForwardForTraining(Tensor<T> input) => EncodeImage(input);

    /// <summary>The prompt of a first turn: the official begin text, then the instruction.</summary>
    private static string FirstTurn(string instruction) => BeginText + instruction;

    /// <summary>
    /// One answer as the official app produces it: the instruction (which may name regions with <c>&lt;bbox&gt;</c>, one per
    /// entry of <paramref name="regions"/>), greedy decoding until &lt;/s&gt; or <paramref name="maxNewTokens"/>, the text
    /// after the last "ASSISTANT: ", and a mask for each [SEG] generated.
    /// </summary>
    public GlammAnswer<T> Chat(Tensor<T> image, string instruction, IReadOnlyList<double[]>? regions = null, int maxNewTokens = 512)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(instruction)) throw new ArgumentException("The instruction is empty.", nameof(instruction));
        var batch = Single(image);
        var prompt = GlammTokenizer.Conversation(new[] { (FirstTurn(instruction), (string?)null) });
        var ids = Tokenizer.EncodeWithImage(prompt);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        var (features, levels) = Network.EncodeGlobal(GlobalInput(batch));
        var imageFeatures = Engine.Reshape(features, new[] { features.Shape[1], features.Shape[2] });
        var regionFeatures = RegionFeatures(levels, regions);
        for (int step = 0; step < maxNewTokens; step++)
        {
            var (embeddings, _, _) = Network.Multimodal(ids, null, imageFeatures, regionFeatures, Tokenizer.BboxId);
            var (_, logits) = Network.Llm.Forward(embeddings);
            int last = logits.Shape[1] - 1, vocabulary = logits.Shape[2], next = 0;
            for (int v = 1; v < vocabulary; v++)
                if (NumOps.GreaterThan(logits[0, last, v], logits[0, last, next])) next = v;
            ids.Add(next);
            if (next == Tokenizer.EosId) break;
        }
        var text = Tokenizer.Decode(ids.Where(id => id != GlammTokenizer.ImageTokenIndex)).Replace("\n", "").Replace("  ", " ");
        int split = text.LastIndexOf("ASSISTANT: ", StringComparison.Ordinal);
        if (split >= 0) text = text.Substring(split + "ASSISTANT: ".Length);
        return new GlammAnswer<T>(text, Masks(batch, ids, imageFeatures, regionFeatures));
    }

    private Tensor<T>? RegionFeatures(List<Tensor<T>> levels, IReadOnlyList<double[]>? regions)
    {
        if (regions is null || regions.Count == 0) return null;
        foreach (var box in regions)
            if (box.Length != 4 || !(box[2] > box[0] && box[3] > box[1]))
                throw new ArgumentException("A region is (x1, y1, x2, y2) as fractions of the image, with positive area.", nameof(regions));
        return Network.Region.Forward(levels, new[] { regions.Select(b => (double[])b.Clone()).ToArray() });
    }

    // The [SEG] masks of a finished sequence: binary at the image's size (the app thresholds the logits at 0).
    private List<Tensor<T>> Masks(Tensor<T> batch, List<int> ids, Tensor<T> imageFeatures, Tensor<T>? regions)
    {
        var (embeddings, _, _) = Network.Multimodal(ids, null, imageFeatures, regions, Tokenizer.BboxId);
        var (hidden, _) = Network.Llm.Forward(embeddings);
        var seg = Network.SegEmbeddings(hidden, ids, Tokenizer.SegId, imageFeatures.Shape[0]);
        var masks = new List<Tensor<T>>();
        if (seg is null) return masks;
        var (input, inputH, inputW) = GroundingInput(batch);
        int h = batch.Shape[2], w = batch.Shape[3];
        var logits = Network.DecodeMasks(seg, Network.Sam.EncodeImage(input), inputH, inputW, h, w);
        for (int i = 0; i < seg.Shape[0]; i++)
        {
            var mask = new Tensor<T>(new[] { h, w });
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) mask[y, x] = NumOps.GreaterThan(logits[i, y, x], NumOps.Zero) ? NumOps.One : NumOps.Zero;
            masks.Add(mask);
        }
        return masks;
    }

    /// <summary>Referring segmentation with the official prompt ("Can you segment the {expression} in this image?"): the
    /// union of the answer's masks <c>[H, W]</c> (zero when it wrote no [SEG]).</summary>
    public Tensor<T> SegmentExpression(Tensor<T> image, string expression)
    {
        if (string.IsNullOrWhiteSpace(expression)) throw new ArgumentException("The expression is empty.", nameof(expression));
        var batch = Single(image);
        var answer = Chat(batch, $"Can you segment the {expression.Trim().ToLowerInvariant()} in this image?");
        var union = new Tensor<T>(new[] { batch.Shape[2], batch.Shape[3] });
        foreach (var mask in answer.Masks)
            for (int i = 0; i < union.Length; i++)
                if (!NumOps.Equals(mask[i], NumOps.Zero)) union[i] = NumOps.One;
        return union;
    }

    /// <summary>Grounded conversation generation with the official prompt: a detailed description whose phrases
    /// (<c>&lt;p&gt; … &lt;/p&gt;</c>) each have a mask.</summary>
    public GlammAnswer<T> DescribeGrounded(Tensor<T> image, int maxNewTokens = 512)
        => Chat(image, "Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.", null, maxNewTokens);

    /// <summary>Region captioning with the official prompt, for one box (x1, y1, x2, y2) as fractions of the image.</summary>
    public string CaptionRegion(Tensor<T> image, double[] box, int maxNewTokens = 512)
        => Chat(image, "Can you provide me with a detailed description of the region in the picture marked by <bbox>?", new[] { box }, maxNewTokens).Text;

    /// <summary>The ids, labels and targets of a training sample (the official collate: the first turn after the begin text,
    /// instructions masked, and the first ModelMaxLength − (patches − 1) ids kept).</summary>
    private (List<int> Ids, int[] Labels, Tensor<T> Image, Tensor<T>? Masks) Supervision(GlammSample<T> sample)
    {
        if (sample is null) throw new ArgumentNullException(nameof(sample));
        if (sample.Turns is null || sample.Turns.Count == 0) throw new ArgumentException("A sample needs at least one turn.", nameof(sample));
        var turns = sample.Turns.Select((t, i) => (i == 0 ? FirstTurn(t.User) : t.User, (string?)t.Assistant)).ToList();
        var conversation = GlammTokenizer.Conversation(turns);
        var ids = Tokenizer.EncodeWithImage(conversation);
        var labels = Tokenizer.MaskInstructions(conversation, ids);
        int keep = _options.ModelMaxLength - (Network.Clip.Patches - 1);
        if (ids.Count > keep)
        {
            ids = ids.Take(keep).ToList();
            labels = labels.Take(keep).ToArray();
        }
        int segs = ids.Count(id => id == Tokenizer.SegId), boxes = ids.Count(id => id == Tokenizer.BboxId);
        int given = sample.Masks?.Count ?? 0;
        if (given != segs) throw new ArgumentException($"The conversation has {segs} [SEG] tokens and {given} masks.", nameof(sample));
        if ((sample.Regions?.Count ?? 0) != boxes) throw new ArgumentException($"The conversation has {boxes} <bbox> tokens and {sample.Regions?.Count ?? 0} regions.", nameof(sample));
        var batch = Single(sample.Image);
        Tensor<T>? masks = null;
        if (segs > 0)
        {
            int h = batch.Shape[2], w = batch.Shape[3];
            masks = new Tensor<T>(new[] { segs, h, w });
            for (int m = 0; m < segs; m++)
            {
                var mask = sample.Masks![m];
                if (mask.Length != h * w) throw new ArgumentException($"Mask {m} must be [{h}, {w}], the image's size.", nameof(sample));
                for (int i = 0; i < h * w; i++) masks[m * h * w + i] = mask[i];
            }
        }
        return (ids, labels, batch, masks);
    }

    /// <summary>GLaMM's loss on a sample (the official model_forward): CE + 2 BCE + 0.5 DICE.</summary>
    private Tensor<T> Loss(Tensor<T> image, List<int> ids, int[] labels, IReadOnlyList<double[]>? regions, Tensor<T>? masks)
    {
        var (features, levels) = Network.EncodeGlobal(GlobalInput(image));
        var imageFeatures = Engine.Reshape(features, new[] { features.Shape[1], features.Shape[2] });
        var (embeddings, _, fullLabels) = Network.Multimodal(ids, labels, imageFeatures, RegionFeatures(levels, regions), Tokenizer.BboxId);
        var (hidden, logits) = Network.Llm.Forward(embeddings);
        var labels2D = new int[1, fullLabels!.Length];
        for (int i = 0; i < fullLabels.Length; i++) labels2D[0, i] = fullLabels[i];
        var loss = Engine.TensorMultiplyScalar(Network.Llm.CausalLmLoss(logits, labels2D), NumOps.FromDouble(_options.CeLossWeight));
        if (masks is null) return loss;
        var seg = Network.SegEmbeddings(hidden, ids, Tokenizer.SegId, imageFeatures.Shape[0])!;
        var (input, inputH, inputW) = GroundingInput(image);
        Tensor<T> samEmbedding;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())   // the official get_grounding_encoder_embs runs under no_grad
            samEmbedding = Network.Sam.EncodeImage(input);
        var predicted = Network.DecodeMasks(seg, samEmbedding, inputH, inputW, image.Shape[2], image.Shape[3]);
        var (bce, dice) = Network.MaskLosses(predicted, masks);
        loss = Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(bce, NumOps.FromDouble(_options.BceLossWeight)));
        return Engine.TensorAdd(loss, Engine.TensorMultiplyScalar(dice, NumOps.FromDouble(_options.DiceLossWeight)));
    }

    /// <summary>One training step on a conversation with masks, with GLaMM's loss and the official trainable modules.</summary>
    public void TrainConversation(GlammSample<T> sample)
    {
        ThrowIfDisposed();
        var (ids, labels, image, masks) = Supervision(sample);
        TrainWithCustomObjective(image, image, (x, _) => Loss(x, ids, labels, sample.Regions, masks), _optimizer);
    }

    /// <summary>GLaMM's loss on a sample, without training (LoRA dropout off).</summary>
    public T EvaluateConversationLoss(GlammSample<T> sample)
    {
        ThrowIfDisposed();
        var (ids, labels, image, masks) = Supervision(sample);
        using var _ = new Tensors.Engines.Autodiff.NoGradScope<T>();
        return Loss(image, ids, labels, sample.Regions, masks)[0];
    }

    /// <summary>GLaMM trains on conversations with masks, which a tensor target cannot carry.</summary>
    /// <exception cref="NotSupportedException">Always: use <see cref="TrainConversation"/>.</exception>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expected)
        => throw new NotSupportedException("GLaMM trains on an image, a conversation and a mask per [SEG]: use TrainConversation(sample).");

    /// <inheritdoc />
    /// <remarks>Every layer is sized at construction and <see cref="NeuralNetworkBase{T}.Layers"/> is not a chain.</remarks>
    protected override void ResolveLazyLayerShapes()
    {
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input) => ObservedLayerActivations(input);

    protected override bool SupportsParameterMutation => true;

    protected override Tensor<T> PreprocessImage(Tensor<T> image) => GlobalInput(Single(image));

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "GLaMM",
            Description = "GLaMM (Rasheed et al., 2024): a grounding large multimodal model answering with text and masks.",
            FeatureCount = _options.LlmHidden,
            Complexity = _options.ClipLayers + _options.LlmLayers + _options.SamEncoderDepth,
        };
        m.AdditionalInfo["Architecture"] = "GLaMM";
        m.AdditionalInfo["LoraRank"] = _options.LoraRank.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GLaMM<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
