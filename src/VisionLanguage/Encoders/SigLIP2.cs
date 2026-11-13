using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// SigLIP 2 (Multilingual Vision-Language Encoders with Improved Semantic Understanding)
/// for contrastive encoding, zero-shot classification, and captioning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SigLIP 2 (Tschannen et al., 2025) extends SigLIP with a multi-objective training framework
/// that combines three losses: (1) sigmoid contrastive loss for image-text alignment, (2) an
/// autoregressive captioning loss via a lightweight decoder for semantic understanding, and
/// (3) a self-supervised masked image modeling (MIM) loss for spatial understanding. The model
/// supports 32+ languages through an extended multilingual vocabulary.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// <list type="number">
/// <item><b>Vision encoder</b>: ViT with patch sizes 14 or 16, supporting 256-512px images</item>
/// <item><b>Text encoder</b>: Transformer with multilingual SentencePiece tokenization (250K vocab)</item>
/// <item><b>Captioning decoder</b>: Lightweight 4-layer autoregressive transformer that attends
/// to vision encoder features via cross-attention, trained with next-token prediction</item>
/// <item><b>MIM decoder</b>: 2-layer MLP decoder that predicts masked patch features from
/// unmasked patch embeddings</item>
/// <item><b>Sigmoid contrastive loss</b>: Per-pair binary classification as in SigLIP</item>
/// </list>
/// </para>
/// <para>
/// <b>Key Innovation:</b> By training with multiple objectives simultaneously, SigLIP 2 produces
/// vision encoders with richer semantic understanding than contrastive-only training. The captioning
/// objective forces the encoder to capture fine-grained details, while MIM improves spatial awareness.
/// Online data curation dynamically adjusts the training data mixture for better representation quality.
/// </para>
/// <para>
/// <b>For Beginners:</b> SigLIP 2 improves on SigLIP by learning from three tasks at once:
/// matching images with text, describing what's in images, and filling in missing parts of images.
/// This multi-task approach produces better visual representations. The model also supports
/// many languages, making it useful for global applications.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 768, outputSize: 768);
/// var siglip2 = new SigLIP2&lt;float&gt;(arch, "siglip2_vit_b16.onnx");
/// var probs = siglip2.ZeroShotClassify(imageTensor, new[] { "a dog", "a cat", "a bird" });
/// var caption = siglip2.GenerateCaption(imageTensor);
/// </code>
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>Paper: "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding"
/// (Tschannen et al., 2025)</item>
/// <item>Original SigLIP: "Sigmoid Loss for Language Image Pre-Training" (Zhai et al., ICCV 2023)</item>
/// <item>Repository: https://github.com/google-research/big_vision</item>
/// </list>
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SigLIP 2 model for multi-objective contrastive training
/// // with captioning, MIM, and multilingual support for 32+ languages
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SigLIP2&lt;double&gt;(architecture, "siglip2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SigLIP2&lt;double&gt;(architecture, new SigLIP2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding",
    "https://arxiv.org/abs/2502.14786",
    Year = 2025,
    Authors = "Tschannen et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-3, WeightDecay = 1e-4,
                ReferenceBatchSize = 32768, WarmupSteps = 20000, MinLearningRate = 0,
                MaxGradientNorm = 1.0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Tschannen et al. 2025, Sec. 3: a learning rate of 1e-3, decoupled weight "
                        + "decay of 1e-4 and gradient clipping to norm 1, at a batch size of 32k under a "
                        + "cosine schedule with 20k warmup steps. The paper names Adam, but the weight "
                        + "decay it specifies is decoupled, which is AdamW; declaring Adam here would "
                        + "apply the decay as coupled L2 instead. The model keeps its own "
                        + "optimizer rather than being built from this record: building one "
                        + "from the recipe snapshots a parameter buffer inside the "
                        + "constructor, where this model reports 64 parameters against the 50 "
                        + "it trains with, and the buffer is rejected on the first step.")]
public partial class SigLIP2<T> : VisionLanguageModelBase<T>, IContrastiveVisionLanguageModel<T>
{
    #region Fields

    private readonly SigLIP2Options _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;

    // Layer boundary indices for the multi-component architecture
    private int _visionEncoderEnd;
    private int _textEncoderEnd;
    private int _captioningDecoderEnd;
    private int _mimDecoderEnd;

    // The LocCa decoder's and the masked-prediction head's own weights (Tschannen et al. 2025, section 2):
    // the vision-token projection to the decoder width, the start-of-caption and position embeddings, and the
    // learned mask token. They live outside Layers and are registered as an auxiliary stream.
    private readonly List<ILayer<T>> _headLayers = new();
    private DenseLayer<T>? _captionMemoryProjection;
    private DenseLayer<T>? _captionStartEmbedding;
    private DenseLayer<T>? _captionPositionEmbedding;
    private DenseLayer<T>? _maskTokenEmbedding;
    private EmaTeacher<T>? _teacher;
    private Tensor<T>? _maskedCenter;

    /// <summary>The masked-prediction loss's running centre of the teacher's outputs (null before the first loss).</summary>
    internal Tensor<T>? MaskedPredictionCenter => _maskedCenter;

    // The EMA teacher and the masked-prediction loss follow iBOT (Zhou et al. 2022), which TIPS and SigLIP 2 adopt:
    // teacher momentum 0.996, student temperature 0.1, teacher temperature 0.04, centre momentum 0.9.
    private const double TeacherMomentum = 0.996;
    private const double StudentTemperature = 0.1;
    private const double TeacherTemperature = 0.04;
    private const double CenterMomentum = 0.9;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SigLIP 2 model in ONNX inference mode from a pre-trained model file.
    /// </summary>
    public SigLIP2(
        NeuralNetworkArchitecture<T> architecture,
        string imageEncoderModelPath,
        SigLIP2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SigLIP2Options();
        Options = _options;
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;

        if (string.IsNullOrWhiteSpace(imageEncoderModelPath))
            throw new ArgumentException(
                "Image encoder model path cannot be null or empty.",
                nameof(imageEncoderModelPath)
            );
        if (!File.Exists(imageEncoderModelPath))
            throw new FileNotFoundException(
                $"ONNX model not found: {imageEncoderModelPath}",
                imageEncoderModelPath
            );

        _options.ImageEncoderModelPath = imageEncoderModelPath;
        OnnxImageEncoder = new OnnxModel<T>(imageEncoderModelPath, _options.OnnxOptions);

        if (_options.TextEncoderModelPath is { } tp && !string.IsNullOrEmpty(tp))
        {
            if (!File.Exists(tp))
                throw new FileNotFoundException($"Text encoder ONNX model not found: {tp}", tp);
            OnnxTextEncoder = new OnnxModel<T>(tp, _options.OnnxOptions);
        }

        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SigLIP 2 model in native training mode.
    /// </summary>
    public SigLIP2(
        NeuralNetworkArchitecture<T> architecture,
        SigLIP2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SigLIP2Options();
        Options = _options;
        _useNativeMode = true;
        // SigLIP2 is a paper-scale ViT (Tschannen et al. 2025): VisionEmbeddingDim
        // 768 with many transformer blocks. The framework AdamW default LR (1e-3)
        // is too aggressive for a from-scratch ViT of this width — gradients
        // accumulate over a handful of steps and the weights blow up to NaN
        // (Training_ShouldReduceLoss / ForwardPass_ShouldBeFinite_AfterTraining).
        // SigLIP/ViT fine-tuning uses 1e-4..1e-5; without LR warmup a deep ViT
        // overshoots a far target in the first few un-warmed steps (loss rises),
        // so pin the conservative end (1e-5) as the paper-faithful stable default.
        // Gradient clipping (norm 1.0) is on by AdamW default.
        _optimizer =
            optimizer
            ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                    this,
                    new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                    {
                        InitialLearningRate = 1e-5,
                    }
                ));
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.VisionEmbeddingDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    #endregion

    #region IContrastiveVisionLanguageModel

    /// <inheritdoc />
    public int EmbeddingDimension => _options.VisionEmbeddingDim;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;

    /// <inheritdoc />
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <inheritdoc />
    public int MaxSequenceLength => _options.MaxSequenceLength;

    /// <inheritdoc />
    public int TextEmbeddingDimension => _options.TextEmbeddingDim;

    /// <inheritdoc />
    public int ProjectionDimension => _options.ProjectionDim;

    /// <inheritdoc />
    public T Temperature => NumOps.FromDouble(_options.Temperature);

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);

        if (IsOnnxMode && OnnxImageEncoder is not null)
            return L2Normalize(OnnxImageEncoder.Run(preprocessed));

        var output = ForwardVisionEncoder(preprocessed);
        return L2Normalize(ProjectVision(output));
    }

    /// <inheritdoc />
    public Tensor<T> EncodeText(string text)
    {
        ThrowIfDisposed();
        var tokenized = TokenizeText(text);

        if (IsOnnxMode && OnnxTextEncoder is not null)
            return L2Normalize(OnnxTextEncoder.Run(tokenized));

        var output = ForwardTextEncoder(tokenized);
        return L2Normalize(ProjectText(output));
    }

    /// <inheritdoc />
    public Tensor<T>[] EncodeTexts(string[] texts)
    {
        var embeddings = new Tensor<T>[texts.Length];
        for (int i = 0; i < texts.Length; i++)
            embeddings[i] = EncodeText(texts[i]);
        return embeddings;
    }

    /// <inheritdoc />
    public T ComputeSimilarity(Tensor<T> image, string text)
    {
        var imageEmb = EncodeImage(image);
        var textEmb = EncodeText(text);
        return CosineSimilarity(imageEmb, textEmb);
    }

    /// <inheritdoc />
    public Dictionary<string, T> ZeroShotClassify(Tensor<T> image, string[] labels)
    {
        var imageEmb = EncodeImage(image);
        var textEmbs = EncodeTexts(labels);

        // SigLIP 2 scores every image-text pair on its own: sigmoid(sim / t + b) is the probability that the
        // pair matches (Zhai et al., 2023), so the scores are not renormalized across the labels.
        T[] probs = EmbeddingScores<T>.SigmoidPairScores(Engine, imageEmb, textEmbs,
            NumOps.FromDouble(_options.Temperature), NumOps.FromDouble(_options.SigmoidBias)).ToArray();
        var result = new Dictionary<string, T>();
        for (int i = 0; i < labels.Length; i++)
            result[labels[i]] = probs[i];

        return result;
    }

    #endregion

    #region SigLIP 2 Specific: Captioning

    /// <summary>
    /// Generates a caption for the given image with SigLIP 2's LocCa decoder.
    /// </summary>
    /// <param name="image">Image tensor to caption.</param>
    /// <returns>The logits of every generated position, <c>[MaxCaptionLength, VocabSize]</c>.</returns>
    /// <remarks>
    /// <para>SigLIP 2 (Tschannen et al. 2025, section 2.2) adds the LocCa decoder (Wan et al. 2024): a transformer
    /// decoder whose layers apply causal self-attention over the caption so far, cross-attention to the un-pooled vision
    /// encoder tokens, and an MLP, each with a residual connection and layer normalization. The caption starts with a
    /// learned start embedding, every position adds a learned position embedding, and generated tokens are fed back
    /// through the embedding tied to the vocabulary projection. Decoding is greedy.</para>
    /// </remarks>
    public Tensor<T> GenerateCaption(Tensor<T> image)
    {
        ThrowIfDisposed();
        if (!_options.IncludeCaptioningDecoder)
            throw new InvalidOperationException(
                "Captioning decoder is not included. Set IncludeCaptioningDecoder = true."
            );

        var preprocessed = PreprocessImage(image);

        if (IsOnnxMode && OnnxImageEncoder is not null)
        {
            // In ONNX mode, run through image encoder and return features
            return OnnxImageEncoder.Run(preprocessed);
        }

        int maxLen = _options.MaxCaptionLength;
        if (maxLen <= 0)
            return new Tensor<T>([0]);

        var start = _captionStartEmbedding ?? throw new InvalidOperationException("The captioning decoder is not initialized.");
        var memory = CaptionMemory(preprocessed);

        int width = _options.CaptioningDecoderDim;
        var sequence = start.Forward(Tensor<T>.CreateDefault([1, 1], NumOps.One));
        var stepLogits = new List<Tensor<T>>(maxLen);
        for (int step = 0; step < maxLen; step++)
        {
            var hidden = DecodeCaption(sequence, memory);
            int positions = hidden.Shape[0];
            var last = Engine.TensorSlice(hidden, [positions - 1, 0], [1, hidden.Shape[1]]);
            var logits = FitLength(VocabularyProjection().Forward(last), _options.VocabSize);
            stepLogits.Add(logits);

            if (step + 1 < maxLen)
            {
                int next = Math.Max(0, ArgExtremumSearch<T>.FirstIndex(Engine, logits, true, out _));
                sequence = Engine.TensorConcatenate([sequence, CaptionTokenEmbedding(next, width)], 0);
            }
        }

        return Engine.TensorConcatenate(stepLogits.ToArray(), 0);
    }

    /// <summary>
    /// LocCa's captioning loss for one image (Wan et al. 2024; SigLIP 2 section 2.2): the decoder reads the caption
    /// teacher-forced (the start token, then the reference tokens shifted by one, under the causal mask) and every
    /// position takes the cross-entropy of the next reference token. With <paramref name="parallel"/> (CapPa's
    /// parallel prediction, which LocCa applies to half of the examples) every decoder input is the start token and
    /// the causal mask is lifted, so each position predicts its token from the image alone.
    /// </summary>
    /// <param name="image">The image.</param>
    /// <param name="captionTokens">The reference caption's token ids.</param>
    /// <param name="parallel">Whether to use parallel prediction instead of teacher forcing.</param>
    /// <returns>The mean token cross-entropy, a one-element tensor.</returns>
    internal Tensor<T> CaptioningLoss(Tensor<T> image, IReadOnlyList<int> captionTokens, bool parallel = false)
    {
        ThrowIfDisposed();
        if (!_options.IncludeCaptioningDecoder)
            throw new InvalidOperationException("Captioning decoder is not included. Set IncludeCaptioningDecoder = true.");
        return CaptioningLossFromMemory(CaptionMemory(PreprocessImage(image)), captionTokens, parallel);
    }

    /// <summary>The loss of <see cref="CaptioningLoss"/> given the decoder's memory <c>[n, CaptioningDecoderDim]</c>.</summary>
    internal Tensor<T> CaptioningLossFromMemory(Tensor<T> memory, IReadOnlyList<int> captionTokens, bool parallel)
    {
        if (memory is null) throw new ArgumentNullException(nameof(memory));
        if (captionTokens is null || captionTokens.Count == 0) throw new ArgumentException("The caption has no tokens.", nameof(captionTokens));
        var start = _captionStartEmbedding ?? throw new InvalidOperationException("The captioning decoder is not initialized.");
        int length = Math.Min(captionTokens.Count, Math.Max(1, _options.MaxCaptionLength));
        var targets = captionTokens.Take(length).ToArray();
        int width = _options.CaptioningDecoderDim;
        var startToken = start.Forward(Tensor<T>.CreateDefault([1, 1], NumOps.One));
        Tensor<T> inputs;
        if (parallel || length == 1)
            inputs = Engine.TensorBroadcastTo(startToken, [length, width]);
        else
            inputs = Engine.TensorConcatenate([startToken, CaptionTokenEmbeddings(targets.Take(length - 1).ToArray(), width)], 0);

        Tensor<T> hidden;
        if (parallel)
        {
            SetCaptionCausalMask(false);
            try { hidden = DecodeCaption(inputs, memory); }
            finally { SetCaptionCausalMask(true); }
        }
        else
        {
            hidden = DecodeCaption(inputs, memory);
        }

        var logits = VocabularyProjection().Forward(hidden);
        int vocabulary = logits.Shape[logits.Rank - 1];
        if (targets.Any(token => token < 0 || token >= vocabulary))
            throw new ArgumentOutOfRangeException(nameof(captionTokens), $"Every caption token must lie in [0, {vocabulary}).");
        var oneHot = Engine.TensorOneHot<T>(new Tensor<int>([length], Vector<int>.FromArray(targets)), vocabulary);
        var logProbabilities = Engine.TensorLogSoftmax(logits.Reshape([length, vocabulary]), 1);
        var total = Engine.ReduceSum(Engine.TensorMultiply(logProbabilities, oneHot), [0, 1], false);
        return Engine.TensorMultiplyScalar(total, NumOps.FromDouble(-1.0 / length));
    }

    // The decoder's memory: the vision encoder's tokens projected to the decoder width.
    private Tensor<T> CaptionMemory(Tensor<T> preprocessed)
    {
        var projection = _captionMemoryProjection ?? throw new InvalidOperationException("The captioning decoder is not initialized.");
        var vision = ForwardVisionEncoder(preprocessed);
        int width = vision.Shape[vision.Rank - 1];
        return projection.Forward(vision.Reshape([vision.Length / width, width]));
    }

    private void SetCaptionCausalMask(bool causal)
    {
        for (int layer = 0; layer < _options.NumCaptioningDecoderLayers; layer++)
            CaptionBlock(layer).SelfAttention.UseCausalMask = causal;
    }

    /// <summary>
    /// The LocCa decoder on the caption so far <c>[L, CaptioningDecoderDim]</c> (start and token embeddings) with the
    /// projected vision tokens <c>[N, CaptioningDecoderDim]</c> as memory: positions added, then per layer
    /// <c>x = LN(x + CausalSelfAttention(x))</c>, <c>x = LN(x + CrossAttention(x, memory))</c>,
    /// <c>x = LN(x + MLP(x))</c>.
    /// </summary>
    internal Tensor<T> DecodeCaption(Tensor<T> sequence, Tensor<T> memory)
    {
        var positionEmbedding = _captionPositionEmbedding ?? throw new InvalidOperationException("The captioning decoder is not initialized.");
        int length = sequence.Shape[0], width = sequence.Shape[1];
        var positions = new Tensor<int>([length], Vector<int>.FromArray(Enumerable.Range(0, length).ToArray()));
        var x = Engine.TensorAdd(sequence,
            positionEmbedding.Forward(Engine.TensorOneHot<T>(positions, Math.Max(1, _options.MaxCaptionLength))));
        var memoryBatch = memory.Reshape([1, memory.Shape[0], memory.Shape[1]]);
        for (int layer = 0; layer < _options.NumCaptioningDecoderLayers; layer++)
        {
            var block = CaptionBlock(layer);
            var self = block.SelfAttention.Forward(x.Reshape([1, length, width])).Reshape([length, width]);
            x = block.SelfNorm.Forward(Engine.TensorAdd(x, self));
            var cross = block.CrossAttention.Forward(x.Reshape([1, length, width]), memoryBatch).Reshape([length, width]);
            x = block.CrossNorm.Forward(Engine.TensorAdd(x, cross));
            x = block.FeedForwardNorm.Forward(Engine.TensorAdd(x, block.FeedForwardDown.Forward(block.FeedForwardUp.Forward(x))));
        }
        return x;
    }

    /// <summary>
    /// The layers of captioning decoder layer <paramref name="layer"/>, in the order the factory builds them:
    /// cross-attention, its norm, self-attention, its norm, the MLP's two projections and its norm.
    /// </summary>
    private (MultiHeadAttentionLayer<T> CrossAttention, ILayer<T> CrossNorm, MultiHeadAttentionLayer<T> SelfAttention,
        ILayer<T> SelfNorm, ILayer<T> FeedForwardUp, ILayer<T> FeedForwardDown, ILayer<T> FeedForwardNorm) CaptionBlock(int layer)
    {
        const int LayersPerBlock = 7;
        int first = _textEncoderEnd + layer * LayersPerBlock;
        if (first + LayersPerBlock > _captioningDecoderEnd - 1 || first + LayersPerBlock > Layers.Count)
            throw new InvalidOperationException("The captioning decoder does not hold the expected layers.");
        var cross = Layers[first] as MultiHeadAttentionLayer<T>;
        var self = Layers[first + 2] as MultiHeadAttentionLayer<T>;
        if (cross is null || self is null)
            throw new InvalidOperationException("A captioning decoder layer must start with cross- and self-attention layers.");
        return (cross, Layers[first + 1], self, Layers[first + 3], Layers[first + 4], Layers[first + 5], Layers[first + 6]);
    }

    /// <summary>The captioning decoder's vocabulary projection, the last layer of the decoder.</summary>
    private ILayer<T> VocabularyProjection()
    {
        int index = _captioningDecoderEnd - 1;
        if (index < _textEncoderEnd || index >= Layers.Count)
            throw new InvalidOperationException("The captioning decoder has no vocabulary projection layer.");
        return Layers[index];
    }

    // The tied input embeddings [n, width] of several tokens: columns of the vocabulary projection.
    private Tensor<T> CaptionTokenEmbeddings(int[] tokens, int width)
    {
        var projection = VocabularyProjection().GetWeights();
        if (projection is null || projection.Rank != 2 || projection.Shape[0] != width || tokens.Any(t => t < 0 || t >= projection.Shape[1]))
            throw new InvalidOperationException(
                "The captioning vocabulary projection must hold [CaptioningDecoderDim, VocabSize] weights covering every token.");
        return Engine.TensorTranspose(Engine.TensorIndexSelect(projection, new Tensor<int>([tokens.Length], Vector<int>.FromArray(tokens)), 1));
    }

    /// <summary>
    /// The tied input embedding of a caption token: its column of the vocabulary projection's
    /// <c>[width, vocab]</c> weights, as a <c>[1, width]</c> row.
    /// </summary>
    private Tensor<T> CaptionTokenEmbedding(int token, int width)
    {
        int projectionIndex = _captioningDecoderEnd - 1;
        if (projectionIndex < _textEncoderEnd || projectionIndex >= Layers.Count)
            throw new InvalidOperationException("The captioning decoder has no vocabulary projection layer.");

        var projection = Layers[projectionIndex].GetWeights();
        if (projection is null || projection.Rank != 2 || projection.Shape[0] != width || token >= projection.Shape[1])
            throw new InvalidOperationException(
                "The captioning vocabulary projection must hold [CaptioningDecoderDim, VocabSize] weights.");

        return Engine.TensorSlice(projection, [0, token], [width, 1]).Reshape([1, width]);
    }

    #endregion

    #region SigLIP 2 Specific: Masked Image Modeling

    /// <summary>
    /// Predicts the features of masked patches with SigLIP 2's masked-prediction head.
    /// </summary>
    /// <param name="image">Image tensor to process.</param>
    /// <param name="maskIndices">Indices of patches to mask (null for a random <see cref="SigLIP2Options.MimMaskRatio"/> of them).</param>
    /// <returns>The head's predictions at the masked patches, <c>[masked, features]</c>, in <paramref name="maskIndices"/> order.</returns>
    /// <remarks>
    /// <para>SigLIP 2 (Tschannen et al. 2025, section 2.3) adds the masked-prediction objective of TIPS (Maninis et al.
    /// 2025) and SILC (Naeem et al. 2024): the student replaces the embeddings of the masked patches with a learned
    /// mask token after the patch embedding, encodes the sequence, and predicts, at the masked positions only, the
    /// per-patch features the teacher computes from the unmasked image. The masking is in embedding space, not on the
    /// pixels.</para>
    /// </remarks>
    public Tensor<T> PredictMaskedPatches(Tensor<T> image, int[]? maskIndices = null)
    {
        ThrowIfDisposed();
        if (Layers.Count == 0)
            throw new InvalidOperationException("The vision encoder has no layers.");
        var preprocessed = PreprocessImage(image);

        // Patch embedding: one token per patch, in the order the patch grid is read.
        var embedded = Layers[0].Forward(preprocessed);
        int width = embedded.Shape[embedded.Rank - 1];
        int totalPatches = embedded.Length / width;
        int numMasked = (int)(totalPatches * _options.MimMaskRatio);

        // Generate random mask if not provided
        if (maskIndices is null)
        {
            var rng = RandomHelper.CreateSecureRandom();
            var indices = new int[totalPatches];
            for (int i = 0; i < totalPatches; i++)
                indices[i] = i;
            // Fisher-Yates shuffle
            for (int i = totalPatches - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            maskIndices = new int[numMasked];
            Array.Copy(indices, maskIndices, numMasked);
        }

        foreach (int patch in maskIndices)
        {
            if (patch < 0 || patch >= totalPatches)
                throw new ArgumentOutOfRangeException(nameof(maskIndices),
                    $"Patch index {patch} is outside the {totalPatches} patches of the image.");
        }

        var masked = MaskTokens(embedded.Reshape([totalPatches, width]), maskIndices).Reshape(embedded.Shape.ToArray());
        return PatchPredictions(masked, maskIndices);
    }

    // The vision encoder after the patch embedding, then the masked-prediction decoder, read at the given patches
    // (every patch when null).
    private Tensor<T> PatchPredictions(Tensor<T> embedded, int[]? patches)
    {
        var encoded = embedded;
        for (int i = 1; i < _visionEncoderEnd && i < Layers.Count; i++)
            encoded = Layers[i].Forward(encoded);
        var predictions = ForwardMimDecoder(encoded);
        int features = predictions.Shape[predictions.Rank - 1];
        var rows = predictions.Reshape([predictions.Length / features, features]);
        if (patches is null)
            return rows;
        if (patches.Length == 0)
            return new Tensor<T>([0, features]);
        return Engine.TensorIndexSelect(rows, new Tensor<int>([patches.Length], Vector<int>.FromArray(patches)), 0);
    }

    /// <summary>
    /// SigLIP 2's masked-prediction loss for one image (section 2.3, following TIPS and iBOT): the student sees the
    /// image with the masked patches replaced by the mask token, the EMA teacher sees the whole image, and at every
    /// masked patch the student's prediction is trained toward the teacher's: the cross-entropy between the teacher's
    /// centred, sharpened softmax <c>softmax((t − c) / 0.04)</c> and the student's <c>log_softmax(s / 0.1)</c>,
    /// averaged over the masked patches. The centre <c>c</c> then follows the teacher's mean output over every patch of
    /// the image (iBOT's patch centre, <c>teacher_patch.mean(1)</c>) with momentum 0.9.
    /// </summary>
    /// <param name="image">The image.</param>
    /// <param name="maskIndices">The patches to mask.</param>
    /// <returns>The loss, a one-element tensor; gradients reach the student only.</returns>
    internal Tensor<T> MaskedPredictionLoss(Tensor<T> image, int[] maskIndices)
    {
        ThrowIfDisposed();
        if (maskIndices is null || maskIndices.Length == 0) throw new ArgumentException("At least one patch must be masked.", nameof(maskIndices));
        var student = PredictMaskedPatches(image, maskIndices);
        var teacherAll = TeacherPatchPredictions(image, null);
        var teacher = Engine.TensorIndexSelect(teacherAll, new Tensor<int>([maskIndices.Length], Vector<int>.FromArray(maskIndices)), 0);
        int masked = student.Shape[0], features = student.Shape[1];
        var center = _maskedCenter is not null && _maskedCenter.Shape[1] == features ? _maskedCenter : new Tensor<T>([1, features]);
        var targets = Engine.Softmax(Engine.TensorMultiplyScalar(
            Engine.TensorSubtract(teacher, Engine.TensorBroadcastTo(center, [masked, features])), NumOps.FromDouble(1.0 / TeacherTemperature)), 1);
        var logProbabilities = Engine.TensorLogSoftmax(Engine.TensorMultiplyScalar(student, NumOps.FromDouble(1.0 / StudentTemperature)), 1);
        var loss = Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(targets, logProbabilities), [0, 1], false),
            NumOps.FromDouble(-1.0 / masked));
        var updated = Engine.TensorAdd(Engine.TensorMultiplyScalar(center, NumOps.FromDouble(CenterMomentum)),
            Engine.TensorMultiplyScalar(Engine.ReduceMean(teacherAll, [0], true), NumOps.FromDouble(1 - CenterMomentum)));
        // The centre outlives this call: copy it out of the engine's result, which may be per-step arena scratch.
        _maskedCenter = new Tensor<T>([1, features], Vector<T>.FromArray(updated.ToArray()));
        return loss;
    }

    /// <summary>The EMA teacher's predictions at <paramref name="patches"/> (every patch when null) of the whole
    /// (unmasked) image, without gradient.</summary>
    internal Tensor<T> TeacherPatchPredictions(Tensor<T> image, int[]? patches)
        => Teacher().Run(() => PatchPredictions(Layers[0].Forward(PreprocessImage(image)), patches));

    /// <summary>Moves the EMA teacher toward the student after an optimizer step (momentum 0.996).</summary>
    internal void UpdateTeacher() => Teacher().Update();

    // The teacher of the student's vision encoder and masked-prediction decoder.
    internal EmaTeacher<T> Teacher()
    {
        if (_teacher is not null) return _teacher;
        var layers = Layers.Take(Math.Min(_visionEncoderEnd, Layers.Count))
            .Concat(Layers.Skip(_captioningDecoderEnd).Take(Math.Max(0, Math.Min(_mimDecoderEnd, Layers.Count) - _captioningDecoderEnd)))
            .ToList();
        _teacher = new EmaTeacher<T>(Engine, layers, TeacherMomentum);
        return _teacher;
    }

    /// <summary>
    /// Replaces the listed rows of the patch tokens <c>[patches, width]</c> with the learned mask token.
    /// </summary>
    internal Tensor<T> MaskTokens(Tensor<T> tokens, int[] maskIndices)
    {
        var maskEmbedding = _maskTokenEmbedding ?? throw new InvalidOperationException("The masked-prediction head is not initialized.");
        int patches = tokens.Shape[0], width = tokens.Shape[1];
        var maskToken = maskEmbedding.Forward(Tensor<T>.CreateDefault([1, 1], NumOps.One));
        if (maskIndices.Length == 0)
            return tokens;
        var keep = Engine.TensorIndexFill(Tensor<T>.CreateDefault([patches], NumOps.One), 0,
            new Tensor<int>([maskIndices.Length], Vector<int>.FromArray(maskIndices)), NumOps.Zero);
        // Select rather than blend, so a non-finite embedding of a masked patch never leaks through.
        return Engine.TensorWhere(Engine.TensorBroadcastTo(keep.Reshape([patches, 1]), [patches, width]), tokens,
            Engine.TensorBroadcastTo(maskToken, [patches, width]));
    }


    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultSigLIP2Layers(
                    visionEmbeddingDim: _options.VisionEmbeddingDim,
                    textEmbeddingDim: _options.TextEmbeddingDim,
                    projectionDim: _options.ProjectionDim,
                    numVisionLayers: _options.NumVisionLayers,
                    numTextLayers: _options.NumTextLayers,
                    numVisionHeads: _options.NumVisionHeads,
                    numTextHeads: _options.NumTextHeads,
                    captioningDecoderDim: _options.CaptioningDecoderDim,
                    numCaptioningDecoderLayers: _options.NumCaptioningDecoderLayers,
                    numCaptioningDecoderHeads: _options.NumCaptioningDecoderHeads,
                    mimDecoderDim: _options.MimDecoderDim,
                    numMimDecoderLayers: _options.NumMimDecoderLayers,
                    vocabSize: _options.VocabSize,
                    includeCaptioningDecoder: _options.IncludeCaptioningDecoder,
                    dropoutRate: _options.DropoutRate,
                    patchSize: _options.PatchSize,
                    inputChannels: 3
                )
            );
        }

        ComputeLayerBoundaries();

        if (_maskTokenEmbedding is null)
        {
            // Learned tables held as bias-free linear maps of a one-hot input, built lazily like the other layers.
            DenseLayer<T> Table(int width)
            {
                var layer = new DenseLayer<T>(width, (IActivationFunction<T>?)null, null, BiasMode.Never);
                _headLayers.Add(layer);
                return layer;
            }
            if (_options.IncludeCaptioningDecoder)
            {
                _captionMemoryProjection = new DenseLayer<T>(_options.CaptioningDecoderDim, (IActivationFunction<T>?)null);
                _headLayers.Add(_captionMemoryProjection);
                _captionStartEmbedding = Table(_options.CaptioningDecoderDim);
                _captionPositionEmbedding = Table(_options.CaptioningDecoderDim);
            }
            _maskTokenEmbedding = Table(_options.VisionEmbeddingDim);
            RegisterAuxiliaryEncoderStream(_headLayers);
        }

        // The decoder's self-attention sees only the caption so far.
        if (_options.IncludeCaptioningDecoder)
            for (int layer = 0; layer < _options.NumCaptioningDecoderLayers; layer++)
            {
                int index = _textEncoderEnd + layer * 7 + 2;
                if (index < Layers.Count && Layers[index] is MultiHeadAttentionLayer<T> selfAttention)
                    selfAttention.UseCausalMask = true;
            }
    }

    /// <summary>
    /// Computes the layer boundary indices for vision encoder, text encoder,
    /// captioning decoder, and MIM decoder sections.
    /// </summary>
    private void ComputeLayerBoundaries()
    {
        // Vision encoder: PatchEmbed + LN + (MHA + LN + FFN_up + FFN_down + LN [+ Dropout]) * numLayers + projection
        int lpb = _options.DropoutRate > 0 ? 6 : 5; // layers per block
        _visionEncoderEnd = 2 + _options.NumVisionLayers * lpb + 1; // +1 for PatchEmbed, +1 for initial LN, +1 for projection

        // Text encoder: LN + (MHA + LN + FFN_up + FFN_down + LN [+ Dropout]) * numLayers + projection
        _textEncoderEnd = _visionEncoderEnd + 1 + _options.NumTextLayers * lpb + 1;

        if (_options.IncludeCaptioningDecoder)
        {
            // Captioning decoder: (cross-MHA + LN + self-MHA + LN + FFN_up + FFN_down + LN) * numLayers + vocab_proj
            int captLpb = 7; // cross-attn + LN + self-attn + LN + FFN_up + FFN_down + LN
            _captioningDecoderEnd =
                _textEncoderEnd + _options.NumCaptioningDecoderLayers * captLpb + 1;
        }
        else
        {
            _captioningDecoderEnd = _textEncoderEnd;
        }

        // MIM decoder: (Dense + LN) * numLayers + prediction_head
        _mimDecoderEnd = _captioningDecoderEnd + _options.NumMimDecoderLayers * 2 + 1;
    }

    /// <inheritdoc />
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxImageEncoder is not null)
            return OnnxImageEncoder.Run(input);

        var current = input;
        for (int i = 0; i < _visionEncoderEnd && i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <inheritdoc />
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        ThrowIfDisposed();
        var activations = new Dictionary<string, Tensor<T>>();
        var current = input;

        // SigLIP2 stores the vision tower, text tower, caption decoder, and MIM
        // decoder in one layer collection, but Predict/ForwardForTraining intentionally
        // execute only the vision tower for an image tensor. Mirror that public forward
        // contract here; chaining the image embedding into the text tower feeds the
        // projection width into attention weights sized for TextEmbeddingDim.
        for (int i = 0; i < _visionEncoderEnd && i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Predict returns the VISION-ENCODER output (the first <c>_visionEncoderEnd</c>
    /// layers), not a sequential pass over the whole Layers list (which also
    /// contains the text encoder, captioning decoder, and MIM decoder). The base
    /// <see cref="NeuralNetworks.NeuralNetworkBase{T}.ForwardForTraining"/> runs
    /// every layer in order, so training would optimize the full-stack output
    /// while the test measures the vision-encoder output — the loss being
    /// minimized would not be the loss being measured, and the measured loss could
    /// rise. Mirror Predict here so training optimizes the same output.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var current = input;
        for (int i = 0; i < _visionEncoderEnd && i < Layers.Count; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var meta = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SigLIP2-Native" : "SigLIP2-ONNX",
            Description =
                "SigLIP 2: Multilingual Vision-Language Encoders with "
                + "Improved Semantic Understanding (Tschannen et al., 2025)",
            FeatureCount = _options.ProjectionDim,
            Complexity =
                _options.NumVisionLayers
                + _options.NumTextLayers
                + _options.NumCaptioningDecoderLayers
                + _options.NumMimDecoderLayers,
        };
        meta.AdditionalInfo["Architecture"] = "SigLIP2";
        meta.AdditionalInfo["VisionEncoder"] = _options.VisionEncoderVariant.ToString();
        meta.AdditionalInfo["LossType"] = _options.LossType.ToString();
        meta.AdditionalInfo["ProjectionDim"] = _options.ProjectionDim.ToString();
        meta.AdditionalInfo["CaptioningLossWeight"] = _options.CaptioningLossWeight.ToString();
        meta.AdditionalInfo["SelfSupervisedLossWeight"] =
            _options.SelfSupervisedLossWeight.ToString();
        meta.AdditionalInfo["Multilingual"] = _options.Multilingual.ToString();
        meta.AdditionalInfo["NumCaptioningDecoderLayers"] =
            _options.NumCaptioningDecoderLayers.ToString();
        meta.AdditionalInfo["MimMaskRatio"] = _options.MimMaskRatio.ToString();
        return meta;
    }

    #endregion

    #region Private Helpers

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");

        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = TokenIdsToTensor(encoding.TokenIds, seqLen);
        return tokens;
    }

    /// <summary>
    /// Forwards input through the vision encoder layers.
    /// </summary>
    private Tensor<T> ForwardVisionEncoder(Tensor<T> input)
    {
        int end = Math.Min(_visionEncoderEnd, Layers.Count);
        var current = input;
        for (int i = 0; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <summary>
    /// Forwards input through the text encoder layers.
    /// </summary>
    private Tensor<T> ForwardTextEncoder(Tensor<T> tokens)
    {
        int start = _visionEncoderEnd;
        int end = Math.Min(_textEncoderEnd, Layers.Count);
        var current = tokens;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    /// <summary>
    /// Projects vision encoder output to the shared embedding space.
    /// The vision projection layer is the last layer of the vision encoder section.
    /// </summary>
    private Tensor<T> ProjectVision(Tensor<T> visionOutput)
    {
        // The last layer before text encoder is the projection
        return visionOutput; // Already projected in ForwardVisionEncoder
    }

    /// <summary>
    /// Projects text encoder output to the shared embedding space.
    /// The text projection layer is the last layer of the text encoder section.
    /// </summary>
    private Tensor<T> ProjectText(Tensor<T> textOutput)
    {
        return textOutput; // Already projected in ForwardTextEncoder
    }

    /// <summary>
    /// Forwards through the MIM decoder layers to predict masked patch features.
    /// </summary>
    private Tensor<T> ForwardMimDecoder(Tensor<T> input)
    {
        int start = _captioningDecoderEnd;
        int end = Math.Min(_mimDecoderEnd, Layers.Count);
        var current = input;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SigLIP2<T>));
    }

    #endregion

    #region Disposal

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
