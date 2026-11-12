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
    /// Generates a caption for the given image using the autoregressive captioning decoder.
    /// </summary>
    /// <param name="image">Image tensor to caption.</param>
    /// <returns>Tensor of predicted token logits for caption generation.</returns>
    /// <remarks>
    /// <para>SigLIP 2's captioning decoder is a lightweight 4-layer transformer that attends to
    /// vision encoder features via cross-attention. During training, this is supervised with
    /// next-token prediction loss. At inference, tokens are generated autoregressively.</para>
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

        // Get vision encoder features (before projection, since decoder cross-attends to full features)
        var visionFeatures = ForwardVisionEncoder(preprocessed);

        // Greedy autoregressive decoding: the sequence is a start-of-sequence embedding followed by the
        // embedding of every token generated so far. Each step cross-attends the whole sequence to the vision
        // tokens, runs the decoder, and takes the last position's logits; their argmax is the next token,
        // embedded with the vocabulary projection's weights (tied input/output embeddings).
        int maxLen = _options.MaxCaptionLength;
        if (maxLen <= 0)
            return new Tensor<T>([0]);

        int width = _options.CaptioningDecoderDim;
        var sequence = Tensor<T>.CreateDefault([1, width], NumOps.FromDouble(0.01));
        var stepLogits = new Tensor<T>[maxLen];
        for (int step = 0; step < maxLen; step++)
        {
            int positions = sequence.Shape[0];
            var decoderOutput = ForwardCaptioningDecoder(CrossAttend(sequence, visionFeatures));
            int perPosition = decoderOutput.Length / positions;
            var last = Engine.TensorSlice(decoderOutput.Reshape([positions, perPosition]),
                [positions - 1, 0], [1, perPosition]);
            stepLogits[step] = FitLength(last, _options.VocabSize);

            if (step + 1 < maxLen)
            {
                int next = Math.Max(0, ArgExtremumSearch<T>.FirstIndex(Engine, last, true, out _));
                sequence = Engine.TensorConcatenate([sequence, CaptionTokenEmbedding(next, width)], 0);
            }
        }

        return Engine.TensorConcatenate(stepLogits, 0);
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
    /// Computes masked image modeling predictions for self-supervised training.
    /// </summary>
    /// <param name="image">Image tensor to process.</param>
    /// <param name="maskIndices">Indices of patches to mask (null for random masking).</param>
    /// <returns>Predicted features for the masked patches.</returns>
    /// <remarks>
    /// <para>The MIM objective masks a fraction of image patches (default 50%) and predicts their
    /// features from the unmasked context. This is similar to MAE (Masked Autoencoder) but uses
    /// the SigLIP 2 vision encoder as the backbone. The MIM decoder is a lightweight 2-layer MLP
    /// that maps from encoder features to predicted patch features.</para>
    /// </remarks>
    public Tensor<T> PredictMaskedPatches(Tensor<T> image, int[]? maskIndices = null)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessImage(image);

        int patchSize = _options.PatchSize;
        int totalPatches = (preprocessed.Shape[preprocessed.Rank - 2] / patchSize)
            * (preprocessed.Shape[preprocessed.Rank - 1] / patchSize);
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

        // Forward through vision encoder with the masked patches zeroed
        var encodedFeatures = ForwardVisionEncoder(MaskPatches(preprocessed, maskIndices));

        // Forward through MIM decoder to predict masked patch features
        var mimPredictions = ForwardMimDecoder(encodedFeatures);

        return mimPredictions;
    }

    /// <summary>
    /// Zeroes the listed patches of a <c>[C, H, W]</c> or <c>[N, C, H, W]</c> image in every channel. Patch
    /// <c>p</c> is the <c>PatchSize x PatchSize</c> window at row <c>p / (W / PatchSize)</c> and column
    /// <c>p % (W / PatchSize)</c> of the patch grid, the order the patch embedding reads patches in.
    /// </summary>
    /// <param name="image">The image to mask.</param>
    /// <param name="maskIndices">Patch indices to zero; each must lie in the patch grid.</param>
    /// <returns>A masked copy of <paramref name="image"/>.</returns>
    internal Tensor<T> MaskPatches(Tensor<T> image, int[] maskIndices)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (maskIndices is null) throw new ArgumentNullException(nameof(maskIndices));
        if (image.Rank < 2)
            throw new ArgumentException("Expected a [C, H, W] or [N, C, H, W] image.", nameof(image));

        int patchSize = _options.PatchSize;
        int height = image.Shape[image.Rank - 2];
        int width = image.Shape[image.Rank - 1];
        int gridHeight = height / patchSize;
        int gridWidth = width / patchSize;
        int totalPatches = gridHeight * gridWidth;
        foreach (int patch in maskIndices)
        {
            if (patch < 0 || patch >= totalPatches)
                throw new ArgumentOutOfRangeException(nameof(maskIndices),
                    $"Patch index {patch} is outside the {gridHeight}x{gridWidth} patch grid.");
        }

        if (maskIndices.Length == 0)
            return image.Clone();

        var keepPatch = Engine.TensorIndexFill(
            Tensor<T>.CreateDefault([totalPatches], NumOps.One), 0,
            new Tensor<int>(new[] { maskIndices.Length }, Vector<int>.FromArray(maskIndices)), NumOps.Zero);
        var keepPixel = Engine.TensorRepeatInterleave(
            Engine.TensorRepeatInterleave(keepPatch.Reshape([gridHeight, gridWidth]), patchSize, 0), patchSize, 1);

        // Pixels past the last whole patch belong to no patch and are kept.
        if (gridHeight * patchSize < height)
            keepPixel = Engine.TensorConcatenate(
                [keepPixel, Tensor<T>.CreateDefault([height - gridHeight * patchSize, gridWidth * patchSize], NumOps.One)], 0);
        if (gridWidth * patchSize < width)
            keepPixel = Engine.TensorConcatenate(
                [keepPixel, Tensor<T>.CreateDefault([height, width - gridWidth * patchSize], NumOps.One)], 1);

        // Select rather than multiply, so a non-finite pixel in a masked patch still becomes zero.
        return Engine.TensorWhere(Engine.TensorBroadcastTo(keepPixel, image.Shape.ToArray()), image,
            new Tensor<T>(image.Shape.ToArray()));
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

    /// <inheritdoc />


    /// <inheritdoc />


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
    /// Cross-attention: decoder query attends to vision encoder key/value features.
    /// Implements scaled dot-product cross-attention for the captioning decoder.
    /// </summary>
    /// <param name="query">Decoder positions, read as rows of the last axis' width.</param>
    /// <param name="context">Vision tokens, read as rows of the same width; they are both keys and values.</param>
    /// <returns>The <c>[positions, width]</c> residual sum query + softmax(Q K^T / sqrt(d)) V.</returns>
    internal Tensor<T> CrossAttend(Tensor<T> query, Tensor<T> context)
    {
        int width = query.Shape[query.Rank - 1];
        int contextWidth = context.Shape[context.Rank - 1];
        if (width <= 0 || contextWidth != width)
            throw new InvalidOperationException(
                $"Cross-attention needs vision tokens as wide as the decoder ({width}); got {contextWidth}. "
                + "Set CaptioningDecoderDim equal to ProjectionDim.");

        var queries = query.Reshape([query.Length / width, width]);
        var tokens = context.Reshape([context.Length / width, width]);

        // Scale the queries before the product so the scores stay small in narrow types.
        var scaledQueries = Engine.TensorMultiplyScalar(queries, NumOps.FromDouble(1.0 / Math.Sqrt(width)));
        var weights = Engine.Softmax(Engine.TensorMatMul(scaledQueries, Engine.TensorTranspose(tokens)), 1);
        return Engine.TensorAdd(queries, Engine.TensorMatMul(weights, tokens));
    }

    /// <summary>
    /// Forwards through the captioning decoder layers.
    /// </summary>
    private Tensor<T> ForwardCaptioningDecoder(Tensor<T> input)
    {
        if (!_options.IncludeCaptioningDecoder)
            return input;

        int start = _textEncoderEnd;
        int end = Math.Min(_captioningDecoderEnd, Layers.Count);
        var current = input;
        for (int i = start; i < end; i++)
            current = Layers[i].Forward(current);
        return current;
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
