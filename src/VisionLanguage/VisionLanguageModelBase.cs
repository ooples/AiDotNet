using System.Collections.Generic;
// AiDotNet.Attributes is REQUIRED for [TensorLayout] to bind to the right type: two other Tensors
// namespaces declare a TensorLayout, and without this using the attribute silently resolves to one
// of those and the contract is never seen.
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;

namespace AiDotNet.VisionLanguage;

/// <summary>
/// Base class for vision-language neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide vision-language-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Vision-language models process both images and text together. This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Image preprocessing utilities (normalization, resizing)
/// - Dual-encoder architecture support (separate image and text encoders)
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input,
    Note = "An image batch. ImageChannels x ImageSize x ImageSize is the declared per-sample shape, "
         + "though probing showed the spatial axes are free rather than pinned to ImageSize.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output,
    Note = "The pooled joint-embedding law: one EmbeddingDim-wide vector per sample. Two other "
         + "measured laws exist in this family - see PatchTokenContract and TrailingFeatureContract.")]
public abstract partial class VisionLanguageModelBase<T> : NeuralNetworkBase<T>, IShapeContract
{
    /// <summary>
    /// The family's default output law: one pooled <see cref="EmbeddingDim"/>-wide embedding per sample.
    /// </summary>
    /// <remarks>
    /// <para>
    /// MEASURED, not assumed. Probing this family found THREE distinct output laws, so a single
    /// declaration for all of it would have been wrong - which is exactly why this is virtual:
    /// </para>
    /// <list type="bullet">
    /// <item><description><b>Pooled</b> (this default) - ALIGN: <c>[1,3,8,8] -&gt; [1,640]</c>, and 640
    /// is its EmbeddingDim. The classic dual-encoder law.</description></item>
    /// <item><description><b>Patch tokens</b> - BiomedCLIP and CLIPA: <c>[1,3,8,8] -&gt; [1,64,512]</c>
    /// but <c>[1,3,12,8] -&gt; [1,96,512]</c>. That middle axis is the PRODUCT of the two spatial axes,
    /// not a constant - see <see cref="PatchTokenContract"/>.</description></item>
    /// <item><description><b>Trailing feature</b> - BASIC: <c>[1,3,8,8] -&gt; [1,3,8,1024]</c>. Every
    /// input axis is carried and only the LAST is replaced - see
    /// <see cref="TrailingFeatureContract"/>.</description></item>
    /// </list>
    /// <para>
    /// NONE of the three is the default, and that is the measurement talking. Two of the first FOUR
    /// models probed did not follow the pooled law, so defaulting to it would attach a wrong contract
    /// to an unknown fraction of 170 models. The conformance sweep that would settle the rest cannot
    /// finish: these are LLaVA-scale networks, and a full-family pass exceeded thirty minutes without
    /// reporting a single model. Declaring an unverified family law here to look complete is exactly
    /// the mistake that put a /32 stride on ten segmentation models that did not have it.
    /// </para>
    /// <para>
    /// So this declines, each MEASURED model states its own law in one line, and the remaining backlog
    /// stays visible in the sweep instead of being hidden behind a confident default.
    /// </para>
    /// </remarks>
    [ShapeContractUnavailable(
        "Vision-language models have several measured output laws, so an unmeasured derived model must decline rather than inherit a guessed family default.")]
    public virtual IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank) => null;

    /// <summary>The pooled law: <c>[Batch, EmbeddingDim]</c>, whatever the input rank above 1.</summary>
    protected IReadOnlyList<OutputAxisContract>? PooledEmbeddingContract(int inputRank)
    {
        if (inputRank < 2 || EmbeddingDim <= 0) return null;
        return
        [
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(EmbeddingDim)),
        ];
    }

    // The other two laws probing found are deliberately NOT offered as helpers, because the sweep
    // refuted the versions of them that seemed to fit:
    //
    //   PATCH TOKENS - BiomedCLIP and CLIPA looked like [Batch, Product(Height,Width), EmbeddingDim]
    //   from [1,3,8,8] -> [1,64,512] and [1,3,12,8] -> [1,96,512]. At extent 64 the real answer was
    //   [1,256,512], not [1,4096,768]: the token axis is a patch grid that does not track H*W, and the
    //   width is not EmbeddingDim.
    //
    //   TRAILING FEATURE - BASIC looked like [..., EmbeddingDim] from [1,3,8,8] -> [1,3,8,1024]. At
    //   extent 64 EmbeddingDim had moved to 1536 while the output width stayed 1024, so the two were
    //   never the same quantity.
    //
    // Leaving those as protected helpers would hand the next author a ready-made wrong answer with a
    // reassuring name. They come back when the real quantities are traced, not before.

    /// <summary>
    /// Gets the expected input image size (height = width in pixels).
    /// </summary>
    public int ImageSize { get; protected set; } = 224;

    /// <summary>
    /// Gets the number of image channels expected (typically 3 for RGB).
    /// </summary>
    public int ImageChannels { get; protected set; } = 3;

    /// <summary>
    /// Gets the embedding dimensionality for this model.
    /// </summary>
    public int EmbeddingDim { get; protected set; } = 512;

    /// <inheritdoc/>
    protected override string? TrainingUnavailableReason =>
        IsOnnxMode ? OnnxTrainingUnavailableReason : base.TrainingUnavailableReason;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode =>
        OnnxImageEncoder is not null || OnnxTextEncoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX image encoder model (for dual-encoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxImageEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX text encoder model (for dual-encoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxTextEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Initializes a new instance of the VisionLanguageModelBase class.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected VisionLanguageModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0
    )
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm) { }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses a raw image tensor for model input.
    /// </summary>
    /// <param name="image">Raw image tensor in [channels, height, width] or [height, width, channels] format.</param>
    /// <returns>Preprocessed image tensor suitable for model input.</returns>
    protected abstract Tensor<T> PreprocessImage(Tensor<T> image);

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Normalizes each channel of each image using CLIP/OpenAI statistics by default.
    /// </summary>
    /// <param name="image">Image tensor in [C,H,W] or [N,C,H,W] layout with values in [0,1].</param>
    /// <param name="mean">Finite per-channel mean values.</param>
    /// <param name="std">Positive finite per-channel standard deviations.</param>
    /// <returns>Normalized image tensor.</returns>
    protected Tensor<T> NormalizeImage(Tensor<T> image, double[]? mean = null, double[]? std = null)
    {
        mean ??= [0.48145466, 0.4578275, 0.40821073]; // CLIP/OpenAI default
        std ??= [0.26862954, 0.26130258, 0.27577711];

        if (image is null) throw new ArgumentNullException(nameof(image));
        int channels = ImageChannels;
        if ((image.Rank != 3 && image.Rank != 4) || channels <= 0
            || image.Shape[image.Rank == 4 ? 1 : 0] != channels || image.Length == 0)
            throw new ArgumentException("Expected a nonempty [C,H,W] or [N,C,H,W] image with the model's channel count.", nameof(image));
        if (mean.Length != channels || std.Length != channels)
            throw new ArgumentException("Image normalization requires one mean and standard deviation per channel.");
        for (int c = 0; c < channels; c++)
        {
            if (double.IsNaN(mean[c]) || double.IsInfinity(mean[c]))
                throw new ArgumentOutOfRangeException(nameof(mean), "Channel means must be finite.");
            if (double.IsNaN(std[c]) || double.IsInfinity(std[c]) || std[c] <= 0)
                throw new ArgumentOutOfRangeException(nameof(std), "Channel standard deviations must be positive and finite.");
        }

        var result = new Tensor<T>(image._shape);
        int batches = image.Rank == 4 ? image.Shape[0] : 1;
        int spatialSize = image.Length / batches / channels;
        for (int batch = 0; batch < batches; batch++)
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = (batch * channels + c) * spatialSize;
                for (int i = 0; i < spatialSize; i++)
                    result[offset + i] = NumOps.FromDouble((NumOps.ToDouble(image[offset + i]) - mean[c]) / std[c]);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes cosine similarity between two embedding tensors.
    /// </summary>
    /// <param name="a">First embedding.</param>
    /// <param name="b">Second embedding.</param>
    /// <returns>Cosine similarity score in [-1, 1].</returns>
    protected T CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        return NumOps.FromDouble(VectorHelper.CosineSimilarity(a.ToVector(), b.ToVector()));
    }

    /// <summary>
    /// Applies softmax to convert logits to probabilities.
    /// </summary>
    /// <param name="logits">Raw scores.</param>
    /// <returns>Probabilities that sum to 1.</returns>
    protected Tensor<T> Softmax(Tensor<T> logits)
    {
        if (logits.Length == 0)
            return new Tensor<T>(logits._shape);

        // Softmax over every element (not one axis). NaN logits are ignored when finding the max,
        // and when no finite or +inf logit exists the shift is zero, so exp(-inf) stays 0.
        var negInf = new Tensor<T>(logits._shape);
        negInf.Fill(NumOps.FromDouble(double.NegativeInfinity));
        T maxVal = Engine.TensorMaxValue(Engine.TensorWhere(Engine.TensorIsNan(logits), negInf, logits));
        if (!NumOps.GreaterThan(maxVal, NumOps.FromDouble(double.NegativeInfinity)))
            maxVal = NumOps.Zero;

        var result = Engine.TensorExp(Engine.TensorSubtractScalar(logits, maxVal));
        T sum = Engine.TensorSum(result);
        if (NumOps.GreaterThan(sum, NumOps.FromDouble(1e-8)))
            result = Engine.TensorDivideScalar(result, sum);

        return result;
    }

    /// <summary>
    /// L2-normalizes an embedding tensor.
    /// </summary>
    /// <param name="embedding">Embedding to normalize.</param>
    /// <returns>Unit-normalized embedding.</returns>
    protected Tensor<T> L2Normalize(Tensor<T> embedding)
    {
        T norm = NumOps.Sqrt(Engine.TensorSum(Engine.TensorMultiply(embedding, embedding)));
        if (NumOps.LessThan(norm, NumOps.FromDouble(1e-8)))
            return embedding;

        return Engine.TensorDivideScalar(embedding, norm);
    }

    /// <summary>
    /// Converts the first <paramref name="length"/> tokenizer ids into a rank-1 tensor of ids.
    /// </summary>
    /// <param name="tokenIds">Token ids produced by the tokenizer.</param>
    /// <param name="length">Number of leading ids to keep.</param>
    /// <returns>A [length] tensor holding the ids converted to <typeparamref name="T"/>.</returns>
    protected static Tensor<T> TokenIdsToTensor(List<int> tokenIds, int length)
    {
        if (tokenIds is null) throw new ArgumentNullException(nameof(tokenIds));
        if (length <= 0)
            return new Tensor<T>([length]);

        return new Tensor<int>(new[] { length }, Vector<int>.FromList(tokenIds.GetRange(0, length))).Cast<T>();
    }

    /// <summary>
    /// Returns the position of the first maximum of <paramref name="values"/> (flattened), matching a
    /// scalar scan that starts at negative infinity and replaces the best only on a strictly greater value:
    /// NaN never wins, ties keep the lowest position, and -1 is returned when no value exceeds -infinity.
    /// </summary>
    /// <param name="values">Scores to search.</param>
    /// <returns>The first position holding the maximum, or -1 when there is none.</returns>
    protected int FirstArgMax(Tensor<T> values)
    {
        int n = values.Length;
        if (n == 0)
            return -1;
        var flat = values.Reshape([n]);
        T negInfValue = NumOps.FromDouble(double.NegativeInfinity);
        var comparable = Engine.TensorWhere(Engine.TensorIsNan(flat), Tensor<T>.CreateDefault([n], negInfValue), flat);
        T best = Engine.TensorMaxValue(comparable);
        if (!NumOps.GreaterThan(best, negInfValue))
            return -1;
        var isBest = Engine.TensorEquals(comparable, best);
        var positions = Engine.TensorArange<T>(0, n, 1);
        var sentinel = Tensor<T>.CreateDefault([n], NumOps.FromDouble(n));
        return NumOps.ToInt(Engine.TensorMinValue(Engine.TensorWhere(isBest, positions, sentinel)));
    }

    /// <summary>
    /// Validates image/token settings used to derive patch sizes for native VLM layer factories.
    /// </summary>
    private protected static void ValidateVisualPatchOptions(int imageSize, int maxVisualTokens)
    {
        if (imageSize <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(imageSize),
                imageSize,
                "ImageSize must be greater than 0."
            );
        if (maxVisualTokens <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(maxVisualTokens),
                maxVisualTokens,
                "MaxVisualTokens must be greater than 0."
            );
    }

    /// <summary>
    /// Computes a patch size from an image size and target visual-token budget.
    /// </summary>
    private protected static int ComputeVisualPatchSize(
        int imageSize,
        int maxVisualTokens,
        bool roundUp = false
    )
    {
        ValidateVisualPatchOptions(imageSize, maxVisualTokens);
        double targetTokensPerSide = Math.Sqrt(maxVisualTokens);
        double patchSize = imageSize / targetTokensPerSide;
        return Math.Max(1, roundUp ? (int)Math.Ceiling(patchSize) : (int)patchSize);
    }

    /// <summary>
    /// Returns the layer count contributed by a standard transformer block.
    /// </summary>
    private protected static int TransformerBlockLayerCount(double dropoutRate) =>
        dropoutRate > 0 ? 6 : 5;

    /// <summary>
    /// Returns the layer count contributed by a cross-attention resampler block.
    /// </summary>
    private protected static int ResamplerBlockLayerCount(double dropoutRate) =>
        dropoutRate > 0 ? 8 : 7;

    /// <summary>
    /// Computes an encoder/decoder split boundary from repeated block counts.
    /// </summary>
    private protected static int ComputeVisionLanguageBoundary(
        int leadingLayerCount,
        int visionLayerCount,
        int visionBlockLayerCount,
        int auxiliaryBlockCount = 0,
        int auxiliaryBlockLayerCount = 0,
        int trailingLayerCount = 0
    )
    {
        if (leadingLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(leadingLayerCount));
        if (visionLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(visionLayerCount));
        if (visionBlockLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(visionBlockLayerCount));
        if (auxiliaryBlockCount < 0)
            throw new ArgumentOutOfRangeException(nameof(auxiliaryBlockCount));
        if (auxiliaryBlockLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(auxiliaryBlockLayerCount));
        if (trailingLayerCount < 0)
            throw new ArgumentOutOfRangeException(nameof(trailingLayerCount));

        return checked(
            leadingLayerCount
            + visionLayerCount * visionBlockLayerCount
            + auxiliaryBlockCount * auxiliaryBlockLayerCount
            + trailingLayerCount
        );
    }

    /// <summary>
    /// Verifies the computed encoder/decoder split is inside the current layer list.
    /// </summary>
    private protected void ValidateEncoderDecoderBoundary(int encoderLayerEnd)
    {
        if (encoderLayerEnd <= 0 || encoderLayerEnd > Layers.Count)
            throw new InvalidOperationException(
                $"Invalid encoder boundary {encoderLayerEnd} for {Layers.Count} layers."
            );
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    // --- Dual-stream layer split (CLIP / SigLIP / BLIP / etc.) ---
    //
    // CLIP-style models concatenate vision-encoder + text-encoder layers into
    // a single LayerHelper.CreateDefaultOpenCLIPLayers() output. Walking that
    // sequentially is incorrect: visionEmbeddingDim ≠ textEmbeddingDim in the
    // OpenCLIP defaults, so vision-projection output mismatches text-encoder
    // input on the first text-encoder MultiHeadAttention. This base provides
    // the storage + accessors so subclasses' Predict / Train walk the vision
    // stack only, and EncodeText walks the text stack independently — mirrors
    // PyTorch / HuggingFace CLIPModel where vision_model and text_model are
    // separate nn.Modules instead of one concatenated list.

    /// <summary>
    /// Text-encoder layers, walked by <c>EncodeText</c> in subclasses.
    /// Lives outside <see cref="NeuralNetworkBase{T}.Layers"/> so the inherited
    /// forward / TrainWithTape paths operate on the vision-only stack.
    /// Surface to streaming-pool / weight-registry by overriding
    /// <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/> in the
    /// concrete subclass (e.g. via <see cref="EnumerateTextEncoderTrainableLayers"/>).
    /// </summary>
    protected readonly List<ILayer<T>> TextEncoderLayers = new List<ILayer<T>>();

    /// <summary>
    /// Splits an OpenCLIP-shaped layer factory output (vision pre-norm +
    /// N×vision-block + vision-projection + text pre-norm + N×text-block +
    /// text-projection) into the model's <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list (vision portion) and <see cref="TextEncoderLayers"/> (text portion).
    /// </summary>
    /// <param name="allLayers">The combined factory output to split.</param>
    /// <param name="visionLayerCount">Count of vision-portion layers; for the
    /// standard OpenCLIP / SigLIP factory this is
    /// <c>2 + numVisionLayers × blockSize</c> where <c>blockSize = 5</c> at
    /// dropoutRate = 0 and <c>6</c> with dropout (the +2 is the pre-norm and
    /// the projection head).</param>
    protected void SplitDualStreamLayers(IEnumerable<ILayer<T>> allLayers, int visionLayerCount)
    {
        if (allLayers is null)
            throw new System.ArgumentNullException(nameof(allLayers));
        if (visionLayerCount < 0)
            throw new System.ArgumentOutOfRangeException(
                nameof(visionLayerCount),
                "visionLayerCount must be ≥ 0."
            );

        // Materialize once so we can validate before mutating Layers /
        // TextEncoderLayers. An oversized visionLayerCount that the
        // previous loop silently accepted (everything piles into Layers,
        // TextEncoderLayers stays empty) would surface as the same
        // class of "EncodeText silently degrades" bug the per-encoder
        // overrides try to prevent — fail fast here so a layer-factory
        // drift becomes an obvious construction error instead of a
        // runtime no-op.
        var layerList = allLayers as IList<ILayer<T>> ?? new List<ILayer<T>>(allLayers);
        if (visionLayerCount > layerList.Count)
            throw new System.ArgumentOutOfRangeException(
                nameof(visionLayerCount),
                $"visionLayerCount ({visionLayerCount}) exceeds the supplied layer sequence "
                    + $"({layerList.Count}). The supplied factory's layer count and the dual-stream "
                    + "split point have drifted apart — check the OpenCLIP-style block-size / layer-count "
                    + "math against the architecture's NumVisionLayers / DropoutRate."
            );

        for (int idx = 0; idx < layerList.Count; idx++)
        {
            if (idx < visionLayerCount)
                Layers.Add(layerList[idx]);
            else
                TextEncoderLayers.Add(layerList[idx]);
        }
    }

    /// <summary>
    /// Helper for subclasses overriding <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/>
    /// to surface their <see cref="TextEncoderLayers"/> to the base weight-registry walker.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateTextEncoderTrainableLayers()
    {
        foreach (var layer in TextEncoderLayers)
            if (layer is LayerBase<T> lb)
                yield return lb;
    }

    // --- Auxiliary-stream support (BLIP-2 / Q-Former / fusion VL models) ---
    //
    // Triple-stream architectures (vision encoder → Q-Former → decoder, or
    // vision + text + cross-modal fusion bridge) need MORE than the dual-stream
    // split. Subclasses register additional streams via
    // <see cref="RegisterAuxiliaryEncoderStream"/>; the base's auxiliary
    // enumerator walks them all so the inherited GetExtraTrainableLayers /
    // streaming-pool / weight-registry hooks stay correct without per-subclass
    // boilerplate.

    private readonly List<List<ILayer<T>>> _auxiliaryEncoderStreams = new List<List<ILayer<T>>>();

    /// <summary>
    /// Registers an auxiliary encoder stream that lives outside
    /// <see cref="NeuralNetworkBase{T}.Layers"/>. The base does NOT walk these
    /// in <c>Predict</c> / <c>TrainWithTape</c>; subclasses use them in their
    /// own forward methods (e.g. <c>GenerateFromImage</c> for Q-Former models,
    /// <c>FuseVisionAndText</c> for BridgeTower-style fusion). Registered
    /// streams are surfaced through
    /// <see cref="EnumerateAuxiliaryStreamTrainableLayers"/> so subclasses can
    /// override <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/>
    /// to yield from auxiliary + text streams in one call.
    /// </summary>
    /// <param name="stream">A non-null stream to register. Adds a reference,
    /// not a copy — subsequent mutations to <paramref name="stream"/> are
    /// visible.</param>
    protected void RegisterAuxiliaryEncoderStream(List<ILayer<T>> stream)
    {
        if (stream is null)
            throw new System.ArgumentNullException(nameof(stream));
        _auxiliaryEncoderStreams.Add(stream);
    }

    /// <summary>
    /// Iterates the registered auxiliary streams (Q-Former, decoder, fusion
    /// bridge, etc.) yielding each layer's <see cref="LayerBase{T}"/> view.
    /// Use alongside <see cref="EnumerateTextEncoderTrainableLayers"/> when
    /// the subclass owns BOTH a text encoder and additional streams.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateAuxiliaryStreamTrainableLayers()
    {
        foreach (var stream in _auxiliaryEncoderStreams)
        foreach (var layer in stream)
            if (layer is LayerBase<T> lb)
                yield return lb;
    }

    /// <summary>
    /// Combined helper that yields trainable-layer references for both the
    /// dual-stream <see cref="TextEncoderLayers"/> and any registered
    /// auxiliary streams. Most subclasses override
    /// <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/> to return
    /// this directly.
    /// </summary>
    protected IEnumerable<LayerBase<T>?> EnumerateAllAuxiliaryTrainableLayers()
    {
        foreach (var l in EnumerateTextEncoderTrainableLayers())
            yield return l;
        foreach (var l in EnumerateAuxiliaryStreamTrainableLayers())
            yield return l;
    }

    /// <summary>
    /// Surfaces the text-encoder and auxiliary streams to the parameter walk for EVERY subclass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This used to be advice rather than behaviour: the helpers above existed and each subclass was
    /// expected to override <see cref="NeuralNetworkBase{T}.GetExtraTrainableLayers"/> itself to call
    /// one of them. Opt-in is the wrong default for this. A subclass that forgot kept working --
    /// nothing threw, nothing failed -- while its Q-Former, decoder, fusion bridge and text encoder
    /// were absent from ParameterCount, from GetParameters and from every checkpoint. That is the
    /// invisible failure this whole surface exists to make impossible.
    /// </para>
    /// <para>
    /// A subclass that already overrides this shadows it and is unaffected. One that needs a
    /// different set still overrides; it just no longer has to in order to be correct.
    /// </para>
    /// </remarks>
    protected override IEnumerable<LayerBase<T>?> GetExtraTrainableLayers()
        => EnumerateAllAuxiliaryTrainableLayers();

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxImageEncoder?.Dispose();
            OnnxTextEncoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
