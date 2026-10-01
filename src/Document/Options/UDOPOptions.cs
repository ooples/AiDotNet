using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the UDOP document model.
/// </summary>
public class UDOPOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="UDOPOptions"/> class with the published UDOP-large
    /// configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These values come from HF <c>UdopConfig</c> for <c>microsoft/udop-large</c>, which is a T5-large
    /// encoder-decoder with: d_model 1024, 16 heads of d_kv 64, d_ff 4096, 24 encoder and 24 decoder blocks, ReLU
    /// feed-forward, RMSNorm epsilon 1e-6, and a vocabulary of 33201. Pages are 224x224 with 16x16 patches. The
    /// relative biases use 32 buckets: a 1-D maximum distance of 128, and 100 for the horizontal and vertical
    /// layout biases. The 2-D cell tables have 1024 positions each.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> These defaults build the full-size model, about 740 million parameters. For
    /// experiments, shrink <see cref="DocumentNeuralNetworkOptions.HiddenDim"/>, the layer counts and the
    /// vocabulary together.
    /// </para>
    /// </remarks>
    public UDOPOptions()
    {
        ImageSize = 224;
        PatchSize = 16;
        MaxSequenceLength = 2048;
        HiddenDim = 1024;
        NumEncoderLayers = 24;
        NumDecoderLayers = 24;
        NumHeads = 16;
        VocabSize = 33201;
    }

    /// <summary>
    /// Width of each attention head (T5 <c>d_kv</c>). Defaults to 64. T5 does not require
    /// <c>NumHeads * KeyValueDim</c> to equal the hidden size.
    /// </summary>
    public int KeyValueDim { get; set; } = 64;

    /// <summary>Inner width of each ReLU feed-forward block (T5 <c>d_ff</c>). Defaults to 4096.</summary>
    public int FeedForwardDim { get; set; } = 4096;

    /// <summary>Number of relative-position buckets shared by every bias table. Defaults to 32.</summary>
    public int RelativeAttentionBuckets { get; set; } = 32;

    /// <summary>Largest 1-D token distance with its own bucket range (T5 <c>max_distance</c>). Defaults to 128.</summary>
    public int RelativeAttentionMaxDistance { get; set; } = 128;

    /// <summary>
    /// Largest horizontal or vertical layout distance (0-1 coordinates scaled by 100) before buckets saturate.
    /// Defaults to 100, as in <c>RelativePositionBiasHorizontal</c> and <c>RelativePositionBiasVertical</c>.
    /// </summary>
    public int RelativeAttentionMaxDistance2D { get; set; } = 100;

    /// <summary>Rows in each 2-D cell-embedding table (<c>max_2d_position_embeddings</c>). Defaults to 1024.</summary>
    public int Max2DPositions { get; set; } = 1024;

    /// <summary>RMSNorm epsilon (<c>layer_norm_epsilon</c>). Defaults to 1e-6.</summary>
    public double LayerNormEpsilon { get; set; } = 1e-6;

    /// <summary>
    /// Number of layout location tokens, quantising a 0-1 coordinate into this many bins. Defaults to 501, the
    /// UDOP tokenizer's <c>&lt;loc_0&gt;</c>..<c>&lt;loc_500&gt;</c>.
    /// </summary>
    /// <remarks>
    /// This implementation places the location tokens at the end of the vocabulary:
    /// <c>[VocabSize - NumLocationBins, VocabSize)</c>. When you run a checkpoint, map its tokenizer's location ids
    /// onto that range.
    /// </remarks>
    public int NumLocationBins { get; set; } = 501;

    /// <summary>
    /// Largest number of tokens a prompted decode (question answering, layout analysis) generates before it stops.
    /// Defaults to 128. This cap belongs to this implementation, not to the paper.
    /// </summary>
    public int MaxGenerationLength { get; set; } = 128;
    /// <summary>
    /// Initial learning rate. Defaults to the paper's 5e-5.
    /// </summary>
    /// <remarks>
    /// <para>
    /// UDOP (Tang et al., arXiv:2212.02623 S4.1) trains with "learning rate 5e-5, 1000 warmup
    /// steps, batch size 512, weight decay of 1e-2, beta1 = 0.9, and beta2 = 0.98". The model
    /// built its optimizer with no options at all, so it ran at Adam's own 1e-3 default -- twenty
    /// times the paper's rate, with beta2 at 0.999 and no weight decay.
    /// </para>
    /// <para>
    /// The paper's 1000-step warmup is deliberately NOT applied by default. It is calibrated to
    /// batch-512 pretraining, and over a short run it would hold the rate near zero for the whole
    /// run; callers reproducing the paper's schedule can attach a warmup scheduler explicitly.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Too big and
    /// training gets worse the longer it runs.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 5e-5;


    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        // This model renders the page as an image, so it needs a size. The family base
        // cannot require this: 15 of the 29 document models work from text and layout
        // coordinates and have no image at all.
        Require(ImageSize, nameof(ImageSize));
        Require(PatchSize, nameof(PatchSize));
        Require(KeyValueDim, nameof(KeyValueDim));
        Require(FeedForwardDim, nameof(FeedForwardDim));
        Require(RelativeAttentionBuckets, nameof(RelativeAttentionBuckets));
        Require(RelativeAttentionMaxDistance, nameof(RelativeAttentionMaxDistance));
        Require(RelativeAttentionMaxDistance2D, nameof(RelativeAttentionMaxDistance2D));
        Require(Max2DPositions, nameof(Max2DPositions));
        Require(MaxGenerationLength, nameof(MaxGenerationLength));
        if (ImageSize % PatchSize != 0)
            throw new ArgumentException($"ImageSize ({ImageSize}) must be a multiple of PatchSize ({PatchSize}).", nameof(PatchSize));
        if (NumLocationBins < 2 || NumLocationBins > VocabSize - 2)
            throw new ArgumentException(
                $"NumLocationBins ({NumLocationBins}) must be at least 2 and leave room for pad, EOS and text tokens in a vocabulary of {VocabSize}.",
                nameof(NumLocationBins));
        if (LayerNormEpsilon <= 0 || double.IsNaN(LayerNormEpsilon))
            throw new ArgumentException("LayerNormEpsilon must be positive.", nameof(LayerNormEpsilon));
        ValidateCore();
    }
}
