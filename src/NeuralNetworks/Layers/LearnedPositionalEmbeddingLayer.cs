using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Adds a LEARNED per-position vector to a sequence of feature vectors — the position embedding used
/// by BERT, ViT, BEiT and everything descended from them.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// This is the trainable counterpart of <see cref="PositionalEncodingLayer{T}"/>. That layer computes
/// fixed sinusoids and declares <c>SupportsTraining =&gt; false</c>, which is right for the original
/// Transformer (Vaswani et al. 2017) and wrong for the very large family that learns positions
/// instead: BERT, ViT, BEiT, and the document models built on them all hold a
/// <c>[maxSequenceLength, dim]</c> table and train it.
/// </para>
/// <para>
/// The distinction is not cosmetic. A sinusoid encodes position as a fixed function of the index, so
/// the model can only use whatever relationships that function happens to expose. A learned table
/// lets position 0 mean "start of document" and position 7 mean whatever the data says, which is why
/// the models above chose it. Substituting sinusoids leaves the architecture unable to represent what
/// the paper trained.
/// </para>
/// <para>
/// Several models in this repository declared exactly this table as a model FIELD — allocated,
/// randomly initialised, counted as a parameter, serialized into every checkpoint and stepped by the
/// optimizer — while a sinusoidal layer sat in the stack doing the work and nothing ever read the
/// field. Both halves of that are fixed by having one trainable layer in the stack.
/// </para>
/// <para><b>For Beginners:</b> A transformer sees a sequence as an unordered set unless you tell it
/// where each item sits. This layer adds a small learned vector to each position, so the model can
/// tell the first word from the fifth. "Learned" means those vectors are trained with everything
/// else rather than computed from a formula.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Positional)]
[LayerTask(LayerTask.PositionalEncoding)]
[LayerProperty(IsTrainable = true,
    TestInputShape = "16, 8", TestConstructorArgs = "16, 8")]
// Shape-preserving at every rank: one vector is added per position and nothing is reshaped, so the
// analyzer's ElementWiseShape declaration says all there is to say.
[ElementWiseShape]
[AutoParameters]
public partial class LearnedPositionalEmbeddingLayer<T> : LayerBase<T>
{
    private readonly EmbeddingLayer<T> _positionTable;
    private readonly int _maxSequenceLength;
    private readonly int _embeddingDim;

    /// <summary>Longest sequence the table can index; later positions reuse the last row.</summary>
    public int MaxSequenceLength => _maxSequenceLength;

    /// <summary>Width of each position vector, matching the features it is added to.</summary>
    public int EmbeddingDim => _embeddingDim;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>The table is sized entirely by constructor arguments, so it needs no input shape.</summary>
    protected override bool ParametersAreConstructionSized => true;

    /// <summary>
    /// Initializes a new <see cref="LearnedPositionalEmbeddingLayer{T}"/>.
    /// </summary>
    /// <param name="maxSequenceLength">Number of positions the table covers (512 for BERT).</param>
    /// <param name="embeddingDim">Width of each position vector; must match the input's feature axis.</param>
    public LearnedPositionalEmbeddingLayer(
        [LayerState] int maxSequenceLength = 512,
        [LayerState] int embeddingDim = 768)
        // MaxSequenceLength sizes the lookup table; it is a capacity, not a promise that every
        // caller supplies exactly that many positions. The layer is element-wise and preserves the
        // real sequence extent, while the feature width is the one binding dimension.
        : base([-1, embeddingDim], [-1, embeddingDim])
    {
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (embeddingDim <= 0) throw new ArgumentOutOfRangeException(nameof(embeddingDim));

        _maxSequenceLength = maxSequenceLength;
        _embeddingDim = embeddingDim;

        _positionTable = new EmbeddingLayer<T>(maxSequenceLength, embeddingDim);

        // Register the child so TapeTrainingStep.CollectParameters walks into it: an unregistered
        // sub-layer is invisible to the optimizer and trains silently never.
        RegisterSubLayer(_positionTable);
    }

    /// <summary>Gets the position table, exposed so a model can load pretrained vectors.</summary>
    public EmbeddingLayer<T> PositionTable => _positionTable;

    /// <summary>
    /// Adds the learned position vector for each step to that step's features.
    /// </summary>
    /// <param name="input">
    /// <c>[seq, dim]</c> or <c>[batch, seq, dim]</c>. The last axis is the feature width and must
    /// equal <see cref="EmbeddingDim"/>; the axis before it is the sequence.
    /// </param>
    /// <returns>The input with one learned vector added per position — same shape.</returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureParametersMaterialized();

        int rank = input.Rank;
        if (rank < 2)
            throw new ArgumentException(
                $"LearnedPositionalEmbeddingLayer expects [seq, dim] or [batch, seq, dim]; got rank {rank}.",
                nameof(input));

        int dim = input.Shape[rank - 1];
        if (dim != _embeddingDim)
            throw new ArgumentException(
                $"LearnedPositionalEmbeddingLayer was built for a feature width of {_embeddingDim} " +
                $"but the input's last axis is {dim}. The position vector is ADDED to the features, " +
                "so the two widths have to agree.",
                nameof(input));

        int seqLen = input.Shape[rank - 2];

        // Index grid = the input shape without its feature axis, so the lookup produces exactly the
        // input's shape back and the add needs no broadcasting rule of its own.
        var indexShape = new int[rank - 1];
        for (int i = 0; i < rank - 1; i++) indexShape[i] = input.Shape[i];

        var positions = new Tensor<T>(indexShape);
        var span = positions.Data.Span;
        for (int i = 0; i < span.Length; i++)
        {
            // Position within the sequence; wraps per batch row. Sequences longer than the table
            // reuse its last row rather than throwing — a caller who feeds 600 tokens to a
            // 512-position model gets a degraded tail, not a crash mid-inference.
            span[i] = NumOps.FromDouble(Math.Min(i % seqLen, _maxSequenceLength - 1));
        }

        return Engine.TensorAdd(input, _positionTable.Forward(positions));
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) => _positionTable.UpdateParameters(learningRate);

    /// <inheritdoc/>
    public override void ResetState() => _positionTable.ResetState();
}
