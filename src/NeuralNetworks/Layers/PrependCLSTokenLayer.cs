using AiDotNet.ActivationFunctions;
// File-level, deliberately: two Tensors namespaces in the project's global usings also define a
// TensorLayout, so [TensorLayout(...)] only binds when this import shadows them from a nearer scope.
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Prepends a learnable <c>[CLS]</c> token to a sequence-of-embeddings
/// input, as introduced by BERT (Devlin et al. 2018 §3.1) and adopted by
/// ViT / AST (Dosovitskiy et al. 2020 §3.1; Gong et al. 2021 §2.2). The
/// CLS token starts as a learnable parameter <c>[1, embedDim]</c>; the
/// layer broadcasts it across the batch and concatenates it at sequence
/// position 0 so the transformer's first output position becomes the
/// classification representation.
/// </summary>
/// <typeparam name="T">Numeric type for tensor data.</typeparam>
/// <remarks>
/// <para>
/// Pairs with <see cref="SequenceTokenSliceLayer{T}"/> using
/// <see cref="SequenceTokenSliceLayer{T}.Position.First"/> after the
/// transformer stack to extract the trained classification embedding —
/// the canonical AST / ViT classification head.
/// </para>
/// <para><b>For Beginners:</b> Most transformer classifiers prepend a
/// special learnable token to the input sequence; the network learns to
/// use that one token as a "summary slot" for the whole sequence. After
/// the transformer runs, you read just that one position to get the
/// classification feature — no mean-pooling required, and gradient flow
/// during training teaches the CLS token to aggregate task-relevant
/// information from the rest of the sequence.</para>
/// </remarks>
// Rank 3 only, and stated by the layer itself: ForwardTraced throws unless
// input.Shape.Length == 3, with the message "expects rank-3 [batch, seq, embedDim]". Time and Features
// match the roles SequenceTokenSliceLayer declares, which is the layer this one is documented to pair
// with - the CLS token is written at position 0 here and read back from position 0 there.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class PrependCLSTokenLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _embedDim;

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// One extra sequence position, nothing else. From <c>ForwardTraced</c>'s
    /// <c>Engine.TensorConcatenate(new[] { clsTiled, input }, axis: 1)</c> where <c>clsTiled</c> is
    /// <c>[batch, 1, embedDim]</c>: the sequence axis gains exactly one element and the other two axes
    /// are untouched.
    /// </para>
    /// <para>
    /// <c>seq + 1</c> looks like it falls outside the relation vocabulary - there is no "offset" form, and
    /// the obvious <c>Window(kernel: 1, stride: 1, padding: p)</c> only ever yields <c>seq + 2p</c>, an
    /// EVEN increment. It does not: with stride 1 the window formula reduces to
    /// <c>seq + 2*padding - dilation*(kernel-1)</c>, so <c>padding: 1, kernel: 2</c> gives
    /// <c>seq + 2 - 1 = seq + 1</c>. Spelling it that way keeps the contract exact rather than
    /// approximating a one-token prepend as no change at all.
    /// </para>
    /// <para>
    /// The feature axis is <c>Same</c>, not <c>Fixed(_embedDim)</c>. This layer does not SET the width -
    /// it requires the input to already carry <c>_embedDim</c> (it throws otherwise) and returns that same
    /// width, which is the distinction <c>Same</c> records and <c>Fixed</c> would lose.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 3) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(
                TensorAxis.Time,
                AxisRelation.Window(TensorAxis.Time, kernel: 2, stride: 1, padding: 1, dilation: 1)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Same(TensorAxis.Features)),
        };
    }

    // Trainable CLS token — shape [1, embedDim]. Held by reference so the
    // gradient tape can track parameter identity, which is why it is readonly:
    // the generated restore copies values into a readonly tensor in place and
    // rebinds a mutable one, and rebinding is what would break that identity.
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Tensor<T> _cls;

    /// <summary>Creates a CLS-token prepender for embedDim-wide inputs.</summary>
    /// <param name="embedDim">Embedding dimension (must match the input's last axis).</param>
    /// <param name="initScale">Gaussian init std. ViT / BERT both use 0.02.</param>
    /// <param name="seed">Optional RNG seed for reproducibility.</param>
    public PrependCLSTokenLayer(int embedDim, double initScale = 0.02, int? seed = null)
        : base(
            inputShape: [-1, -1, -1],
            outputShape: [-1, -1, -1],
            scalarActivation: new IdentityActivation<T>())
    {
        if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
        _embedDim = embedDim;
        _cls = new Tensor<T>(new[] { 1, embedDim });

        var rng = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
        for (int i = 0; i < embedDim; i++)
            _cls[0, i] = NumOps.FromDouble(rng.NextGaussian() * initScale);
    }

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Shape.Length != 3)
            throw new ArgumentException(
                $"PrependCLSTokenLayer expects rank-3 [batch, seq, embedDim]; got rank {input.Shape.Length}.",
                nameof(input));
        if (input.Shape[2] != _embedDim)
            throw new ArgumentException(
                $"PrependCLSTokenLayer embedDim mismatch: layer={_embedDim}, input[2]={input.Shape[2]}.",
                nameof(input));

        int batch = input.Shape[0];
        int seq = input.Shape[1];

        // Tile the [1, embedDim] CLS to [batch, 1, embedDim], then concat
        // along axis 1 (sequence) with input.
        var clsRow = Engine.Reshape(_cls, new[] { 1, 1, _embedDim });
        var clsTiled = Engine.TensorTile(clsRow, new[] { batch, 1, 1 });
        return Engine.TensorConcatenate(new[] { clsTiled, input }, axis: 1);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (ParameterGradients is null) return;
        if (ParameterGradients.Length != ParameterCount)
            throw new InvalidOperationException(
                $"PrependCLSTokenLayer.UpdateParameters: gradient buffer length " +
                $"{ParameterGradients.Length} does not match ParameterCount {ParameterCount}.");
        for (int i = 0; i < _embedDim; i++)
            _cls[0, i] = NumOps.Subtract(_cls[0, i],
                NumOps.Multiply(learningRate, ParameterGradients[i]));
    }

    /// <inheritdoc/>
    public override void ResetState() { }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["EmbedDim"] = _embedDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
