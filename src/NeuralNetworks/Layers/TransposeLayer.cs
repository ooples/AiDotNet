using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Reorders the axes of the input tensor according to a fixed permutation. Zero-parameter
/// utility layer, primarily used to expose a different axis as the "last" dimension so a
/// <see cref="DenseLayer{T}"/> can operate on it (enables MLP-Mixer-style cross-axis MLPs
/// without bespoke kernels).
/// </summary>
/// <remarks>
/// <para>
/// The <paramref name="permutation"/> passed to the constructor uses logical axis indices
/// (excluding the batch axis). For rank-N inputs with a batch axis at position 0, this layer
/// keeps the batch axis at index 0 and permutes the remaining N-1 axes per
/// <paramref name="permutation"/>.
/// </para>
/// <para>
/// Common pattern (MLP-Mixer temporal mixer):
/// <code>
///   // [B, numPatches, hiddenDim] -> [B, hiddenDim, numPatches]
///   new TransposeLayer&lt;T&gt;(new[] { numPatches, hiddenDim }, new[] { 1, 0 });
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric element type.</typeparam>
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.Projection)]
[LayerProperty(
    IsTrainable = false,
    ChangesShape = true,
    TestInputShape = "1, 2, 3",
    TestConstructorArgs = "new[] { 1, 0 }")]
// Rank is _permutation.Length + 1 - one batch axis plus the permuted logical axes - and it is EXACT,
// not a minimum. OnFirstForward's guard reads ">= 1 + _permutation.Length", but ForwardTraced then
// calls Engine.TensorPermute(input, _fullPermutation) with a permutation of exactly that length, so a
// higher rank would fail there. Only the rank-3 form is declared, which is the layer's own documented
// example ("[B, numPatches, hiddenDim] -> [B, hiddenDim, numPatches]") and its TestInputShape.
//
// The roles come from that example: numPatches is the token axis (Time) and hiddenDim the feature axis.
//
// THE OUTPUT LAYOUT IS THE SWAP, because the permutation is a CONSTRUCTOR ARGUMENT and a C# attribute
// takes compile-time constants only - so one static ordering has to be chosen for a type whose whole
// job is to reorder. At rank 3 there are exactly two permutations: the swap (what TestConstructorArgs
// builds, what the class docs demonstrate, and the only one that does anything) and the identity, which
// makes this layer a no-op. OutputAxesFor below does NOT hardcode either: it reads _permutation and
// emits the real order, so an identity-permutation instance is caught by
// ShapeInference.ContractMatchesLayout as a reported disagreement rather than resolving to a wrong
// shape - inference itself never consults the output layout.
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, TensorAxis.Time,
    Direction = TensorLayoutDirection.Output,
    Note = "The permutation is a constructor argument; this declares the swap that TestConstructorArgs builds.")]
[AutoParameters]
public partial class TransposeLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Every axis is carried through unchanged - only the ORDER moves. That is
    /// <c>OnFirstForward</c>'s own statement of the output shape,
    /// <c>for (int i = 0; i &lt; rank; i++) outShape[i] = logical[_permutation[i]];</c>, and
    /// <c>ForwardTraced</c>'s <c>Engine.TensorPermute(input, _fullPermutation)</c>, which moves data
    /// without resizing anything. So every relation is <see cref="AxisRelation.Same"/>; the contract's
    /// content is which ROLE sits at which position.
    /// </para>
    /// <para>
    /// This is the pattern the addendum calls out: a permutation stored by the constructor means the
    /// output axes are the input roles REORDERED, and <c>OutputAxesFor</c> is an instance method, so it
    /// can simply index the declared input roles by <c>_permutation</c>. <c>_fullPermutation</c> is the
    /// same array with the batch axis pinned at position 0, which is why Batch leads the result here.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // TensorPermute needs the ranks to agree exactly, and only rank 3 has declared roles to permute.
        if (inputRank != _permutation.Length + 1 || inputRank != 3) return null;

        // The declared input layout with its batch axis removed - the logical axes _permutation indexes.
        var logicalRoles = new[] { TensorAxis.Time, TensorAxis.Features };

        var axes = new OutputAxisContract[inputRank];
        axes[0] = new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch));
        for (int i = 0; i < _permutation.Length; i++)
        {
            var role = logicalRoles[_permutation[i]];
            axes[i + 1] = new OutputAxisContract(role, AxisRelation.Same(role));
        }

        return axes;
    }

    private int[] _logicalInputShape;
    private readonly int[] _permutation;
    private readonly int[] _fullPermutation;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="TransposeLayer{T}"/>.
    /// </summary>
    /// <param name="inputShape">Input shape excluding the batch axis.</param>
    /// <param name="permutation">
    /// Permutation of logical axis indices (all values must be in [0, inputShape.Length) and each
    /// must appear exactly once). Axis 0 here refers to the first non-batch axis of the input.
    /// </param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="permutation"/> is not a valid permutation of
    /// <c>0..inputShape.Length - 1</c>.
    /// </exception>
    public TransposeLayer(int[] permutation)
        : base(MakeUnknown(permutation.Length), MakeUnknown(permutation.Length))
    {
        ValidatePermutation(permutation);
        _logicalInputShape = Array.Empty<int>();
        _permutation = (int[])permutation.Clone();

        // Expand to include the batch axis at position 0 for Engine.TensorPermute.
        _fullPermutation = new int[permutation.Length + 1];
        _fullPermutation[0] = 0;
        for (int i = 0; i < permutation.Length; i++)
            _fullPermutation[i + 1] = permutation[i] + 1;
    }

    private static int[] MakeUnknown(int rank)
    {
        var s = new int[rank];
        for (int i = 0; i < rank; i++) s[i] = -1;
        return s;
    }

    private static void ValidatePermutation(int[] permutation)
    {
        if (permutation is null) throw new ArgumentNullException(nameof(permutation));
        var seen = new bool[permutation.Length];
        for (int i = 0; i < permutation.Length; i++)
        {
            int p = permutation[i];
            if (p < 0 || p >= permutation.Length)
                throw new ArgumentException(
                    $"Permutation index {p} at position {i} is out of range [0, {permutation.Length}).",
                    nameof(permutation));
            if (seen[p])
                throw new ArgumentException(
                    $"Permutation index {p} appears more than once.", nameof(permutation));
            seen[p] = true;
        }
    }

    /// <summary>
    /// Resolves logical input shape on first forward and computes the output shape via the permutation.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        BindToActualInput(input);
    }

    /// <inheritdoc />
    protected override void ReconcileShapeOnlyResolution(Tensor<T> input)
    {
        // A custom model forward can feed a different sequence extent than the architecture's
        // sequential shape walk predicted. The permutation is fixed, but the extents are not.
        BindToActualInput(input);
    }

    private void BindToActualInput(Tensor<T> input)
    {
        // Treat the leading axis of input as the batch axis; logical input shape is the rest.
        var fullShape = input.Shape.ToArray();
        if (fullShape.Length < 1 + _permutation.Length)
            throw new ArgumentException(
                $"TransposeLayer expects input rank >= {1 + _permutation.Length} (batch + {_permutation.Length} logical axes); got {fullShape.Length}.",
                nameof(input));

        int rank = _permutation.Length;
        var logical = new int[rank];
        Array.Copy(fullShape, fullShape.Length - rank, logical, 0, rank);
        _logicalInputShape = logical;

        var outShape = new int[rank];
        for (int i = 0; i < rank; i++) outShape[i] = logical[_permutation[i]];
        ResolveShapes(logical, outShape);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        return Engine.TensorPermute(input, _fullPermutation);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        // Stateless apart from construction-time fields.
    }

    /// <summary>
    /// Emits the permutation alongside the base metadata so deserialization can
    /// reconstruct the layer exactly. Shape-only inference would fail on
    /// permutations that leave the output shape equal to the input shape (e.g.
    /// axis swaps of two equal-size dims) or on ambiguous cases where multiple
    /// source axes share the same extent.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["Permutation"] = string.Join(",", _permutation);
        return metadata;
    }
}
