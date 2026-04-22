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
    TestConstructorArgs = "new[] { 2, 3 }, new[] { 1, 0 }")]
public class TransposeLayer<T> : LayerBase<T>
{
    private readonly int[] _logicalInputShape;
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
    public TransposeLayer(int[] inputShape, int[] permutation)
        : base(inputShape, ComputeOutputShape(inputShape, permutation))
    {
        _logicalInputShape = (int[])inputShape.Clone();
        _permutation = (int[])permutation.Clone();

        // Expand to include the batch axis at position 0 for Engine.TensorPermute.
        _fullPermutation = new int[permutation.Length + 1];
        _fullPermutation[0] = 0;
        for (int i = 0; i < permutation.Length; i++)
            _fullPermutation[i + 1] = permutation[i] + 1;
    }

    private static int[] ComputeOutputShape(int[] inputShape, int[] permutation)
    {
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));
        if (permutation is null) throw new ArgumentNullException(nameof(permutation));
        if (permutation.Length != inputShape.Length)
            throw new ArgumentException(
                $"Permutation length ({permutation.Length}) must match inputShape length ({inputShape.Length}).",
                nameof(permutation));

        var seen = new bool[inputShape.Length];
        for (int i = 0; i < permutation.Length; i++)
        {
            int p = permutation[i];
            if (p < 0 || p >= inputShape.Length)
                throw new ArgumentException(
                    $"Permutation index {p} at position {i} is out of range [0, {inputShape.Length}).",
                    nameof(permutation));
            if (seen[p])
                throw new ArgumentException(
                    $"Permutation index {p} appears more than once.",
                    nameof(permutation));
            seen[p] = true;
        }

        var outputShape = new int[inputShape.Length];
        for (int i = 0; i < permutation.Length; i++)
            outputShape[i] = inputShape[permutation[i]];
        return outputShape;
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        return Engine.TensorPermute(input, _fullPermutation);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // No trainable parameters.
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return Vector<T>.Empty();
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
