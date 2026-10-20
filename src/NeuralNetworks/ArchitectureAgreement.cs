namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Checks that a <see cref="NeuralNetworkArchitecture{T}"/> agrees with the sizes a model built its layers from (#2147).
/// </summary>
/// <remarks>
/// Most models size their layers from options; the architecture repeats some of those sizes. Where the two disagree,
/// the model used to build from its options and ignore the architecture without a word. These checks make that a
/// failure that names both values. A size the architecture leaves at zero is "not stated" and is never refused.
/// </remarks>
internal static class ArchitectureAgreement
{
    /// <summary>
    /// Requires an architecture's input description to match a [batch, <paramref name="sequenceLength"/>,
    /// <paramref name="numFeatures"/>] model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// An architecture describes such an input as a grid (<c>InputHeight</c> time steps by <c>InputWidth</c> features,
    /// as the financial models document) or as a flat <c>InputSize</c>, and it converts one into the other itself: a
    /// two-dimensional architecture given only 96 records a 1 x 96 grid, and one given 784 records 28 x 28. So a grid
    /// that matches exactly agrees. A grid of one of those two inferred forms (1 x n, or square) is read as the flat
    /// size it came from, which agrees when it is one of the three readings callers use - the feature count, the
    /// sequence length (the generated forecasting fixtures pass the context length) or the whole window. Any other
    /// grid was stated as a grid, so it must match both axes: 48 x 14 holds as many values as 96 x 7 and is still a
    /// different input.
    /// </para>
    /// <para>
    /// An architecture whose spatial size is unstated (zero, or the lazy -1 of a three-dimensional input) is never
    /// refused.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">The architecture states an input size the model does not have.</exception>
    internal static void RequireSequenceInput<T>(NeuralNetworkArchitecture<T> architecture, string model,
        int sequenceLength, int numFeatures)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        int height = architecture.InputHeight, width = architecture.InputWidth;
        bool grid = height > 0 && width > 0;
        if (grid && height == sequenceLength && width == numFeatures) return;
        bool inferredGrid = grid && (height == 1 || height == width);
        if (grid && !inferredGrid)
            throw new ArgumentException(
                $"{model}: the architecture describes a {height} x {width} (time steps x features) input, but the model " +
                $"was built for {sequenceLength} x {numFeatures} from its options. State the options' sequence length and " +
                "feature count in the architecture, or leave its input size unset.", nameof(architecture));

        long stated = grid ? (long)height * width * Math.Max(1, architecture.InputDepth) : architecture.InputSize;
        if (stated <= 0) return;
        long window = (long)sequenceLength * numFeatures;
        if (stated == numFeatures || stated == sequenceLength || stated == window) return;

        string described = grid
            ? $"a {height} x {width} (time steps x features) input"
            : $"an InputSize of {stated}";
        throw new ArgumentException(
            $"{model}: the architecture describes {described}, but the model was built for {sequenceLength} x " +
            $"{numFeatures} (time steps x features, {window} values) from its options. State the options' sequence " +
            "length and feature count in the architecture, or leave its input size unset.", nameof(architecture));
    }
}
