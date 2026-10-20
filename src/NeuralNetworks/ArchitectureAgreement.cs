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
    /// An architecture describes such an input either as a grid (<c>InputHeight</c> time steps by <c>InputWidth</c>
    /// features, as the financial models document) or as a flat <c>InputSize</c>. A flat size cannot say how it
    /// splits across time and features, so it agrees when it equals either the feature count or the whole window.
    /// </remarks>
    /// <exception cref="ArgumentException">The architecture states an input size the model does not have.</exception>
    internal static void RequireSequenceInput<T>(NeuralNetworkArchitecture<T> architecture, string model,
        int sequenceLength, int numFeatures)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        if (architecture.InputHeight > 0 && architecture.InputWidth > 0)
        {
            if (architecture.InputHeight != sequenceLength || architecture.InputWidth != numFeatures)
                throw new ArgumentException(
                    $"{model}: the architecture describes a {architecture.InputHeight} x {architecture.InputWidth} " +
                    $"(time steps x features) input, but the model was built for {sequenceLength} x {numFeatures} " +
                    "from its options. Make the architecture's InputHeight and InputWidth match the options' sequence " +
                    "length and feature count, or leave them unset.", nameof(architecture));
            return;
        }

        int flat = architecture.InputSize;
        if (flat > 0 && flat != numFeatures && (long)flat != (long)sequenceLength * numFeatures)
            throw new ArgumentException(
                $"{model}: the architecture's InputSize is {flat}, but the model was built for {numFeatures} features " +
                $"over {sequenceLength} time steps ({(long)sequenceLength * numFeatures} values) from its options. " +
                "Make InputSize the feature count or the whole window, or leave it unset.", nameof(architecture));
    }
}
