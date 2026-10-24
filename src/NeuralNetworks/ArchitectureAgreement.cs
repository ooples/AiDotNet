using System.Linq;

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
    /// The feature count an architecture states for a [batch, time steps, features] input: the width of a grid stated
    /// as a grid, otherwise the flat <c>InputSize</c>.
    /// </summary>
    /// <remarks>
    /// A forecasting architecture of 96 x 7 has an <c>InputSize</c> of 672, the whole window; its features are the 7
    /// columns. A grid of one of the two forms the architecture infers from a flat size (1 x n, or square) carries no
    /// split, so its flat size is the feature count.
    /// </remarks>
    internal static int SequenceFeatureCount<T>(NeuralNetworkArchitecture<T> architecture)
    {
        if (architecture is null) throw new ArgumentNullException(nameof(architecture));
        int height = architecture.InputHeight, width = architecture.InputWidth;
        bool statedGrid = height > 0 && width > 0 && height != 1 && height != width;
        return statedGrid ? width : architecture.InputSize;
    }

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
        int depth = Math.Max(1, architecture.InputDepth);
        bool grid = height > 0 && width > 0;
        // A grid with more than one channel holds depth times the window's values, so it never matches exactly.
        if (grid && depth == 1 && height == sequenceLength && width == numFeatures) return;
        bool inferredGrid = grid && (height == 1 || height == width);
        if (grid && (!inferredGrid || depth > 1))
            throw new ArgumentException(
                $"{model}: the architecture describes a {height} x {width}{(depth > 1 ? $" x {depth}" : "")} (time steps x features{(depth > 1 ? " x channels" : "")}) input, but the model " +
                $"was built for {sequenceLength} x {numFeatures} from its options. State the options' sequence length and " +
                "feature count in the architecture, or leave its input size unset.", nameof(architecture));

        long stated = grid ? (long)height * width * depth : architecture.InputSize;
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

    /// <summary>
    /// Requires a size the architecture states (<c>InputSize</c> or <c>OutputSize</c>) to be one reading of a tensor
    /// the model actually took or produced.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A model that sizes its layers from its options takes and produces tensors of its options' sizes, so the tensor
    /// is the authority and no per-model guess at what a flat size means is needed. The readings are each axis after
    /// the batch axis, the per-sample count, and the whole count: a forecast of [batch, horizon, features] agrees with
    /// an architecture stating the horizon, the feature count or horizon x features. Any other stated size describes
    /// a different model, which used to be built from the options without a word (#2147).
    /// </para>
    /// <para>A size the architecture leaves at zero is not stated and is never refused.</para>
    /// </remarks>
    /// <param name="model">The model's name, for the message.</param>
    /// <param name="what">"InputSize" or "OutputSize".</param>
    /// <param name="stated">The architecture's value.</param>
    /// <param name="shape">The shape of the tensor the model took or produced.</param>
    /// <exception cref="ArgumentException">The stated size is none of the tensor's readings.</exception>
    internal static void RequireStatedSizeObserved(string model, string what, int stated, IReadOnlyList<int> shape)
    {
        if (stated <= 0 || shape is null || shape.Count == 0) return;
        var readings = Readings(shape);
        if (readings.Contains(stated)) return;
        bool output = what == nameof(NeuralNetworkArchitecture<double>.OutputSize);
        throw new ArgumentException(
            $"{model}: the architecture states an {what} of {stated}, but the model sizes its layers from its options and " +
            $"{(output ? "produces" : "takes")} [{string.Join(", ", shape)}]. State one of " +
            $"{string.Join(", ", readings.OrderBy(r => r))} in the architecture, or leave its {what} unset.", "architecture");
    }

    /// <summary>
    /// Requires an architecture's <c>InputSize</c> to be one of the sizes a model sized from its options declares.
    /// </summary>
    /// <param name="model">The model's name, for the message.</param>
    /// <param name="stated">The architecture's <c>InputSize</c>; zero is not stated and is never refused.</param>
    /// <param name="declared">The option values the model accepts as an input size.</param>
    /// <exception cref="ArgumentException">The stated size is none of the declared ones.</exception>
    internal static void RequireStatedSizeDeclared(string model, int stated, IReadOnlyCollection<int> declared)
    {
        if (stated <= 0 || declared is null || declared.Count == 0 || declared.Contains(stated)) return;
        throw new ArgumentException(
            $"{model}: the architecture states an InputSize of {stated}, but the model sizes its layers from its options, " +
            $"which give {string.Join(", ", declared.Distinct().OrderBy(d => d))}. State one of those in the architecture, " +
            "or set the options to the size you mean.", "architecture");
    }

    private static HashSet<long> Readings(IReadOnlyList<int> shape)
    {
        var readings = new HashSet<long>();
        long whole = 1, perSample = 1;
        for (int axis = 0; axis < shape.Count; axis++)
        {
            whole *= shape[axis];
            if (axis == 0 && shape.Count > 1) continue;
            readings.Add(shape[axis]);
            perSample *= shape[axis];
        }
        readings.Add(perSample);
        readings.Add(whole);
        return readings;
    }
}
