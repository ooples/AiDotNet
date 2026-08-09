using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Checks that a chain of layers agrees on tensor LAYOUT — each layer's declared output feeding the next
/// layer's declared input — and reports the first disagreement by layer index and role.
/// </summary>
/// <remarks>
/// <para>
/// WHY THIS RUNS AT RUNTIME RATHER THAN IN THE SOURCE GENERATOR. The chain does not exist at compile
/// time: <c>InitializeLayers</c> builds it through <c>LayerHelper</c>, and several layer types resolve
/// their input depth lazily on the first forward pass. Roslyn cannot execute that. So validation happens
/// where the chain actually becomes real — as it is assembled.
/// </para>
/// <para>
/// It compares DECLARATIONS, never materialized tensors, and that is what makes it work with lazy layers:
/// a layer can declare that it consumes <c>[Batch, Channels, Height, Width]</c> long before it knows how
/// many channels that will be.
/// </para>
/// <para>
/// WHAT IT BUYS. ABCNet failed 16 scaffold tests with two different messages —
/// <c>Expected input depth 1, but got 256</c> and <c>Tensor shapes must match. Got [128, 32, 97] and
/// [17, 32, 32]</c> — thrown from inside a forward pass, naming neither the layer nor the mismatch. The
/// same defect through this validator reads "layer 5 emits [Batch, Time, Features], layer 6 expects
/// [Batch, Channels, Height, Width]".
/// </para>
/// <para>
/// UNANNOTATED LAYERS ARE SKIPPED, not failed. Annotation is being rolled out incrementally, and a
/// validator that rejected every un-annotated layer would break the entire library on the first commit.
/// A skipped layer breaks the chain into independently-validated runs rather than invalidating it.
/// </para>
/// <para><b>For Beginners:</b> Neural network layers pass tensors to each other, and if one produces data
/// shaped differently from what the next one expects, the model fails somewhere deep inside with a
/// confusing message. This checks the hand-off up front and says exactly which two layers disagree.</para>
/// </remarks>
internal static class LayerContractValidator
{
    /// <summary>One layer-to-layer disagreement.</summary>
    /// <param name="ProducerIndex">Index of the layer producing the tensor.</param>
    /// <param name="ProducerName">Type name of the producing layer.</param>
    /// <param name="ConsumerIndex">Index of the layer consuming it.</param>
    /// <param name="ConsumerName">Type name of the consuming layer.</param>
    /// <param name="Message">Human-readable description of the mismatch.</param>
    public readonly record struct LayoutMismatch(
        int ProducerIndex,
        string ProducerName,
        int ConsumerIndex,
        string ConsumerName,
        string Message);

    /// <summary>
    /// Validates a layer chain and returns every disagreement found, in order. An empty result means
    /// every ANNOTATED adjacent pair agrees.
    /// </summary>
    /// <param name="layers">The chain, in execution order.</param>
    public static IReadOnlyList<LayoutMismatch> Validate<T>(IReadOnlyList<ILayer<T>> layers)
    {
        var found = new List<LayoutMismatch>();
        if (layers is null || layers.Count < 2) return found;

        // ONLY DIRECTLY ADJACENT ANNOTATED PAIRS. An unannotated layer between two annotated ones
        // BREAKS the chain here rather than being stepped over, and that correction is the difference
        // between a useful check and a noise generator.
        //
        // This used to scan forward past unannotated layers to find "the real consumer", justified as
        // letting a gap in annotation reduce coverage without inventing failures. It did the opposite.
        // Measured across every constructible model: 14 of 182 reported a conv -> dense mismatch, and
        // every one was false - CreateDefaultCNNLayers puts a MaxPoolingLayer AND a FlattenLayer between
        // them, both unannotated, and flattening is exactly the transformation that makes the hand-off
        // correct. Skipping an unannotated layer assumes it is shape-preserving, which is an assumption
        // about a layer that has told us nothing.
        //
        // Same rule as LayerGraph.ContiguousRuns, and for the same reason: a run breaks wherever the
        // next layer does not receive the previous one's output unchanged.
        for (int i = 0; i < layers.Count - 1; i++)
        {
            var producer = layers[i];
            var consumer = layers[i + 1];
            if (producer is null || consumer is null) continue;

            var outputs = OutputLayouts(producer.GetType());
            if (outputs.Count == 0) continue;

            var inputs = InputLayouts(consumer.GetType());
            if (inputs.Count == 0) continue;   // unannotated consumer: the run ends here, silently

            if (!AnyCompatible(outputs, inputs))
            {
                found.Add(new LayoutMismatch(
                    i, producer.GetType().Name,
                    i + 1, consumer.GetType().Name,
                    $"layer {i} ({producer.GetType().Name}) emits {Describe(outputs)} but layer {i + 1} "
                    + $"({consumer.GetType().Name}) expects {Describe(inputs)}"));
            }
        }
        return found;
    }

    /// <summary>
    /// Validates a chain and throws when it disagrees, naming every mismatch.
    /// </summary>
    /// <param name="layers">The chain, in execution order.</param>
    /// <param name="ownerName">Model name, for the message.</param>
    /// <exception cref="InvalidOperationException">A layout disagreement was found.</exception>
    public static void ValidateOrThrow<T>(IReadOnlyList<ILayer<T>> layers, string ownerName)
    {
        var problems = Validate(layers);
        if (problems.Count == 0) return;

        var sb = new StringBuilder();
        sb.Append(ownerName).Append(": layer layout contract violated (")
          .Append(problems.Count).AppendLine(problems.Count == 1 ? " mismatch):" : " mismatches):");
        foreach (var p in problems) sb.Append("  - ").AppendLine(p.Message);
        sb.Append("Layouts are declared with [TensorLayout]; a layer with no declaration is skipped ")
          .Append("rather than failed, so this names only pairs that both declare and disagree.");
        throw new InvalidOperationException(sb.ToString());
    }

    /// <summary>
    /// True when at least one declared output form can feed at least one declared input form.
    /// </summary>
    /// <remarks>
    /// ANY-to-ANY rather than all-to-all, because both sides legitimately declare alternatives: a layer
    /// that emits either <c>[N,F]</c> or <c>[N,T,F]</c> feeding one that accepts either is fine so long as
    /// some pairing works.
    /// </remarks>
    private static bool AnyCompatible(
        IReadOnlyList<TensorLayoutAttribute> outputs, IReadOnlyList<TensorLayoutAttribute> inputs)
    {
        foreach (var o in outputs)
            foreach (var i in inputs)
                if (Compatible(o, i)) return true;
        return false;
    }

    private static bool Compatible(TensorLayoutAttribute produced, TensorLayoutAttribute expected)
    {
        // Try every rank the produced side can present, so an optional batch axis on either side lines up
        // with the other's expectation instead of reading as a rank mismatch.
        for (int rank = 1; rank <= Math.Max(produced.Rank, expected.Rank); rank++)
        {
            var a = produced.AxesForRank(rank);
            var b = expected.AxesForRank(rank);
            if (a is null || b is null) continue;

            bool same = true;
            for (int k = 0; k < a.Length && same; k++)
            {
                // Other is a declared escape hatch, so it matches anything at that position. It still
                // participates in the RANK check above, which is the part that catches most errors.
                if (a[k] == TensorAxis.Other || b[k] == TensorAxis.Other) continue;
                if (a[k] != b[k]) same = false;
            }
            if (same) return true;
        }
        return false;
    }

    private static string Describe(IReadOnlyList<TensorLayoutAttribute> layouts)
    {
        if (layouts.Count == 1) return layouts[0].ToString();
        var parts = new string[layouts.Count];
        for (int i = 0; i < layouts.Count; i++) parts[i] = layouts[i].ToString();
        return "one of " + string.Join(" | ", parts);
    }

    /// <summary>Declared input layouts for a type, or an empty list when unannotated.</summary>
    public static IReadOnlyList<TensorLayoutAttribute> InputLayouts(Type type)
        => LayoutsFor(type, TensorLayoutDirection.Input);

    /// <summary>Declared output layouts for a type, or an empty list when unannotated.</summary>
    public static IReadOnlyList<TensorLayoutAttribute> OutputLayouts(Type type)
        => LayoutsFor(type, TensorLayoutDirection.Output);

    private static IReadOnlyList<TensorLayoutAttribute> LayoutsFor(Type type, TensorLayoutDirection dir)
    {
        var result = new List<TensorLayoutAttribute>();
        if (type is null) return result;

        // Generic layers are annotated on the open generic definition, so read through it: the closed
        // ConvolutionalLayer<double> would otherwise report no attributes at all.
        var source = type.IsGenericType && !type.IsGenericTypeDefinition
            ? type.GetGenericTypeDefinition()
            : type;

        foreach (var a in source.GetCustomAttributes<TensorLayoutAttribute>(inherit: true))
            if (a.Direction == dir) result.Add(a);

        return result;
    }
}
