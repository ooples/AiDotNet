using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Recovers a layer's symbolic shape relation by RUNNING it on probe shapes and fitting the simplest
/// relation that reproduces every observation.
/// </summary>
/// <remarks>
/// <para>
/// Hand-writing an <see cref="IShapeContract"/> for every layer type does not scale and is not what
/// "automatic" means: this codebase has ~199 layer types and only 6 had even the coarse relation kind
/// declared, so annotation-only coverage would sit near zero indefinitely and each new layer would start
/// uncovered. Discovery inverts that — every layer is covered the moment it exists, and a layer that
/// changes its shape behaviour is caught without anyone remembering to update a declaration.
/// </para>
/// <para>
/// PROBES MUST VARY ONE AXIS AT A TIME or the fit is ambiguous rather than wrong. If every probe has
/// Height equal to Width, <c>Same(Height)</c> and <c>Same(Width)</c> reproduce the observations equally
/// well and there is no evidence to choose between them; the fitted relation would then be right on the
/// probes and wrong on the first non-square input. <see cref="ProbeShapes"/> generates a discriminating
/// set, and an axis that stays ambiguous is reported as such instead of being guessed.
/// </para>
/// <para>
/// A FIT IS EVIDENCE, NOT PROOF. Simplest-that-explains is an inductive step: a layer could follow the
/// window formula on every probe and something else beyond them. That is exactly why discovery does not
/// replace <see cref="IShapeContract"/> — where a layer declares one, the declaration is authoritative
/// and discovery becomes an independent check on it. Where none is declared, a discovered relation is
/// strictly better than the nothing that was there before.
/// </para>
/// </remarks>
public static class ShapeRelationDiscovery
{
    /// <summary>The largest kernel, stride and padding the window search will consider.</summary>
    /// <remarks>
    /// Bounded because the search is a brute force over (kernel, stride, padding, dilation). These cover
    /// every convolution and pooling configuration in this codebase; beyond them a layer is better served
    /// declaring a contract than being guessed at.
    /// </remarks>
    private const int MaxWindowTerm = 16;

    /// <summary>What discovery concluded about one output axis.</summary>
    /// <param name="Relation">The fitted relation, or <c>null</c> when nothing fit.</param>
    /// <param name="Ambiguous">True when several DIFFERENT relations fit every observation equally.</param>
    /// <param name="Detail">Human-readable explanation, always populated on failure or ambiguity.</param>
    public readonly record struct AxisFinding(AxisRelation? Relation, bool Ambiguous, string Detail);

    /// <summary>
    /// Generates probe shapes that vary each axis independently, so fits are discriminating.
    /// </summary>
    /// <param name="baseShape">A shape the layer is known to accept.</param>
    /// <returns>The base shape plus one variant per axis.</returns>
    /// <remarks>
    /// Perturbations are deliberately UNEQUAL across axes and include a non-multiple: a probe set where
    /// every axis moves by the same amount cannot separate <c>Same(Height)</c> from <c>Same(Width)</c>,
    /// and one where every size is a multiple of the stride cannot separate the true window formula from
    /// "divide by stride" — the exact off-by-one a shape system exists to catch.
    /// </remarks>
    public static IReadOnlyList<int[]> ProbeShapes(IReadOnlyList<int> baseShape)
    {
        if (baseShape is null) throw new ArgumentNullException(nameof(baseShape));

        var probes = new List<int[]> { baseShape.ToArray() };

        // Distinct deltas per axis so no two axes ever move together.
        var deltas = new[] { 1, 2, 3, 5, 7, 11 };
        for (int axis = 0; axis < baseShape.Count; axis++)
        {
            var variant = baseShape.ToArray();
            variant[axis] += deltas[axis % deltas.Length];
            probes.Add(variant);
        }

        return probes;
    }

    /// <summary>
    /// Fits a relation for every output axis from observed (input shape, output shape) pairs.
    /// </summary>
    /// <param name="inputAxes">Axis roles of the input, in order.</param>
    /// <param name="observations">Input shape paired with the output shape the layer actually produced.</param>
    /// <param name="outputAxes">Axis roles of the output, in order.</param>
    /// <returns>One finding per output axis.</returns>
    public static IReadOnlyList<AxisFinding> Fit(
        IReadOnlyList<TensorAxis> inputAxes,
        IReadOnlyList<TensorAxis> outputAxes,
        IReadOnlyList<(int[] Input, int[] Output)> observations)
    {
        if (inputAxes is null) throw new ArgumentNullException(nameof(inputAxes));
        if (outputAxes is null) throw new ArgumentNullException(nameof(outputAxes));
        if (observations is null) throw new ArgumentNullException(nameof(observations));
        if (observations.Count == 0)
            throw new ArgumentException("Fitting needs at least one observation.", nameof(observations));

        var findings = new List<AxisFinding>(outputAxes.Count);

        for (int outPos = 0; outPos < outputAxes.Count; outPos++)
        {
            findings.Add(FitOneAxis(inputAxes, observations, outPos));
        }

        return findings;
    }

    private static AxisFinding FitOneAxis(
        IReadOnlyList<TensorAxis> inputAxes,
        IReadOnlyList<(int[] Input, int[] Output)> observations,
        int outPos)
    {
        var usable = observations
            .Where(o => o.Output is not null && outPos < o.Output.Length)
            .ToList();

        if (usable.Count == 0)
            return new AxisFinding(null, false, $"no observation produced an axis at position {outPos}");

        var outputs = usable.Select(o => o.Output[outPos]).ToList();

        // Candidates are collected in increasing complexity and ALL of them are kept, because finding
        // two that fit is the ambiguity signal — stopping at the first would hide it.
        var candidates = new List<AxisRelation>();

        // Same: this axis simply copies an input axis.
        for (int a = 0; a < inputAxes.Count; a++)
        {
            int axisIndex = a;
            if (usable.All(o => axisIndex < o.Input.Length && o.Input[axisIndex] == o.Output[outPos]))
            {
                candidates.Add(AxisRelation.Same(inputAxes[axisIndex]));
            }
        }

        if (candidates.Count > 0) return Resolve(candidates, inputAxes, "same-as-input");

        // Fixed: constant across probes. Only meaningful when the probes actually VARIED - a single
        // observation makes every axis look fixed, which is a statement about the probe set, not the layer.
        if (outputs.Distinct().Count() == 1 && usable.Count > 1 && InputsVaried(usable))
        {
            candidates.Add(AxisRelation.Fixed(outputs[0]));
            return Resolve(candidates, inputAxes, "constant");
        }

        // Scaled: a small rational multiple of one input axis.
        for (int a = 0; a < inputAxes.Count; a++)
        {
            int axisIndex = a;
            var first = usable[0];
            if (axisIndex >= first.Input.Length) continue;
            int from = first.Input[axisIndex];
            int to = first.Output[outPos];
            if (from <= 0 || to <= 0) continue;

            int g = Gcd(to, from);
            int num = to / g;
            int den = from / g;
            if (num > MaxWindowTerm || den > MaxWindowTerm) continue;

            bool fitsAll = usable.All(o =>
                axisIndex < o.Input.Length
                && (long)o.Input[axisIndex] * num % den == 0
                && (long)o.Input[axisIndex] * num / den == o.Output[outPos]);

            if (fitsAll) candidates.Add(AxisRelation.Scaled(inputAxes[axisIndex], num, den));
        }

        if (candidates.Count > 0) return Resolve(candidates, inputAxes, "scaled");

        // Window: the sliding-window formula. Brute force over small terms.
        for (int a = 0; a < inputAxes.Count; a++)
        {
            int axisIndex = a;
            for (int kernel = 1; kernel <= MaxWindowTerm; kernel++)
            for (int stride = 1; stride <= MaxWindowTerm; stride++)
            for (int padding = 0; padding <= MaxWindowTerm; padding++)
            {
                int k = kernel, s = stride, p = padding;
                bool fitsAll = usable.All(o =>
                {
                    if (axisIndex >= o.Input.Length) return false;
                    long effective = k - 1L + 1L;   // dilation 1; dilated forms are declared, not guessed
                    long numerator = o.Input[axisIndex] + (2L * p) - effective;
                    if (numerator < 0) return false;
                    return (numerator / s) + 1 == o.Output[outPos];
                });

                if (fitsAll)
                {
                    candidates.Add(AxisRelation.Window(inputAxes[axisIndex], k, s, p));
                    // One canonical (k,s,p) per axis is enough; padding and kernel trade off against each
                    // other for a fixed output, and enumerating every equivalent triple would report
                    // ambiguity where there is only one behaviour.
                    goto nextAxis;
                }
            }
            nextAxis: ;
        }

        if (candidates.Count > 0) return Resolve(candidates, inputAxes, "window");

        return new AxisFinding(
            null, false,
            $"no Same/Fixed/Scaled/Window relation reproduces the observed sizes "
            + $"[{string.Join(", ", outputs)}] at output position {outPos}");
    }

    private static AxisFinding Resolve(
        List<AxisRelation> candidates, IReadOnlyList<TensorAxis> inputAxes, string family)
    {
        if (candidates.Count == 1) return new AxisFinding(candidates[0], false, family);

        // Several relations of the same family fit. That is a PROBE weakness, not a layer property: two
        // input axes held equal across every probe are indistinguishable by construction.
        var rendered = string.Join(" or ", candidates.Select(c => c.ToString()));
        return new AxisFinding(
            candidates[0], true,
            $"ambiguous ({family}): {rendered} all reproduce every observation, so the probes do not "
            + "separate them. Vary those input axes independently to decide.");
    }

    private static bool InputsVaried(IReadOnlyList<(int[] Input, int[] Output)> observations)
    {
        var first = observations[0].Input;
        return observations.Any(o => !o.Input.SequenceEqual(first));
    }

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return a < 0 ? -a : a;
    }
}
