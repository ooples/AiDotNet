using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Derives a model's shape relation by PROBING it - running Predict over several input shapes and
/// several CONSTRUCTION profiles, then fitting the simplest relation that reproduces every observation.
/// </summary>
/// <remarks>
/// <para>
/// WHY THE PROBE IS THE GROUND TRUTH. Models do not declare shape information - none of the 1825 do -
/// and 902 override PredictCore, so a model's output is not its last layer's output. There is nothing
/// to read; the shape law has to be OBSERVED. Predict is the only authority, so it is what this asks,
/// with zero annotation.
/// </para>
/// <para>
/// WHY IT VARIES CONSTRUCTION, NOT JUST BATCH. An output axis that never moves is only <i>Fixed</i> if
/// something that could have moved it actually moved. Probing one model at one size makes every
/// interior axis look constant - which is how the layer sweep produced 50 false <c>Fixed(4)</c>s and a
/// MaxPooling layer declared shape-preserving, until multiple construction profiles showed the
/// constants tracking a constructor argument.
/// </para>
/// <para>
/// WHY PROFILES ARE COMPARED ONLY WITHIN A FAMILY. Two profiles are a controlled experiment only when
/// they differ in ONE thing. A model built <c>OneDimensional</c> and the same model built
/// <c>ThreeDimensional</c> are different models with different layer stacks, so a structural
/// difference between them is a category error, not a finding. Profiles are therefore grouped by input
/// type and reconciled within the group, where the only variable is extent.
/// </para>
/// <para>
/// AND WHY THE FIT IS STRUCTURED, NOT A STRING. An earlier revision formatted each fit to text and
/// re-parsed it to verify; <c>Scaled(in[0], 1/2)</c> contains the separator the parser split on, so a
/// correct fit was reported as a 6-axis fit against a 4-axis observation. The self-check caught it, and
/// the response is to delete the round-trip rather than repair the parser - the fit is a value, and
/// text is only ever produced for display.
/// </para>
/// <para>
/// The self-check itself stays: every fitted relation is re-evaluated against the observations it came
/// from, and a relation that cannot reproduce them fails the test as a defect in the METHOD. That is
/// the direct lesson from the chain-fold measurement deleted earlier, which was never shown correct on
/// a case with a known answer and drove five changes and five reverts.
/// </para>
/// </remarks>
public class ModelShapeDiscoveryProbeTests
{
    private readonly ITestOutputHelper _out;
    public ModelShapeDiscoveryProbeTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// How many models to probe. Bounded on purpose - this proves the METHOD, not the inventory.
    /// Measured cost is ~14 s per model across all its profiles, so the full tensor-to-tensor
    /// inventory is a multi-hour job and needs its own lane rather than a bigger number here.
    /// </summary>
    private const int ModelBudget = 40;

    /// <summary>
    /// Optional window over the candidate list, for running the probe in passes.
    /// </summary>
    /// <remarks>
    /// The DEFAULTS are the real configuration and are what CI runs - these only let a developer split
    /// one long pass into several short ones on a machine that cannot hold a ten-minute run open. They
    /// never reduce what a default run covers.
    /// </remarks>
    private static int EnvInt(string name, int fallback, int minimum) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= minimum ? v : fallback;

    /// <summary>Cap on any probed axis, mirroring NeuralNetworkModelTestBase.MaxFreeAxisExtent.</summary>
    private const int BaseAxisExtent = 8;

    /// <summary>Second extent used to move a free axis, so a constant output axis can be told from a tracked one.</summary>
    private const int AltAxisExtent = 12;

    private enum FitKind { Fixed, Same, Scaled }

    /// <summary>One output axis's fitted law. A value, never re-parsed from text.</summary>
    private readonly record struct AxisFit(FitKind Kind, int Value, int InAxis, int Numerator, int Denominator)
    {
        public static AxisFit FixedAt(int value) => new(FitKind.Fixed, value, -1, 1, 1);
        public static AxisFit SameAs(int inAxis) => new(FitKind.Same, 0, inAxis, 1, 1);
        public static AxisFit ScaledFrom(int inAxis, int numerator, int denominator)
            => new(FitKind.Scaled, 0, inAxis, numerator, denominator);

        /// <summary>The axis size this law predicts, or null if it cannot apply to that input.</summary>
        public int? Evaluate(int[] inputShape)
        {
            if (Kind == FitKind.Fixed) return Value;
            if (InAxis < 0 || InAxis >= inputShape.Length) return null;
            return Kind == FitKind.Same
                ? inputShape[InAxis]
                : inputShape[InAxis] * Numerator / Denominator;
        }

        /// <summary>The law with its constant erased, for telling "parameterised" from "different".</summary>
        public AxisFit WithoutConstant() => Kind == FitKind.Fixed ? this with { Value = 0 } : this;

        public override string ToString() => Kind switch
        {
            FitKind.Fixed => $"Fixed({Value})",
            FitKind.Same => $"Same(in[{InAxis}])",
            _ => Denominator == 1
                ? $"Scaled(in[{InAxis}], {Numerator})"
                : $"Scaled(in[{InAxis}], 1/{Denominator})",
        };
    }

    /// <summary>A fitted relation, or a stated reason for declining to fit one.</summary>
    private sealed record Relation(IReadOnlyList<AxisFit>? Axes, string? Declined)
    {
        public override string ToString() => Axes is null
            ? $"DECLINED({Declined})"
            : "[" + string.Join(", ", Axes) + "]";
    }

    /// <summary>
    /// Construction profiles. Two per input type, differing ONLY in extent, so any output constant that
    /// moves between them is a constructor-parameterised value rather than a fixed one.
    /// </summary>
    private static IReadOnlyList<(string Family, string Name, Func<NeuralNetworkArchitecture<double>> Build)> Profiles =>
    [
        ("1D", "1D-8", () => new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: BaseAxisExtent, outputSize: 4)),
        ("1D", "1D-12", () => new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: AltAxisExtent, outputSize: 4)),
        ("3D", "3D-3x8x8", () => new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: BaseAxisExtent, inputWidth: BaseAxisExtent, outputSize: 4)),
        ("3D", "3D-3x12x12", () => new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: AltAxisExtent, inputWidth: AltAxisExtent, outputSize: 4)),
    ];

    [Trait("Category", "Sweep")]
    [Fact(Timeout = 1800000)]
    public async Task ProbingAModelYieldsAShapeRelationThatReproducesWhatPredictReturned()
    {
        await Task.Yield();

        var candidates = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromNeuralNetworkBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        int budget = EnvInt("ADNSHAPE_PROBE_BUDGET", ModelBudget, 1);
        int offset = EnvInt("ADNSHAPE_PROBE_OFFSET", 0, 0);
        if (offset > 0) candidates = candidates.Skip(offset).ToList();
        _out.WriteLine($"candidates={candidates.Count}  budget={budget}  offset={offset}");

        int probed = 0, reproduced = 0, confirmed = 0, parameterised = 0, ambiguous = 0, unconfirmed = 0;
        int reachedByArchitectureCtor = 0;
        var failures = new List<string>();
        var skipped = new List<string>();
        var selfInconsistent = new List<string>();

        foreach (var open in candidates)
        {
            if (probed >= budget) break;

            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            // Fit ONCE PER CONSTRUCTION PROFILE. Pooling profiles would hide the very thing the
            // profiles exist to expose - a constant that moves with a constructor argument.
            var fits = new List<(string Family, string Profile, Relation Relation, List<(int[] In, int[] Out)> Obs)>();
            string? skipReason = null;
            bool usedArchitectureCtor = false;

            foreach (var (family, profileName, buildArch) in ConstructionAttemptsFor(closed))
            {
                object? model;
                try { model = Construct(closed, buildArch); }
                catch (Exception ex)
                {
                    skipReason ??= $"{Unwrap(ex).GetType().Name} constructing";
                    continue;
                }
                if (model is null) continue;
                if (buildArch is not null) usedArchitectureCtor = true;

                try
                {
                    int[]? perSampleInput = TryArchitectureInputShape(model);
                    if (perSampleInput is null || perSampleInput.Length == 0 || perSampleInput.Any(d => d <= 0))
                    {
                        skipReason ??= "no concrete declared input shape to probe from";
                        continue;
                    }

                    var (observations, predictFailure) = ProbeModel(model, perSampleInput);
                    if (observations.Count < 2)
                    {
                        // Name the actual Predict failure. "fewer than 2 successful probes" says
                        // nothing about WHY, and a skip reason that cannot be acted on is a
                        // limitation being recorded rather than closed.
                        skipReason ??= predictFailure is null
                            ? "fewer than 2 successful probes (Predict returned null)"
                            : $"Predict failed: {predictFailure}";
                        continue;
                    }

                    fits.Add((family, profileName, FitRelation(observations), observations));
                }
                finally { (model as IDisposable)?.Dispose(); }
            }

            if (fits.Count == 0)
            {
                string reason = skipReason ?? "no construction profile applied";

                // A RANK complaint is a finding, not a harness limitation. Probing clamps EXTENTS but
                // never changes RANK, so "requires rank-3, got rank 2" means the model's own declared
                // input rank is rejected by its own layers - it cannot consume the shape it advertises.
                // Reported separately and counted; not asserted on yet, per the report-then-enforce
                // ladder that worked for the layer contracts.
                if (reason.Contains("rank", StringComparison.OrdinalIgnoreCase))
                    selfInconsistent.Add($"{open.Name}: {reason}");
                else
                    skipped.Add($"{open.Name}: {reason}");
                continue;
            }

            probed++;
            if (usedArchitectureCtor) reachedByArchitectureCtor++;

            string verdict = Reconcile(fits.Select(f => (f.Family, f.Relation)).ToList());
            if (verdict.StartsWith("CONFIRMED", StringComparison.Ordinal)) confirmed++;
            else if (verdict.StartsWith("PARAMETERISED", StringComparison.Ordinal)) parameterised++;
            else if (verdict.StartsWith("AMBIGUOUS", StringComparison.Ordinal)) ambiguous++;
            else unconfirmed++;

            _out.WriteLine($"{open.Name}   {verdict}");
            foreach (var (_, profileName, relation, obs) in fits)
            {
                _out.WriteLine($"  [{profileName}] {relation}");
                foreach (var (inS, outS) in obs)
                {
                    _out.WriteLine($"       [{string.Join(",", inS)}]  ->  [{string.Join(",", outS)}]");
                }

                // SELF-CHECK: re-evaluate the fit against the observations it came from. A fit that
                // cannot reproduce its own inputs is a broken method, and saying so is the point.
                string? mismatch = CheckFitReproducesObservations(relation, obs);
                if (mismatch is null) { reproduced++; }
                else { failures.Add($"{open.Name} [{profileName}]: {mismatch}"); }
            }
        }

        _out.WriteLine("");
        _out.WriteLine($"probed={probed}  confirmed={confirmed}  parameterised={parameterised}  "
            + $"ambiguous={ambiguous}  unconfirmed (single profile)={unconfirmed}");
        _out.WriteLine($"reached only via the architecture ctor={reachedByArchitectureCtor}  "
            + $"fits that reproduced their observations={reproduced}  failed={failures.Count}  "
            + $"skipped={skipped.Count}  self-inconsistent (declared rank rejected by own layers)="
            + $"{selfInconsistent.Count}");
        foreach (var s in selfInconsistent) _out.WriteLine($"  SELF-INCONSISTENT: {s}");
        foreach (var s in skipped.Take(25)) _out.WriteLine($"  skipped: {s}");
        foreach (var f in failures) _out.WriteLine($"  FAILED: {f}");

        Assert.True(probed > 0, "no model was probed - the harness, not the method, is broken");

        // Assert the EXERCISED count, not just green. Without this, a harness whose profiles all
        // silently failed to construct would still pass while proving nothing - the vacuous-sweep
        // failure mode that hid 13 dead [ElementWiseShape] contracts until the layer sweep printed
        // its own exercised counts.
        Assert.True(
            confirmed + parameterised + ambiguous > 0,
            "no model was probed under more than one construction profile within a single input-type "
            + "family, so every 'Fixed' here is unfalsified. The multi-profile machinery is not running.");

        Assert.True(
            failures.Count == 0,
            $"{failures.Count} fitted relation(s) do not reproduce the observations they were fitted "
            + "from. That is a defect in the FITTING METHOD, not in the models."
            + Environment.NewLine + string.Join(Environment.NewLine, failures));
    }

    /// <summary>
    /// The construction attempts for a model: its parameterless ctor if it has one, otherwise the
    /// architecture ctor once per profile.
    /// </summary>
    /// <remarks>
    /// A parameterless model builds its own architecture, so profiles cannot move it - its axis
    /// variation has to come from the input side alone, and its fit is reported UNCONFIRMED.
    /// </remarks>
    private static IEnumerable<(string Family, string Name, Func<NeuralNetworkArchitecture<double>>? Build)>
        ConstructionAttemptsFor(Type closed)
    {
        if (closed.GetConstructor(Type.EmptyTypes) is not null)
        {
            yield return ("default", "default-ctor", null);
            yield break;
        }

        if (FindArchitectureCtor(closed) is null) yield break;

        foreach (var (family, name, build) in Profiles) yield return (family, name, build);
    }

    private static ConstructorInfo? FindArchitectureCtor(Type closed) =>
        closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });

    private static object? Construct(Type closed, Func<NeuralNetworkArchitecture<double>>? buildArch)
    {
        if (buildArch is null) return Activator.CreateInstance(closed);

        var ctor = FindArchitectureCtor(closed);
        if (ctor is null) return null;

        var ps = ctor.GetParameters();
        var args = new object?[ps.Length];
        args[0] = buildArch();
        for (int i = 1; i < ps.Length; i++) args[i] = ps[i].DefaultValue;
        return ctor.Invoke(args);
    }

    /// <summary>
    /// Probes a constructed model: batch sizes 1/2/3 at its declared shape, then each per-sample axis
    /// moved on its own. Also returns the first Predict failure, so a skip can name its cause.
    /// </summary>
    /// <remarks>
    /// Moving an axis a weight is sized against makes Predict throw, and that probe is simply dropped -
    /// so "try it and keep what worked" needs no guess about which axes are free. Dropping a probe
    /// loses information; it never invents any.
    /// </remarks>
    private static (List<(int[] In, int[] Out)> Observations, string? FirstFailure) ProbeModel(
        object model, int[] perSampleInput)
    {
        var observations = new List<(int[] In, int[] Out)>();
        string? firstFailure = null;

        int[] BaseShape(int batch)
        {
            var shape = new int[perSampleInput.Length + 1];
            shape[0] = batch;
            for (int i = 0; i < perSampleInput.Length; i++)
            {
                shape[i + 1] = Math.Min(perSampleInput[i], BaseAxisExtent);
            }
            return shape;
        }

        void Probe(int[] shape)
        {
            var (outShape, failure) = TryPredict(model, shape);
            if (outShape is not null) observations.Add((shape, outShape));
            else firstFailure ??= failure;
        }

        foreach (int batch in new[] { 1, 2, 3 }) Probe(BaseShape(batch));

        // Move ONE per-sample axis at a time. This is what makes a Fixed output axis falsifiable:
        // without it every interior axis is constant simply because nothing varied.
        for (int axis = 0; axis < perSampleInput.Length; axis++)
        {
            var shape = BaseShape(1);
            if (shape[axis + 1] >= AltAxisExtent) continue;
            shape[axis + 1] = AltAxisExtent;
            Probe(shape);
        }

        return (observations, firstFailure);
    }

    /// <summary>Fits the simplest relation over the observed (input, output) shape pairs.</summary>
    private static Relation FitRelation(List<(int[] In, int[] Out)> obs)
    {
        int outRank = obs[0].Out.Length;
        if (obs.Any(o => o.Out.Length != outRank))
            return new Relation(null, "output rank varies with input");

        var axes = new AxisFit[outRank];
        for (int axis = 0; axis < outRank; axis++)
        {
            var values = obs.Select(o => o.Out[axis]).ToArray();

            if (values.Distinct().Count() == 1) { axes[axis] = AxisFit.FixedAt(values[0]); continue; }

            AxisFit? found = null;
            int inRank = obs[0].In.Length;

            for (int inAxis = 0; inAxis < inRank && found is null; inAxis++)
            {
                if (obs.All(o => inAxis < o.In.Length && o.Out[axis] == o.In[inAxis]))
                {
                    found = AxisFit.SameAs(inAxis);
                    break;
                }

                // A small integer ratio is still an exact relation, not curve fitting: 1/d covers
                // striding and pooling, n covers upsampling.
                for (int d = 2; d <= 8 && found is null; d++)
                {
                    if (obs.All(o => inAxis < o.In.Length && o.Out[axis] == o.In[inAxis] / d))
                        found = AxisFit.ScaledFrom(inAxis, 1, d);
                }
                for (int n = 2; n <= 8 && found is null; n++)
                {
                    if (obs.All(o => inAxis < o.In.Length && o.Out[axis] == o.In[inAxis] * n))
                        found = AxisFit.ScaledFrom(inAxis, n, 1);
                }
            }

            if (found is null) return new Relation(null, $"axis {axis} varies with no matching input axis");
            axes[axis] = found.Value;
        }

        return new Relation(axes, null);
    }

    /// <summary>
    /// Compares the per-profile fits WITHIN each input-type family and says what the evidence supports.
    /// </summary>
    /// <remarks>
    /// Only a same-family pair is a controlled comparison - see the class remarks. A family with one
    /// usable fit proves nothing about constants and is reported as such rather than counted.
    /// </remarks>
    private static string Reconcile(List<(string Family, Relation Relation)> fits)
    {
        var verdicts = new List<string>();

        foreach (var group in fits.GroupBy(f => f.Family))
        {
            var fitted = group.Where(g => g.Relation.Axes is not null).Select(g => g.Relation.Axes!).ToList();
            if (fitted.Count < 2) continue;

            bool identical = fitted.All(a => a.SequenceEqual(fitted[0]));
            if (identical) { verdicts.Add($"CONFIRMED in {group.Key} across {fitted.Count} profiles"); continue; }

            bool sameStructure = fitted.All(a =>
                a.Count == fitted[0].Count
                && a.Select(f => f.WithoutConstant()).SequenceEqual(fitted[0].Select(f => f.WithoutConstant())));

            verdicts.Add(sameStructure
                ? $"PARAMETERISED in {group.Key} (structure holds, constants move with construction)"
                : $"AMBIGUOUS in {group.Key} (structure differs at the same input type)");
        }

        if (verdicts.Count == 0)
            return "UNCONFIRMED (no input-type family produced two comparable fits - constants unfalsified)";

        // Weakest evidence wins: one ambiguous family is a real finding even if another confirmed.
        if (verdicts.Any(v => v.StartsWith("AMBIGUOUS", StringComparison.Ordinal)))
            return string.Join("; ", verdicts.Where(v => v.StartsWith("AMBIGUOUS", StringComparison.Ordinal)));
        if (verdicts.Any(v => v.StartsWith("PARAMETERISED", StringComparison.Ordinal)))
            return string.Join("; ", verdicts.Where(v => v.StartsWith("PARAMETERISED", StringComparison.Ordinal)));
        return string.Join("; ", verdicts);
    }

    /// <summary>Re-evaluates a fitted relation against the observations it was fitted from.</summary>
    private static string? CheckFitReproducesObservations(Relation relation, List<(int[] In, int[] Out)> obs)
    {
        if (relation.Axes is null) return null;   // declining is honest

        var axes = relation.Axes;
        foreach (var (inS, outS) in obs)
        {
            if (axes.Count != outS.Length)
                return $"fit has {axes.Count} axes, observation has {outS.Length}";

            for (int axis = 0; axis < axes.Count; axis++)
            {
                int? expected = axes[axis].Evaluate(inS);
                if (expected is null)
                    return $"axis {axis}: fit {axes[axis]} cannot apply to input [{string.Join(",", inS)}]";

                if (expected.Value != outS[axis])
                {
                    return $"axis {axis}: fit {axes[axis]} says {expected.Value}, observed {outS[axis]} "
                        + $"for in [{string.Join(",", inS)}]";
                }
            }
        }

        return null;
    }

    private static Exception Unwrap(Exception ex) =>
        ex is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : ex;

    private static int[]? TryArchitectureInputShape(object model)
    {
        try
        {
            dynamic arch = ((dynamic)model).GetArchitecture();
            int[] shape = arch.GetInputShape();
            return shape;
        }
        catch { return null; }
    }

    private static (int[]? Shape, string? Failure) TryPredict(object model, int[] shape)
    {
        try
        {
            var probe = new Tensor<double>(shape);

            // WHOLE NUMBERS, not fractions. A fractional fill made every token-driven model reject
            // the probe outright - "EmbeddingLayer is in Indices mode but element 1 is 0.538..., which
            // is not a token index in [0, 256)" - which was the single largest skip cluster. Small
            // integers are valid token indices for every vocabulary in the inventory (the smallest
            // measured is 128) and are equally valid as continuous features, so one fill serves both
            // and the models stop being unreachable. Shape discovery does not depend on the values.
            for (int i = 0; i < probe.Length; i++) probe[i] = (i * 7) % 13;
            var result = ((dynamic)model).Predict(probe);
            return result is null ? (null, "Predict returned null") : ((int[])result._shape, null);
        }
        catch (Exception ex)
        {
            var root = Unwrap(ex);
            return (null, $"{root.GetType().Name}: {Summarise(root.Message)}");
        }
    }

    private static string Summarise(string message)
    {
        var firstLine = message.Split('\n')[0].Trim();
        return firstLine.Length <= 140 ? firstLine : firstLine.Substring(0, 140) + "...";
    }

    private static bool DerivesFromNeuralNetworkBase(Type openGeneric)
    {
        for (var t = openGeneric.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
        }
        return false;
    }
}
