using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Sweeps every constructible layer, DISCOVERS its shape relations by probing, and emits the
/// <c>[TensorLayout]</c> / <c>OutputAxesFor</c> text to paste into it.
/// </summary>
/// <remarks>
/// <para>
/// Two of 246 layers declare a shape contract. Hand-deriving the rest is the manual work this effort
/// exists to remove, so this does the deriving: construct the layer, probe it with shapes that vary one
/// axis at a time, watch what comes out, and fit the simplest relation reproducing every observation.
/// That is <c>ShapeRelationDiscovery</c>, already built and already tested â€” this points it at the whole
/// inventory.
/// </para>
/// <para>
/// TWO CONSTRUCTION PROFILES, because the first version of this sweep was WRONG in a way that would have
/// been written into fifty files. Constructing every layer with the same synthesized parameters made 50
/// of 151 fits report a constant of exactly 4 â€” which was not the layer's contract, it was this
/// harness's default parameter leaking into the answer. <c>AdaptiveAveragePoolingLayer</c> "fixes" its
/// output at 4 only because 4 is what it was handed. Transcribing that would bake a test artifact into
/// product code as a shape law, the same failure as writing a model to satisfy an invariant.
/// </para>
/// <para>
/// So every layer is built TWICE with different parameter magnitudes. A constant that moves with the
/// parameter is PARAMETERISED and must not be declared <c>Fixed</c>; a constant that holds across both
/// is genuinely fixed. That distinction is invisible to a single-profile probe, and it is exactly the
/// kind of thing that only shows up when you vary the thing you suspect.
/// </para>
/// <para>
/// WHAT PROBING STILL CANNOT DO. Discovery recovers RELATIONS because they are visible in the numbers.
/// It cannot recover ROLES: nothing in a shape says axis 2 is Height rather than Width. Roles below are
/// POSITIONAL STAND-INS and the emitted text is a proposal. Renaming them is the judgement this cannot
/// automate, and pretending otherwise would bake a guess into every annotated file.
/// </para>
/// <para>
/// This is a REPORT, not an assertion. A layer that declines to construct, refuses every probe, or fits
/// ambiguously is recorded with the reason rather than failing the build â€” the choice ADNGEN001 makes,
/// for the same reason: a list of what is not covered is useful, a red build naming nothing is not.
/// </para>
/// </remarks>
public class LayerShapeDiscoverySweepTests
{
    private readonly ITestOutputHelper _out;
    public LayerShapeDiscoverySweepTests(ITestOutputHelper output) => _out = output;

    /// <summary>Positional stand-ins. Distinct per position so Fit can tell the axes apart.</summary>
    private static readonly TensorAxis[] PositionalRoles =
    {
        TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
        TensorAxis.Depth, TensorAxis.Time,
    };

    /// <summary>Base shapes tried until one forwards. Deliberately non-square and non-uniform.</summary>
    private static readonly int[][] CandidateBaseShapes =
    {
        new[] { 3, 8, 9 },
        new[] { 2, 3, 8, 9 },
        new[] { 6, 7 },
        new[] { 12 },
        new[] { 2, 6, 7 },
    };

    /// <summary>One construction profile: the parameter values synthesized for a layer's constructor.</summary>
    private readonly struct Profile
    {
        public Profile(string name, int size, int kernel, int stride, int padding, int axis)
        {
            Name = name; Size = size; Kernel = kernel; Stride = stride; Padding = padding; Axis = axis;
        }

        public string Name { get; }
        public int Size { get; }
        public int Kernel { get; }
        public int Stride { get; }
        public int Padding { get; }
        public int Axis { get; }
    }

    /// <summary>
    /// Construction profiles. Each varies ONE family of parameters so a relation that moves can be
    /// attributed to what moved.
    /// </summary>
    /// <remarks>
    /// <para>
    /// SIZE alone is not enough, and assuming it was produced a wrong answer. With only size varying,
    /// MaxPoolingLayer fitted every axis as Same - true for the synthesized kernel=3/stride=1/padding=1,
    /// which happens to preserve spatial dims, and false for the layer. Transcribing that would have
    /// declared a pooling layer shape-preserving.
    /// </para>
    /// <para>
    /// So WINDOW varies kernel/stride/padding, and AXIS varies the reduction axis for layers like
    /// MeanLayer that take one as a constructor argument. A relation that differs across any pair is
    /// derived from construction, not from the layer's contract, and must not be declared as fixed.
    /// </para>
    /// </remarks>
    private static readonly Profile[] Profiles =
    {
        new Profile("size=4",          size: 4, kernel: 3, stride: 1, padding: 1, axis: 0),
        new Profile("size=6",          size: 6, kernel: 3, stride: 1, padding: 1, axis: 0),
        new Profile("window k2 s2 p0", size: 4, kernel: 2, stride: 2, padding: 0, axis: 0),
        new Profile("axis=1",          size: 4, kernel: 3, stride: 1, padding: 1, axis: 1),
    };

    private sealed class LayerFit
    {
        public int InRank;
        public int OutRank;
        public int[] BaseShape = Array.Empty<int>();
        public List<ShapeRelationDiscovery.AxisFinding> Findings = new();
    }

    [Fact]
    public void SweepEmitsCopyReadyShapeDeclarations()
    {
        var layerTypes = typeof(LayerBase<>).Assembly
            .GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromLayerBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        var report = new StringBuilder();
        int constructed = 0, probed = 0, clean = 0, ambiguous = 0, parameterised = 0;
        var skipped = new List<string>();

        foreach (var open in layerTypes)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch (Exception ex) { skipped.Add($"{open.Name}: {ex.GetType().Name} closing generic"); continue; }

            if (TryConstruct(closed, Profiles[0], out string? why) is null)
            {
                skipped.Add($"{open.Name}: {why}");
                continue;
            }

            constructed++;

            // One fit per profile, so the only thing that changed is the layer's construction
            // parameters. Any relation that moves between profiles is config-derived, not contractual.
            var perProfile = new List<LayerFit?>();
            foreach (var profile in Profiles)
            {
                perProfile.Add(FitLayer(closed, profile));
            }

            var primary = perProfile[0];
            if (primary is null)
            {
                skipped.Add($"{open.Name}: no candidate base shape produced 2+ successful forwards");
                continue;
            }

            probed++;

            // Every other profile that produced a comparable fit is a witness against the primary.
            var witnesses = perProfile
                .Select((fit, i) => (fit, name: Profiles[i].Name))
                .Skip(1)
                .Where(x => x.fit is not null
                            && x.fit.OutRank == primary.OutRank
                            && x.fit.Findings.Count == primary.Findings.Count)
                .ToList();

            var inputRoles = PositionalRoles.Take(primary.InRank).ToArray();
            var outputRoles = PositionalRoles.Take(primary.OutRank).ToArray();

            bool anyAmbiguous = false, anyParameterised = false;
            var lines = new List<string>();

            for (int i = 0; i < primary.Findings.Count; i++)
            {
                var f = primary.Findings[i];
                string rendered = f.Relation?.ToString() ?? "UNRESOLVED";
                string note = string.Empty;

                if (f.Ambiguous || f.Relation is null) { anyAmbiguous = true; note = "  <-- AMBIGUOUS"; }

                if (witnesses.Count > 0)
                {
                    var disagreements = witnesses
                        .Select(w => (w.name, other: w.fit!.Findings[i].Relation?.ToString() ?? "UNRESOLVED"))
                        .Where(x => !string.Equals(rendered, x.other, StringComparison.Ordinal))
                        .Select(x => $"{x.name} gives '{x.other}'")
                        .ToList();

                    if (disagreements.Count > 0)
                    {
                        anyParameterised = true;
                        note = $"  <-- PARAMETERISED: {Profiles[0].Name} gives '{rendered}' but "
                               + string.Join("; ", disagreements)
                               + ". NOT a fixed contract - this axis is derived from a constructor "
                               + "argument, so do NOT transcribe the constant or the relation.";
                    }
                }
                else
                {
                    note += "  (no comparable second profile - relation UNVERIFIED)";
                }

                lines.Add($"        [{i}] {outputRoles[i],-9} = {rendered}{note}   ({f.Detail})");
            }

            if (anyAmbiguous) ambiguous++; else if (anyParameterised) parameterised++; else clean++;

            string banner = anyAmbiguous ? "   *** AMBIGUOUS - needs a hand-written declaration ***"
                          : anyParameterised ? "   *** PARAMETERISED - do NOT transcribe the constants ***"
                          : string.Empty;

            report.AppendLine();
            report.AppendLine($"{open.Name}  rank {primary.InRank} -> {primary.OutRank}   "
                              + $"probe base [{string.Join(",", primary.BaseShape)}]{banner}");
            report.AppendLine($"    [TensorLayout({string.Join(", ", inputRoles.Select(r => "TensorAxis." + r))},");
            report.AppendLine($"        Direction = TensorLayoutDirection.Input)]");
            report.AppendLine($"    [TensorLayout({string.Join(", ", outputRoles.Select(r => "TensorAxis." + r))},");
            report.AppendLine($"        Direction = TensorLayoutDirection.Output)]");
            report.AppendLine($"    OutputAxesFor({primary.OutRank}):");
            foreach (var l in lines) report.AppendLine(l);
        }

        var dir = Path.Combine(Path.GetTempPath(), "adn-shape-sweep");
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "layer-shape-declarations.txt");

        var header = new StringBuilder();
        header.AppendLine("LAYER SHAPE DISCOVERY SWEEP");
        header.AppendLine("Roles are POSITIONAL STAND-INS, not discovered facts - probing recovers relations,");
        header.AppendLine("never axis names. Rename them to the layer's real axes before pasting.");
        header.AppendLine($"Every layer built under {Profiles.Length} profiles ({string.Join("", "", Profiles.Select(p => p.Name))}) so a");
        header.AppendLine("constant derived from a constructor argument is flagged PARAMETERISED, not Fixed.");
        header.AppendLine();
        header.AppendLine($"  layer types found     : {layerTypes.Count}");
        header.AppendLine($"  constructed           : {constructed}");
        header.AppendLine($"  probed (2+ fwd)       : {probed}");
        header.AppendLine($"  CLEAN (transcribable) : {clean}");
        header.AppendLine($"  parameterised         : {parameterised}");
        header.AppendLine($"  ambiguous             : {ambiguous}");
        header.AppendLine($"  skipped               : {skipped.Count}");

        File.WriteAllText(path, header + report.ToString() + Environment.NewLine
                                + "SKIPPED" + Environment.NewLine
                                + string.Join(Environment.NewLine, skipped.Select(s => "  " + s)));

        _out.WriteLine(header.ToString());
        _out.WriteLine($"report written to: {path}");

        Assert.True(layerTypes.Count > 100, $"only {layerTypes.Count} layer types discovered; expected the full inventory.");
        Assert.True(clean + parameterised + ambiguous > 0, "no layer produced a fit at all - the sweep measured nothing.");
    }

    /// <summary>Probes one layer built at a given parameter size, returning its fit or null.</summary>
    /// <summary>
    /// Every layer that DECLARES a shape contract must have a contract that is TRUE: resolving it
    /// against a real input shape has to reproduce the shape a real forward pass actually produces.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Before this test, exactly two contracts (Dense and Convolution) were checked against a forward
    /// pass, by hand, in <c>ShapeContractConformanceTests</c>. The rest were transcribed from the
    /// discovery sweep or hand-derived from constructor fields, and a transcription error in one of
    /// them is strictly worse than no contract at all - it makes the shape system confidently wrong
    /// instead of silent. Checking them one at a time does not scale to the whole layer set, so this
    /// checks all of them the only way that stays honest as layers are added: by construction.
    /// </para>
    /// <para>
    /// It calls <see cref="ShapeInference.InferOutputShape"/> - the same production entry point
    /// <c>LayerGraph.ResolveShapes</c> uses - rather than re-implementing resolution here. A test that
    /// resolved relations its own way could agree with itself while disagreeing with the library.
    /// </para>
    /// <para>
    /// A null prediction is NOT a failure. <c>OutputAxesFor</c> returning null is a layer explicitly
    /// declining to claim anything at that rank, which is the honest answer for a rank it does not
    /// support. Only a prediction that DISAGREES with the forward pass fails.
    /// </para>
    /// </remarks>
    [Fact]
    public void EveryDeclaredContractPredictsTheRealForwardShape()
    {
        var contractTypes = typeof(LayerBase<>).Assembly
            .GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromLayerBase(t)
                        && t.GetInterfaces().Any(i => i == typeof(IShapeContract)))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        var mismatches = new List<string>();
        var checkedShapes = new List<string>();
        var unexercised = new List<string>();

        foreach (var open in contractTypes)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch (Exception ex) { unexercised.Add($"{open.Name}: {ex.GetType().Name} closing generic"); continue; }

            bool exercised = false;
            bool everForwarded = false;
            string? lastFailure = null;

            // Every profile, because a contract built from constructor fields (Fixed(_outputSize),
            // Window(_poolSize, ...)) is only proven correct if it tracks those fields as they change.
            // A contract that happens to be right for one configuration is the exact bug this catches.
            foreach (var profile in Profiles)
            {
                if (TryConstruct(closed, profile, out _) is not LayerBase<double> layer) continue;

                foreach (var shape in BaseShapeCandidates(closed, profile))
                {
                    if (TryConstruct(closed, profile, out _) is not LayerBase<double> fresh) break;

                    var observed = TryForward(fresh, shape, out string? failure);
                    if (observed is null)
                    {
                        lastFailure ??= failure is null ? null : $"in[{string.Join(",", shape)}] {failure}";
                        continue;
                    }

                    int[]? predicted;
                    try { predicted = ShapeInference.InferOutputShape(fresh, shape); }
                    catch (Exception ex)
                    {
                        mismatches.Add(
                            $"{open.Name} in[{string.Join(",", shape)}]: contract THREW " +
                            $"{ex.GetType().Name}: {Short(ex.Message)}");
                        exercised = true;
                        break;
                    }

                    // Declining to predict is allowed; predicting wrongly is not.
                    everForwarded = true;
                    if (predicted is null) continue;

                    exercised = true;
                    if (!predicted.SequenceEqual(observed))
                    {
                        mismatches.Add(
                            $"{open.Name} [{profile.Name}] in[{string.Join(",", shape)}]: " +
                            $"contract says [{string.Join(",", predicted)}] but forward produced " +
                            $"[{string.Join(",", observed)}]");
                    }
                    else
                    {
                        checkedShapes.Add($"{open.Name} [{string.Join(",", shape)}]");
                    }

                    break;   // one verified shape per profile is enough to prove tracking
                }
            }

            if (!exercised)
            {
                // Two different situations, and conflating them misreads the layer. A layer that
                // forwarded but predicted nothing is DECLINING - OutputAxesFor returned null for that
                // rank, which is the honest answer for a rank it does not model, and no defect.
                // A layer that never forwarded is a probe the harness could not satisfy.
                unexercised.Add(everForwarded
                    ? $"{open.Name}: forwards, but its contract declines to predict at that rank"
                    : lastFailure is null
                        ? $"{open.Name}: no candidate shape forwarded"
                        : $"{open.Name}: first probe failure - {lastFailure}");
            }
        }

        // Recorded, not asserted: a layer this harness cannot construct or forward is a gap in the
        // HARNESS, not proof the layer is wrong, and silently treating it as passing would let the
        // verified count rot as layers are added. Printing it keeps the number visible.
        _out.WriteLine(
            $"contracts declared: {contractTypes.Count}   verified against a forward pass: " +
            $"{contractTypes.Count - unexercised.Count}   not exercised by this harness: {unexercised.Count}");
        foreach (var u in unexercised) _out.WriteLine($"  not exercised: {u}");

        Assert.True(
            mismatches.Count == 0,
            $"{mismatches.Count} declared shape contract(s) disagree with the actual forward pass. " +
            "A wrong contract is worse than no contract - it makes shape inference confidently wrong." +
            Environment.NewLine + string.Join(Environment.NewLine, mismatches));
    }

    private static LayerFit? FitLayer(Type closed, Profile profile)
    {
        foreach (var candidate in BaseShapeCandidates(closed, profile))
        {
            var probes = ShapeRelationDiscovery.ProbeShapes(candidate);
            var collected = new List<(int[], int[])>();

            foreach (var probe in probes)
            {
                // A fresh instance per probe: layers resolve lazy shapes on first forward, so reusing one
                // would measure every later probe against the first probe's frozen shape.
                if (TryConstruct(closed, profile, out _) is not LayerBase<double> fresh) break;

                var output = TryForward(fresh, probe);
                if (output is null) break;
                collected.Add((probe, output));
            }

            if (collected.Count < 2) continue;

            int inRank = candidate.Length;
            int outRank = collected[0].Item2.Length;

            try
            {
                var findings = ShapeRelationDiscovery.Fit(
                    PositionalRoles.Take(inRank).ToArray(),
                    PositionalRoles.Take(outRank).ToArray(),
                    collected);

                return new LayerFit
                {
                    InRank = inRank,
                    OutRank = outRank,
                    BaseShape = candidate,
                    Findings = findings.ToList(),
                };
            }
            catch
            {
                return null;
            }
        }

        return null;
    }

    /// <summary>The layer's own declared input shape first, then the generic fallbacks.</summary>
    private static IEnumerable<int[]> BaseShapeCandidates(Type closed, Profile profile)
    {
        int[]? declared = null;
        try
        {
            if (TryConstruct(closed, profile, out _) is ILayer<double> probe) declared = probe.GetInputShape();
        }
        catch { /* layer declines to say */ }

        if (declared is { Length: > 0 } && Array.TrueForAll(declared, d => d > 0))
        {
            yield return declared;
        }

        foreach (var c in CandidateBaseShapes) yield return c;
    }

    private static bool DerivesFromLayerBase(Type openGeneric)
    {
        for (var t = openGeneric.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(LayerBase<>)) return true;
        }
        return false;
    }

    /// <summary>
    /// Builds a layer from the constructor with the fewest parameters it can satisfy.
    /// </summary>
    /// <remarks>
    /// Values are chosen by parameter NAME because that is the only signal available - a bare
    /// <c>int</c> could be a kernel size or a class count, and a kernel of 12 over an 8-wide probe
    /// simply throws. Names are the codebase's own convention, so using them reads the code rather than
    /// guessing at it. <paramref name="size"/> varies between profiles; kernel/stride/padding do NOT,
    /// because changing those changes the RELATION rather than exposing a parameterised constant.
    /// </remarks>
    private static object? TryConstruct(Type closed, Profile profile, out string? why)
    {
        why = null;
        var ctors = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
            .OrderBy(c => c.GetParameters().Length)
            .ToList();

        if (ctors.Count == 0) { why = "no public constructor"; return null; }

        string? lastError = null;

        foreach (var ctor in ctors)
        {
            var pars = ctor.GetParameters();
            var args = new object?[pars.Length];
            bool usable = true;

            for (int i = 0; i < pars.Length; i++)
            {
                if (!TryValueFor(pars[i], profile, out args[i])) { usable = false; break; }
            }

            if (!usable) continue;

            try { return ctor.Invoke(args); }
            catch (Exception ex) { lastError = Short((ex.InnerException ?? ex).Message); }
        }

        why = lastError is null ? "no constructor whose parameters could be supplied" : $"ctor threw - {lastError}";
        return null;
    }

    private static bool TryValueFor(ParameterInfo p, Profile profile, out object? value)
    {
        var t = p.ParameterType;
        string n = p.Name?.ToLowerInvariant() ?? string.Empty;

        if (p.HasDefaultValue && !t.IsValueType) { value = p.DefaultValue; return true; }

        if (t == typeof(int))
        {
            value = n switch
            {
                var s when s.Contains("kernel") || s.Contains("pool") => profile.Kernel,
                var s when s.Contains("stride") => profile.Stride,
                var s when s.Contains("padding") || s.Contains("pad") => profile.Padding,
                var s when s.Contains("dilation") => 1,
                var s when s.Contains("head") => 2,
                var s when s.Contains("group") => 1,
                var s when s.Contains("axis") || (s.Contains("dim") && s.Contains("index")) => profile.Axis,
                _ => profile.Size,
            };
            return true;
        }

        if (t == typeof(double) || t == typeof(float)) { value = Convert.ChangeType(0.5, t); return true; }
        if (t == typeof(bool)) { value = false; return true; }
        // Array parameters MUST vary between profiles too. Holding them fixed is how PaddingLayer
        // fitted a "window" relation containing 2*8 and 2*9 - the probe shape's own dimensions, fed in
        // as a padding array and then read back out as if it were the layer's contract. A relation
        // derived from a constant input looks perfectly stable, which is exactly what makes it
        // dangerous: it passes every consistency check and is still an artifact.
        if (t == typeof(int[]))
        {
            value = new[] { profile.Kernel, profile.Size, profile.Size + 1 };
            return true;
        }

        if (t == typeof(int[][]))
        {
            var row = new[] { profile.Kernel, profile.Size, profile.Size + 1 };
            value = new[] { row, row };
            return true;
        }
        if (!t.IsValueType) { value = p.HasDefaultValue ? p.DefaultValue : null; return true; }
        if (p.HasDefaultValue) { value = p.DefaultValue; return true; }

        value = null;
        return false;
    }

    /// <summary>
    /// Forwards one probe, feeding EVERY input port for multi-input layers.
    /// </summary>
    /// <remarks>
    /// Add, Concatenate, Multiply and the cross-attention family were unreachable while this called only
    /// the single-input overload - they are exactly the layers that join branches, so omitting them would
    /// leave the joins undiscovered for the same reason tracing used to miss them. The probe varies the
    /// FIRST port and holds the rest at their declared shapes, so a fitted relation is attributable to
    /// one input rather than to several moving together.
    /// </remarks>
    private static int[]? TryForward(LayerBase<double> layer, int[] shape)
        => TryForward(layer, shape, out _);

    /// <summary>
    /// As <see cref="TryForward(LayerBase{double}, int[])"/>, but reports WHY a probe failed.
    /// </summary>
    /// <remarks>
    /// A swallowed exception turns "this layer's contract is unverified" into an unanswerable question:
    /// the count of unexercised layers is visible but the cause is not, so the gap cannot be closed
    /// without re-instrumenting the harness by hand each time. The reason is carried out instead.
    /// </remarks>
    private static int[]? TryForward(LayerBase<double> layer, int[] shape, out string? failure)
    {
        failure = null;
        try
        {
            int ports = 1;
            try { ports = Math.Max(1, layer.InputPorts?.Count ?? 1); } catch { ports = 1; }

            if (ports == 1) return layer.Forward(Probe(shape))?._shape;

            // MUST be the PORT overload, not Forward(params Tensor[]). Multi-port layers override
            // ForwardTracedPorts; they do NOT override ForwardTracedMany, whose base implementation
            // concatenates its inputs along dimension 1. Calling the array overload therefore measures
            // LayerBase's generic concat rather than the layer, and reports a doubled axis as if it
            // were the layer's shape law: MemoryRead/MemoryWrite/TransformerDecoder each "produced"
            // [3,16,9] from [3,8,9], and DiffusionResBlock [1,8,4,4] from [1,4,4,4]. Four correct
            // contracts looked wrong, and transcribing that would have written a base-class default
            // into four layers as their contract.
            var declaredPorts = layer.InputPorts;
            var inputs = new Dictionary<string, Tensor<double>>(StringComparer.Ordinal);
            for (int i = 0; i < declaredPorts.Count; i++)
            {
                var port = declaredPorts[i];

                // Port 0 carries the shape under test.
                if (i == 0) { inputs[port.Name] = Probe(shape); continue; }

                // OPTIONAL ports are omitted, even when they declare a resolved shape. A layer that
                // marks a port Required: false documents a default for it, and that default path is
                // the one a shape contract has to describe. Synthesizing the input instead tests a
                // tensor the harness invented: MemoryReadLayer's 'memory' port declares
                // GetOutputShape(), which is rank 1, and the layer's own transpose then rejects it -
                // reporting the layer as unverifiable over an input it never asked for.
                if (!port.Required) continue;

                // A required secondary port uses the shape it declares, since it is rarely the same
                // shape as the primary input.
                inputs[port.Name] = port.Shape is { Count: > 0 } portDeclared && portDeclared.All(d => d > 0)
                    ? Probe(portDeclared.ToArray())
                    : Probe(shape);
            }

            return layer.Forward(inputs)?._shape;
        }
        catch (Exception ex)
        {
            failure = $"{ex.GetType().Name}: {Short(ex.Message)}";
            return null;
        }
    }

    private static Tensor<double> Probe(int[] shape)
    {
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = ((i * 7) % 13) / 13.0;
        return t;
    }

    private static string Short(string s) =>
        s.Length <= 90 ? s.Replace(Environment.NewLine, " ") : s.Substring(0, 90).Replace(Environment.NewLine, " ") + "...";
}

