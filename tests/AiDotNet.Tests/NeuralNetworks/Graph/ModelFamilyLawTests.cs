using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Asks what shape law a model FAMILY actually obeys, before any law is declared on its base.
/// </summary>
/// <remarks>
/// <para>
/// A base declaration has enormous leverage - SegmentationModelBase sits above 69 models,
/// AudioNeuralNetworkBase above 178 - and that leverage cuts both ways: a wrong law on a base is wrong
/// for every member at once. So the law is MEASURED here rather than assumed from the family's name.
/// </para>
/// <para>
/// TWO THINGS ARE VARIED, and the second is the one that makes a constant falsifiable.
/// </para>
/// <para>
/// (1) The INPUT shape, which reveals which output axes track which input axes.
/// </para>
/// <para>
/// (2) The CONSTRUCTION, by building each model twice with a different class count. Input variation
/// alone cannot tell <c>Fixed(21)</c> - a genuine constant - from a value that merely happens to equal
/// whatever the constructor was handed. Building at 7 and at 13 settles it: an output axis that MOVES
/// with the argument is <c>Fixed(_numClasses)</c>, and one that HOLDS is a real constant. This is the
/// same discipline that exposed 50 false <c>Fixed(4)</c>s in the layer sweep, where every interior axis
/// looked constant purely because nothing had moved it.
/// </para>
/// <para>
/// Reports and asserts only that it ran. A coverage number is evidence for a decision, not a property
/// to enforce - the enforcing comes later, once a law is actually declared.
/// </para>
/// </remarks>
public class ModelFamilyLawTests
{
    private readonly ITestOutputHelper _out;
    public ModelFamilyLawTests(ITestOutputHelper output) => _out = output;

    /// <summary>How many members of a family to probe. Bounded - this is evidence, not an inventory.</summary>
    private const int PerFamilyBudget = 10;

    private const int Extent = 8;

    /// <summary>
    /// Three construction profiles. Each pair differs in exactly ONE variable, so each output axis can
    /// be attributed to the thing that moved it.
    /// </summary>
    /// <remarks>
    /// A and B differ only in class count (isolates the class axis), A and C only in input geometry
    /// (isolates the spatial axes), A and D only in batch (isolates the batch axis).
    /// </para>
    /// <para>
    /// THE GEOMETRIES STRADDLE THE STRIDE, 64 and 128, and that is not arbitrary. A first attempt used
    /// 8 and 16, and every spatial axis came back a constant 1 - which looked like proof of Fixed(1)
    /// and was nothing of the kind: a /32-stride encoder floors BOTH 8 and 16 to 1, so the measurement
    /// could not separate Fixed(1) from Scaled(Height, 1/32). Probing below the stride cannot falsify a
    /// spatial constant, however many sizes are tried.
    /// </remarks>
    private static readonly (string Name, int Classes, int Extent, int Batch)[] Profiles =
    [
        ("A: classes=7,  extent=64,  batch=1", 7,  64,  1),
        ("B: classes=13, extent=64,  batch=1", 13, 64,  1),
        ("C: classes=7,  extent=128, batch=1", 7,  128, 1),
        ("D: classes=7,  extent=64,  batch=2", 7,  64,  2),
    ];

    [Trait("Category", "Sweep")]
    [Fact(Timeout = 1800000)]
    public async Task DoTheMembersOfAFamilyShareOneShapeLaw()
    {
        await Task.Yield();

        var asm = typeof(NeuralNetworkBase<>).Assembly;

        var concrete = asm.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && Ancestors(t).Any(a => a.Name == "NeuralNetworkBase`1"))
            .ToList();

        // TRANSITIVE grouping, so a model counts under EVERY abstract ancestor. Grouping by nearest
        // base alone hid the segmentation family: its 69 models split across SemanticSegmentationBase
        // (15), PanopticSegmentationBase (12) and six more, so SegmentationModelBase - the base all 69
        // inherit, and the one a contract would go on - never appeared at all.
        var families = new Dictionary<string, List<Type>>(StringComparer.Ordinal);
        foreach (var t in concrete)
        {
            foreach (var a in Ancestors(t))
            {
                if (!a.IsAbstract || a.Name == "NeuralNetworkBase`1") continue;
                if (!families.TryGetValue(a.Name, out var list)) families[a.Name] = list = new List<Type>();
                list.Add(t);
            }
        }

        string? only = Environment.GetEnvironmentVariable("ADNSHAPE_FAMILY");
        var selected = string.IsNullOrWhiteSpace(only)
            ? families.OrderByDescending(f => f.Value.Count).Take(4)
            : families.Where(f => f.Key.StartsWith(only, StringComparison.OrdinalIgnoreCase));

        int familiesProbed = 0;

        foreach (var family in selected.OrderByDescending(f => f.Value.Count))
        {
            _out.WriteLine("");
            _out.WriteLine($"=== {family.Key}  ({family.Value.Count} models) ===");

            var rows = new List<(string Model, int[][] In, int[][] Out)>();
            var skipped = new List<string>();

            foreach (var open in family.Value.Distinct().OrderBy(t => t.Name, StringComparer.Ordinal))
            {
                if (rows.Count >= PerFamilyBudget) break;

                Type closed;
                try { closed = open.MakeGenericType(typeof(double)); }
                catch { continue; }

                var ins = new List<int[]>();
                var outs = new List<int[]>();
                string? failure = null;

                foreach (var (_, classes, extent, batch) in Profiles)
                {
                    object? model = null;
                    try { model = Construct(closed, classes, extent); }
                    catch (Exception ex) { failure ??= $"{Unwrap(ex).GetType().Name} constructing"; break; }
                    if (model is null) { failure ??= "no usable constructor"; break; }

                    try
                    {
                        int[]? perSample = TryArchitectureInputShape(model);
                        if (perSample is null || perSample.Length == 0 || perSample.Any(d => d <= 0))
                        {
                            failure ??= "no concrete declared input shape";
                            break;
                        }

                        var shape = new int[perSample.Length + 1];
                        shape[0] = batch;
                        for (int i = 0; i < perSample.Length; i++) shape[i + 1] = Math.Min(perSample[i], extent);

                        var (o, f) = TryPredict(model, shape);
                        if (o is null) { failure ??= f; break; }
                        ins.Add(shape);
                        outs.Add(o);
                    }
                    finally { (model as IDisposable)?.Dispose(); }
                }

                if (outs.Count < Profiles.Length)
                {
                    skipped.Add($"{open.Name}: {failure ?? "incomplete profile set"}");
                    continue;
                }

                rows.Add((open.Name, ins.ToArray(), outs.ToArray()));
            }

            if (rows.Count == 0)
            {
                _out.WriteLine("  no member produced both profiles - no evidence either way");
                foreach (var s in skipped.Take(6)) _out.WriteLine($"    skipped: {s}");
                continue;
            }

            familiesProbed++;

            foreach (var (m, i, o) in rows)
            {
                _out.WriteLine($"    {m,-28} A {Fmt(i[0])}->{Fmt(o[0])}   B {Fmt(o[1])}   C {Fmt(i[2])}->{Fmt(o[2])}");
            }

            var ranks = rows.Select(r => r.Out[0].Length).Distinct().OrderBy(r => r).ToList();
            _out.WriteLine($"  probed={rows.Count}  skipped={skipped.Count}  output ranks observed={string.Join(",", ranks)}");
            _out.WriteLine("  NOTE: several ranks is NOT a blocker - a base declares one [TensorLayout] per");
            _out.WriteLine("        accepted rank and switches on it in OutputAxesFor, exactly as layers do.");

            foreach (int rank in ranks)
            {
                var group = rows.Where(r => r.Out[0].Length == rank).ToList();
                var verdicts = new List<string>();

                for (int axis = 0; axis < rank; axis++)
                {
                    // A vs B isolates class count, A vs C geometry, A vs D batch.
                    bool tracksClasses = group.All(r => r.Out[0][axis] == 7 && r.Out[1][axis] == 13);
                    bool movedWithGeometry = group.Any(r => r.Out[0][axis] != r.Out[2][axis]);
                    bool movedWithBatch = group.Any(r => r.Out[0][axis] != r.Out[3][axis]);

                    if (movedWithBatch && group.All(r => r.Out[0][axis] == 1 && r.Out[3][axis] == 2))
                    {
                        verdicts.Add($"axis{axis}=Same(Batch)");
                        continue;
                    }

                    if (tracksClasses) { verdicts.Add($"axis{axis}=Fixed(_numClasses)"); continue; }

                    if (!movedWithGeometry && !movedWithBatch)
                    {
                        // Constant under class count, geometry AND batch - now genuinely falsified.
                        var vals = group.Select(r => r.Out[0][axis]).Distinct().ToList();
                        verdicts.Add(vals.Count == 1
                            ? $"axis{axis}=Fixed({vals[0]}) [falsified vs classes, geometry AND batch]"
                            : $"axis{axis}=Fixed(per-model)");
                        continue;
                    }

                    // Moved with geometry - name the exact relation to the corresponding input axis.
                    var rel = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var r in group)
                    {
                        int inA = axis < r.In[0].Length ? r.In[0][axis] : -1;
                        int inC = axis < r.In[2].Length ? r.In[2][axis] : -1;
                        if (inA <= 0 || inC <= 0) { rel.Add("?"); continue; }
                        if (r.Out[0][axis] == inA && r.Out[2][axis] == inC) { rel.Add("Same"); continue; }

                        int dA = inA / Math.Max(1, r.Out[0][axis]);
                        int dC = inC / Math.Max(1, r.Out[2][axis]);
                        rel.Add(dA == dC && dA > 0 && r.Out[0][axis] * dA == inA ? $"Scaled(1/{dA})" : "varies");
                    }
                    verdicts.Add($"axis{axis}=" + (rel.Count == 1 ? rel.First() : "varies-per-model")
                        + "(in[" + axis + "])");
                }

                _out.WriteLine($"    rank {rank} ({group.Count} models): {string.Join("  ", verdicts)}");
            }
        }

        Assert.True(familiesProbed > 0,
            "no family produced a complete profile set, so this measured nothing - harness problem, not a finding");
    }

    private static string Fmt(int[] s) => "[" + string.Join(",", s) + "]";

    private static IEnumerable<Type> Ancestors(Type t)
    {
        for (var a = t.BaseType; a is not null; a = a.BaseType)
            yield return a.IsGenericType ? a.GetGenericTypeDefinition() : a;
    }

    /// <summary>
    /// Builds the model, overriding any constructor parameter that names a class count.
    /// </summary>
    /// <remarks>
    /// Overriding BY PARAMETER NAME rather than position, because the class count sits at a different
    /// index in almost every model and a positional guess would silently pass 7 as a dropout rate.
    /// </remarks>
    private static object? Construct(Type closed, int classes, int extent)
    {
        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });

        if (ctor is null)
        {
            return closed.GetConstructor(Type.EmptyTypes) is not null
                ? Activator.CreateInstance(closed)
                : null;
        }

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: extent, inputWidth: extent, outputSize: classes);

        for (int i = 1; i < pars.Length; i++)
        {
            var p = pars[i];
            bool isClassCount = p.ParameterType == typeof(int)
                && (p.Name?.IndexOf("numClasses", StringComparison.OrdinalIgnoreCase) >= 0
                    || p.Name?.IndexOf("classCount", StringComparison.OrdinalIgnoreCase) >= 0);
            args[i] = isClassCount ? classes : p.DefaultValue;
        }

        return ctor.Invoke(args);
    }

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
            for (int i = 0; i < probe.Length; i++) probe[i] = (i * 7) % 13;
            var result = ((dynamic)model).Predict(probe);
            return result is null ? (null, "Predict returned null") : ((int[])result._shape, null);
        }
        catch (Exception ex)
        {
            var root = Unwrap(ex);
            var msg = root.Message.Split('\n')[0].Trim();
            return (null, $"{root.GetType().Name}: {(msg.Length > 80 ? msg.Substring(0, 80) + "..." : msg)}");
        }
    }

    private static Exception Unwrap(Exception ex) =>
        ex is TargetInvocationException { InnerException: not null } tie ? tie.InnerException : ex;
}
