using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Measures how much of the model inventory a BASE-CLASS shape declaration could reach.
/// </summary>
/// <remarks>
/// <para>
/// This exists to answer one question before any declaration is written: if a shape contract is
/// declared on a model's base class and inherited, how many concrete models does that actually cover?
/// A base-class strategy is only worth building if the answer is most of them.
/// </para>
/// <para>
/// WHY BASES RATHER THAN MODELS. Declaring per model would mean ~1490 hand-written contracts, which is
/// the opposite of automatic. A layer precedent already settled this: one contract on LoRAAdapterBase
/// covered 34 adapters in a single edit, because [TensorLayout] is inherited and the resolver reads it
/// with inherit: true. The same mechanism applies here - the question is only how many models sit under
/// a shared base.
/// </para>
/// <para>
/// This measures and reports. It asserts only that the measurement RAN, because a coverage number is
/// evidence for a design decision, not a property to enforce.
/// </para>
/// </remarks>
public class ModelBaseCoverageTests
{
    private readonly ITestOutputHelper _out;
    public ModelBaseCoverageTests(ITestOutputHelper output) => _out = output;

    [Trait("Category", "Sweep")]
    [Fact(Timeout = 300000)]
    public async Task HowManyModelsWouldABaseClassDeclarationReach()
    {
        await Task.Yield();

        var asm = typeof(NeuralNetworkBase<>).Assembly;

        // Concrete, single-type-parameter model types - the same population the shape probe walks.
        var concrete = asm.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFrom(t, "NeuralNetworkBase`1"))
            .ToList();

        // TRANSITIVE coverage, counting EVERY abstract ancestor rather than just the nearest one.
        //
        // Counting only the nearest base is the wrong instrument for an inheritance design, and it
        // produced a badly wrong reading: segmentation models were all attributed to their family base
        // (SemanticSegmentationBase and siblings), which made SegmentationModelBase - the generic base
        // those families themselves derive from - look like it had NO models at all. [TensorLayout] is
        // inherited, so a declaration on a base reaches every descendant at any depth; the number that
        // matters is therefore how many concrete models sit ANYWHERE beneath it.
        var byBase = new Dictionary<string, List<string>>(StringComparer.Ordinal);
        var rootOnly = new List<string>();

        foreach (var t in concrete)
        {
            bool any = false;
            foreach (var a in Ancestors(t))
            {
                if (!a.IsAbstract || a.Name == "NeuralNetworkBase`1") continue;
                any = true;
                if (!byBase.TryGetValue(a.Name, out var list)) byBase[a.Name] = list = new List<string>();
                list.Add(t.Name);
            }

            if (!any) rootOnly.Add(t.Name);
        }

        // A model beneath three nested bases is counted under all three, so this is a DISTINCT count.
        int covered = concrete.Count - rootOnly.Count;

        _out.WriteLine($"concrete models                                        : {concrete.Count}");
        _out.WriteLine($"sit under at least one family base                     : {covered}");
        _out.WriteLine($"derive straight from NeuralNetworkBase (no family base): {rootOnly.Count}");
        _out.WriteLine($"distinct abstract bases in the tree                    : {byBase.Count}");
        _out.WriteLine("");
        // Which bases ALREADY declare a contract. Without this the ranking answers "where is the
        // leverage" but not "where is the leverage LEFT", and the top of the list stays dominated by
        // bases that were declared several commits ago.
        var declaring = new HashSet<string>(StringComparer.Ordinal);
        foreach (var t in asm.GetTypes())
        {
            if (t.IsAbstract && t.GetInterfaces().Any(i => i.Name == "IShapeContract"))
                declaring.Add(t.Name);
        }

        _out.WriteLine("--- TRANSITIVE coverage per base, richest first (a declaration here reaches all of them) ---");
        foreach (var kv in byBase.OrderByDescending(k => k.Value.Count).Take(40))
        {
            string mark = declaring.Contains(kv.Key) ? "declared" : "        ";
            _out.WriteLine($"  {kv.Value.Count,4}  {mark}  {kv.Key}");
        }

        _out.WriteLine("");
        _out.WriteLine("--- richest bases with NO contract yet (this is the remaining work) ---");
        foreach (var kv in byBase.Where(k => !declaring.Contains(k.Key))
                                 .OrderByDescending(k => k.Value.Count).Take(25))
        {
            _out.WriteLine($"  {kv.Value.Count,4}  {kv.Key}");
        }

        // The greedy set-cover: how few declarations actually cover the inventory, given that a base
        // high in the tree subsumes every base below it. This is the number that decides how much hand
        // work the model half really costs.
        _out.WriteLine("");
        _out.WriteLine("--- fewest bases needed to cover them all (greedy, highest-leverage first) ---");
        var remaining = new HashSet<string>(concrete.Select(t => t.Name), StringComparer.Ordinal);
        foreach (var n in rootOnly) remaining.Remove(n);
        int picks = 0;
        while (remaining.Count > 0 && picks < 40)
        {
            var best = byBase
                .Select(kv => (Base: kv.Key, Gain: kv.Value.Count(m => remaining.Contains(m))))
                .OrderByDescending(x => x.Gain).First();
            if (best.Gain == 0) break;
            picks++;
            _out.WriteLine($"  {picks,2}. {best.Base,-42} covers {best.Gain,4} more  ({remaining.Count - best.Gain} left)");
            foreach (var m in byBase[best.Base]) remaining.Remove(m);
        }
        _out.WriteLine($"  => {picks} declarations cover {covered} of {concrete.Count} models");

        _out.WriteLine("");
        _out.WriteLine("--- a sample of the models with NO family base ---");
        foreach (var n in rootOnly.OrderBy(n => n, StringComparer.Ordinal).Take(20)) _out.WriteLine($"  {n}");

        Assert.True(concrete.Count > 0, "no concrete models found - the harness is broken, not the inventory");
    }

    private static IEnumerable<Type> Ancestors(Type t)
    {
        for (var a = t.BaseType; a is not null; a = a.BaseType)
            yield return a.IsGenericType ? a.GetGenericTypeDefinition() : a;
    }

    private static bool DerivesFrom(Type type, string openGenericName)
    {
        for (var a = type.BaseType; a is not null; a = a.BaseType)
        {
            var def = a.IsGenericType ? a.GetGenericTypeDefinition() : a;
            if (def.Name == openGenericName) return true;
        }
        return false;
    }
}
