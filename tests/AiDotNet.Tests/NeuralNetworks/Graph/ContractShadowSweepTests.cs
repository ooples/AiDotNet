using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Runs the declared shape CONTRACTS alongside the imperative shape RESOLUTION across assembled
/// models, and reports where the two disagree.
/// </summary>
/// <remarks>
/// <para>
/// THE LIBRARY CARRIES TWO SHAPE SYSTEMS THAT NEVER MET. The declarative one - <c>IShapeContract</c>
/// resolved by <c>ShapeInference.InferOutputShape</c> - is verified layer-by-layer against real
/// forward passes, and had ZERO production callers. The operational one - <c>OnFirstForward</c>
/// populating the field <c>GetOutputShape()</c> returns - is what every model, graph resolution and
/// chain check actually uses. 317 verified contracts that nothing consults are a decoration.
/// </para>
/// <para>
/// This is the parallel run that earns them authority. It is deliberately a REPORT first: the point is
/// to find out whether the two agree before <c>InferOutputShape</c> is made the source of truth in
/// <c>TryAdvanceLayerShape</c> and <c>LayerGraph.ResolveShapes</c>, because there was previously no
/// evidence either way. Making the swap first and discovering the disagreements afterwards would be
/// changing the shape of every model in the library on an assumption.
/// </para>
/// <para>
/// IN AN ASSEMBLED CHAIN, not on a probe. The per-layer conformance sweep already checks a contract
/// against a forward pass on a synthesized input. What this adds is the case that actually ships: a
/// layer whose input shape came from its PREDECESSOR, resolved through the model's own wiring.
/// </para>
/// </remarks>
public class ContractShadowSweepTests
{
    private readonly ITestOutputHelper _out;
    public ContractShadowSweepTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// How many models to construct. Bounded because construction is cheap but shape resolution can
    /// allocate: the full inventory is ~1825 models and some build paper-scale variants.
    /// </summary>
    private const int MaxModels = 400;

    /// <remarks>
    /// CATEGORY=SWEEP, and it belongs there rather than in the gating shard. This constructs hundreds
    /// of models reflectively, and at least one of them takes the test HOST down - not a failing
    /// assertion but an abort, which is uncatchable and destroys the whole run. Putting a model-level
    /// sweep in a gating shard is exactly what the plan said not to do, and doing it anyway turned an
    /// 84-test green shard into "Test Run Aborted" after 59 tests.
    ///
    /// Category!=Sweep already appears in 45 shard filters, so this is excluded from all of them and
    /// runs in the existing non-gating parameter-enumeration-sweep job: its own runner, 60-minute
    /// budget, continue-on-error, findings uploaded as artifacts.
    /// </remarks>
    [Trait("Category", "Sweep")]
    [Fact(Timeout = 300000)]
    public async Task ContractsAgreeWithResolvedShapes_AcrossAssembledModels()
    {
        await Task.Yield();   // xunit applies Timeout only to async tests.

        var open = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromNeuralNetworkBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .Take(MaxModels)
            .ToList();

        int constructed = 0, resolved = 0;
        int agreed = 0, declined = 0, unresolved = 0;
        var disagreements = new List<string>();
        var skipped = new List<string>();

        // A CHAIN-FOLD comparison lived here and was REMOVED as unsound, not as unfinished.
        //
        // It propagated each layer's contract output into the next layer's input and compared against
        // the imperative walk, to license making contracts authoritative. But contracts answer in
        // BATCHED terms while chain resolution propagates PER-SAMPLE shapes, so the fold acquired a
        // phantom batch axis at layer 0 and carried it the length of every chain. Its 46 "disagreements"
        // were that phantom: DenseLayer resolves [128] -> [256] correctly and was accused of
        // [1,128] -> [1,256], and a "fix" applied against that reading moved the number to 96.
        //
        // A measurement that has not been shown correct on a case with a known answer must not drive
        // changes. This one was not, and it drove five. The per-layer comparison below WAS validated
        // against real forward passes before being trusted, and it found four genuine defects.

        // Measurement 1: contracts that lead with Same(Batch) and answer identically either way.
        var ignoresFlagLeadingBatch = new HashSet<string>(StringComparer.Ordinal);
        var honoursFlag = new HashSet<string>(StringComparer.Ordinal);

        // Measurement 2: WHY layers go unresolved - the blind spot a cutover would inherit.
        var unresolvedByType = new Dictionary<string, int>(StringComparer.Ordinal);
        var noContractByType = new Dictionary<string, int>(StringComparer.Ordinal);

        foreach (var t in open)
        {
            Type closed;
            try { closed = t.MakeGenericType(typeof(double)); }
            catch { continue; }

            object? model;
            try { model = Activator.CreateInstance(closed); }
            catch (Exception ex)
            {
                skipped.Add($"{t.Name}: {ex.GetType().Name} constructing");
                continue;
            }
            if (model is null) continue;
            constructed++;

            // Resolve the chain WITHOUT a forward pass. ResolveLazyLayerShapes walks the layers and
            // sizes each from its predecessor, which is exactly the assembled-chain state this is here
            // to inspect, and it does not allocate weights or consume RNG the way a Predict would.
            try
            {
                var resolve = closed.GetMethod(
                    "ResolveLazyLayerShapes",
                    BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.FlattenHierarchy);
                resolve?.Invoke(model, null);
                resolved++;
            }
            catch
            {
                // A model that cannot resolve its own chain has nothing to compare; it is reported by
                // the unresolved count below rather than treated as agreement.
            }

            var layers = closed.GetProperty(
                "Layers",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.FlattenHierarchy)
                ?.GetValue(model) as System.Collections.IEnumerable;
            if (layers is null) continue;

            var typed = layers.Cast<ILayer<double>>().ToList();
            if (typed.Count == 0) continue;

            // MEASUREMENT 1 - which contracts IGNORE isBatched. A contract that answers identically
            // for both is either genuinely batch-agnostic or has simply not implemented the flag, and
            // only the second kind can produce the fold's 46 disagreements. Grepping for contracts that
            // EMIT Same(Batch) first gives 121, but that counts declarations, not behaviour - the
            // question is how many actually ANSWER differently, which only evaluating both can say.
            foreach (var layer in typed)
            {
                if (layer is not IShapeContract sc) continue;

                int[] inShape;
                try { inShape = layer.GetInputShape(); } catch { continue; }
                if (inShape is not { Length: > 0 }) continue;

                IReadOnlyList<OutputAxisContract>? batched, perSample;
                try
                {
                    batched = sc.OutputAxesFor(inShape.Length, isBatched: true);
                    perSample = sc.OutputAxesFor(inShape.Length, isBatched: false);
                }
                catch { continue; }

                if (batched is null || perSample is null) continue;

                string name = layer.GetType().Name;
                bool leadsWithBatch = batched.Count > 0
                    && batched[0].Axis == TensorAxis.Batch
                    && batched[0].Relation.Kind == AxisRelation.Form.Same;

                bool identical = batched.Count == perSample.Count;
                if (identical)
                {
                    for (int k = 0; k < batched.Count; k++)
                    {
                        if (batched[k].Axis != perSample[k].Axis) { identical = false; break; }
                    }
                }

                if (leadsWithBatch && identical) ignoresFlagLeadingBatch.Add(name);
                else if (!identical) honoursFlag.Add(name);
            }

            var shadow = LayerContractValidator.CompareContractsToResolvedShapes(typed);
            agreed += shadow.Agreed;
            declined += shadow.Declined;
            unresolved += shadow.Unresolved;
            foreach (var d in shadow.Disagreements) disagreements.Add($"{t.Name} {d}");

            // MEASUREMENT 2 - why layers go unresolved. "unresolved" is neither agreement nor
            // disagreement; it is surface the comparison never reached, so a clean result says nothing
            // about it. Splitting it into "no contract at all" versus "contract present but the shape
            // never became concrete" says whether the gap is annotation or resolution.
            foreach (var layer in typed)
            {
                if (layer is null) continue;
                string name = layer.GetType().Name;

                if (layer is not IShapeContract)
                {
                    noContractByType[name] = noContractByType.TryGetValue(name, out var c0) ? c0 + 1 : 1;
                    continue;
                }

                int[] inS, outS;
                try { inS = layer.GetInputShape(); outS = layer.GetOutputShape(); }
                catch
                {
                    unresolvedByType[name] = unresolvedByType.TryGetValue(name, out var c1) ? c1 + 1 : 1;
                    continue;
                }

                bool concrete = inS is { Length: > 0 } && outS is { Length: > 0 }
                    && inS.All(d => d > 0) && outS.All(d => d > 0);
                if (!concrete)
                {
                    unresolvedByType[name] = unresolvedByType.TryGetValue(name, out var c2) ? c2 + 1 : 1;
                }
            }


            (model as IDisposable)?.Dispose();
        }

        _out.WriteLine($"models: {constructed} constructed, {resolved} chain-resolved, {skipped.Count} skipped");
        _out.WriteLine($"per-layer: agreed={agreed} declined={declined} unresolved={unresolved} disagreed={disagreements.Count}");
        foreach (var d in disagreements.Take(40)) _out.WriteLine($"  {d}");
        if (disagreements.Count > 40) _out.WriteLine($"  ... and {disagreements.Count - 40} more");


        _out.WriteLine("");
        _out.WriteLine("=== MEASUREMENT 1: does the contract honour isBatched? ===");
        _out.WriteLine($"leads with Same(Batch) and IGNORES the flag : {ignoresFlagLeadingBatch.Count} types");
        foreach (var n in ignoresFlagLeadingBatch.OrderBy(x => x, StringComparer.Ordinal).Take(60))
        {
            _out.WriteLine($"  {n}");
        }
        if (ignoresFlagLeadingBatch.Count > 60) _out.WriteLine($"  ... and {ignoresFlagLeadingBatch.Count - 60} more");
        _out.WriteLine($"answers DIFFERENTLY per flag (already honours) : {honoursFlag.Count} types");
        foreach (var n in honoursFlag.OrderBy(x => x, StringComparer.Ordinal)) _out.WriteLine($"  {n}");

        _out.WriteLine("");
        _out.WriteLine("=== MEASUREMENT 2: what is 'unresolved' made of? ===");
        _out.WriteLine($"contract present, shape never concrete : {unresolvedByType.Values.Sum()} instances "
            + $"across {unresolvedByType.Count} types");
        foreach (var kv in unresolvedByType.OrderByDescending(k => k.Value).Take(20))
        {
            _out.WriteLine($"  {kv.Value,4}  {kv.Key}");
        }
        _out.WriteLine($"no IShapeContract at all              : {noContractByType.Values.Sum()} instances "
            + $"across {noContractByType.Count} types");
        foreach (var kv in noContractByType.OrderByDescending(k => k.Value).Take(20))
        {
            _out.WriteLine($"  {kv.Value,4}  {kv.Key}");
        }

        // NOT VACUOUS. A sweep that constructs nothing also reports zero disagreements, and that is
        // the failure mode this assertion exists to catch - the same one that revealed 13 dead
        // [ElementWiseShape] contracts when the layer sweep started printing its exercised count.
        Assert.True(constructed > 0, "no models were constructed - the harness, not the contracts, is broken");
        Assert.True(
            agreed + declined > 0,
            $"no layer contract was exercised in any assembled chain (unresolved={unresolved}). "
            + "The comparison ran but compared nothing, so a clean result would mean nothing.");

        // FAILING, because the backlog cleared. It entered report-only at 4 disagreements against 607
        // agreements and reached 0 once all four were fixed, which is the point of that ladder. From
        // here a contract that stops matching what the layer resolves fails this test by name.
        //
        // The four were NOT one root cause, which is why each was diagnosed before being touched:
        //   FlattenLayer, GlobalPoolingLayer  contract read rank 3 as batched when chain resolution
        //                                     passes PER-SAMPLE shapes; fixed with BatchOptional.
        //   GRULayer                          the opposite - OnFirstForward ignored _returnSequences
        //                                     and declared a sequence the forward does not return.
        //   DigitCapsuleLayer                 declared its output in THREE places; two disagreed.
        // FAILING, because the backlog cleared. It entered report-only at 4 disagreements against 607
        // agreements and reached 0. From here a contract that stops matching what its layer resolves
        // fails this test by name, in a real assembled chain rather than on a synthesized probe.
        //
        // The four were four DIFFERENT causes, each diagnosed before being touched:
        //   GlobalPoolingLayer  read rank 3 as [Batch, Time, Features] when chain resolution means
        //                       per-sample [C,H,W], and dropped the rank that pooling keeps.
        //   GRULayer            the RESOLUTION was wrong: OnFirstForward ignored _returnSequences and
        //                       declared a sequence the forward does not return.
        //   DigitCapsuleLayer   declared its output in THREE places; two of them disagreed.
        //   FlattenLayer        needed the isBatched flag. It collapses everything after the batch, so
        //                       its answer depends on whether a leading axis IS one - [3,72] for a
        //                       batched tensor, [1568] for a per-sample shape. Rank alone could not
        //                       say, so whichever reading it picked was wrong for the other caller.
        Assert.True(
            disagreements.Count == 0,
            $"{disagreements.Count} layer(s) have a shape contract that disagrees with the shape the "
            + "imperative resolution concluded, in a real assembled chain. A contract that is wrong "
            + "HERE is wrong where it ships." + Environment.NewLine
            + string.Join(Environment.NewLine, disagreements.Take(40)));
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
