using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// A model's own declared input layout must share a rank with its first layer's.
/// </summary>
/// <remarks>
/// <para>
/// BOTH SIDES ARE DECLARATIONS, WHICH IS WHAT MAKES THIS SOUND. An earlier attempt compared the
/// ARCHITECTURE's input shape against the first layer and produced 179 false positives, because the
/// architecture is supplied by the caller: the harness handed every model a 3-D image and then
/// reported TTS models for opening with an EmbeddingLayer that takes rank 1-2. That check was
/// measuring its own input. This one reads <c>[TensorLayout(Direction = Input)]</c> off the MODEL
/// type and off the LAYER type, so neither operand comes from the test.
/// </para>
/// <para>
/// It closes the boundary nothing else looks at.
/// <c>NeuralNetworkBase.ReportLayerContractMismatches</c> validates layer-to-layer ADJACENCY - it
/// returns early when <c>Layers.Count &lt; 2</c>, so it only inspects pairs INSIDE the stack. The
/// conformance sweep checks the far end, the model's output contract against what Predict returns.
/// Neither looks at where the declared input meets the first layer, and
/// <c>TensorLayoutAttribute.AcceptsRank</c> was never called from the model side at all.
/// </para>
/// <para>
/// Models that declare no input layout are COUNTED, not failed - that count is the backlog, the same
/// shape of ladder ADNSHAPE006 climbed from 85 to 0. A model that declares one and contradicts its
/// own first layer is a defect, because Predict cannot succeed on the input the model advertises.
/// </para>
/// </remarks>
public class ModelInputMeetsFirstLayerTests
{
    private readonly ITestOutputHelper _out;
    public ModelInputMeetsFirstLayerTests(ITestOutputHelper output) => _out = output;

    private const int Extent = 8;

    /// <summary>
    /// Per-model ceiling on construction, so ONE pathological constructor cannot consume the sweep.
    /// </summary>
    /// <remarks>
    /// MEASURED, NOT GUESSED AT. Without this the sweep ran past ten minutes and was killed having
    /// printed nothing, which is indistinguishable from a pass - the same failure
    /// <c>ParameterCountContractTests</c> hit ("died with 'Test host process crashed' and produced
    /// NOTHING"), and this is its remedy, reused rather than reinvented. A model that exceeds the
    /// ceiling is REPORTED as skipped with that reason; it is never quietly dropped, and the
    /// exercised-count guard below still refuses a run that measured nothing.
    /// </remarks>
    private static readonly TimeSpan ConstructionTimeout = TimeSpan.FromSeconds(10);

    /// <summary>Above this, a model is named in the report as a contributor to the sweep's wall clock.</summary>
    private static readonly TimeSpan SlowModelThreshold = TimeSpan.FromSeconds(1);

    // async solely to make Timeout effective: xunit rejects it outright on a synchronous test
    // ("Tests marked with Timeout are only supported for async tests"), which is why the sibling
    // ParameterCountContractTests is shaped the same way. The Yield is what makes the method genuinely
    // async; the sweep itself is synchronous and stays that way.
    [Fact(Timeout = 1800000)]
    public async System.Threading.Tasks.Task AModelsDeclaredInputRankIsOneItsFirstLayerAccepts()
    {
        await System.Threading.Tasks.Task.Yield();

        // AUTO-FLUSHED, because the interesting run is the one that dies. Constructing ~900 models in
        // one process can take the host down from inside a single constructor - an AccessViolation or
        // StackOverflow cannot be caught here - and then xunit's captured output is lost entirely. The
        // last line of this file names the model that was in flight when that happened, which is the
        // only way to find a hanging constructor without guessing.
        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "model-input-boundary.txt");
        System.IO.StreamWriter? log = null;
        try
        {
            log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true };
        }
        catch (Exception ex)
        {
            // REPORTED, not swallowed: a silently unopenable log makes every later write go nowhere,
            // and the run then looks like one that simply found nothing.
            _out.WriteLine($"NOTE: could not open {logPath} ({ex.GetType().Name}: {ex.Message}); "
                           + "the output below is the only record of this run.");
        }

        using var _logHandle = log;

        var models = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition
                        && t.GetGenericArguments().Length == 1
                        && DerivesFromNeuralNetworkBase(t))
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        int checkedCount = 0, modelUndeclared = 0, layerUndeclared = 0;
        var mismatched = new List<string>();
        var exempted = new List<string>();
        var skipped = new List<string>();
        var slow = new List<(string Name, TimeSpan Elapsed)>();

        foreach (var open in models)
        {
            Type closed;
            try { closed = open.MakeGenericType(typeof(double)); }
            catch { continue; }

            // The model's OWN declaration, read from the open generic type so no instance is needed
            // for this half - inherited attributes included, since a family base declares for all.
            var modelRanks = AcceptedRanks(open);
            if (modelRanks.Count == 0) { modelUndeclared++; continue; }

            // The model's OWN statement that its stack does not see Predict's input. Chronos tokenizes
            // first - Forward(Tokenize(input)) - so its EmbeddingLayer receives token indices and the two
            // declarations describe different tensors. Read off the OPEN type and BEFORE construction:
            // an exemption that needed an instance would silently lapse the day the model stopped
            // constructing, and the flag is a property of the type, not of the numeric type it closes
            // over. Reasons are printed, so an exemption is visible rather than absent from the total.
            var exemption = open.GetCustomAttributes(typeof(PreprocessesInputAttribute), inherit: true)
                                .Cast<PreprocessesInputAttribute>().FirstOrDefault();
            if (exemption is not null)
            {
                exempted.Add($"{open.Name}: {exemption.Reason}");
                continue;
            }

            log?.WriteLine($"[constructing] {open.Name}");

            // TIMED PER MODEL, because "764 constructions at 25 minutes" does not say whether the cost
            // is spread evenly or concentrated in a handful of constructors - and those two diagnoses
            // call for opposite fixes. Only models over the threshold are logged, so the interesting
            // lines are not buried under 700 fast ones.
            var watch = System.Diagnostics.Stopwatch.StartNew();

            object? model = null;
            try
            {
                if (!TryConstruct(closed, out model, out string? failure))
                {
                    skipped.Add($"{open.Name}: {failure}");
                    // Logged with the REASON, because the reasons are not interchangeable: a model with
                    // no architecture constructor costs nothing, while one that exceeds the ceiling
                    // costs the ceiling plus its drain. Both land in "skipped", and only this line
                    // distinguishes an untestable model from the reason the sweep takes as long as it
                    // does. Measured at 25m08s over 764 constructions - ~2s each against
                    // DefaultConstructionTests' ~0.01s - so the distribution here is the lead.
                    log?.WriteLine($"[skipped] {open.Name}: {failure}");
                    continue;
                }

                var first = FirstLayer(model);
                if (first is null) { skipped.Add($"{open.Name}: no layers"); continue; }

                var layerRanks = AcceptedRanks(first.GetType());
                if (layerRanks.Count == 0) { layerUndeclared++; continue; }

                checkedCount++;
                if (modelRanks.Overlaps(layerRanks)) continue;

                // Written to the log AS FOUND, not only in the summary below: the findings are the
                // deliverable, and a run that dies part-way should still hand over what it proved.
                string finding = $"{open.Name}: declares input rank(s) [{Fmt(modelRanks)}] but its first "
                    + $"layer {first.GetType().Name} accepts [{Fmt(layerRanks)}]";
                mismatched.Add(finding);
                log?.WriteLine($"MISMATCH {finding}");
            }
            catch (Exception ex) { skipped.Add($"{open.Name}: {Unwrap(ex).GetType().Name}"); }
            finally
            {
                (model as IDisposable)?.Dispose();

                // In the FINALLY so every path is measured - including the `continue`s above and a
                // throwing constructor. Timing only the happy path would have understated exactly the
                // models most likely to be slow.
                watch.Stop();
                if (watch.Elapsed >= SlowModelThreshold)
                {
                    slow.Add((open.Name, watch.Elapsed));
                    log?.WriteLine($"[slow] {open.Name}: {watch.Elapsed.TotalSeconds:0.0}s");
                }
            }
        }

        _out.WriteLine($"models checked                        : {checkedCount}");
        _out.WriteLine($"model declares no input layout (backlog): {modelUndeclared}");
        _out.WriteLine($"first layer declares no input layout  : {layerUndeclared}");
        _out.WriteLine($"exempt: [PreprocessesInput]            : {exempted.Count}");
        _out.WriteLine($"MISMATCHED                            : {mismatched.Count}");
        _out.WriteLine($"skipped                               : {skipped.Count}");

        // THE WALL CLOCK, ATTRIBUTED. A sweep whose cost is invisible is one nobody can put in a CI
        // lane with confidence, and "it takes 25 minutes" is not actionable while "these N models take
        // 24 of them" is. Printed slowest-first with the share they account for.
        var slowTotal = TimeSpan.Zero;
        foreach (var s in slow) slowTotal += s.Elapsed;
        _out.WriteLine($"models over {SlowModelThreshold.TotalSeconds:0}s                     : {slow.Count}"
                       + $" (accounting for {slowTotal.TotalSeconds:0}s)");
        foreach (var s in slow.OrderByDescending(x => x.Elapsed).Take(40))
            _out.WriteLine($"  SLOW: {s.Name} {s.Elapsed.TotalSeconds:0.0}s");
        foreach (var e in exempted) _out.WriteLine($"  EXEMPT: {e}");
        foreach (var m in mismatched) _out.WriteLine($"  MISMATCH: {m}");

        // Assert the EXERCISED count. A run where nothing constructed would otherwise pass while
        // proving nothing - the vacuous-sweep failure mode that hid 13 dead [ElementWiseShape]
        // contracts, and that a previous version of this very check tripped over (0 checked, 907
        // "no input shape", caught only because the guard was here).
        Assert.True(checkedCount > 0,
            "no model was checked, so this proves nothing about the declared-input boundary");

        // A RATCHET, NOT A GATE - YET. Measured 2026-08-10: 764 constructed, 24 mismatched, 2 exempt.
        // Failing on 24 would redden the branch against work that is unfinished rather than wrong,
        // which is the exact reason ADNSHAPE006 entered as a suppressed warning at 85 of ~270 layers
        // and was promoted to Error only once it reached zero. Same ladder: the count is printed every
        // run so the backlog stays readable, and REPLACING THIS WITH Assert.Empty IS THE PROMOTION.
        //
        // The 24 fall into three clusters, and each needs a per-model judgement about WHICH SIDE is
        // wrong - never a bulk sweep:
        //   * 14 audio models declaring rank 2 against a rank-3 conv (ACEStep, ASTModel,
        //     AudioSuperResolution, ContextNet, MusicSourceSeparator, NeMoCitrinet, OpenVoiceV2,
        //     PANNs, PANNsModel, RoomImpulseResponse, SileroVad, Wav2Vec2Model, GraFPrint,
        //     SpeechEmotionRecognizer). Some are like CLAPModel - a front end converts before the
        //     stack, so [PreprocessesInput] is the honest answer; others fold input straight in, and
        //     for those one of the two declarations is simply wrong.
        //   * 4 document / vision-language models declaring rank 4 against a rank-1/2 EmbeddingLayer
        //     (LayoutLMv3, LiLT, PICK, plus Florence2 against a rank-3 encoder) - the tokenize-first
        //     shape, like Chronos.
        //   * 6 3-D / point-cloud models declaring rank 4 against rank-2/3 attention (GPT4Point, LEOVL,
        //     PointLLM, SceneLLM, ThreeDGraphLLM, ThreeDLLM).
        // Distinguishing the two causes requires reading each PredictCore, which is why this is worked
        // model by model. AST was WRONGLY assumed to belong here and does not: its first layer is a
        // DenseLayer accepting ranks 1-4, so the inherited rank 2 already overlaps.
        Assert.True(mismatched.Count <= 24,
            $"the declared-input mismatch backlog GREW to {mismatched.Count} (was 24). A new model "
            + "declares an input rank its own first layer rejects, so Predict cannot succeed on the "
            + "input that model advertises."
            + Environment.NewLine + string.Join(Environment.NewLine, mismatched));
    }

    /// <summary>Ranks a type's own <c>[TensorLayout(Input)]</c> declarations accept.</summary>
    private static HashSet<int> AcceptedRanks(Type type)
    {
        var ranks = new HashSet<int>();
        foreach (var l in type.GetCustomAttributes(typeof(TensorLayoutAttribute), inherit: true)
                              .Cast<TensorLayoutAttribute>()
                              .Where(a => a.Direction == TensorLayoutDirection.Input
                                          && a.Axes is { Length: > 0 }))
        {
            // AcceptsRank is the authority - it already encodes BatchOptional, so a
            // [Batch?, Channels, Height, Width] layout accepts both 3 and 4 and neither is
            // reconstructable from Axes.Length alone.
            for (int r = 1; r <= 6; r++)
                if (l.AcceptsRank(r)) ranks.Add(r);
        }
        return ranks;
    }

    private static string Fmt(IEnumerable<int> ranks) => string.Join(", ", ranks.OrderBy(r => r));

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
        }
        return false;
    }

    private static LayerBase<double>? FirstLayer(object model)
    {
        try
        {
            var layers = (model as NeuralNetworkBase<double>)?.Layers;
            if (layers is null) return null;
            foreach (var l in layers) if (l is LayerBase<double> lb) return lb;
            return null;
        }
        catch { return null; }
    }

    /// <summary>
    /// Constructs the model under a per-model time ceiling, reporting WHY when it cannot.
    /// </summary>
    /// <remarks>
    /// The bound is the point. Invoking the constructor inline let a single slow constructor stall the
    /// whole sweep with nothing printed. Note the DRAIN after a timeout rather than an abandon:
    /// <c>Task.Wait</c> only stops waiting, so an abandoned constructor keeps burning CPU and its cost
    /// is then charged to every model after it - which is what made a sibling sweep blame a different
    /// model depending on machine load.
    /// </remarks>
    private static bool TryConstruct(Type closed, out object? instance, out string? failure)
    {
        instance = null;
        failure = null;

        var ctor = closed.GetConstructors().FirstOrDefault(c =>
        {
            var ps = c.GetParameters();
            return ps.Length > 0
                && ps[0].ParameterType == typeof(NeuralNetworkArchitecture<double>)
                && ps.Skip(1).All(p => p.HasDefaultValue);
        });
        if (ctor is null) { failure = "no architecture constructor"; return false; }

        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
            inputDepth: 3, inputHeight: Extent, inputWidth: Extent, outputSize: 4);

        var pars = ctor.GetParameters();
        var args = new object?[pars.Length];
        args[0] = architecture;
        for (int i = 1; i < pars.Length; i++) args[i] = pars[i].DefaultValue;

        object? built = null;
        var task = System.Threading.Tasks.Task.Run(() => built = ctor.Invoke(args));
        if (!task.Wait(ConstructionTimeout))
        {
            task.Wait(TimeSpan.FromSeconds(20));
            failure = $"construction exceeded {ConstructionTimeout.TotalSeconds:0}s";
            return false;
        }

        if (task.Exception is not null)
        {
            failure = Unwrap(task.Exception).GetType().Name;
            return false;
        }

        instance = built;
        if (instance is null) { failure = "constructor returned null"; return false; }
        return true;
    }

    /// <summary>Unwraps reflection and task wrappers so a report names the real exception.</summary>
    private static Exception Unwrap(Exception ex)
        => ex switch
        {
            System.Reflection.TargetInvocationException { InnerException: not null } tie => Unwrap(tie.InnerException),
            AggregateException { InnerException: not null } ae => Unwrap(ae.InnerException),
            _ => ex,
        };
}
