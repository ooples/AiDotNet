using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.NeMo;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.VisionLanguage.Foundational;
using AiDotNet.VisionLanguage.Generative;
using AiDotNet.VisionLanguage.InstructionTuned;
using AiDotNet.VisionLanguage.Unified;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Sweeps every model through DeepCopy to find which ones lose construction state.
/// </summary>
/// <remarks>
/// <para>
/// The point is the coverage number, not a pass, for the same reason the layer sweep exists: a
/// green build says nothing about whether a clone carries what it was built with.
/// </para>
/// <para>
/// DeepCopy now rebuilds a model from its generated construction plan and generated state payload.
/// Concrete models have no factory, clone, or serialization override to keep synchronized with
/// their constructors. What this measures is how many models survive that shared round trip with
/// their parameter count and architecture intact, and stay independent of the original afterwards.
/// </para>
/// <para>
/// Models that cannot be constructed from a standard architecture are reported separately from
/// models that fail to clone. Conflating a shortfall in this harness with a shortfall in the
/// feature is how a coverage number stops meaning anything.
/// </para>
/// </remarks>
public class AllModelsCloneTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes a new instance of the <see cref="AllModelsCloneTests"/> class.</summary>
    /// <param name="output">Sink for the coverage summary.</param>
    public AllModelsCloneTests(ITestOutputHelper output) => _output = output;

    /// <summary>Returned by <see cref="Attempt"/> for a model this harness cannot build.</summary>
    private const string SkipMarker = "\0skip";

    /// <summary>How many shards the model list is split across.</summary>
    /// <remarks>
    /// The sweep is split rather than given a longer clock. One run over every model needed more
    /// than the 45-minute ceiling a shard gets, and a single test that cannot finish inside its
    /// budget reports nothing at all -- the 15-minute attempt died at the letter C. Sharding also
    /// lets the runner work on them in parallel, so the wall-clock is one shard, not the sum.
    /// </remarks>
    private const int ShardCount = 24;

    private string ReportPath =
        System.IO.Path.Combine(System.IO.Path.GetTempPath(), "aidotnet-model-clone-sweep.txt");

    private readonly List<string> cloned = new();
    private readonly List<string> failed = new();
    private readonly List<string> notConstructed = new();
    private readonly List<string> budgetExceeded = new();

    /// <summary>Maximum observation window for one model in this diagnostic sweep.</summary>
    /// <remarks>
    /// Exceeding this budget is deliberately not called a hang. The number is a property of this
    /// harness and runner capacity; changing it changes the count without changing model behavior.
    /// </remarks>
    private static readonly TimeSpan PerModelProbeBudget = TimeSpan.FromSeconds(20);

    /// <summary>
    /// How long to let a single over-budget attempt finish before giving up on the sweep entirely.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="System.Threading.Tasks.Task.Wait(TimeSpan)"/> does NOT cancel. An attempt that
    /// exceeds <see cref="PerModelProbeBudget"/> keeps running and keeps its model rooted, so
    /// starting the next attempt on top of it stacks a second model-sized allocation, and the one
    /// after that a third. Shard 13 hit the budget thirteen times and the pile-up, not any single
    /// model, is what exhausted the heap: four attempts threw <c>OutOfMemoryException</c> long after
    /// their own LIMIT line, and the test host then died outright with no report written at all.
    /// </para>
    /// <para>
    /// So the budget now bounds CLASSIFICATION only, never concurrency: at most one abandoned
    /// attempt is ever in flight, and it must finish before the next model starts. If it will not
    /// finish even within this much larger drain window it is a genuine stall. The assertion must
    /// fire before CI's five-minute inactivity watchdog so the failure keeps the model name.
    /// </para>
    /// </remarks>
    private static readonly TimeSpan AbandonedAttemptDrainBudget = TimeSpan.FromMinutes(2);

    // A timed-out Task can outlive its [InlineData] invocation and still own a materialized model.
    // Keep that process-wide lifetime visible to later theory rows so they never stack another model
    // on top of an attempt that this test can no longer cancel.
    private static readonly object OutstandingAttemptSync = new();
    private static System.Threading.Tasks.Task? OutstandingAttempt;
    private static string OutstandingAttemptName = string.Empty;

    /// <summary>
    /// Models whose original never materialized under the probe, so the two sides are not comparable.
    /// </summary>
    /// <remarks>
    /// A HARNESS LIMIT, kept out of the failure count for the same reason <see cref="notConstructed"/>
    /// is. The probe below is a 1x4 tensor; a vision-language model refuses it, so the original sits
    /// at whatever its constructor sized while DeepCopy returns a copy that has materialized. The
    /// sweep read that as "BlipNeuralNetwork: 23441664 parameters against 768" and counted a
    /// failure, when what it had actually measured was one side resolved and the other not. Nothing
    /// is known about those models' cloning either way until the harness can drive them with an
    /// input they accept -- which is a statement about this test, not about the copy.
    /// </remarks>
    private readonly List<string> unresolved = new();

    /// <summary>Appends one line to the progress file as the sweep runs.</summary>
    /// <remarks>
    /// Written as it goes rather than at the end. The first two runs timed out having written
    /// nothing, which said only that the sweep was slow and not which model it was stuck on.
    /// </remarks>
    private void Note(string line)
    {
        lock (ReportPath) System.IO.File.AppendAllLines(ReportPath, new[] { line });
    }

    private string Fail(Type open, string why)
    {
        var line = $"{open.Name}: {why}";
        Note($"FAIL  {line}");
        return line;
    }

    /// <summary>Reports how many models survive a DeepCopy with their construction state.</summary>
    /// <returns>A task representing the test.</returns>
    // ONE SHARD, WELL INSIDE THE CEILING. The single-test version died at CTCSegmentation after 15
    // minutes -- 104 of 200+ models -- and raising its clock past the 45-minute shard ceiling would
    // only have moved where it died. Each shard now takes 1/24th of the list; 8 shards still overran.
    [Theory(Timeout = 900000)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(6)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(10)]
    [InlineData(11)]
    [InlineData(12)]
    [InlineData(13)]
    [InlineData(14)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(18)]
    [InlineData(19)]
    [InlineData(20)]
    [InlineData(21)]
    [InlineData(22)]
    [InlineData(23)]
    public async System.Threading.Tasks.Task EveryModel_ReportsWhetherItSurvivesADeepCopy(int shard)
    {
        await System.Threading.Tasks.Task.Yield();

        // ONE SHARD PER PROCESS. `dotnet test --filter` cannot reliably address an individual
        // [InlineData] case, so parallel runs each executed every shard and overwrote one
        // another's report. The runner sets AIDOTNET_SWEEP_SHARD and the shards that do not
        // match return immediately, so each process does 1/24th of the work and owns one file.
        // Unset means run every shard, which is what a plain `dotnet test` should still do.
        var only = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_SHARD");
        if (!string.IsNullOrEmpty(only) && int.TryParse(only, out var wanted) && wanted != shard) return;
        EnsureNoPriorAttemptIsStillRunning();

        // AIDOTNET_SWEEP_DIR keeps the reports off the system drive, which nine parallel runs
        // filled to zero bytes free.
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_DIR");
        if (string.IsNullOrEmpty(dir)) dir = System.IO.Path.GetTempPath();
        System.IO.Directory.CreateDirectory(dir);

        ReportPath = System.IO.Path.Combine(dir, $"aidotnet-model-clone-sweep-{shard}.txt");

        var all = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToList();

        // STRIDED, not carved into contiguous blocks. The expensive models cluster by name (the
        // whole BLIP/Blip2/BLIP3 family lands together), so contiguous blocks would put every slow
        // one in the same shard and leave that shard timing out while the others idle.
        var candidates = all.Where((_, i) => i % ShardCount == shard).ToList();

        // Optional local/CI diagnostic selector. It is deliberately applied after sharding, so an
        // invalid model/shard pairing runs zero candidates instead of silently testing another row.
        var onlyModel = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_MODEL");
        if (!string.IsNullOrWhiteSpace(onlyModel))
            candidates = candidates.Where(t => string.Equals(t.Name, onlyModel, StringComparison.Ordinal)).ToList();

        // APPENDED AS IT GOES. The first run timed out at 15 minutes with nothing written, which
        // told us only that the sweep is slow -- not which model it was on. A progress file costs
        // nothing and turns a timeout into a result plus a culprit.
        System.IO.File.WriteAllText(ReportPath, string.Empty);


        // The single over-budget attempt still running, if any, and the model it belongs to.
        System.Threading.Tasks.Task? draining = null;
        string drainingName = string.Empty;

        // The abandoned attempt's RESULT travels with the task. Without it the drain below could
        // only clear the task and move on, so an attempt that went over budget and then failed was
        // recorded in budgetExceeded and nowhere else -- and budgetExceeded is not asserted on,
        // while failed is. A slow clone failure could therefore pass this test.
        string?[]? drainingSlot = null;

        foreach (var open in candidates)
        {
            Type closed;
            try
            {
                closed = open.MakeGenericType(typeof(float));
            }
            catch (Exception)
            {
                notConstructed.Add($"{open.Name}: constraints reject float"); Note($"skip  {open.Name}: constraints reject float");
                continue;
            }

            // Let any previously abandoned attempt finish BEFORE allocating another model on top of
            // it. This is the whole pile-up fix; see AbandonedAttemptDrainBudget.
            if (draining is not null)
            {
                bool drained;
                bool faulted = false;
                try
                {
                    drained = draining.Wait(AbandonedAttemptDrainBudget);
                }
                catch (Exception)
                {
                    faulted = true;
                    // The attempt faulted after we stopped watching it. It has already been counted
                    // as over-budget and its memory is released either way; observing the exception
                    // here just keeps it from resurfacing as an unobserved task exception.
                    drained = true;
                }

                if (drained)
                {
                    // Read the result we stopped waiting for. Every attempt is classified exactly
                    // once, here or at the point it completed in time.
                    if (faulted)
                        failed.Add(
                            $"{drainingName}: attempt faulted after exceeding its observation budget");
                    else if (drainingSlot is not null)
                        Classify(drainingName, drainingSlot[0]);
                }
                else
                {
                    Note($"STOP  sweep at {drainingName}: abandoned attempt would not drain");
                    Assert.True(
                        drained,
                        $"{drainingName}: clone attempt was still running "
                            + $"{AbandonedAttemptDrainBudget.TotalMinutes:0} minutes after its observation budget; "
                            + "the process-wide guard will prevent later theory rows from materializing another model.");
                    return;
                }

                ClearOutstandingAttempt(draining);
                draining = null;
                drainingName = string.Empty;
                drainingSlot = null;

                // The drained attempt's model is unreachable now; reclaim it before the next one
                // allocates rather than letting several generations of them coexist.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }

            // BEFORE the attempt, so a shard-level timeout still names the model it was observing.
            Note($"try   {open.Name}");

            // A BUDGET PER MODEL. This keeps one slow or stuck attempt from consuming the whole
            // shard, but it is only an observation budget. It cannot distinguish a true deadlock
            // from valid work that needs more time on this runner, so report it separately and do
            // not turn the budget-sensitive count into a claimed hang rate.
            var slot = new string?[1];
            var work = System.Threading.Tasks.Task.Run(() => slot[0] = Attempt(open, closed));

            if (!work.Wait(PerModelProbeBudget))
            {
                budgetExceeded.Add(
                    $"{open.Name}: exceeded {PerModelProbeBudget.TotalSeconds:0}s observation budget");
                Note($"LIMIT {open.Name}");
                draining = work;
                drainingName = open.Name;
                drainingSlot = slot;
                TrackOutstandingAttempt(work, open.Name);
                continue;
            }

            Classify(open.Name, slot[0]);
        }

        // Don't leave a final abandoned attempt allocating underneath the report write-out.
        bool finalDrainCompleted = true;
        if (draining is not null)
        {
            bool finalFaulted = false;
            try { finalDrainCompleted = draining.Wait(AbandonedAttemptDrainBudget); }
            catch (Exception) { finalDrainCompleted = true; finalFaulted = true; }

            if (finalFaulted)
                failed.Add($"{drainingName}: attempt faulted after exceeding its observation budget");
            else if (finalDrainCompleted && drainingSlot is not null)
                Classify(drainingName, drainingSlot[0]);

            if (finalDrainCompleted)
                ClearOutstandingAttempt(draining);
        }

        // A STILL-RUNNING ATTEMPT IS NOT A REPORTABLE STATE.
        //
        // Wait returning false means the background Attempt is still going. It appends to failed,
        // notConstructed and unresolved and writes to the report file, all of which the write-out
        // below reads -- so the published counts would be a snapshot taken while they were still
        // changing, and enumerating a List<string> mid-Add can throw outright. Stop here instead of
        // reporting numbers that are not yet true. Unconditional: a completed drain passes it.
        Assert.True(
            finalDrainCompleted,
            $"{drainingName}: abandoned attempt was still running "
                + $"{AbandonedAttemptDrainBudget.TotalMinutes:0} minutes after the sweep ended; "
                + "report withheld because its counts were still being written.");

        _output.WriteLine($"model types        : {candidates.Count}");
        _output.WriteLine($"cloned OK          : {cloned.Count}");
        _output.WriteLine($"clone FAILED       : {failed.Count}");
        _output.WriteLine($"not constructed    : {notConstructed.Count} (harness limit, not a clone result)");
        _output.WriteLine($"probe did not run  : {unresolved.Count} (harness limit, not a clone result)");
        _output.WriteLine($"probe budget limit : {budgetExceeded.Count} (budget-sensitive; not a hang rate)");
        _output.WriteLine(string.Empty);

        foreach (var line in failed) _output.WriteLine($"FAIL  {line}");
        foreach (var line in notConstructed) _output.WriteLine($"skip  {line}");
        foreach (var line in unresolved) _output.WriteLine($"lazy  {line}");

        // ALSO to a file. xunit only surfaces ITestOutputHelper on a failing test or under
        // `verbosity=detailed`, and detailed logs every one of 72,000 discovered cases -- 17MB of
        // noise to read four numbers out of. A report worth running is worth being able to read.
        var report = new List<string>
        {
            $"model types        : {candidates.Count}",
            $"cloned OK          : {cloned.Count}",
            $"clone FAILED       : {failed.Count}",
            $"not constructed    : {notConstructed.Count} (harness limit, not a clone result)",
            $"probe did not run  : {unresolved.Count} (harness limit, not a clone result)",
            $"probe budget limit : {budgetExceeded.Count} (budget-sensitive; not a hang rate)",
            string.Empty,
        };
        report.AddRange(failed.Select(f => $"FAIL  {f}"));
        report.AddRange(notConstructed.Select(n => $"skip  {n}"));
        report.AddRange(unresolved.Select(u => $"lazy  {u}"));

        report.AddRange(budgetExceeded.Select(t => $"LIMIT {t}"));
        System.IO.File.WriteAllLines(ReportPath, report);

        // THE SWEEP NOW FAILS WHEN A MODEL FAILS TO CLONE. Until now this test had no assertion at
        // all: it collected failures, wrote them to a report, and returned green regardless, so a
        // model that could not be cloned was recorded and ignored. That made it documentation
        // rather than regression proof -- the same state the layer sweep was in before it was
        // fixed, and the reason its own comment warns about a 'measurement-only assertion'.
        //
        // No budget and no allowlist here, deliberately: every entry is a model that cannot be
        // copied, which is the defect this whole surface exists to prevent.
        Assert.True(
            failed.Count == 0,
            $"{failed.Count} model(s) failed cloning in shard {shard}:{Environment.NewLine}"
                + string.Join(Environment.NewLine, failed));
    }

    private static void EnsureNoPriorAttemptIsStillRunning()
    {
        System.Threading.Tasks.Task? prior;
        string priorName;
        lock (OutstandingAttemptSync)
        {
            prior = OutstandingAttempt;
            priorName = OutstandingAttemptName;
        }

        if (prior is null) return;

        bool completed = prior.IsCompleted;
        Assert.True(
            completed,
            $"{priorName}: a prior clone attempt is still running; refusing to materialize another model in this process.");

        // Observe a late fault before releasing the process-wide lifetime guard. Its owning theory
        // row already failed, so this row must not classify the same outcome a second time.
        try { prior.GetAwaiter().GetResult(); }
        catch (Exception) { }
        ClearOutstandingAttempt(prior);
    }

    private static void TrackOutstandingAttempt(System.Threading.Tasks.Task attempt, string name)
    {
        lock (OutstandingAttemptSync)
        {
            bool canReplace = OutstandingAttempt is null || OutstandingAttempt.IsCompleted;
            Assert.True(canReplace, $"{OutstandingAttemptName}: another clone attempt is already running.");
            OutstandingAttempt = attempt;
            OutstandingAttemptName = name;
        }
    }

    private static void ClearOutstandingAttempt(System.Threading.Tasks.Task attempt)
    {
        lock (OutstandingAttemptSync)
        {
            if (!ReferenceEquals(OutstandingAttempt, attempt)) return;
            OutstandingAttempt = null;
            OutstandingAttemptName = string.Empty;
        }
    }

    /// <summary>Records one completed attempt's result.</summary>
    /// <remarks>
    /// Every attempt passes through here exactly once -- whether it finished inside its observation
    /// budget or was drained afterwards -- so that no completed attempt's result goes unread.
    /// </remarks>
    private void Classify(string name, string? outcome)
    {
        if (outcome is null) cloned.Add(name);
        else if (!ReferenceEquals(outcome, SkipMarker) && outcome != SkipMarker) failed.Add(outcome);
    }

    /// <summary>Constructs, copies and checks one model. Returns null when it cloned cleanly.</summary>
    private string? Attempt(Type open, Type closed)
    {
        var model = TryConstruct(closed);
        if (model is null)
        {
            notConstructed.Add($"{open.Name}: no constructor takes a standard architecture");
            Note($"skip  {open.Name}: not constructible");
            return SkipMarker;
        }

        object? copiedModel = null;
        try
        {
            var probed = Resolve(model);
            var before = model.ParameterCount;
            copiedModel = model.DeepCopy();
            var copy = copiedModel as NeuralNetworkBase<float>;

            if (copy is null) return Fail(open, "DeepCopy returned null");
            if (copy.GetType() != closed) return Fail(open, $"copy is {copy.GetType().Name}");

            Resolve(copy);

            // COMPARE THE SAME STATE. DeepCopy materializes the source as a side effect of reading
            // its parameter surface, and `before` is captured BEFORE that. Measured on
            // TimeGANGenerator: 192 before, 8640 after, copy also 8640, both sides structurally
            // identical (3 FullyConnectedLayers, 320/4160/4160). Comparing the pre-copy number
            // against the post-copy one measured materialization, not copying.
            var originalAfterCopy = model.ParameterCount;

            if (copy.ParameterCount != originalAfterCopy)
            {
                // AN UNRESOLVED ORIGINAL IS NOT A FAILED COPY. Every model in this bucket reports
                // the copy as the LARGER side -- DeepCopy materialized layers the original had left
                // lazy because the probe never ran on it. Comparing those two counts measures
                // materialization, not copying.
                if (!probed)
                {
                    unresolved.Add($"{open.Name}: probe did not run ({copy.ParameterCount} against {before})");
                    Note($"lazy  {open.Name}");
                    return SkipMarker;
                }

                return Fail(open, $"{copy.ParameterCount} parameters against {originalAfterCopy}");
            }

            if (ReferenceEquals(copy, model) || !IsIndependent(model, copy))
                return Fail(open, "copy is not independent of the original");

            Note($"ok    {open.Name}");
            return null;
        }
        catch (Exception ex)
        {
            var inner = ex.InnerException ?? ex;
            var message = inner.Message;
            return Fail(open, $"{inner.GetType().Name}: {message.Substring(0, Math.Min(90, message.Length))}");
        }
        finally
        {
            try
            {
                if (copiedModel is IDisposable disposableCopy && !ReferenceEquals(copiedModel, model))
                    disposableCopy.Dispose();
            }
            finally
            {
                model.Dispose();
            }
        }
    }

    /// <summary>Runs one probe input through the model so lazy layers materialise.</summary>
    /// <remarks>
    /// A model that cannot accept the standard probe is left as it is; the comparison below then
    /// still holds, because both sides are measured in the same unresolved state.
    /// </remarks>
    /// <returns>
    /// True only when the probe leaves a materialized or legitimately parameter-free surface.
    /// A successful Predict call can still be a no-op for an unfitted model.
    /// </returns>
    private static bool Resolve(NeuralNetworkBase<float> model)
    {
        try
        {
            var input = new Tensor<float>(new[] { 1, 4 });
            model.Predict(input);

            return model.ParameterLayout.Readiness is
                ParameterReadiness.Materialized or
                ParameterReadiness.ParameterFree or
                ParameterReadiness.ConditionalAbsent;
        }
        catch (Exception)
        {
            // Not every model predicts from a 1x4 probe. Both sides get the same treatment, but the
            // caller needs to KNOW that neither side was driven -- an unresolved original compared
            // against a materialized copy is not a result about cloning.
            return false;
        }
    }

    /// <summary>The independence check must reject a model compared against itself.</summary>
    /// <remarks>
    /// The chunked check above is INVERTED -- it asserts the original still differs from the
    /// mutated copy -- and an inverted test wired up wrongly does not fail loudly, it passes
    /// everything, leaving the sweep reporting a clean run over hundreds of models while checking
    /// nothing. A model handed to the check as its own copy shares every tensor by definition.
    /// </remarks>
    [Fact]
    public void IsIndependent_RejectsAModelComparedWithItself()
    {
        var wanted = new[] { "FeedForwardNeuralNetwork`1", "NeuralNetwork`1", "Autoencoder`1" };

        var candidates = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .Where(t => wanted.Contains(t.Name))
            .OrderBy(t => Array.IndexOf(wanted, t.Name));

        foreach (var candidate in candidates)
        {
            Type closed;
            try { closed = candidate.MakeGenericType(typeof(float)); }
            catch (Exception) { continue; }

            var model = TryConstruct(closed);
            if (model is null) continue;

            Resolve(model);

            if (model.ParameterCount is 0 or > 5_000_000)
            {
                (model as IDisposable)?.Dispose();
                continue;
            }

            try
            {
                Assert.False(
                    IsIndependent(model, model),
                    $"{candidate.Name} compared against itself was reported independent, so the "
                        + "independence check cannot detect shared storage and the sweep is vacuous");
            }
            finally
            {
                (model as IDisposable)?.Dispose();
            }

            return;
        }

        Assert.Fail("no small constructible model was available to validate the independence check");
    }

    /// <summary>Dumps the bounded option values for the type named by AIDOTNET_DUMP_OPTIONS.</summary>
    /// <remarks>
    /// Comparing what the generator produces against what a hand-written branch produced is the only
    /// way to settle which knob differs and in which DIRECTION. Guessing from property names put
    /// four wrong hypotheses into this file's history.
    /// </remarks>
    [Fact]
    public void DumpBoundedOptions()
    {
        var wanted = Environment.GetEnvironmentVariable("AIDOTNET_DUMP_OPTIONS");
        if (string.IsNullOrEmpty(wanted)) return;

        var optionType = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .FirstOrDefault(t => string.Equals(t.Name, wanted, StringComparison.Ordinal));
        Assert.True(optionType is not null, $"no type named {wanted}");

        // The table size and its DECLARING ASSEMBLY, both load-bearing. An empty table is what an
        // over-attached generator produces in a compilation with no options types, and the copy in
        // the test assembly then shadows the real one -- silently, since it still compiles and still
        // returns an object.
        _output.WriteLine(
            $"table: types={AiDotNet.Testing.ModelTestScale.BoundedTypeCount} "
                + $"knobs={AiDotNet.Testing.ModelTestScale.KnobCount} from "
                + $"{typeof(AiDotNet.Testing.ModelTestScale).Assembly.GetName().Name}");

        var generated = AiDotNet.Testing.ModelTestScale.CreateBoundedOptions(optionType!);
        var plain = Activator.CreateInstance(optionType!);

        foreach (var property in optionType!.GetProperties().OrderBy(x => x.Name))
        {
            if (property.PropertyType != typeof(int) || !property.CanRead) continue;

            var defaultValue = property.GetValue(plain);
            var boundedValue = generated is null ? null : property.GetValue(generated);
            var mark = Equals(defaultValue, boundedValue) ? "  " : "->";
            _output.WriteLine($"{mark} {property.Name}: default={defaultValue} bounded={boundedValue}");
        }
    }

    /// <summary>Probes ONE model named by AIDOTNET_PROBE_MODEL, for fast iteration.</summary>
    /// <remarks>
    /// A shard is 24 models and minutes long; chasing a single model's behaviour through it wastes
    /// most of that time. No-ops when the variable is unset, so a normal run is unaffected.
    /// </remarks>
    [Fact]
    public void ProbeOneModel()
    {
        var wanted = Environment.GetEnvironmentVariable("AIDOTNET_PROBE_MODEL");
        if (string.IsNullOrEmpty(wanted)) return;

        var open = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .FirstOrDefault(t => string.Equals(t.Name, wanted, StringComparison.Ordinal));

        Assert.True(open is not null, $"no model type named {wanted}");

        Type closed;
        try { closed = open!.MakeGenericType(typeof(float)); }
        catch (Exception) { return; }

        var started = System.Diagnostics.Stopwatch.StartNew();
        var model = TryConstruct(closed);
        var constructed = started.ElapsedMilliseconds;
        Assert.True(model is not null, $"{wanted} could not be constructed");

        Resolve(model!);
        var resolved = started.ElapsedMilliseconds;

        var copy = model!.DeepCopy() as NeuralNetworkBase<float>;
        var copied = started.ElapsedMilliseconds;

        _output.WriteLine(
            $"{wanted}: construct={constructed}ms resolve={resolved - constructed}ms "
                + $"copy={copied - resolved}ms params={model.ParameterCount}");

        Assert.True(copy is not null, "DeepCopy returned null");
    }

    /// <summary>Whether writing through one model leaves the other alone.</summary>
    private static bool IsIndependent(
        NeuralNetworkBase<float> original,
        NeuralNetworkBase<float> copy)
    {
        // Chunk-by-chunk, so nothing full-length is ever allocated. PaLI3 and SenseVoiceLarge need
        // a 2.4-2.6 GB CONTIGUOUS block for a flat vector on top of ~5 GB of resident model, and
        // could not get one: both killed the test host outright rather than failing.
        var originalChunks = new List<Tensor<float>>();
        foreach (var chunk in original.GetParameterStateChunks())
        {
            // A chunk that is not writable in place -- an fp16-resident or sparse component handing
            // out a transient snapshot -- would swallow the mutation silently, which reads as
            // "the copy never changed" and would be reported as SHARED storage: a false failure.
            if (!chunk.IsWritableInPlace) return IsIndependentViaFlatSurface(original, copy);
            originalChunks.Add(chunk.Tensor);
        }

        if (originalChunks.Count == 0) return true;

        var copyChunks = new List<Tensor<float>>();
        foreach (var chunk in copy.GetParameterStateChunks())
        {
            if (!chunk.IsWritableInPlace) return IsIndependentViaFlatSurface(original, copy);
            copyChunks.Add(chunk.Tensor);
        }

        if (copyChunks.Count != originalChunks.Count) return false;
        for (var c = 0; c < copyChunks.Count; c++)
        {
            if (copyChunks[c].Length != originalChunks[c].Length) return false;
        }

        // Write through the COPY. Independent means it ends at original + 1 everywhere; shared means
        // the same write moved the original too, so afterwards the two read equal. Comparing the two
        // live surfaces is the whole test and needs no baseline copy.
        for (var c = 0; c < copyChunks.Count; c++)
        {
            var target = copyChunks[c];
            for (var i = 0; i < target.Length; i++) target[i] = target[i] + 1.0f;
        }

        for (var c = 0; c < originalChunks.Count; c++)
        {
            var before = originalChunks[c];
            var after = copyChunks[c];
            for (var i = 0; i < before.Length; i++)
            {
                float originalValue = before[i];
                if (originalValue + 1.0f == originalValue) continue;   // cannot discriminate
                if (Math.Abs(after[i] - originalValue) <= 1e-5f) return false;
            }
        }

        return true;
    }

    /// <summary>Independence check for models whose chunks cannot be written in place.</summary>
    private static bool IsIndependentViaFlatSurface(
        NeuralNetworkBase<float> original,
        NeuralNetworkBase<float> copy)
    {
        var parameters = original.GetParameters();
        if (parameters.Length == 0) return true;

        var mutated = new Vector<float>(parameters.Length);
        for (var i = 0; i < parameters.Length; i++) mutated[i] = parameters[i] + 1.0f;

        copy.UpdateParameters(mutated);

        var after = original.GetParameters();
        if (after.Length != parameters.Length) return false;

        for (var i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - parameters[i]) > 1e-5f) return false;
        }

        return true;
    }

    private static NeuralNetworkBase<float>? TryConstruct(Type closed)
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);

        // The widest constructor whose every remaining argument is optional, so a model is built
        // through the one carrying the most configuration rather than the narrowest.
        foreach (var ctor in closed.GetConstructors()
                     .OrderByDescending(c => c.GetParameters().Length))
        {
            var formal = ctor.GetParameters();
            if (formal.Length == 0) continue;

            var args = new object?[formal.Length];
            var usable = true;

            for (var i = 0; i < formal.Length && usable; i++)
            {
                if (formal[i].ParameterType.IsInstanceOfType(architecture)) args[i] = architecture;
                else if (TryCreateBoundedArgument(closed, formal[i], out var bounded)) args[i] = bounded;
                else if (CreateBoundedOptions(formal[i].ParameterType) is { } options) args[i] = options;
                else if (formal[i].HasDefaultValue)
                {
                    // A model that takes its dimensions as plain constructor ints is reached by no
                    // options type and no name rule. MATCHA built at its full 1536-wide default and
                    // killed the host. Scaling the DECLARED DEFAULT proportionally needs no
                    // vocabulary and keeps ratios intact, so width % heads still divides.
                    args[i] = formal[i].ParameterType == typeof(int)
                        && formal[i].DefaultValue is int declaredInt
                            ? AiDotNet.Testing.ModelTestScale.ScaleDeclaredInteger(declaredInt)
                            : formal[i].DefaultValue;
                }
                else usable = false;
            }

            if (!usable || args.All(a => a is null)) continue;
            PreserveConstructorRelationships(formal, args);

            try
            {
                return ctor.Invoke(args) as NeuralNetworkBase<float>;
            }
            catch (Exception)
            {
                // A model that rejects the standard architecture is a harness limit, not a defect.
            }
        }

        return null;
    }

    /// <summary>
    /// Restores relationships that independent integer scaling can cross at its minimum floor.
    /// </summary>
    /// <remarks>
    /// Options declare this through <see cref="DimensionDivisibilityAttribute"/>. Constructors that
    /// expose the same architecture as scalar parameters have no options instance to carry that
    /// metadata, so the reflection scaffold assigns typed semantic roles and aligns the dimension
    /// after all defaults have been scaled. The role, not a model name or string operation, selects
    /// behavior; SGPT is merely the first model that proved the floor could turn 768 / 12 into 32 / 12.
    /// </remarks>
    private static void PreserveConstructorRelationships(
        IReadOnlyList<ParameterInfo> parameters,
        object?[] arguments)
    {
        var headIndices = new List<int>();
        for (int i = 0; i < parameters.Count; i++)
        {
            if (GetModelDimensionRole(parameters[i]) == ModelDimensionRole.AttentionHeadCount
                && arguments[i] is int headCount
                && headCount > 0)
            {
                headIndices.Add(i);
            }
        }

        // Multiple independent attention groups need explicit declarative pairing; guessing which
        // width belongs to which head count would be the same stringly-typed bug in another form.
        if (headIndices.Count != 1) return;
        int divisor = (int)arguments[headIndices[0]]!;

        for (int i = 0; i < parameters.Count; i++)
        {
            if (GetModelDimensionRole(parameters[i]) != ModelDimensionRole.AttentionDimension
                || arguments[i] is not int dimension
                || dimension <= 0
                || dimension % divisor == 0)
            {
                continue;
            }

            arguments[i] = AiDotNet.Testing.ModelTestScale.AlignDimensionToDivisor(
                dimension,
                divisor);
        }
    }

    private static ModelDimensionRole? GetModelDimensionRole(ParameterInfo parameter)
        => parameter.GetCustomAttribute<ModelDimensionRoleAttribute>()?.Role;

    private static bool TryCreateBoundedArgument(
        Type modelType,
        ParameterInfo parameter,
        out object? value)
    {
        // DocOwl deliberately treats a small image as its public smoke-test configuration and
        // scales the 7B-style dimensions down internally. The reflection sweep used the 448px
        // default, bypassed that contract and started materializing the production-scale graph.
        if (modelType.Name == "DocOwl`1" && parameter.Name == "imageSize")
        {
            value = 64;
            return true;
        }

        value = null;
        return false;
    }

    /// <summary>Creates explicit small configurations for paper-scale models in the clone sweep.</summary>
    /// <remarks>
    /// This sweep verifies construction-state cloning, not whether a CI runner can materialize a
    /// published foundation model. Passing null used the paper defaults: Chameleon alone exposed
    /// more than <see cref="int.MaxValue"/> parameters, while DeepSeek-VL2 built sixty 4096-wide
    /// decoder layers. Those attempts cannot be cancelled and can terminate the whole test host
    /// before any clone result is reported. Small but structurally representative options exercise
    /// the same generated construction plan and state payload without turning the test into a
    /// multi-billion-parameter allocation benchmark.
    /// </remarks>
    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.IsGenericType && b.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
            if (b.Name.StartsWith("NeuralNetworkBase", StringComparison.Ordinal)) return true;
        }

        return false;
    }

    /// <summary>Builds a size-bounded options instance for a model the sweep is about to clone.</summary>
    /// <remarks>
    /// TestScaleOptionsGenerator emits bounds for every options type, which is how NEW models get
    /// covered without anyone editing this file. The hand-written branches are kept ahead of it and
    /// are not yet redundant: deleting them regressed CSM from passing to stalling past the drain
    /// budget, because CSMOptions inherits its dimensions from CodecTtsOptions and the hand-written
    /// branch caught the whole family through IsAssignableFrom.
    ///
    /// Do not add a branch here. Extend the generator; these shrink as generated bounds are shown
    /// equivalent family by family.
    /// </remarks>

    /// <summary>Builds a size-bounded options instance for a model the sweep is about to clone.</summary>
    /// <remarks>
    /// Entirely generated. This method held one hand-written branch per model family, each naming a
    /// family's option type and poking its dimensions down, and every new large model needed
    /// another. TestScaleOptionsGenerator now emits bounds for every options type in the library, so
    /// a new model is covered without touching this file.
    ///
    /// Do not add a branch here. Extend the generator's vocabulary instead and every model sharing
    /// that property name is fixed at once.
    /// </remarks>
    private static object? CreateBoundedOptions(Type optionType)
        => AiDotNet.Testing.ModelTestScale.CreateBoundedOptions(optionType);
}
