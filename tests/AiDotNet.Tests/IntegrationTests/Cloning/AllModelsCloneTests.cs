using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
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
    /// How long a model gets to FINISH after it has already blown its observation budget.
    /// </summary>
    /// <remarks>
    /// Separate from <see cref="PerModelProbeBudget"/> on purpose. The budget answers "is this model
    /// slow" and is allowed to be tight. This answers "will it ever finish", and crossing it means
    /// the sweep stops rather than accumulating abandoned work.
    /// </remarks>
    private static readonly TimeSpan StuckModelCeiling = TimeSpan.FromMinutes(5);

    /// <summary>Per-model cost: how long it took and how much it grew the heap.</summary>
    /// <remarks>
    /// PyTorch records per-test DURATION centrally (test-times.json) and shards by it. Recording
    /// PEAK BYTES alongside is the part that catches this failure: a model can stay fast while its
    /// memory regresses, and duration alone would never show it. Committing the file turns "a shard
    /// died again" into "this model regressed at this commit".
    /// </remarks>
    private const string Tab = "\t";

    private readonly List<string> costs = new();

    /// <summary>Per-phase allocation from the most recent Attempt, for the cost manifest.</summary>
    private string LastPhases = string.Empty;

    private void RecordCost(string model, TimeSpan elapsed, long allocBefore, bool stuck)
    {
        // ALLOCATED, not GetTotalMemory. GetTotalMemory(false) reports the live heap WITHOUT
        // collecting, so its delta is dominated by whatever garbage happens to be uncollected and
        // can even read negative when a GC lands mid-window -- useless as a regression signal.
        // GetTotalAllocatedBytes is monotonic and counts everything this model's probe allocated,
        // which is the quantity that actually regresses. Cheap enough to call per model.
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - allocBefore;
        costs.Add(model + Tab + elapsed.TotalSeconds.ToString("0.00") + Tab
            + allocated.ToString() + Tab + (stuck ? "stuck" : "ok") + Tab + LastPhases);
    }

    private void WriteCostManifest(string dir, int shard)
    {
        try
        {
            var path = System.IO.Path.Combine(dir, $"aidotnet-model-clone-cost-{shard}.tsv");
            var nl = Environment.NewLine;
            System.IO.File.WriteAllText(
                path,
                "model" + Tab + "seconds" + Tab + "bytes" + Tab + "status" + Tab + "phases" + nl
                    // bytes = allocation volume in-process, PEAK working set when isolated.
                    + string.Join(nl, costs) + nl);
        }
        catch (System.IO.IOException) { }
    }

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
    /// <summary>
    /// Probes ONE model, named by AIDOTNET_PROBE_MODEL. The isolated half of the sweep.
    /// </summary>
    /// <remarks>
    /// WHY A SEPARATE PROCESS AT ALL. A probe cannot be cancelled: Task.Wait(timeout) returns but
    /// the work keeps running, holding the original AND its DeepCopy (IsIndependent needs both, so
    /// the 2x is inherent). Measured on this suite: 256 abandoned attempts across 15 shards, 10-27
    /// outstanding at once, a 49.6 GB peak, and on CI a runner shutdown with no TRX and no culprit.
    /// Bounding the loop stops repeated leaking inside one case but cannot reclaim a stuck attempt.
    /// A child process can simply be killed, which is what PyTorch relies on too -- pytest-timeout
    /// kills the process rather than abandoning in-process work.
    ///
    /// Exit code is the result: 0 probed cleanly, non-zero did not. The PARENT reads the child's
    /// peak working set after exit, which is how the cost manifest gets a real memory number
    /// instead of an allocation estimate.
    /// </remarks>
    [Fact]
    public void ProbeOneModel()
    {
        var wanted = Environment.GetEnvironmentVariable("AIDOTNET_PROBE_MODEL");
        if (string.IsNullOrEmpty(wanted)) return;   // no-op in a normal run

        var open = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && !t.IsNested)
            .Where(t => t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(DerivesFromNeuralNetworkBase)
            .FirstOrDefault(t => string.Equals(t.Name, wanted, StringComparison.Ordinal));

        Assert.True(open is not null, $"no model type named {wanted}");

        Type closed;
        try { closed = open!.MakeGenericType(typeof(float)); }
        catch (Exception) { return; }   // constraints reject float: not a clone failure

        var outcome = Attempt(open, closed);
        Assert.True(outcome is null || ReferenceEquals(outcome, SkipMarker), outcome ?? string.Empty);
    }

    /// <summary>
    /// Set AIDOTNET_SWEEP_ISOLATED=1 to probe each model in its own process. NOT YET TRUSTWORTHY.
    /// </summary>
    /// <remarks>
    /// EXPERIMENTAL, AND DELIBERATELY NOT ENABLED IN CI. Measured on shards 13 and 18, it does bound
    /// a stuck model and leaves no orphaned processes, but two defects make its output unusable as a
    /// gate today:
    ///
    /// 1. The peak it records is the `dotnet vstest` LAUNCHER, about 50 MB, not the testhost
    ///    grandchild that does the work. A model measured at 9.27 GB of allocation in-process shows
    ///    as 0.04 GB here, so the memory column is wrong by two orders of magnitude.
    /// 2. It reports false failures. CLAP and Chameleon come back "failed" in isolation and pass
    ///    in-process, so the child's exit code is not reliably the probe's verdict.
    ///
    /// It also costs about 20 s of process startup per model (25 models took 14m32s), which is
    /// affordable for a sweep that used to kill the runner but is not free.
    ///
    /// The in-process path is what carries the fix: it takes the shard from 49,586 MB to 8,646 MB
    /// and from 4 completed tests to 876. Isolation remains the correct end state, because only a
    /// child can be killed - an in-process attempt cannot be cancelled and keeps its two
    /// materialized models - but it needs the grandchild measured and the verdict plumbed through
    /// something other than the exit code before it can be turned on.
    /// </remarks>
    private static bool IsolatedMode =>
        !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_ISOLATED"));

    /// <summary>Heap cap handed to each child, as a hex byte count for DOTNET_GCHeapHardLimit.</summary>
    /// <remarks>
    /// 2 GB. A model needing more than that inside a probe is the problem being hunted, and capping
    /// it means the child dies cheaply and named instead of taking the runner with it. The CI runner
    /// has 15 GB TOTAL, so anything approaching that is already fatal in practice.
    /// </remarks>
    private const string ChildHeapCap = "80000000";

    /// <summary>
    /// Runs one model's probe in a child process and returns its outcome, duration and PEAK memory.
    /// </summary>
    /// <remarks>
    /// This is the only construct here that genuinely bounds a stuck model. An in-process budget can
    /// report that a model is slow but cannot reclaim it: Task.Wait(timeout) does not cancel, so the
    /// attempt keeps running and keeps its two materialized models. A child can simply be killed.
    ///
    /// Peak working set is read from the CHILD after it exits, so the cost manifest records what the
    /// probe actually cost rather than an allocation estimate - the number that would have named
    /// this problem on day one.
    /// </remarks>
    private (string Status, TimeSpan Elapsed, long PeakBytes) ProbeInChildProcess(string modelName, string dir)
    {
        var assembly = typeof(AllModelsCloneTests).Assembly.Location;
        var childResults = System.IO.Path.Combine(dir, "isolated", modelName);

        var psi = new System.Diagnostics.ProcessStartInfo("dotnet")
        {
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        psi.ArgumentList.Add("vstest");
        psi.ArgumentList.Add(assembly);
        psi.ArgumentList.Add("--Tests:ProbeOneModel");
        psi.ArgumentList.Add($"--ResultsDirectory:{childResults}");
        psi.Environment["AIDOTNET_PROBE_MODEL"] = modelName;
        psi.Environment["DOTNET_GCHeapHardLimit"] = ChildHeapCap;
        // The child must not recurse into isolation.
        psi.Environment["AIDOTNET_SWEEP_ISOLATED"] = string.Empty;

        var watch = System.Diagnostics.Stopwatch.StartNew();
        long peak = 0;
        using var proc = System.Diagnostics.Process.Start(psi);
        if (proc is null) return ("spawn-failed", watch.Elapsed, 0);

        // Drain the pipes, or a chatty child blocks on a full buffer and looks like a hang.
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        // SAMPLED WHILE ALIVE, not read after exit. PeakWorkingSet64 throws
        // InvalidOperationException once the process has exited, and swallowing that leaves the
        // column reading 0 for every model -- which is exactly what the first isolated run produced.
        // Polling costs nothing next to a ~20s probe and yields the number the manifest exists for.
        bool exited = false;
        var deadline = DateTime.UtcNow + StuckModelCeiling;
        while (DateTime.UtcNow < deadline)
        {
            try
            {
                proc.Refresh();
                if (proc.PeakWorkingSet64 > peak) peak = proc.PeakWorkingSet64;
            }
            catch (InvalidOperationException) { /* exited between Refresh and read */ }

            if (proc.WaitForExit(250)) { exited = true; break; }
        }

        if (!exited)
        {
            try { proc.Kill(entireProcessTree: true); } catch (Exception) { }
            return ("stuck", watch.Elapsed, peak);
        }

        return (proc.ExitCode == 0 ? "ok" : "failed", watch.Elapsed, peak);
    }

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

        // AIDOTNET_SWEEP_DIR keeps the reports off the system drive, which nine parallel runs
        // filled to zero bytes free.
        var dir = Environment.GetEnvironmentVariable("AIDOTNET_SWEEP_DIR");
        if (string.IsNullOrEmpty(dir)) dir = System.IO.Path.GetTempPath();

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

        // APPENDED AS IT GOES. The first run timed out at 15 minutes with nothing written, which
        // told us only that the sweep is slow -- not which model it was on. A progress file costs
        // nothing and turns a timeout into a result plus a culprit.
        System.IO.File.WriteAllText(ReportPath, string.Empty);


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

            // BEFORE the attempt, so a shard-level timeout still names the model it was observing.
            Note($"try   {open.Name}");

            // A BUDGET PER MODEL. This keeps one slow or stuck attempt from consuming the whole
            // shard, but it is only an observation budget. It cannot distinguish a true deadlock
            // from valid work that needs more time on this runner, so report it separately and do
            // not turn the budget-sensitive count into a claimed hang rate.
            // ISOLATED PATH. A child process is the only thing here that can be reclaimed: an
            // in-process attempt cannot be cancelled, so a stuck one keeps its two materialized
            // models for the life of the host. Killing a child costs nothing beyond that child.
            if (IsolatedMode)
            {
                var isolated = ProbeInChildProcess(open.Name, dir);
                costs.Add(open.Name + Tab + isolated.Elapsed.TotalSeconds.ToString("0.00") + Tab
                    + isolated.PeakBytes.ToString() + Tab + isolated.Status);

                switch (isolated.Status)
                {
                    case "ok":
                        cloned.Add(open.Name);
                        Note($"ok    {open.Name} (isolated)");
                        break;
                    case "stuck":
                        budgetExceeded.Add(
                            $"{open.Name}: killed after {StuckModelCeiling.TotalMinutes:0} min in an isolated probe");
                        Note($"STUCK {open.Name} (isolated, killed)");
                        break;
                    default:
                        failed.Add($"{open.Name}: isolated probe reported {isolated.Status}");
                        Note($"fail  {open.Name} (isolated)");
                        break;
                }

                continue;
            }

            string? outcome = null;
            var started = System.Diagnostics.Stopwatch.StartNew();
            long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
            var work = System.Threading.Tasks.Task.Run(() => outcome = Attempt(open, closed));

            if (!work.Wait(PerModelProbeBudget))
            {
                budgetExceeded.Add(
                    $"{open.Name}: exceeded {PerModelProbeBudget.TotalSeconds:0}s observation budget");
                Note($"LIMIT {open.Name}");

                // ABANDONING IS WHAT KILLED THE RUNNER. Task.Wait(timeout) returns false but does
                // NOT cancel the task: Attempt keeps running, and it holds the original AND its
                // DeepCopy, both fully materialized (IsIndependent needs both, so the 2x is
                // inherent). Every overrun therefore leaked two models and the next iteration
                // started another. Measured: 256 abandoned attempts across 15 shard files, 10-27
                // outstanding at once, driving a 49.6 GB peak on a 64 GB box -- and on CI,
                // "The runner has received a shutdown signal" with no TRX and no named culprit.
                //
                // So the budget now bounds ATTRIBUTION, not memory: it records that this model is
                // slow and moves on only once the work has actually finished. Within one shard case
                // that holds memory to a single attempt no matter how many models are slow.
                //
                // WHAT THIS DOES NOT FIX: a genuinely stuck attempt cannot be cancelled, so the
                // throw below fails this case while that task keeps running and keeps its two models
                // alive for the rest of the process. Twenty-three sibling [InlineData] cases still
                // execute in the same host and can each strand one. Bounding the loop is necessary
                // but not sufficient - only running each probe in its own process makes a stuck
                // model cost nothing beyond that process.
                if (!work.Wait(StuckModelCeiling))
                {
                    // Past this it is not "slow", it is stuck. Continuing would resume leaking, and
                    // a shard that dies later reports nothing at all -- so stop here, while the
                    // model that did it can still be named.
                    Note($"STUCK {open.Name} (no completion within {StuckModelCeiling.TotalMinutes:0} min)");
                    RecordCost(open.Name, started.Elapsed, allocBefore, stuck: true);
                    WriteCostManifest(dir, shard);
                    throw new Xunit.Sdk.XunitException(
                        $"{open.Name} did not complete within {StuckModelCeiling.TotalMinutes:0} minutes. "
                        + "Aborting the shard rather than abandoning the attempt, which would leak two "
                        + "materialized models and kill the runner without naming a culprit.");
                }

                RecordCost(open.Name, started.Elapsed, allocBefore, stuck: false);
                continue;
            }

            RecordCost(open.Name, started.Elapsed, allocBefore, stuck: false);

            if (outcome is null) cloned.Add(open.Name);
            else if (!ReferenceEquals(outcome, SkipMarker) && outcome != SkipMarker) failed.Add(outcome);
        }

        // Always written, not only on the stuck path: the value is the TREND across runs, and a
        // manifest that only appears on failure cannot establish one.
        WriteCostManifest(dir, shard);

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

    /// <summary>Constructs, copies and checks one model. Returns null when it cloned cleanly.</summary>
    private string? Attempt(Type open, Type closed)
    {
        // RESET FIRST. LastPhases is a field, and Attempt returns early on SkipMarker and on every
        // Fail path. Without this the next model's row inherits the previous model's numbers -
        // observed as S4 and PaLI3 reporting byte-identical phases while their totals differed.
        LastPhases = "construct=? resolve1=? copy=? resolve2=?";

        // CONSTRUCTION IS INSTRUMENTED TOO. The first cut started its counter AFTER TryConstruct,
        // which hid the cost of the biggest allocators entirely: WhisperLargeV3Turbo, KotobaWhisper
        // and Chirp3 each reported tens of GB in total against essentially zero across all measured
        // phases, because everything they spend is spent before the first counter was read.
        long pc0 = GC.GetTotalAllocatedBytes(precise: true);
        var model = TryConstruct(closed);
        long pc1 = GC.GetTotalAllocatedBytes(precise: true);
        LastPhases = $"construct={pc1 - pc0} resolve1=? copy=? resolve2=?";
        if (model is null)
        {
            notConstructed.Add($"{open.Name}: no constructor takes a standard architecture");
            Note($"skip  {open.Name}: not constructible");
            return SkipMarker;
        }

        try
        {
            // PHASE SPLIT. The sweep shows models allocating tens of GB for one construct+clone -
            // SenseVoiceLarge 46 GB - against weights that are a fraction of that. Attributing the
            // cost to a PHASE is what turns "a model is expensive" into something fixable, and it
            // costs four counter reads. GetTotalAllocatedBytes is monotonic, so these are volumes,
            // not peaks: a phase can allocate 20 GB while never holding more than a GB at once,
            // and that distinction is the whole point - churn and residency need different fixes.
            long p0 = GC.GetTotalAllocatedBytes(precise: true);

            // DO NOT read ParameterCount here. Doing so forces lazy shape resolution, and that
            // single read is enough to change the outcome: with it, TimeGANGenerator's original
            // materialized early and matched its copy, turning a real 8640-against-192 mismatch
            // into a pass. The probe must observe the model, not resolve it.
            // Non-mutating: reports what is materialized without resolving anything, so unlike
            // ParameterCount it can be read before the probe without deciding the outcome.
            var matBeforeProbe = model.MaterializedParameterCount();
            var probed = Resolve(model);
            var matAfterProbe = model.MaterializedParameterCount();
            long p1 = GC.GetTotalAllocatedBytes(precise: true);

            var before = model.ParameterCount;
            var copy = model.DeepCopy() as NeuralNetworkBase<float>;
            // The ORIGINAL re-read after DeepCopy. DeepCopy has to read the source's full parameter
            // surface to copy it, which materializes the source too, so `before` (captured earlier)
            // and the copy's count are not measured in the same state.
            var originalAfterCopy = model.ParameterCount;
            var matAfterCopy = copy is null ? -1 : copy.MaterializedParameterCount();
            var afterCopy = copy is null ? -1 : copy.ParameterCount;
            long p2 = GC.GetTotalAllocatedBytes(precise: true);

            if (copy is null) return Fail(open, "DeepCopy returned null");
            if (copy.GetType() != closed) return Fail(open, $"copy is {copy.GetType().Name}");

            Resolve(copy);
            long p3 = GC.GetTotalAllocatedBytes(precise: true);

            // Parameter counts alongside the allocation phases. A count that MOVES across DeepCopy
            // means the two sides materialized different amounts of a lazy surface, which is a
            // different defect from a copy that lost state, and the totals alone cannot tell them
            // apart -- TimeGANGenerator reports 8640 against 192 with the probe having run on both.
            LastPhases = $"construct={pc1 - pc0} resolve1={p1 - p0} copy={p2 - p1} resolve2={p3 - p2}"
                + $" mat(beforeProbe={matBeforeProbe} afterProbe={matAfterProbe} afterCopy={matAfterCopy})"
                + $" counts(afterResolve1={before} originalAfterCopy={originalAfterCopy}"
                + $" afterCopy={afterCopy} afterResolve2={copy.ParameterCount})";

            // COMPARE THE SAME STATE. `before` is captured BEFORE DeepCopy, and DeepCopy
            // materializes the source as a side effect of reading its parameter surface -- measured
            // on TimeGANGenerator: 192 before, 8640 after, with the copy also at 8640 and both
            // sides structurally identical (3 FullyConnectedLayers, 320/4160/4160). Comparing the
            // pre-copy number against the post-copy one measured materialization, not copying, and
            // reported a 45x "mismatch" between two models that were the same.
            if (copy.ParameterCount != originalAfterCopy)
            {
                // AN UNRESOLVED ORIGINAL IS NOT A FAILED COPY. Every model in this bucket reports
                // the copy as the LARGER side -- DeepCopy materialized layers the original had left
                // lazy because the probe never ran on it. Comparing those two counts measures
                // materialization, not copying.
                if (!probed)
                {
                    unresolved.Add(
                        $"{open.Name}: probe did not run ({copy.ParameterCount} against {originalAfterCopy})");
                    Note($"lazy  {open.Name}");
                    return SkipMarker;
                }

                return Fail(
                    open,
                    $"{copy.ParameterCount} parameters against {originalAfterCopy} || {LastPhases}"
                        + $" || original {Structure(model)} || copy {Structure(copy)}");
            }

            // Live heap immediately before the independence check: the two resident models plus
            // whatever DeepCopy left behind. The phase counters measure allocation VOLUME and
            // cannot say how much of it is still resident; this can, and it is the number that
            // decides whether the flat parameter vectors are the problem or merely the last straw.
            LastPhases += $" liveBeforeIndependence={GC.GetTotalMemory(false) / (1024 * 1024)}MB";

            if (ReferenceEquals(copy, model) || !IsIndependent(model, copy))
                return Fail(open, "copy is not independent of the original");

            Note($"ok    {open.Name}");
            return null;
        }
        catch (Exception ex)
        {
            var inner = ex.InnerException ?? ex;
            var message = inner.Message;
            // 90 chars used to cut clone-rejection reasons mid-word ("trainable layer 0
            // (PatchEmbeddingLa"), which hid the very detail the sweep exists to report.
            var trimmed = message.Substring(0, Math.Min(400, message.Length));

            // An OutOfMemoryException carries no useful message, so without the originating frames
            // the sweep cannot say whether the model died in the copy-on-write attempt or in the
            // serialize fallback -- which is the only distinction that matters for this failure.
            var frames = (inner.StackTrace ?? string.Empty)
                .Split('\n')
                .Select(line => line.Trim())
                .Where(line => line.StartsWith("at AiDotNet.", StringComparison.Ordinal))
                .Take(4);
            var origin = string.Join(" <- ", frames);

            // An OutOfMemoryException cannot afford to describe itself -- building the message
            // allocates, and the heap is exactly what has run out. LayerBase records the shape it
            // was about to allocate beforehand, so the culprit weight survives the failure.
            var attempted = AiDotNet.NeuralNetworks.Layers.LayerBase<float>.LastAttemptedWeightShape;
            var attemptedOwner = AiDotNet.NeuralNetworks.Layers.LayerBase<float>.LastAttemptedWeightOwner;
            var weight = attempted is null
                ? string.Empty
                : $" || last weight attempt: {attemptedOwner} [{string.Join(",", attempted)}]";

            return Fail(
                open,
                origin.Length == 0
                    ? $"{inner.GetType().Name}: {trimmed}{weight} || {LastPhases}"
                    : $"{inner.GetType().Name}: {trimmed} || {origin}{weight} || {LastPhases}");
        }
        finally
        {
            (model as IDisposable)?.Dispose();
        }
    }

    /// <summary>Runs one probe input through the model so lazy layers materialise.</summary>
    /// <remarks>
    /// A model that cannot accept the standard probe is left as it is; the comparison below then
    /// still holds, because both sides are measured in the same unresolved state.
    /// </remarks>
    /// <returns>True when the probe ran, so the model's parameter surface is materialized.</returns>
    private static bool Resolve(NeuralNetworkBase<float> model)
    {
        try
        {
            var input = new Tensor<float>(new[] { 1, 4 });
            model.Predict(input);

            return true;
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
    /// Guards the rewrite that removed the baseline vector. The verification loop is INVERTED -- it
    /// asserts the original still differs from the mutated buffer -- and an inverted test that is
    /// wired up wrongly does not fail loudly, it passes everything. The sweep would then report a
    /// clean run over hundreds of models while checking nothing at all. A model handed to the check
    /// as its own copy shares every tensor by definition, so this must come back false.
    /// </remarks>
    [Fact]
    public void IsIndependent_RejectsAModelComparedWithItself()
    {
        // Named types, not a sweep over every model. Walking the whole assembly meant constructing
        // and resolving arbitrary multi-gigabyte networks just to find a small one, and this test
        // ran out of memory before it could validate anything.
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

            // Small on purpose. This validates the checker, not a model, and the sweep itself is
            // where large models get exercised.
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


    /// <summary>Compact layer-by-layer shape of a model, for comparing an original with its copy.</summary>
    /// <remarks>
    /// A parameter-count mismatch has two very different causes with the same symptom: the copy
    /// built MORE layers, or it built the SAME layers at larger shapes. Those need different central
    /// fixes, and the totals alone cannot tell them apart. Uses the non-mutating counter so reading
    /// this does not resolve the very lazy shapes under investigation.
    /// </remarks>
    private static string Structure(NeuralNetworkBase<float> model)
    {
        var parts = new List<string>();
        for (var i = 0; i < model.Layers.Count && i < 24; i++)
        {
            var layer = model.Layers[i];
            var name = layer.GetType().Name;
            var materialized = layer is AiDotNet.NeuralNetworks.Layers.LayerBase<float> lb
                ? lb.MaterializedParameterCount().ToString()
                : "n/a";
            parts.Add($"{name}={materialized}");
        }

        return $"[{model.Layers.Count} layers: {string.Join(" ", parts)}]";
    }
    /// <summary>Whether writing through one model leaves the other alone.</summary>
    /// <remarks>
    /// This method, not <c>DeepCopy</c>, is where FishSpeech ran out of memory. It used to hold a
    /// baseline vector, a mutation vector and a re-read vector of the model's ENTIRE parameter
    /// surface at once, on top of the two resident models. Measured on FishSpeech: 4.88 GB live
    /// after cloning, a 330,193,920-element parameter surface, and three 1.26 GB contiguous
    /// large-object allocations demanded on top of that. It never got them.
    ///
    /// One full-length vector remains, because <c>UpdateParameters</c> accepts nothing else. The
    /// baseline and the verification read are streamed through <c>GetParameterStateChunks</c>,
    /// whose scope contract guarantees it covers exactly the parameters the flat APIs do.
    ///
    /// The assertion is unchanged in strength: every parameter is still written through the copy,
    /// and every parameter of the original is still checked elementwise at the same tolerance. Only
    /// the direction of the comparison moved -- see the verification loop.
    /// </remarks>
    private static bool IsIndependent(
        NeuralNetworkBase<float> original,
        NeuralNetworkBase<float> copy)
    {
        // Both sides are walked chunk-by-chunk, so nothing full-length is ever allocated. That
        // matters: PaLI3 and SenseVoiceLarge need a 2.4-2.6 GB CONTIGUOUS large-object block for a
        // flat vector, on top of ~5 GB of legitimate resident model, and could not get one.
        var originalChunks = new List<Tensor<float>>();
        bool streamable = true;
        foreach (var chunk in original.GetParameterStateChunks())
        {
            // A chunk that is not writable in place -- an fp16-resident or sparse component handing
            // out a transient snapshot, or a layer that is not a LayerBase -- would swallow the
            // mutation silently. Left unchecked that reads as "the copy never changed", which this
            // method would report as SHARED storage: a false failure, not a missed one.
            if (!chunk.IsWritableInPlace)
            {
                streamable = false;
                break;
            }

            originalChunks.Add(chunk.Tensor);
        }

        if (originalChunks.Count == 0 && streamable) return true;
        if (!streamable) return IsIndependentViaFlatSurface(original, copy);

        var copyChunks = new List<Tensor<float>>();
        foreach (var chunk in copy.GetParameterStateChunks())
        {
            if (!chunk.IsWritableInPlace) return IsIndependentViaFlatSurface(original, copy);
            copyChunks.Add(chunk.Tensor);
        }

        // A copy that does not present the same chunk surface as its original is not a faithful
        // copy, and pairing the overlapping prefix would quietly pass that.
        if (copyChunks.Count != originalChunks.Count) return false;
        for (var c = 0; c < copyChunks.Count; c++)
        {
            if (copyChunks[c].Length != originalChunks[c].Length) return false;
        }

        // Write through the COPY. If the two are independent it ends at original + 1 everywhere; if
        // they share storage the same write moved the original too, so afterwards they read equal.
        // Comparing the two live surfaces is therefore the whole test, and needs no baseline copy.
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

                // Where +1.0f is a no-op the element cannot discriminate either way, so skip it
                // rather than report false sharing. Real weights never reach that magnitude.
                if (originalValue + 1.0f == originalValue) continue;

                if (Math.Abs(after[i] - originalValue) <= 1e-5f) return false;
            }
        }

        return true;
    }

    /// <summary>Independence check for models whose chunks cannot be written in place.</summary>
    /// <remarks>
    /// The flat surface is the only way to reach an fp16-resident or sparse component, or a layer
    /// that is not a LayerBase. It costs one contiguous full-length vector, which is exactly what
    /// the chunked path above exists to avoid, so it runs only when that path cannot apply.
    /// </remarks>
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
                else if (formal[i].HasDefaultValue) args[i] = formal[i].DefaultValue;
                else usable = false;
            }

            if (!usable || args.All(a => a is null)) continue;

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

    private static bool DerivesFromNeuralNetworkBase(Type type)
    {
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            if (b.IsGenericType && b.GetGenericTypeDefinition() == typeof(NeuralNetworkBase<>)) return true;
            if (b.Name.StartsWith("NeuralNetworkBase", StringComparison.Ordinal)) return true;
        }

        return false;
    }
}
