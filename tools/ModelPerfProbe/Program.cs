using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.Tools.ModelPerfProbe;

/// <summary>
/// Validates and aggregates fixture-native model performance census records. Model construction and
/// workload execution deliberately live in the generated model-family tests, because those fixtures
/// already know each model's legal constructor, input domain, shape and target semantics.
/// </summary>
internal static class Program
{
    private const int CurrentBaselineSchemaVersion = 3;

    /// <summary>
    /// Baseline schemas whose numbers this build can still read.
    /// </summary>
    /// <remarks>
    /// Schema 3 adds the commit a baseline was measured at. Schema 2 carries the same metrics
    /// without it, so it stays comparable - refusing it would have silently disabled the gate for
    /// the whole window it takes for a new baseline to be published, which is the opposite of what
    /// a version check is for.
    /// </remarks>
    private static readonly int[] ComparableBaselineSchemaVersions = { 2, 3 };

    private static readonly string[] RequiredMetrics =
    {
        "constructMs", "targetPreparationMs", "coldForwardMs", "steadyForwardMedianMs", "steadyForwardP95Ms",
        "tapeForwardMs", "backwardMs", "trainStepMs", "allocatedBytes", "wallMs",
        "runnerElapsedMs", "peakWorkingSetBytes", "peakPrivateMemoryBytes", "runnerCpuMs", "runnerCpuToWallRatio",
    };

    private static int Main(string[] args)
    {
        Options? options = Options.Parse(args);
        if (options is null) return PrintUsage();
        if (options.SelfTest) return RunSelfTest();

        try
        {
            IReadOnlyList<CensusRecord> records = LoadRecords(options.ResultsDirectory!);
            var diagnostics = new List<Diagnostic>();
            ValidateCoverage(records, options.ExpectedCount, diagnostics);
            ValidateRecords(records, diagnostics);

            BaselineDocument? baseline = options.BaselinePath is null
                ? null
                : JsonSerializer.Deserialize<BaselineDocument>(
                    File.ReadAllText(options.BaselinePath), JsonOptions);

            IReadOnlyList<BaselineDocument> history = LoadHistory(options.HistoryDirectory, diagnostics);
            PerfIntent[] intents = LoadIntents(options.PerfIntentPath, diagnostics);

            CompareBaseline(records, baseline, options, diagnostics, history, intents);
            DetectCohortOutliers(records, diagnostics);
            ValidateAbsoluteCeilings(records, options, diagnostics);

            var summary = BuildSummary(records, diagnostics, options.ExpectedCount);
            WriteJson(options.OutputPath!, summary);
            if (options.WriteBaselinePath is not null)
                WriteJson(options.WriteBaselinePath, BuildBaseline(records, options.Commit));

            PrintSummary(summary, options.OutputPath!);
            return diagnostics.Any(d => d.Severity == "error") ? 1 : 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"model performance census failed: {ex.Message}");
            return 2;
        }
    }

    private static IReadOnlyList<CensusRecord> LoadRecords(string directory)
    {
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Results directory does not exist: {directory}");

        var records = new List<CensusRecord>();
        foreach (string path in Directory.EnumerateFiles(directory, "*.json", SearchOption.AllDirectories))
        {
            using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("fixture", out _)) continue;
            records.Add(CensusRecord.FromJson(path, root));
        }
        return records;
    }

    private static void ValidateCoverage(
        IReadOnlyList<CensusRecord> records,
        int? expectedCount,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (IGrouping<string, CensusRecord> duplicate in records.GroupBy(r => r.Fixture)
                     .Where(group => group.Count() > 1))
        {
            diagnostics.Add(Diagnostic.Error(duplicate.Key, "coverage",
                $"duplicate fixture result ({duplicate.Count()} records)"));
        }

        if (expectedCount.HasValue && records.Select(r => r.Fixture).Distinct().Count() != expectedCount.Value)
        {
            diagnostics.Add(Diagnostic.Error("<census>", "coverage",
                $"expected {expectedCount.Value} fixture result(s), found " +
                $"{records.Select(r => r.Fixture).Distinct().Count()}"));
        }
        if (records.Count == 0)
            diagnostics.Add(Diagnostic.Error("<census>", "coverage", "no fixture records were produced"));
    }

    private static void ValidateRecords(
        IReadOnlyList<CensusRecord> records,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (CensusRecord record in records)
        {
            if (!string.Equals(record.Status, "ok", StringComparison.Ordinal))
            {
                string phase = string.IsNullOrWhiteSpace(record.Phase) ? "unknown" : record.Phase;
                string detail = string.IsNullOrWhiteSpace(record.Error) ? "no process diagnostic" : record.Error;
                string resources = double.IsFinite(record.Metric("peakWorkingSetBytes"))
                    ? $"; peak working set {record.Metric("peakWorkingSetBytes") / (1024.0 * 1024.0):F1} MiB, " +
                      $"CPU/wall {record.Metric("runnerCpuToWallRatio"):F2}x"
                    : "";
                diagnostics.Add(Diagnostic.Error(record.Fixture, "status",
                    $"status is '{record.Status}' after {record.ElapsedMs:F0} ms in phase '{phase}'{resources}: {detail}"));
                continue;
            }

            foreach (string metric in RequiredMetrics)
            {
                if (!record.Metrics.TryGetValue(metric, out double value)
                    || double.IsNaN(value) || double.IsInfinity(value) || value < 0.0)
                {
                    diagnostics.Add(Diagnostic.Error(record.Fixture, metric,
                        "required metric is missing, negative, NaN or infinite"));
                }
            }
        }
    }

    /// <summary>
    /// How reliably a metric survives being compared across two runs on two different CI machines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Measured, not guessed. All 863 census fixtures were run twice over functionally identical
    /// code (master d843c805 and PR #2009, whose diff touches optimizers only). Every wall-clock
    /// metric came back with a median run-over-run ratio of 0.99 - no systematic shift whatsoever -
    /// but a p90 of 1.30-1.54 and a maximum of 4.60x, and the spread was symmetric: for
    /// <c>coldForwardMs</c>, 183 fixtures looked more than 1.25x slower and 134 looked more than
    /// 1.25x faster. Over the same pairs <c>allocatedBytes</c> had a median ratio of 1.000 and
    /// <c>parameterCount</c> was identical for all 863 fixtures.
    /// </para>
    /// <para>
    /// A single 1.25x limit applied to every metric therefore sat BELOW the noise floor of its own
    /// measurement, and reported 759-842 regressions per run on code that did the same work. The
    /// classes below give each metric a limit its own measurement can actually support.
    /// </para>
    /// </remarks>
    private enum MetricStability
    {
        /// <summary>Exact counts - any increase is a real change, so the limit is 1.0.</summary>
        Exact,

        /// <summary>Deterministic byte counts. Median ratio 1.000, p99 1.49 across the null pairs.</summary>
        Allocation,

        /// <summary>Process memory peaks: allocator-quantised, but the tightest timing-free signal (p99 1.23).</summary>
        Memory,

        /// <summary>Wall-clock. Machine-variance dominated (p99 1.51-1.61); needs corroboration to be an error.</summary>
        Timing,

        /// <summary>Wall-clock over millisecond-scale magnitudes, so noisier again (p99 2.55, max 4.60).</summary>
        VolatileTiming,

        /// <summary>Measures the runner, not the code under test. Never an error on its own.</summary>
        RunnerScoped,

        /// <summary>A restatement of another metric - gating it counts one measurement several times.</summary>
        Derived,
    }

    /// <summary>
    /// Assigns a metric to the stability class whose limit its measurement can support.
    /// </summary>
    private static MetricStability ClassifyMetric(string metric) => metric switch
    {
        // projected*Ms are trainStepMs multiplied by a requested iteration count. They carry no
        // independent measurement, so gating them made one slow train step count three times -
        // visible in the raw data as three diagnostics with identical ratios (2.06 / 2.06 / 2.06).
        "projectedTrainingReduceLossMs" or "projectedMoreDataMs" => MetricStability.Derived,

        // Whole-process timings: these move when the runner is busy, whatever the code does.
        "runnerCpuMs" or "runnerElapsedMs" or "wallMs" => MetricStability.RunnerScoped,

        "parameterCount" or "parameterSlots" or "tapeEntries" or "gradientTensorCount"
            => MetricStability.Exact,

        "allocatedBytes" => MetricStability.Allocation,
        "peakWorkingSetBytes" or "peakPrivateMemoryBytes" => MetricStability.Memory,

        // Steady-state forward timings are medians of millisecond-scale samples, where a fixed
        // per-sample overhead is a large relative share - the noisiest metrics in the census.
        "steadyForwardMedianMs" or "steadyForwardP95Ms" => MetricStability.VolatileTiming,

        _ => MetricStability.Timing,
    };

    /// <summary>
    /// Compares this run against the stored baseline, applying each metric's own regression limit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A wall-clock regression on its own is not evidence of slower library code. It is an error only when the
    /// same fixture's allocation or whole-workload process CPU moved too. Those are independent
    /// signals that the measured code did more work. A timing-only movement is still reported as a
    /// warning, regardless of size, because even a gross single-phase stopwatch result can be caused
    /// by descheduling while process CPU and allocation remain flat.
    /// </para>
    /// <para>
    /// A declared cost increase is a scoped, expiring lease: it names the exact baseline commit,
    /// fixture, metric, and reason. It suppresses only that comparison and expires when the baseline
    /// advances; unrelated metrics and later regressions still fail.
    /// </para>
    /// <para>
    /// The case this is built from: MeshCNN's cold forward measured 3.29x its baseline on PR #2009,
    /// while the same fixture allocated 532,453,184 bytes against master's 532,453,792 - a 608-byte
    /// difference on 532 MB. Identical work, different machine.
    /// </para>
    /// </remarks>
    private static void CompareBaseline(
        IReadOnlyList<CensusRecord> records,
        BaselineDocument? baseline,
        Options options,
        ICollection<Diagnostic> diagnostics,
        IReadOnlyList<BaselineDocument>? historyDocuments = null,
        IReadOnlyList<PerfIntent>? declaredIntents = null)
    {
        IReadOnlyList<BaselineDocument> history = historyDocuments ?? Array.Empty<BaselineDocument>();
        IReadOnlyList<PerfIntent> intents = declaredIntents ?? Array.Empty<PerfIntent>();
        if (baseline is null) return;
        if (Array.IndexOf(ComparableBaselineSchemaVersions, baseline.SchemaVersion) < 0)
        {
            diagnostics.Add(Diagnostic.Warning("<census>", "baseline",
                $"baseline schema {baseline.SchemaVersion} is not comparable to schema " +
                $"{CurrentBaselineSchemaVersion}; refresh the environment-qualified baseline"));
            return;
        }

        Dictionary<(string Fixture, string EnvironmentKey, string Metric), List<SeriesPoint>> seriesIndex =
            IndexHistory(history);

        var index = baseline.Entries.ToDictionary(
            entry => (entry.Fixture, entry.Environment),
            entry => entry,
            new FixtureEnvironmentComparer());

        var missingByEnvironment = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (CensusRecord record in records.Where(r => r.Status == "ok"))
        {
            if (!index.TryGetValue((record.Fixture, record.Environment), out BaselineEntry? prior))
            {
                missingByEnvironment.TryGetValue(record.Environment, out int missingCount);
                missingByEnvironment[record.Environment] = missingCount + 1;
                continue;
            }

            // Does this fixture's allocation confirm that it is genuinely doing more work?
            bool allocationRegressed =
                record.Metrics.TryGetValue("allocatedBytes", out double allocNow)
                && prior.Metrics.TryGetValue("allocatedBytes", out double allocWas)
                && allocWas > 0.0
                && allocNow / allocWas > options.CorroborationRatio;

            // A timing-only slowdown can be real without allocating more (for example, an
            // accidentally quadratic loop). Whole-workload process CPU is an independent signal
            // that separates that case from a descheduled Stopwatch phase.
            bool cpuRegressed =
                record.Metrics.TryGetValue("cpuMs", out double cpuNow)
                && prior.Metrics.TryGetValue("cpuMs", out double cpuWas)
                && cpuWas > 0.0
                && cpuNow / cpuWas > options.CorroborationRatio;

            foreach ((string metric, double current) in record.Metrics)
            {
                // Utilization ratios are diagnostic direction signals, not monotonic costs: a
                // higher CPU/wall ratio generally means the engine used the machine better.
                if (metric.EndsWith("Ratio", StringComparison.Ordinal)) continue;
                if (!prior.Metrics.TryGetValue(metric, out double previous) || previous <= 0.0) continue;

                MetricStability stability = ClassifyMetric(metric);
                if (stability is MetricStability.Derived or MetricStability.RunnerScoped) continue;

                double limit = stability switch
                {
                    MetricStability.Exact => 1.0,
                    MetricStability.Allocation => options.MaxAllocationRegressionRatio,
                    MetricStability.Memory => options.MaxMemoryRegressionRatio,
                    MetricStability.VolatileTiming => options.MaxVolatileTimingRegressionRatio,
                    _ => options.MaxRegressionRatio,
                };

                double noiseFloor = metric switch
                {
                    "allocatedBytes" => 1_048_576.0,
                    "peakWorkingSetBytes" or "peakPrivateMemoryBytes" => 67_108_864.0,
                    _ => 25.0,
                };

                // WHAT THIS RUN IS JUDGED AGAINST.
                //
                // A single prior run cannot tell a step from a spike, and it cannot tell a step
                // that somebody meant to make from one nobody noticed. With a series it can do
                // both: find the last SUSTAINED level shift, say which commits it happened
                // between, and judge this run against the level the fixture has actually been
                // sitting at since - not against a point from before a change that everyone has
                // already accepted.
                //
                // With too little history this is exactly the old comparison, so nothing loosens
                // on a repository that has not accumulated a window yet.
                (string Fixture, string EnvironmentKey, string Metric) seriesKey =
                    (record.Fixture, SeriesEnvironmentKey(record.Environment, stability), metric);
                IReadOnlyList<SeriesPoint> series = seriesIndex.TryGetValue(seriesKey, out List<SeriesPoint>? cached)
                    ? cached
                    : Array.Empty<SeriesPoint>();
                if (series.Count >= MinimumSeriesPoints)
                {
                    StepChange? step = FindLastStep(series, limit, noiseFloor);
                    if (step is not null)
                    {
                        PerfIntent? intent = IntentFor(intents, step, record.Fixture, metric);
                        if (intent is not null)
                        {
                            diagnostics.Add(Diagnostic.Warning(record.Fixture, metric,
                                $"stepped {step.Ratio:F2}x ({step.Before:F2} -> {step.After:F2}) at "
                                + $"{step.Range}"
                                + (intent.Origin.Length > 0 ? $", caused by {intent.Origin}" : "")
                                + $", declared: {intent.Reason}"));
                        }
                        else if (Same(options.Commit, step.ToCommit))
                        {
                            // Only the commit under test can be required to declare its own step.
                            // Re-failing every later PR for an older main-branch change turns
                            // historical context into unrelated permanent CI debt.
                            diagnostics.Add(Diagnostic.Error(record.Fixture, metric,
                                $"stepped {step.Ratio:F2}x ({step.Before:F2} -> {step.After:F2}) at "
                                + $"{step.Range} and stayed there; limit is {limit:F2}x. Fix it, or "
                                + "declare it in .github/model-performance-intent.json with the census commit, "
                                + "fixture, metric, and reason"));
                        }
                        else
                        {
                            diagnostics.Add(Diagnostic.Warning(record.Fixture, metric,
                                $"historical step {step.Ratio:F2}x ({step.Before:F2} -> {step.After:F2}) at "
                                + $"{step.Range}; current commit {options.Commit} is judged against the post-step level"));
                        }

                        previous = ReferenceLevel(series, step);
                        if (previous <= 0.0) continue;
                    }
                    // If history contains no defensible sustained step, retain the exact latest,
                    // environment-qualified baseline. Replacing it with a stale series median can
                    // report a regression even when current and latest-prior values are identical.
                }

                double ratio = current / previous;
                if (ratio <= limit || current - previous <= noiseFloor) continue;

                bool timing = stability is MetricStability.Timing or MetricStability.VolatileTiming;
                if (timing && !allocationRegressed && !cpuRegressed)
                {
                    diagnostics.Add(Diagnostic.Warning(record.Fixture, metric,
                        $"timing moved {ratio:F2}x ({previous:F2} -> {current:F2}) but neither "
                        + "allocation nor whole-workload process CPU regressed; treating the "
                        + $"Stopwatch-only movement as runner variance (limit is {limit:F2}x)"));
                    continue;
                }

                // WITHIN AN ENVELOPE THE FIXTURE HAS ALREADY OCCUPIED.
                //
                // A regression claim is "this run is worse than what this fixture does". When the
                // series contains no sustained step, the comparison falls back to a SINGLE prior
                // point - and if that point happened to be the series minimum, the ratio measures
                // where the baseline landed rather than anything about this run.
                //
                // Measured: PaLI's constructMs over twelve environment-qualified runs spans 48.0 to
                // 477.2 ms, a 9.94x spread, with two points on the very same EPYC 9V74 at 360.8 and
                // 477.2. The baseline drew 48.0, the lowest of all thirteen, so a mid-range 142.4
                // read as a 2.97x regression. Every one of those higher values was itself a passing
                // baseline at the time, so calling this one a regression contradicts the runs the
                // window already accepted.
                //
                // Scoped to the timing classes on purpose. Their run-over-run spread is machine
                // dominated, which is what makes a single low point unrepresentative. Allocation and
                // memory are stable enough that their envelope is not a licence - RepViT-SAM's peak
                // held inside 397.8-461.3 MB across the same twelve runs - so those keep comparing
                // against the exact latest point and a genuine step there still fails.
                //
                // Still reported, because a fixture creeping toward the top of its envelope is worth
                // seeing before it leaves it.
                if (timing && series.Count >= MinimumSeriesPoints)
                {
                    double envelope = series.Max(point => point.Value);
                    if (current <= envelope)
                    {
                        diagnostics.Add(Diagnostic.Warning(record.Fixture, metric,
                            $"moved {ratio:F2}x ({previous:F2} -> {current:F2}), but {current:F2} is "
                            + $"inside the {series.Count}-run envelope this fixture has already "
                            + $"occupied on this environment (max {envelope:F2}); the baseline drew a "
                            + "low point rather than this run regressing"));
                        continue;
                    }
                }

                PerfIntent? directIntent = IntentForBaseline(intents, baseline.Commit, record.Fixture, metric);
                if (directIntent is not null)
                {
                    diagnostics.Add(Diagnostic.Warning(record.Fixture, metric,
                        $"regressed {ratio:F2}x ({previous:F2} -> {current:F2}) from declared baseline "
                        + $"{baseline.Commit}"
                        + (directIntent.Origin.Length > 0 ? $", caused by {directIntent.Origin}" : "")
                        + $"; declared: {directIntent.Reason}"));
                    continue;
                }

                diagnostics.Add(Diagnostic.Error(record.Fixture, metric,
                    $"regressed {ratio:F2}x ({previous:F2} -> {current:F2}); limit is {limit:F2}x"));
            }
        }

        foreach ((string environment, int missingCount) in missingByEnvironment)
        {
            diagnostics.Add(Diagnostic.Warning("<census>", "baseline",
                $"no environment-qualified baseline for {environment}; " +
                $"skipped {missingCount} fixture comparison(s)"));
        }
    }

    /// <summary>
    /// Checks the series behaviour the point comparison could not have: a spike is not a step, a
    /// sustained step is attributed, a declared step is accepted, and an undeclared one is not.
    /// </summary>
    private static int RunSeriesSelfTest()
    {
        int Fail(string what)
        {
            Console.Error.WriteLine($"self-test failed: {what}");
            return 1;
        }

        const string Fixture = "SeriesFixture";
        const string Environment = "test";
        const string Metric = "peakWorkingSetBytes";
        double noiseFloor = 67_108_864.0;
        double limit = 1.6;

        BaselineDocument Point(string commit, int day, double value,
            int schemaVersion = CurrentBaselineSchemaVersion) => new()
        {
            SchemaVersion = schemaVersion,
            GeneratedUtc = new DateTimeOffset(2026, 1, day, 0, 0, 0, TimeSpan.Zero),
            Commit = commit,
            Entries = new[]
            {
                new BaselineEntry
                {
                    Fixture = Fixture,
                    Environment = Environment,
                    Metrics = new Dictionary<string, double> { [Metric] = value },
                },
            },
        };

        // Duplicate fixture entries in one artifact retain the first usable metric, matching the
        // pre-index scan. This guards the one-time index against silently changing malformed-input
        // behavior.
        BaselineDocument duplicate = Point("fffffff", 6, 300_000_000);
        duplicate.Entries = duplicate.Entries.Concat(new[]
        {
            new BaselineEntry
            {
                Fixture = Fixture,
                Environment = Environment,
                Metrics = new Dictionary<string, double> { [Metric] = 900_000_000 },
            },
        }).ToArray();
        IReadOnlyList<SeriesPoint> duplicateSeries = SeriesFor(
            new[] { duplicate }, Fixture, Environment, Metric);
        if (duplicateSeries.Count != 1 || Math.Abs(duplicateSeries[0].Value - 300_000_000) > 0.5)
            return Fail("the history index must preserve first-entry-per-document behavior");

        // One high reading between low ones is a spike. A point comparison calls that a regression;
        // a series must not, because the level did not change.
        var spike = new[]
        {
            Point("aaaaaaa", 1, 300_000_000), Point("bbbbbbb", 2, 300_000_000),
            Point("ccccccc", 3, 900_000_000),
            Point("ddddddd", 4, 300_000_000), Point("eeeeeee", 5, 300_000_000),
        };
        if (FindLastStep(SeriesFor(spike, Fixture, Environment, Metric), limit, noiseFloor) is not null)
            return Fail("a single spike must not read as a step change");

        // A level that rises and STAYS is a step, and it happened between the last low commit and
        // the first high one.
        var step = new[]
        {
            Point("aaaaaaa", 1, 300_000_000), Point("bbbbbbb", 2, 300_000_000),
            Point("ccccccc", 3, 900_000_000), Point("ddddddd", 4, 900_000_000),
            Point("eeeeeee", 5, 900_000_000),
        };
        StepChange? found = FindLastStep(
            SeriesFor(step.Reverse().ToArray(), Fixture, Environment, Metric), limit, noiseFloor);
        if (found is null) return Fail("a sustained level shift must read as a step change");
        if (found.FromCommit != "bbbbbbb" || found.ToCommit != "ccccccc")
            return Fail($"the step must be attributed to bbbbbbb..ccccccc, not {found.FromCommit}..{found.ToCommit}");

        // Judged against the level it has been sitting at since the step, a run AT that level is
        // not a fresh regression - which is what stops one accepted change failing forever.
        double reference = ReferenceLevel(SeriesFor(step, Fixture, Environment, Metric), found);
        if (Math.Abs(reference - 900_000_000) > 0.5)
            return Fail($"the reference level after a step must be the level since it, got {reference}");

        // Schema-2 points have no commit. The boundary index, not an empty commit lookup, must
        // still exclude the old level from the post-step reference median.
        var unstamped = new[]
        {
            Point("", 1, 300_000_000, schemaVersion: 2), Point("", 2, 300_000_000, schemaVersion: 2),
            Point("", 3, 900_000_000, schemaVersion: 2), Point("", 4, 900_000_000, schemaVersion: 2),
        };
        IReadOnlyList<SeriesPoint> unstampedSeries = SeriesFor(unstamped, Fixture, Environment, Metric);
        StepChange? unstampedStep = FindLastStep(unstampedSeries, limit, noiseFloor);
        if (unstampedStep is null || Math.Abs(ReferenceLevel(unstampedSeries, unstampedStep) - 900_000_000) > 0.5)
            return Fail("an unstamped schema-2 step must use its boundary index for the new reference level");

        // An intent naming the census commit accepts it; one naming a different commit does not.
        var declared = new[] { new PerfIntent { Commit = "ccccccc", Reason = "paper fidelity" } };
        if (IntentFor(declared, found, Fixture, Metric) is null)
            return Fail("an intent naming the step's commit must cover it");
        var elsewhere = new[] { new PerfIntent { Commit = "zzzzzzz", Reason = "something else" } };
        if (IntentFor(elsewhere, found, Fixture, Metric) is not null)
            return Fail("an intent naming another commit must not cover this step");
        var otherMetric = new[] { new PerfIntent { Commit = "ccccccc", Metric = "allocatedBytes", Reason = "scoped" } };
        if (IntentFor(otherMetric, found, Fixture, Metric) is not null)
            return Fail("an intent scoped to another metric must not cover this step");
        if (Same("ccccccc", "c"))
            return Fail("a commit reference shorter than the collision-safe floor must never match");

        // A reason is mandatory, exactly as a tolerance's why is.
        var reasonless = new List<Diagnostic>();
        string path = Path.Combine(Path.GetTempPath(), "perf-intent-" + Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllText(path,
            """[{"commit":"ccccccc","reason":"   "},{"commit":"c","reason":"short"},{"commit":"","reason":"missing"}]""");
        try
        {
            PerfIntent[] loaded = LoadIntents(path, reasonless);
            if (loaded.Length != 0 || reasonless.Count(d => d.Severity == "error") != 3)
                return Fail("reasonless, short, and empty commit intents must each be refused with a diagnostic");
        }
        finally
        {
            File.Delete(path);
        }


        CensusRecord Current(params (string Name, double Value)[] values) => new()
        {
            Fixture = Fixture,
            Model = "Synthetic",
            Status = "ok",
            Environment = Environment,
            Cohort = "System.Single|10^1",
            Phase = "",
            Error = "",
            Metrics = values.ToDictionary(pair => pair.Name, pair => pair.Value, StringComparer.Ordinal),
        };

        BaselineDocument DirectBaseline(string commit, params (string Name, double Value)[] values) => new()
        {
            SchemaVersion = CurrentBaselineSchemaVersion,
            Commit = commit,
            Entries =
            [
                new BaselineEntry
                {
                    Fixture = Fixture,
                    Environment = Environment,
                    Metrics = values.ToDictionary(pair => pair.Name, pair => pair.Value, StringComparer.Ordinal),
                },
            ],
        };

        var comparisonDiagnostics = new List<Diagnostic>();

        // A historical step that predates the commit under test is context, not permanent CI debt.
        CompareBaseline(
            [Current((Metric, 900_000_000))],
            Point("eeeeeee", 5, 900_000_000),
            new Options(),
            comparisonDiagnostics,
            step);
        if (comparisonDiagnostics.Any(d => d.Severity == "error")
            || !comparisonDiagnostics.Any(d => d.Message.Contains("historical step", StringComparison.Ordinal)))
            return Fail("an old sustained step must be warning-only for an unrelated later commit");

        // If history has no sustained step, the exact newest environment-qualified point wins. A
        // stale whole-window median must not manufacture a regression against an unchanged latest value.
        comparisonDiagnostics.Clear();
        CompareBaseline(
            [Current((Metric, 900_000_000))],
            Point("fffffff", 6, 900_000_000),
            new Options(),
            comparisonDiagnostics,
            spike);
        if (comparisonDiagnostics.Any(d => d.Severity == "error"))
            return Fail("a history window without a step must retain the exact latest baseline");

        // A TIMING VALUE INSIDE THE ENVELOPE THE FIXTURE ALREADY OCCUPIES IS NOT A REGRESSION,
        // and one above that envelope still is. Both arms matter: the first is the PaLI case, where
        // a 9.94x-spread metric drew its series minimum as the baseline; the second is the
        // StreamDiffVSR case, which must keep failing or this rule would blunt real regressions.
        const string TimingMetric = "constructMs";
        const string Allocation = "allocatedBytes";

        BaselineDocument TimingPoint(string commit, int day, double timing, double allocated) => new()
        {
            SchemaVersion = CurrentBaselineSchemaVersion,
            GeneratedUtc = new DateTimeOffset(2026, 2, day, 0, 0, 0, TimeSpan.Zero),
            Commit = commit,
            Entries =
            [
                new BaselineEntry
                {
                    Fixture = Fixture,
                    Environment = Environment,
                    Metrics = new Dictionary<string, double>(StringComparer.Ordinal)
                    {
                        [TimingMetric] = timing,
                        [Allocation] = allocated,
                    },
                },
            ],
        };

        // Corroborated by allocation (1.30x, over the 1.25x corroboration ratio but under the 2.5x
        // allocation limit), so these reach the envelope rule instead of stopping at the
        // uncorroborated-Stopwatch branch ahead of it.
        CensusRecord TimingRun(double timing) => Current((TimingMetric, timing), (Allocation, 130_000_000));

        // PaLI's real numbers: baseline 48.0 is the minimum of a series reaching 477.2.
        BaselineDocument[] dispersed =
        [
            TimingPoint("aaaaaaa", 1, 100.8, 100_000_000),
            TimingPoint("bbbbbbb", 2, 477.2, 100_000_000),
            TimingPoint("ccccccc", 3, 95.1, 100_000_000),
            TimingPoint("ddddddd", 4, 360.8, 100_000_000),
            TimingPoint("eeeeeee", 5, 48.0, 100_000_000),
        ];

        comparisonDiagnostics.Clear();
        CompareBaseline(
            [TimingRun(142.4)],
            TimingPoint("eeeeeee", 5, 48.0, 100_000_000),
            new Options(),
            comparisonDiagnostics,
            dispersed);
        if (comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == TimingMetric))
            return Fail("a timing value inside the fixture's own measured envelope must not be an error");
        if (!comparisonDiagnostics.Any(d => d.Metric == TimingMetric
                && d.Message.Contains("envelope", StringComparison.Ordinal)))
            return Fail("suppressing on the envelope must still report why");

        // StreamDiffVSR's real numbers: 2765.9 is 2.5x above everything the fixture has ever done.
        BaselineDocument[] tight =
        [
            TimingPoint("aaaaaaa", 1, 833.5, 100_000_000),
            TimingPoint("bbbbbbb", 2, 646.3, 100_000_000),
            TimingPoint("ccccccc", 3, 684.6, 100_000_000),
            TimingPoint("ddddddd", 4, 1108.7, 100_000_000),
            TimingPoint("eeeeeee", 5, 825.9, 100_000_000),
        ];

        comparisonDiagnostics.Clear();
        CompareBaseline(
            [TimingRun(2765.9)],
            TimingPoint("eeeeeee", 5, 825.9, 100_000_000),
            new Options(),
            comparisonDiagnostics,
            tight);
        if (!comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == TimingMetric))
            return Fail("a timing value above everything the fixture has measured must remain an error");

        // The envelope is a timing-only allowance. Memory keeps comparing against the exact latest
        // point, so RepViT-SAM's kind of step still fails even though its series contains it.
        comparisonDiagnostics.Clear();
        CompareBaseline(
            [Current((Metric, 739_753_984))],
            Point("eeeeeee", 5, 397_791_232),
            new Options(),
            comparisonDiagnostics,
            [Point("aaaaaaa", 1, 461_254_656), Point("bbbbbbb", 2, 406_716_416),
             Point("ccccccc", 3, 437_084_160), Point("ddddddd", 4, 441_790_464),
             Point("eeeeeee", 5, 397_791_232)]);
        if (!comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == Metric))
            return Fail("the envelope allowance must not extend to memory");

        const string Allocated = "allocatedBytes";
        BaselineDocument allocationBaseline = DirectBaseline("base000", (Allocated, 100_000_000));
        CensusRecord allocationIncrease = Current((Allocated, 300_000_000));

        comparisonDiagnostics.Clear();
        CompareBaseline([allocationIncrease], allocationBaseline, new Options(), comparisonDiagnostics);
        if (!comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == Allocated))
            return Fail("a genuine undeclared allocation regression must remain an error");

        var allocationIntent = new[]
        {
            new PerfIntent
            {
                Commit = "base000",
                Origin = "change00",
                Fixture = Fixture,
                Metric = Allocated,
                Reason = "paper-faithful capacity",
            },
        };
        comparisonDiagnostics.Clear();
        CompareBaseline(
            [allocationIncrease],
            allocationBaseline,
            new Options(),
            comparisonDiagnostics,
            declaredIntents: allocationIntent);
        if (comparisonDiagnostics.Any(d => d.Severity == "error")
            || !comparisonDiagnostics.Any(d => d.Severity == "warning"
                && d.Message.Contains("paper-faithful capacity", StringComparison.Ordinal)))
            return Fail("a fixture-and-metric-scoped intent must accept only its exact baseline comparison");

        allocationIntent[0].Commit = "other00";
        comparisonDiagnostics.Clear();
        CompareBaseline(
            [allocationIncrease],
            allocationBaseline,
            new Options(),
            comparisonDiagnostics,
            declaredIntents: allocationIntent);
        if (!comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == Allocated))
            return Fail("a declared intent for another baseline must not suppress a real allocation regression");

        // Stopwatch time alone is not proof: a 5x phase spike with identical allocation and total CPU is
        // warning-only, but the same spike with corroborating whole-process CPU remains an error.
        const string Cold = "coldForwardMs";
        const string Cpu = "cpuMs";
        BaselineDocument timingBaseline = DirectBaseline(
            "timing0", (Cold, 100), (Allocated, 100_000_000), (Cpu, 100));

        comparisonDiagnostics.Clear();
        CompareBaseline(
            [Current((Cold, 500), (Allocated, 100_000_000), (Cpu, 100))],
            timingBaseline,
            new Options(),
            comparisonDiagnostics);
        if (comparisonDiagnostics.Any(d => d.Severity == "error")
            || !comparisonDiagnostics.Any(d => d.Severity == "warning" && d.Metric == Cold))
            return Fail("an uncorroborated stopwatch spike must be visible but warning-only");

        comparisonDiagnostics.Clear();
        CompareBaseline(
            [Current((Cold, 500), (Allocated, 100_000_000), (Cpu, 130))],
            timingBaseline,
            new Options(),
            comparisonDiagnostics);
        if (!comparisonDiagnostics.Any(d => d.Severity == "error" && d.Metric == Cold))
            return Fail("a timing spike corroborated by whole-process CPU must remain an error");

        return 0;
    }

    // ---------------------------------------------------------------- declared intent

    /// <summary>Shortest commit prefix that an intent may match without unsafe overreach.</summary>
    /// <remarks>Seven characters is Git's conventional abbreviation floor.</remarks>
    private const int MinimumCommitReferenceLength = 7;

    /// <summary>
    /// A deliberate cost change declared in the repository-owned performance-intent file.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A model that becomes paper-faithful usually gets more expensive, and that is not a
    /// regression - but it is indistinguishable from one by measurement alone. DCRNN is the worked
    /// case: <c>cb9ecf59db</c> gave it real DiffusionConvolutionalGRU layers carrying graph
    /// transition matrices and K diffusion steps, and its peak working set rose 1.67x. Correct, and
    /// more expensive.
    /// </para>
    /// <para>
    /// Deliberate changes are recorded in <c>.github/model-performance-intent.json</c> with the
    /// census commit, fixture, metric, and reason. The checked-in list stays auditable, while the
    /// census-commit key makes each record expire naturally when that measurement leaves the
    /// history window.
    /// </para>
    /// <para>
    /// A reason is mandatory - an intent record without one is refused - for the same reason a
    /// tolerance without a stated why is refused elsewhere in this repository.
    /// </para>
    /// </remarks>
    private sealed class PerfIntent
    {
        /// <summary>
        /// A census endpoint for the cost change: the exact baseline for a first direct comparison,
        /// or either boundary of a sustained step once history contains the new level.
        /// </summary>
        /// <remarks>
        /// Keyed to a measurement rather than to the source commit because that makes the intent
        /// expire. A direct-comparison lease stops matching when the baseline advances; a step lease
        /// stops matching when the step scrolls out of the history window. A suppression keyed only
        /// to the source commit would never expire and could hide unrelated later regressions.
        /// </remarks>
        [JsonPropertyName("commit")] public string Commit { get; set; } = "";


        /// <summary>The commit that actually caused it, for the message. Informational.</summary>
        [JsonPropertyName("origin")] public string Origin { get; set; } = "";

        /// <summary>Optional: restricts the intent to one fixture. Empty means any.</summary>
        [JsonPropertyName("fixture")] public string Fixture { get; set; } = "";

        /// <summary>Optional: restricts the intent to one metric. Empty means any.</summary>
        [JsonPropertyName("metric")] public string Metric { get; set; } = "";

        [JsonPropertyName("reason")] public string Reason { get; set; } = "";

        public bool Covers(string fixture, string metric) =>
            (string.IsNullOrEmpty(Fixture) || string.Equals(Fixture, fixture, StringComparison.Ordinal))
            && (string.IsNullOrEmpty(Metric) || string.Equals(Metric, metric, StringComparison.Ordinal));
    }

    private static PerfIntent[] LoadIntents(string? path, ICollection<Diagnostic> diagnostics)
    {
        if (path is null) return Array.Empty<PerfIntent>();
        if (!File.Exists(path))
        {
            diagnostics.Add(Diagnostic.Warning("<census>", "intent",
                $"no declared-intent file at {path}; every step change will be reported as a regression"));
            return Array.Empty<PerfIntent>();
        }

        PerfIntent[] intents =
            JsonSerializer.Deserialize<PerfIntent[]>(File.ReadAllText(path), JsonOptions)
            ?? Array.Empty<PerfIntent>();

        var usable = new List<PerfIntent>();
        foreach (PerfIntent intent in intents)
        {
            string commit = intent.Commit?.Trim() ?? "";
            if (commit.Length < MinimumCommitReferenceLength)
            {
                string displayed = commit.Length == 0 ? "<empty>" : commit;
                diagnostics.Add(Diagnostic.Error("<census>", "intent",
                    $"the declared intent for fixture '{intent.Fixture ?? ""}' names commit '{displayed}'; "
                    + $"at least {MinimumCommitReferenceLength} characters are required so a short prefix "
                    + "cannot cover unrelated census commits"));
                continue;
            }

            string reason = intent.Reason?.Trim() ?? "";
            if (reason.Length == 0)
            {
                diagnostics.Add(Diagnostic.Error("<census>", "intent",
                    $"the declared intent for {commit} has no reason; a cost change is "
                    + "accepted only with a stated justification"));
                continue;
            }

            intent.Commit = commit;
            intent.Origin = intent.Origin?.Trim() ?? "";
            intent.Fixture = intent.Fixture?.Trim() ?? "";
            intent.Metric = intent.Metric?.Trim() ?? "";
            intent.Reason = reason;
            usable.Add(intent);
        }

        return usable.ToArray();
    }

    /// <summary>The intent covering a step, if any commit it could be attributed to declared one.</summary>
    private static PerfIntent? IntentFor(
        IReadOnlyList<PerfIntent> intents, StepChange step, string fixture, string metric)
    {
        foreach (PerfIntent intent in intents)
        {
            if (!intent.Covers(fixture, metric)) continue;
            if (string.IsNullOrEmpty(intent.Commit)) continue;
            if (Same(step.ToCommit, intent.Commit) || Same(step.FromCommit, intent.Commit))
            {
                return intent;
            }
        }

        return null;
    }


    /// <summary>
    /// Finds an intent leased to the exact baseline used by a direct comparison.
    /// </summary>
    /// <remarks>
    /// This covers the first census run that observes an intentional change, before enough post-change
    /// history exists to detect a sustained step. Naming the baseline makes the lease expire as soon as
    /// a newer baseline is published.
    /// </remarks>
    private static PerfIntent? IntentForBaseline(
        IReadOnlyList<PerfIntent> intents, string baselineCommit, string fixture, string metric)
    {
        foreach (PerfIntent intent in intents)
        {
            if (intent.Covers(fixture, metric) && Same(baselineCommit, intent.Commit)) return intent;
        }

        return null;
    }

    /// <summary>Whether two collision-safe commit references name the same commit.</summary>
    private static bool Same(string? left, string? right)
    {
        if (left is null || right is null
            || left.Length < MinimumCommitReferenceLength || right.Length < MinimumCommitReferenceLength)
            return false;
        return left.StartsWith(right, StringComparison.OrdinalIgnoreCase)
            || right.StartsWith(left, StringComparison.OrdinalIgnoreCase);
    }

    // ---------------------------------------------------------------- the series

    /// <summary>Fewest history points before a series is worth reasoning about as a series.</summary>
    /// <remarks>
    /// Below this a median is barely different from the single prior point, so the comparison stays
    /// exactly what it was. This is what keeps the change from loosening anything: a repository with
    /// no history behaves identically to before.
    /// </remarks>
    private const int MinimumSeriesPoints = 4;

    /// <summary>Fewest points on each side of a level shift before it counts as sustained.</summary>
    /// <remarks>Two, so a single anomalous run cannot be read as a step. This is the property a
    /// point-to-point comparison cannot have at any threshold.</remarks>
    private const int MinimumSegmentPoints = 2;

    /// <summary>One measurement of one fixture, in order.</summary>
    private sealed class SeriesPoint
    {
        public SeriesPoint(string commit, DateTimeOffset takenUtc, double value)
        {
            Commit = commit ?? "";
            TakenUtc = takenUtc;
            Value = value;
        }

        public string Commit { get; }
        public DateTimeOffset TakenUtc { get; }
        public double Value { get; }
    }

    /// <summary>A sustained level shift inside a series, and the commits it happened between.</summary>
    private sealed class StepChange
    {
        public StepChange(double before, double after, string fromCommit, string toCommit, int boundaryIndex)
        {
            Before = before;
            After = after;
            FromCommit = fromCommit ?? "";
            ToCommit = toCommit ?? "";
            BoundaryIndex = boundaryIndex;
        }

        public double Before { get; }
        public double After { get; }

        /// <summary>The last commit measured at the old level.</summary>
        public string FromCommit { get; }

        /// <summary>The first commit measured at the new level; the culprit is in (From, To].</summary>
        public string ToCommit { get; }

        /// <summary>Index of the first measurement at the new level.</summary>
        public int BoundaryIndex { get; }

        public double Ratio => Before > 0.0 ? After / Before : double.PositiveInfinity;

        public string Range =>
            FromCommit.Length == 0 || ToCommit.Length == 0
                ? "an unattributable range (the window predates commit-stamped baselines)"
                : $"{Short(FromCommit)}..{Short(ToCommit)}";

        private static string Short(string sha) => sha.Length <= 10 ? sha : sha.Substring(0, 10);
    }

    /// <summary>
    /// One fixture's history of one metric, oldest measurement first.
    /// </summary>
    /// <remarks>
    /// Environment-qualified like the point comparison is: a measurement from a different processor
    /// is a different population, and mixing them would manufacture steps out of runner allocation.
    /// </remarks>
    private static IReadOnlyList<SeriesPoint> SeriesFor(
        IReadOnlyList<BaselineDocument> history, string fixture, string environment, string metric)
    {
        MetricStability stability = ClassifyMetric(metric);
        var key = (Fixture: fixture, EnvironmentKey: SeriesEnvironmentKey(environment, stability), Metric: metric);
        Dictionary<(string Fixture, string EnvironmentKey, string Metric), List<SeriesPoint>> index =
            IndexHistory(history);
        return index.TryGetValue(key, out List<SeriesPoint>? points)
            ? points
            : Array.Empty<SeriesPoint>();
    }

    /// <summary>Indexes every usable history point once for constant-time series lookup.</summary>
    /// <remarks>
    /// A per-document key set preserves the previous first-matching-entry behavior when a malformed
    /// artifact contains duplicate fixture entries. Documents are ordered here so every cached
    /// series remains chronological even when callers provide an unsorted history collection.
    /// </remarks>
    private static Dictionary<(string Fixture, string EnvironmentKey, string Metric), List<SeriesPoint>>
        IndexHistory(IReadOnlyList<BaselineDocument> history)
    {
        var index = new Dictionary<(string Fixture, string EnvironmentKey, string Metric), List<SeriesPoint>>();
        foreach (BaselineDocument document in history.OrderBy(item => item.GeneratedUtc))
        {
            var seenInDocument = new HashSet<(string Fixture, string EnvironmentKey, string Metric)>();
            foreach (BaselineEntry entry in document.Entries)
            {
                foreach ((string metric, double value) in entry.Metrics)
                {
                    if (value <= 0.0) continue;
                    var key = (
                        Fixture: entry.Fixture,
                        EnvironmentKey: SeriesEnvironmentKey(entry.Environment, ClassifyMetric(metric)),
                        Metric: metric);
                    if (!seenInDocument.Add(key)) continue;
                    if (!index.TryGetValue(key, out List<SeriesPoint>? points))
                    {
                        points = new List<SeriesPoint>();
                        index.Add(key, points);
                    }
                    points.Add(new SeriesPoint(document.Commit, document.GeneratedUtc, value));
                }
            }
        }
        return index;
    }

    /// <summary>
    /// How much of the environment a metric's series has to agree on.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The point comparison qualifies on the whole environment, processor model included, and that
    /// is right for timing: the same code on an EPYC 9V74 and an EPYC 7763 is not the same
    /// measurement. Carrying that rule into a SERIES breaks it, though, and the census showed how:
    /// the runs of 25, 26, 27 and 28 August landed on 7763, 9V74, 7763 and 7763, so a
    /// processor-qualified window of four runs held only three usable points and step detection
    /// never engaged. Runner allocation is not something this repository controls, so a series
    /// keyed that tightly would stay too shallow to detect anything, permanently.
    /// </para>
    /// <para>
    /// Allocated bytes and exact counts are deterministic - the census's own null-pair study puts
    /// allocation at a median ratio of 1.000 - so they are comparable across processor models.
    /// Process memory peaks are allocator-quantised and scale with the degree of parallelism, so
    /// they keep the processor COUNT and drop only the model. Timing keeps everything.
    /// </para>
    /// </remarks>
    private static string SeriesEnvironmentKey(string environment, MetricStability stability)
    {
        if (stability is MetricStability.Timing or MetricStability.VolatileTiming) return environment;

        // frameworkMajor|osPlatform|arch|engine|processorCount|processorModel
        int lastSeparator = environment.LastIndexOf('|');
        return lastSeparator < 0 ? environment : environment.Substring(0, lastSeparator);
    }

    private static double Median(IReadOnlyList<double> values)
    {
        if (values.Count == 0) return 0.0;
        double[] sorted = values.ToArray();
        Array.Sort(sorted);
        int middle = sorted.Length / 2;
        return sorted.Length % 2 == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2.0;
    }

    /// <summary>
    /// The most recent sustained level shift in a series, or null when the series is one level.
    /// </summary>
    /// <remarks>
    /// Split at every interior index and take the latest split whose two sides differ by more than
    /// the metric's own limit and its noise floor, with at least
    /// <see cref="MinimumSegmentPoints"/> measurements on each side. Medians rather than means, so
    /// one outlier on either side cannot manufacture or hide a step.
    /// </remarks>
    private static StepChange? FindLastStep(
        IReadOnlyList<SeriesPoint> series, double limit, double noiseFloor)
    {
        // Segment greedily from the start rather than scanning backwards for any split that
        // qualifies. Scanning backwards attributes a step one measurement too late: with
        // 300, 300, 900, 900, 900 the split before the last two also "qualifies", because the
        // median of 300, 300, 900 is still 300 - so the change gets blamed on the commit AFTER the
        // one that caused it. Taking the earliest split that separates the levels, then looking for
        // further steps only beyond it, names the first run that measured the new level, which is
        // the commit range the culprit is actually in.
        StepChange? latest = null;
        int start = 0;

        while (series.Count - start >= MinimumSegmentPoints * 2)
        {
            int found = -1;
            double was = 0.0;
            double now = 0.0;

            for (int split = start + MinimumSegmentPoints; split <= series.Count - MinimumSegmentPoints; split++)
            {
                var before = new List<double>();
                for (int i = start; i < split; i++) before.Add(series[i].Value);
                var after = new List<double>();
                for (int i = split; i < series.Count; i++) after.Add(series[i].Value);

                double left = Median(before);
                double right = Median(after);
                if (left <= 0.0) continue;
                if (right / left <= limit || right - left <= noiseFloor) continue;

                found = split;
                was = left;
                now = right;
                break;
            }

            if (found < 0) break;
            latest = new StepChange(was, now, series[found - 1].Commit, series[found].Commit, found);
            start = found;
        }

        return latest;
    }

    /// <summary>
    /// The level the current run should be judged against: the series median since its last step.
    /// </summary>
    private static double ReferenceLevel(IReadOnlyList<SeriesPoint> series, StepChange? step)
    {
        if (step is null) return Median(series.Select(point => point.Value).ToList());
        if (step.BoundaryIndex < 0 || step.BoundaryIndex >= series.Count)
            return Median(series.Select(point => point.Value).ToList());

        return Median(series.Skip(step.BoundaryIndex).Select(point => point.Value).ToList());
    }

    private static void DetectCohortOutliers(
        IReadOnlyList<CensusRecord> records,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (IGrouping<string, CensusRecord> cohort in records
                     .Where(r => r.Status == "ok"
                         && r.ParameterCount > 0
                         && r.Metric("trainStepMs") > 0.0
                         && r.Metric("steadyForwardMedianMs") > 0.0)
                     .GroupBy(r => r.Cohort))
        {
            // Parameter-decade peers can have radically different sequence and spatial work.
            // Compare the training overhead relative to each model's own serving forward; absolute
            // forward, train, wall, and memory ceilings below still catch end-to-end slowness.
            double[] values = cohort.Select(r => Math.Log(1.0 + TrainingAmplification(r)))
                .OrderBy(v => v).ToArray();
            if (values.Length < 5) continue;
            double median = MedianOfSorted(values);
            double mad = MedianOfSorted(values.Select(v => Math.Abs(v - median)).OrderBy(v => v).ToArray());
            if (mad <= 1e-9) continue;

            foreach (CensusRecord record in cohort)
            {
                double amplification = TrainingAmplification(record);
                double robustZ = 0.6745 * (Math.Log(1.0 + amplification) - median) / mad;
                if (robustZ > 6.0)
                {
                    diagnostics.Add(Diagnostic.Warning(record.Fixture, "trainingAmplification",
                        $"training is {amplification:F1}x its steady forward " +
                        $"(robust z={robustZ:F1}, cohort={cohort.Key})"));
                }
            }
        }

        foreach (IGrouping<string, CensusRecord> cohort in records
                     .Where(r => r.Status == "ok" && r.ParameterCount > 0 && r.Metric("peakWorkingSetBytes") > 0.0)
                     .GroupBy(r => r.Cohort))
        {
            double[] values = cohort.Select(r => Math.Log(1.0 + r.Metric("peakWorkingSetBytes"))).OrderBy(v => v).ToArray();
            if (values.Length < 5) continue;
            double median = MedianOfSorted(values);
            double mad = MedianOfSorted(values.Select(v => Math.Abs(v - median)).OrderBy(v => v).ToArray());
            if (mad <= 1e-9) continue;

            foreach (CensusRecord record in cohort)
            {
                double robustZ = 0.6745 * (Math.Log(1.0 + record.Metric("peakWorkingSetBytes")) - median) / mad;
                if (robustZ > 6.0)
                {
                    diagnostics.Add(Diagnostic.Warning(record.Fixture, "peakWorkingSetBytes",
                        $"robust cohort memory outlier (z={robustZ:F1}, cohort={cohort.Key})"));
                }
            }
        }
    }

    private static double TrainingAmplification(CensusRecord record)
        => record.Metric("trainStepMs") / record.Metric("steadyForwardMedianMs");

    private static void ValidateAbsoluteCeilings(
        IReadOnlyList<CensusRecord> records,
        Options options,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (CensusRecord record in records.Where(r => r.Status == "ok"))
        {
            double trainStep = record.Metric("trainStepMs");
            if (trainStep > options.MaxTrainStepMs)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "trainStepMs",
                    $"single train step {trainStep:F0} ms exceeds {options.MaxTrainStepMs:F0} ms ceiling"));
            }
            // Same dimensional argument as the memory ceilings below, applied to time: one flat
            // millisecond ceiling across an inventory spanning 4-parameter fixtures and 733M-parameter
            // foundation models is not a like-for-like bound, and the largest model in the census is
            // structurally the one that trips it first. Ordinary models stay bounded by the flat ceiling
            // (nothing under ~469M parameters is affected at the default allowance); larger ones get a
            // per-parameter envelope instead, whichever is greater.
            //
            // This is a bound on scale, NOT a statement that the biggest fixture is efficient. Measured
            // over the 863-fixture census, StableVideoSR needs 41 ns/parameter for a steady forward while
            // the next slowest large model (TortoiseTTS, 165M) needs 7.5 and the rest are under 1, so the
            // default allowance is deliberately loose and does not certify that model's forward path.
            double steadyForwardScaledCeiling = record.ParameterCount > 0
                ? record.ParameterCount * options.MaxSteadyForwardNanosecondsPerParameter / 1_000_000.0
                : 0.0;
            double steadyForwardCeiling = Math.Max(options.MaxSteadyForwardP95Ms, steadyForwardScaledCeiling);
            double steadyForwardP95 = record.Metric("steadyForwardP95Ms");
            if (steadyForwardP95 > steadyForwardCeiling)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "steadyForwardP95Ms",
                    $"steady forward p95 {steadyForwardP95:F0} ms exceeds " +
                    $"{steadyForwardCeiling:F0} ms model-scaled ceiling " +
                    $"({options.MaxSteadyForwardNanosecondsPerParameter:F0} ns/parameter)"));
            }
            double wall = record.Metric("wallMs");
            if (wall > options.MaxFixtureWallMs)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "wallMs",
                    $"fixture workload {wall:F0} ms exceeds {options.MaxFixtureWallMs:F0} ms ceiling"));
            }
            // A fixed process ceiling alone is not dimensionally valid across this inventory:
            // the same lane covers tiny classifiers and 500M+ parameter foundation models. An
            // fp32 training step necessarily carries weights plus some combination of gradients,
            // optimizer state and packed execution weights. Bound ordinary models by the fixed
            // floor and large models by a strict bytes-per-parameter amplification envelope. The
            // environment-qualified baseline above remains the tighter regression detector.
            double parameterScaledCeiling = record.ParameterCount > 0
                ? record.ParameterCount * options.MaxPeakBytesPerParameter
                : 0.0;
            double workingSetCeiling = Math.Max(options.MaxPeakWorkingSetBytes, parameterScaledCeiling);
            double privateMemoryCeiling = Math.Max(options.MaxPeakPrivateMemoryBytes, parameterScaledCeiling);

            double peakWorkingSet = record.Metric("peakWorkingSetBytes");
            if (peakWorkingSet > workingSetCeiling)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "peakWorkingSetBytes",
                    $"peak working set {peakWorkingSet / (1024.0 * 1024.0):F1} MiB exceeds " +
                    $"{workingSetCeiling / (1024.0 * 1024.0):F1} MiB model-scaled ceiling " +
                    $"({options.MaxPeakBytesPerParameter:F0} bytes/parameter)"));
            }
            double peakPrivateMemory = record.Metric("peakPrivateMemoryBytes");
            if (peakPrivateMemory > privateMemoryCeiling)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "peakPrivateMemoryBytes",
                    $"peak private memory {peakPrivateMemory / (1024.0 * 1024.0):F1} MiB exceeds " +
                    $"{privateMemoryCeiling / (1024.0 * 1024.0):F1} MiB model-scaled ceiling " +
                    $"({options.MaxPeakBytesPerParameter:F0} bytes/parameter)"));
            }
            if (record.Metric("projectedTrainingReduceLossMs") > options.MaxCorrectnessProbeMs)
            {
                diagnostics.Add(Diagnostic.Warning(record.Fixture, "projectedTrainingReduceLossMs",
                    $"projected correctness workload is {record.Metric("projectedTrainingReduceLossMs") / 1000.0:F1} s"));
            }
        }
    }

    private static SummaryDocument BuildSummary(
        IReadOnlyList<CensusRecord> records,
        IReadOnlyList<Diagnostic> diagnostics,
        int? expectedCount)
    {
        CensusRecord[] slowest = records.Where(r => r.Status == "ok")
            .OrderByDescending(r => r.Metric("trainStepMs")).Take(25).ToArray();
        CensusRecord[] largest = records.Where(r => r.Status == "ok")
            .OrderByDescending(r => r.Metric("peakWorkingSetBytes")).Take(25).ToArray();
        return new SummaryDocument
        {
            SchemaVersion = 1,
            GeneratedUtc = DateTimeOffset.UtcNow,
            ExpectedFixtures = expectedCount,
            ObservedFixtures = records.Select(r => r.Fixture).Distinct().Count(),
            ErrorCount = diagnostics.Count(d => d.Severity == "error"),
            WarningCount = diagnostics.Count(d => d.Severity == "warning"),
            StatusCounts = records.GroupBy(r => r.Status)
                .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal),
            Diagnostics = diagnostics.ToArray(),
            SlowestTrainSteps = slowest.Select(r => new RankedRecord
            {
                Fixture = r.Fixture,
                Model = r.Model,
                ParameterCount = r.ParameterCount,
                TrainStepMs = r.Metric("trainStepMs"),
                BackwardMs = r.Metric("backwardMs"),
                TapeEntries = r.Metric("tapeEntries"),
                AllocatedBytes = r.Metric("allocatedBytes"),
                PeakWorkingSetBytes = r.Metric("peakWorkingSetBytes"),
            }).ToArray(),
            LargestPeakWorkingSets = largest.Select(r => new RankedRecord
            {
                Fixture = r.Fixture,
                Model = r.Model,
                ParameterCount = r.ParameterCount,
                TrainStepMs = r.Metric("trainStepMs"),
                BackwardMs = r.Metric("backwardMs"),
                TapeEntries = r.Metric("tapeEntries"),
                AllocatedBytes = r.Metric("allocatedBytes"),
                PeakWorkingSetBytes = r.Metric("peakWorkingSetBytes"),
            }).ToArray(),
        };
    }

    /// <summary>
    /// Loads the prior baselines that form the series, newest last.
    /// </summary>
    /// <remarks>
    /// Every file in the directory is a baseline published by an earlier census run. A file that
    /// will not parse is reported and skipped rather than failing the census: losing one point of
    /// history weakens attribution slightly, while refusing to run at all would take the whole gate
    /// down over a corrupt artifact.
    /// </remarks>
    private static IReadOnlyList<BaselineDocument> LoadHistory(
        string? directory, ICollection<Diagnostic> diagnostics)
    {
        if (directory is null || !Directory.Exists(directory)) return Array.Empty<BaselineDocument>();

        var documents = new List<BaselineDocument>();
        foreach (string path in Directory.GetFiles(directory, "*.json", SearchOption.AllDirectories)
                     .OrderBy(p => p, StringComparer.Ordinal))
        {
            BaselineDocument? document;
            try
            {
                document = JsonSerializer.Deserialize<BaselineDocument>(File.ReadAllText(path), JsonOptions);
            }
            catch (Exception ex) when (ex is JsonException or IOException or UnauthorizedAccessException)
            {
                diagnostics.Add(Diagnostic.Warning("<census>", "history",
                    $"could not read {Path.GetFileName(path)} as a baseline ({ex.Message}); "
                    + "that point is missing from the series"));
                continue;
            }

            if (document is null) continue;
            document.Commit ??= "";
            if (Array.IndexOf(ComparableBaselineSchemaVersions, document.SchemaVersion) < 0) continue;
            documents.Add(document);
        }

        documents.Sort((left, right) => left.GeneratedUtc.CompareTo(right.GeneratedUtc));

        // Say how much history the gate actually has. A census that silently fell back to a single
        // prior point looks identical in its output to one reasoning over a window, and the two
        // give different answers - so the depth is stated rather than assumed.
        if (documents.Count >= MinimumSeriesPoints)
        {
            Console.WriteLine($"history: comparing against a series of {documents.Count} prior census run(s)");
        }
        else
        {
            diagnostics.Add(Diagnostic.Warning("<census>", "history",
                $"only {documents.Count} prior census run(s) available; "
                + $"{MinimumSeriesPoints} are needed before step detection engages, so this run "
                + "is compared against the single most recent baseline"));
        }

        return documents;
    }

    private static BaselineDocument BuildBaseline(IReadOnlyList<CensusRecord> records, string commit) => new()
    {
        SchemaVersion = CurrentBaselineSchemaVersion,
        GeneratedUtc = DateTimeOffset.UtcNow,
        Commit = commit,
        Entries = records.Where(record => record.Status == "ok").Select(record => new BaselineEntry
        {
            Fixture = record.Fixture,
            Environment = record.Environment,
            Metrics = new Dictionary<string, double>(record.Metrics),
        }).OrderBy(entry => entry.Fixture, StringComparer.Ordinal).ToArray(),
    };

    private static void WriteJson<T>(string path, T value)
    {
        string fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        string temporary = fullPath + "." + Guid.NewGuid().ToString("N") + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(value, JsonOptions));
        File.Move(temporary, fullPath, overwrite: true);
    }

    private static void PrintSummary(SummaryDocument summary, string outputPath)
    {
        Console.WriteLine($"fixtures: {summary.ObservedFixtures}/{summary.ExpectedFixtures?.ToString() ?? "unbounded"}");
        Console.WriteLine($"diagnostics: {summary.ErrorCount} error(s), {summary.WarningCount} warning(s)");
        foreach (RankedRecord record in summary.SlowestTrainSteps.Take(10))
            Console.WriteLine($"{record.TrainStepMs,10:F1} ms  {record.ParameterCount,14:N0} params  {record.Fixture}");
        Console.WriteLine("largest peak working sets:");
        foreach (RankedRecord record in summary.LargestPeakWorkingSets.Take(10))
            Console.WriteLine($"{record.PeakWorkingSetBytes / (1024.0 * 1024.0),10:F1} MiB  {record.ParameterCount,14:N0} params  {record.Fixture}");
        Console.WriteLine($"summary: {outputPath}");
    }

    /// <summary>Returns the median of an array that the caller has already sorted.</summary>
    private static double MedianOfSorted(double[] sorted) => sorted.Length % 2 == 1
        ? sorted[sorted.Length / 2]
        : (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]) / 2.0;

    private static int RunSelfTest()
    {
        static int Fail(string check)
        {
            Console.Error.WriteLine($"ModelPerfProbe self-test failed: {check}");
            return 1;
        }

        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-perf-selftest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string record = """
            {"schemaVersion":1,"status":"ok","fixture":"F","model":"M","precision":"System.Single",
             "parameterCount":10,"engine":"Cpu","frameworkMajor":10,"osPlatform":"test","processArchitecture":"X64","processorCount":1,"processorModel":"test-cpu",
             "constructMs":1,"targetPreparationMs":0,"coldForwardMs":2,"steadyForwardMedianMs":1,"steadyForwardP95Ms":1,
             "tapeForwardMs":1,"tapeEntries":2,"backwardMs":2,"trainStepMs":3,"allocatedBytes":4,"wallMs":5,
             "runnerElapsedMs":6,"peakWorkingSetBytes":104857600,"peakPrivateMemoryBytes":125829120,
             "runnerCpuMs":5,"runnerCpuToWallRatio":0.83,
             "projectedTrainingReduceLossMs":90}
            """;
            File.WriteAllText(Path.Combine(directory, "record.json"), record);
            IReadOnlyList<CensusRecord> records = LoadRecords(directory);
            var diagnostics = new List<Diagnostic>();
            ValidateCoverage(records, 1, diagnostics);
            ValidateRecords(records, diagnostics);
            if (records.Count != 1 || diagnostics.Count != 0)
                return Fail("valid record loading and validation");

            string timeout = """
            {"schemaVersion":1,"status":"timeout","fixture":"SlowFixture","model":"","precision":"System.Single",
             "frameworkMajor":10,"osPlatform":"test","processArchitecture":"X64","processorCount":1,
             "phase":"backward","elapsedMs":180001,"error":"hard runtime ceiling exceeded"}
            """;
            File.WriteAllText(Path.Combine(directory, "timeout.json"), timeout);
            records = LoadRecords(directory);
            diagnostics.Clear();
            ValidateCoverage(records, 2, diagnostics);
            ValidateRecords(records, diagnostics);
            if (diagnostics.Count != 1
                || diagnostics[0].Metric != "status"
                || !diagnostics[0].Message.Contains("backward", StringComparison.Ordinal))
                return Fail("timeout phase diagnostic");
            BaselineDocument generatedBaseline = BuildBaseline(records, "0123456789abcdef");
            if (generatedBaseline.SchemaVersion != CurrentBaselineSchemaVersion
                || generatedBaseline.Entries.Length != 1
                || generatedBaseline.Commit != "0123456789abcdef"
                || !generatedBaseline.Entries[0].Environment.EndsWith("|test-cpu", StringComparison.Ordinal))
                return Fail("generated baseline schema, commit stamp and processor-qualified environment key");

            diagnostics.Clear();
            CompareBaseline(records, new BaselineDocument { SchemaVersion = 1 }, new Options(), diagnostics);
            if (diagnostics.Count != 1
                || diagnostics[0].Metric != "baseline"
                || !diagnostics[0].Message.Contains("not comparable", StringComparison.Ordinal))
                return Fail("incompatible baseline schema warning");

            // A baseline written by the previous release still has to be readable, or the gate
            // silently stops comparing for as long as it takes a new one to be published.
            diagnostics.Clear();
            CompareBaseline(records, new BaselineDocument { SchemaVersion = 2 }, new Options(), diagnostics);
            if (diagnostics.Any(d => d.Metric == "baseline" && d.Message.Contains("not comparable", StringComparison.Ordinal)))
                return Fail("schema 2 baseline must stay comparable");

            if (RunSeriesSelfTest() is int seriesFailure and not 0) return seriesFailure;

            // Expensive models are not training hot-path outliers when their train/forward ratio
            // matches their peers; a cheap-forward model with extreme training amplification is.
            CensusRecord Synthetic(int index, double trainMs, double forwardMs, long parameters)
                => new()
                {
                    Fixture = $"Synthetic{index}",
                    Model = "Synthetic",
                    Status = "ok",
                    Environment = "test",
                    Cohort = "System.Single|10^8",
                    Phase = "",
                    Error = "",
                    ParameterCount = parameters,
                    Metrics = new Dictionary<string, double>
                    {
                        ["trainStepMs"] = trainMs,
                        ["steadyForwardMedianMs"] = forwardMs,
                        ["peakWorkingSetBytes"] = parameters * (20.0 + index * 0.25),
                    },
                };

            var comparable = new[]
            {
                Synthetic(0, 90, 50, 100_000_000),
                Synthetic(1, 95, 50, 110_000_000),
                Synthetic(2, 100, 50, 120_000_000),
                Synthetic(3, 105, 50, 130_000_000),
                Synthetic(4, 110, 50, 140_000_000),
                Synthetic(5, 10_000, 5_000, 150_000_000),
            };
            diagnostics.Clear();
            DetectCohortOutliers(comparable, diagnostics);
            if (diagnostics.Count != 0)
                return Fail("comparable cohort remains silent");

            var amplified = comparable.ToArray();
            amplified[^1] = Synthetic(6, 50_000, 500, 150_000_000);
            diagnostics.Clear();
            DetectCohortOutliers(amplified, diagnostics);
            if (!diagnostics.Any(d => d.Metric == "trainingAmplification"))
                return Fail("training amplification detection");
            Console.WriteLine("ModelPerfProbe self-test passed.");
            return 0;
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static int PrintUsage()
    {
        Console.WriteLine("ModelPerfProbe --results DIR --output FILE [--expected-count N]");
        Console.WriteLine("  [--baseline FILE] [--write-baseline FILE] [--max-regression-ratio 2.5]");
        Console.WriteLine("  [--history DIR] [--perf-intent FILE] [--commit SHA]");
        Console.WriteLine("  [--max-allocation-regression-ratio 2.5] [--max-memory-regression-ratio 1.6]");
        Console.WriteLine("  [--max-volatile-timing-regression-ratio 5.0]");
        Console.WriteLine("  [--uncorroborated-timing-ratio 4.0] [--corroboration-ratio 1.25]");
        Console.WriteLine("  [--max-train-step-ms 120000] [--max-steady-forward-p95-ms 30000]");
        Console.WriteLine("  [--max-steady-forward-ns-per-parameter 64]");
        Console.WriteLine("  [--max-fixture-wall-ms 120000] [--max-peak-working-set-bytes 8589934592]");
        Console.WriteLine("  [--max-peak-private-memory-bytes 9663676416] [--max-peak-bytes-per-parameter 32]");
        Console.WriteLine("  [--max-correctness-probe-ms 120000]");
        Console.WriteLine("ModelPerfProbe --self-test");
        return 2;
    }

    private sealed class Options
    {
        public string? ResultsDirectory { get; private set; }
        public string? OutputPath { get; private set; }
        public string? BaselinePath { get; private set; }
        public string? WriteBaselinePath { get; private set; }

        /// <summary>Directory of prior baseline documents, forming the series to judge against.</summary>
        public string? HistoryDirectory { get; private set; }

        /// <summary>File of cost changes that were made deliberately, each with its reason.</summary>
        public string? PerfIntentPath { get; private set; }

        /// <summary>The commit these measurements are taken at, stamped into the written baseline.</summary>
        public string Commit { get; private set; } = "";
        public int? ExpectedCount { get; private set; }
        public double MaxRegressionRatio { get; private set; } = 2.5;
        public double MaxAllocationRegressionRatio { get; private set; } = 2.5;
        public double MaxMemoryRegressionRatio { get; private set; } = 1.6;
        public double MaxVolatileTimingRegressionRatio { get; private set; } = 5.0;
        public double UncorroboratedTimingRatio { get; private set; } = 4.0;
        public double CorroborationRatio { get; private set; } = 1.25;
        public double MaxTrainStepMs { get; private set; } = 120_000.0;
        public double MaxSteadyForwardP95Ms { get; private set; } = 30_000.0;

        /// <summary>
        /// Per-parameter steady-forward allowance for foundation-scale fixtures, in nanoseconds.
        /// </summary>
        /// <remarks>
        /// Applied as <c>max(flat ceiling, ParameterCount * this)</c>, so it only ever raises the bound and
        /// only for models large enough for the product to exceed the flat ceiling — above roughly 469M
        /// parameters at the default. 64 ns/parameter leaves headroom over the 41-47 ns/parameter
        /// StableVideoSR was measured at across runs, since this metric carries about +/-15% runner noise.
        /// </remarks>
        public double MaxSteadyForwardNanosecondsPerParameter { get; private set; } = 64.0;
        public double MaxFixtureWallMs { get; private set; } = 120_000.0;
        public double MaxPeakWorkingSetBytes { get; private set; } = 8_589_934_592.0;
        public double MaxPeakPrivateMemoryBytes { get; private set; } = 9_663_676_416.0;
        public double MaxPeakBytesPerParameter { get; private set; } = 32.0;
        public double MaxCorrectnessProbeMs { get; private set; } = 120_000.0;
        public bool SelfTest { get; private set; }

        public static Options? Parse(string[] args)
        {
            var options = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"missing value after {args[i]}");
                switch (args[i])
                {
                    case "--results": options.ResultsDirectory = Next(); break;
                    case "--output": options.OutputPath = Next(); break;
                    case "--baseline": options.BaselinePath = Next(); break;
                    case "--write-baseline": options.WriteBaselinePath = Next(); break;
                    case "--history": options.HistoryDirectory = Next(); break;
                    case "--perf-intent": options.PerfIntentPath = Next(); break;
                    case "--commit": options.Commit = Next() ?? ""; break;
                    case "--expected-count": options.ExpectedCount = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-regression-ratio": options.MaxRegressionRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-allocation-regression-ratio": options.MaxAllocationRegressionRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-memory-regression-ratio": options.MaxMemoryRegressionRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-volatile-timing-regression-ratio": options.MaxVolatileTimingRegressionRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--uncorroborated-timing-ratio": options.UncorroboratedTimingRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--corroboration-ratio": options.CorroborationRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-train-step-ms": options.MaxTrainStepMs = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-steady-forward-p95-ms": options.MaxSteadyForwardP95Ms = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-steady-forward-ns-per-parameter": options.MaxSteadyForwardNanosecondsPerParameter = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-fixture-wall-ms": options.MaxFixtureWallMs = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-peak-working-set-bytes": options.MaxPeakWorkingSetBytes = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-peak-private-memory-bytes": options.MaxPeakPrivateMemoryBytes = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-peak-bytes-per-parameter": options.MaxPeakBytesPerParameter = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-correctness-probe-ms": options.MaxCorrectnessProbeMs = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--self-test": options.SelfTest = true; break;
                    default: return null;
                }
            }
            return options.SelfTest || (!string.IsNullOrWhiteSpace(options.ResultsDirectory)
                && !string.IsNullOrWhiteSpace(options.OutputPath)) ? options : null;
        }
    }

    private sealed class CensusRecord
    {
        public required string Fixture { get; init; }
        public required string Model { get; init; }
        public required string Status { get; init; }
        public required string Environment { get; init; }
        public required string Cohort { get; init; }
        public required string Phase { get; init; }
        public required string Error { get; init; }
        public double ElapsedMs { get; init; }
        public long ParameterCount { get; init; }
        public required IReadOnlyDictionary<string, double> Metrics { get; init; }
        public double Metric(string name) => Metrics.TryGetValue(name, out double value) ? value : 0.0;

        public static CensusRecord FromJson(string path, JsonElement root)
        {
            string Text(string name) => root.TryGetProperty(name, out JsonElement value) ? value.ToString() : "";
            double Number(string name) => root.TryGetProperty(name, out JsonElement value) && value.TryGetDouble(out double parsed) ? parsed : double.NaN;
            long parameters = root.TryGetProperty("parameterCount", out JsonElement parameterElement)
                && parameterElement.TryGetInt64(out long parsedParameters) ? parsedParameters : 0;
            // Baselines must distinguish materially different runners without becoming invalid on
            // every .NET servicing release or Linux kernel update. Keep the full descriptions in
            // the raw record for diagnosis, but key comparisons by runtime major + OS family.
            string frameworkKey = Text("frameworkMajor");
            if (string.IsNullOrWhiteSpace(frameworkKey)) frameworkKey = Text("framework");
            string osKey = Text("osPlatform");
            if (string.IsNullOrWhiteSpace(osKey)) osKey = Text("os");
            string processorModel = Text("processorModel");
            if (string.IsNullOrWhiteSpace(processorModel)) processorModel = "unknown";
            string environment = string.Join("|", frameworkKey, osKey, Text("processArchitecture"),
                Text("engine"), Text("processorCount"), processorModel);
            int magnitude = parameters <= 0 ? 0 : (int)Math.Floor(Math.Log10(parameters));
            var metrics = new Dictionary<string, double>(StringComparer.Ordinal);
            foreach (string metric in RequiredMetrics.Concat(new[]
                     {
                         "tapeEntries", "gradientTensorCount", "cpuMs", "cpuToWallRatio",
                         "projectedTrainingReduceLossMs", "projectedMoreDataMs",
                     }))
                metrics[metric] = Number(metric);

            return new CensusRecord
            {
                Fixture = Text("fixture"),
                Model = Text("model"),
                Status = Text("status"),
                Environment = environment,
                Cohort = $"{Text("precision")}|10^{magnitude}",
                Phase = Text("phase"),
                Error = Text("error"),
                ElapsedMs = Number("elapsedMs"),
                ParameterCount = parameters,
                Metrics = metrics,
            };
        }
    }

    private sealed class FixtureEnvironmentComparer : IEqualityComparer<(string Fixture, string Environment)>
    {
        public bool Equals((string Fixture, string Environment) x, (string Fixture, string Environment) y)
            => StringComparer.Ordinal.Equals(x.Fixture, y.Fixture)
               && StringComparer.Ordinal.Equals(x.Environment, y.Environment);
        public int GetHashCode((string Fixture, string Environment) value)
            => HashCode.Combine(StringComparer.Ordinal.GetHashCode(value.Fixture), StringComparer.Ordinal.GetHashCode(value.Environment));
    }

    private sealed class BaselineDocument
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; set; }
        [JsonPropertyName("generatedUtc")] public DateTimeOffset GeneratedUtc { get; set; }

        /// <summary>The commit these measurements were taken at, so a step can name its culprit.</summary>
        /// <remarks>
        /// Absent in schema 2, which recorded only the numbers. A step detected across a window that
        /// includes schema-2 points can still be detected; it just cannot be attributed, and says so.
        /// </remarks>
        [JsonPropertyName("commit")] public string Commit { get; set; } = "";

        [JsonPropertyName("entries")] public BaselineEntry[] Entries { get; set; } = Array.Empty<BaselineEntry>();
    }

    private sealed class BaselineEntry
    {
        [JsonPropertyName("fixture")] public string Fixture { get; set; } = "";
        [JsonPropertyName("environment")] public string Environment { get; set; } = "";
        [JsonPropertyName("metrics")] public Dictionary<string, double> Metrics { get; set; } = new();
    }

    private sealed class SummaryDocument
    {
        [JsonPropertyName("schemaVersion")] public int SchemaVersion { get; set; }
        [JsonPropertyName("generatedUtc")] public DateTimeOffset GeneratedUtc { get; set; }
        [JsonPropertyName("expectedFixtures")] public int? ExpectedFixtures { get; set; }
        [JsonPropertyName("observedFixtures")] public int ObservedFixtures { get; set; }
        [JsonPropertyName("errorCount")] public int ErrorCount { get; set; }
        [JsonPropertyName("warningCount")] public int WarningCount { get; set; }
        [JsonPropertyName("statusCounts")] public Dictionary<string, int> StatusCounts { get; set; } = new();
        [JsonPropertyName("diagnostics")] public Diagnostic[] Diagnostics { get; set; } = Array.Empty<Diagnostic>();
        [JsonPropertyName("slowestTrainSteps")] public RankedRecord[] SlowestTrainSteps { get; set; } = Array.Empty<RankedRecord>();
        [JsonPropertyName("largestPeakWorkingSets")] public RankedRecord[] LargestPeakWorkingSets { get; set; } = Array.Empty<RankedRecord>();
    }

    private sealed class Diagnostic
    {
        [JsonPropertyName("severity")] public string Severity { get; set; } = "";
        [JsonPropertyName("fixture")] public string Fixture { get; set; } = "";
        [JsonPropertyName("metric")] public string Metric { get; set; } = "";
        [JsonPropertyName("message")] public string Message { get; set; } = "";
        public static Diagnostic Error(string fixture, string metric, string message) => new() { Severity = "error", Fixture = fixture, Metric = metric, Message = message };
        public static Diagnostic Warning(string fixture, string metric, string message) => new() { Severity = "warning", Fixture = fixture, Metric = metric, Message = message };
    }

    private sealed class RankedRecord
    {
        [JsonPropertyName("fixture")] public string Fixture { get; set; } = "";
        [JsonPropertyName("model")] public string Model { get; set; } = "";
        [JsonPropertyName("parameterCount")] public long ParameterCount { get; set; }
        [JsonPropertyName("trainStepMs")] public double TrainStepMs { get; set; }
        [JsonPropertyName("backwardMs")] public double BackwardMs { get; set; }
        [JsonPropertyName("tapeEntries")] public double TapeEntries { get; set; }
        [JsonPropertyName("allocatedBytes")] public double AllocatedBytes { get; set; }
        [JsonPropertyName("peakWorkingSetBytes")] public double PeakWorkingSetBytes { get; set; }
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };
}
