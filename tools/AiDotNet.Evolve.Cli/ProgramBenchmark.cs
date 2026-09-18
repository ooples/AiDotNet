using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolve.Cli;

// Development pilot, not a sealed benchmark or an execution boundary for untrusted code.
internal static class ProgramBenchmark
{
    internal const string Unit = "authored-csharp-worker-dispatch-v1";
    internal const string Method = "authored-fibonacci-end-to-end-v1";
    private static readonly JsonSerializerSettings Json = new()
    {
        Formatting = Formatting.Indented,
        Converters = { new StringEnumConverter() }
    };

    internal static ProgramGenome Baseline => Genome("return n < 2 ? n : F(n - 1) + F(n - 2);");
    internal static ProgramGenome Incorrect => Genome("return -1;");
    internal static ProgramGenome Iterative => Genome("long a = 0, b = 1; for (int i = 0; i < n; i++) { long next = a + b; a = b; b = next; } return a;");
    private static ProgramGenome Genome(string body) => new(
        "public static class P { public static long F(int n) { " + body +
        " } public static void Main() { System.Console.Write(F(int.Parse(System.Console.ReadLine()!, System.Globalization.CultureInfo.InvariantCulture)).ToString(System.Globalization.CultureInfo.InvariantCulture)); } }",
        ProgramLanguage.CSharp);

    internal static void RequireCatalog(ProgramGenome genome)
    {
        if (genome.Language != ProgramLanguage.CSharp ||
            !new[] { Baseline, Incorrect, Iterative }.Any(known => string.Equals(known.Source, genome.Source, StringComparison.Ordinal)))
            throw new ArgumentException("This pilot executes only its exact authored source catalog.", nameof(genome));
    }

    // Independent recurrence (fast doubling), not either candidate's implementation.
    internal static long Reference(int input)
    {
        if (input is < 0 or > 40) throw new ArgumentOutOfRangeException(nameof(input));
        return Pair(input).First;
        static (long First, long Next) Pair(int n)
        {
            if (n == 0) return (0, 1);
            var (a, b) = Pair(n / 2);
            long c = a * (2 * b - a), d = a * a + b * b;
            return n % 2 == 0 ? (c, d) : (d, c + d);
        }
    }

    internal static string RequireWorker(string worker)
    {
        if (!Path.IsPathFullyQualified(worker) || worker.Any(c => char.IsControl(c) || "\"$`{}".Contains(c)))
            throw new ArgumentException("--worker requires an absolute path without shell/template metacharacters.");
        string full = Path.GetFullPath(worker);
        if (!string.Equals(Path.GetFileName(full), "AiDotNet.CSharp.Worker.dll", StringComparison.Ordinal) || !File.Exists(full))
            throw new ArgumentException("--worker must name the built AiDotNet.CSharp.Worker.dll from this checkout.");
        return full;
    }

    internal static async Task<int> RunAsync(string worker, string directory, int runs, int measurements,
        TextWriter output, CancellationToken token)
    {
        if (runs is < 1 or > 12) throw new ArgumentOutOfRangeException(nameof(runs), "--runs must be 1..12.");
        if (measurements is < 1 or > 9) throw new ArgumentOutOfRangeException(nameof(measurements), "--measurements must be 1..9.");
        worker = RequireWorker(worker);
        string destination = Path.GetFullPath(directory);
        if (Directory.Exists(destination) || File.Exists(destination)) throw new ArgumentException("--output must be a new directory.");
        string runtime = RuntimeEnvironment.GetRuntimeDirectory();
        string host = Path.GetFullPath(Path.Combine(runtime, "..", "..", "..", OperatingSystem.IsWindows() ? "dotnet.exe" : "dotnet"));
        if (!File.Exists(host)) throw new FileNotFoundException("Cannot locate the current dotnet host.");
        string workerRoot = Path.GetDirectoryName(worker)!;
        string workerTarget = ReadWorkerTarget(Path.ChangeExtension(worker, ".runtimeconfig.json"));
        var binaries = new[] { worker, Path.ChangeExtension(worker, ".runtimeconfig.json"), Path.ChangeExtension(worker, ".deps.json"),
            Path.Combine(workerRoot, "Microsoft.CodeAnalysis.dll"), Path.Combine(workerRoot, "Microsoft.CodeAnalysis.CSharp.dll"),
            typeof(ProgramBenchmark).Assembly.Location, typeof(ProgramEvolutionOptions).Assembly.Location,
            typeof(EvolutionEngine<>).Assembly.Location, host, typeof(object).Assembly.Location };
        var identity = binaries.Distinct(StringComparer.Ordinal).ToDictionary(path => path, Hash, StringComparer.Ordinal);
        var report = new Report
        {
            Started = DateTimeOffset.UtcNow,
            Measurements = measurements,
            BinaryHashes = identity,
            WorkerRequestedFramework = workerTarget,
            Runtime = RuntimeInformation.FrameworkDescription,
            Platform = RuntimeInformation.OSDescription,
            Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            ProcessorCount = Environment.ProcessorCount,
            Runs = Enumerable.Range(0, runs).Select(index => new Run { Index = index }).ToArray(),
            Catalog = new[] { Baseline, Incorrect, Iterative }.ToDictionary(genome => genome.Id, genome => genome.Source)
        };
        // A write-once plan exists before any dispatch; an interrupted process cannot disappear from the record.
        Directory.CreateDirectory(destination);
        WriteNew(Path.Combine(destination, "plan.json"), report);
        try
        {
            foreach (Run run in report.Runs)
            {
                if (token.IsCancellationRequested) break;
                await ExecuteRun(run, worker, host, Path.Combine(destination, "work", run.Index.ToString(CultureInfo.InvariantCulture)),
                    measurements, EvolutionHash.Combine(identity.OrderBy(pair => pair.Key, StringComparer.Ordinal).Select(pair => pair.Value)),
                    options => new ProcessProgramExecutionEngine(options), token).ConfigureAwait(false);
                WriteNew(Path.Combine(destination, $"run-{run.Index:D2}.json"), run);
                output.WriteLine($"Run {run.Index + 1}/{runs}: {run.Status}; report retained.");
            }
        }
        finally
        {
            report.Finished = DateTimeOffset.UtcNow;
            report.BinariesUnchanged = identity.All(pair => MatchesHash(pair.Key, pair.Value));
            WriteNew(Path.Combine(destination, "report.json"), report);
        }
        output.WriteLine(Path.Combine(destination, "report.json"));
        if (token.IsCancellationRequested) return EvolveCommandLine.ExitCancelled;
        return report.BinariesUnchanged && report.Runs.All(run => run.Status == "completed")
            ? EvolveCommandLine.ExitSuccess : EvolveCommandLine.ExitRunFailed;
    }

    internal static async Task ExecuteRun(Run run, string worker, string host, string workspace, int measurements,
        string binaryIdentity, Func<ProgramSandboxOptions, IProgramExecutionEngine> executionFactory, CancellationToken token)
    {
        var ledger = new EvolutionResourceLedger("authored-runtime-" + run.Index,
            EvolutionResources.Of("cost_units", 3 * (4 + measurements) + 2 * measurements));
        run.Status = "running";
        try
        {
            var sandbox = new ProgramSandboxOptions { WorkingDirectory = workspace };
            sandbox.Limits.TimeLimitSeconds = 30;
            sandbox.Limits.MemoryLimitMb = 4096;
            sandbox.Limits.MaxConcurrentExecutions = 1;
            string command = "\"" + worker + "\" --source {source}";
            sandbox.SetInterpreter(ProgramLanguage.CSharp, new ProgramInterpreterSpecification(host, command, command + " --compile-only"));
            var execution = executionFactory(sandbox);
            using var ownedExecution = execution as IDisposable;
            var correctness = new MeasuredEvaluator(execution, run, "correctness", new[] { 0, 1, 10, 20 }, true, binaryIdentity);
            var fitness = new MeasuredEvaluator(execution, run, "search", Enumerable.Repeat(39, measurements).ToArray(), false, binaryIdentity);
            var proposals = new CatalogVariation();
            var program = new ProgramEvolutionOptions
            {
                Language = ProgramLanguage.CSharp,
                CustomVariation = proposals,
                CustomFitnessEvaluator = fitness,
                ResourceAccounting = new ProgramEvolutionResourceOptions(ledger, 4 + measurements, Unit)
            };
            program.SeedPrograms.Add(Baseline.Source);
            var search = new EvolutionOptions
            {
                RunId = "authored-runtime-" + run.Index,
                Seed = (ulong)run.Index,
                ArchiveDirection = EvolutionOptimizationDirection.Minimize,
                MaxEvaluationAttempts = 3,
                MaxProposals = 3,
                MaxGenerations = 2,
                ProposalBatchSize = 1
            };
            var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(search).ConfigureProgramEvolution(program).ConfigureProgramCorrectness(correctness)
                .BuildAsync(token).ConfigureAwait(false);
            run.Search = result.ProgramEvolution;
            var winner = result.ProgramEvolution?.BestProgram;
            if (token.IsCancellationRequested) { run.Status = "cancelled"; return; }
            if (winner is null || winner.Id == Incorrect.Id || proposals.GetUsage().Proposals != 2 ||
                !SearchSamplesComplete(run, measurements))
            { run.Status = "search-failed"; return; }
            // Public development confirmation, never used to select a new winner; alternate order across runs.
            var confirmation = new MeasuredEvaluator(execution, run, "confirmation", Enumerable.Repeat(40, measurements).ToArray(), false, binaryIdentity);
            var task = new ResourceMeteredEvolutionTask<ProgramGenome>(new ProgramEvolutionTask(confirmation), ledger, new[] { (decimal)measurements });
            string[] order = run.Index % 2 == 0 ? new[] { "baseline", "winner" } : new[] { "winner", "baseline" };
            for (int i = 0; i < order.Length; i++)
            {
                var genome = order[i] == "baseline" ? Baseline : winner;
                long evaluationId = 100 + i;
                var candidate = new EvolutionCandidate<ProgramGenome>(evaluationId, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id),
                    new EvolutionLineage(null, null, "public-confirmation", null, 0, 0, (ulong)run.Index));
                var measurement = await task.EvaluateAsync(candidate, new EvolutionEvaluationContext(evaluationId, (ulong)run.Index, (ulong)i, 1), token).ConfigureAwait(false);
                run.Confirmation.Add(new Confirmation(order[i], genome.Id, measurement.Status, measurement.Quality));
            }
            run.Status = run.Confirmation.All(item => item.Status == EvolutionEvaluationStatus.Completed) &&
                !ledger.Snapshot().MaximumViolated && ledger.Snapshot().Unknown == 0 ? "completed" : "confirmation-failed";
        }
        catch (OperationCanceledException) { run.Status = "cancelled"; }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException and not AccessViolationException)
        {
            run.Status = "failed";
            run.ErrorType = exception.GetType().Name; // No raw execution output, paths, source or exception payload.
        }
        finally { run.Resources = ledger.Snapshot(); }
    }

    internal sealed class MeasuredEvaluator : IProgramFitnessEvaluator
    {
        private readonly IProgramExecutionEngine _execution;
        private readonly Run _run;
        private readonly int[] _inputs;
        private readonly bool _correctness;
        public string Id { get; }
        public string VersionHash { get; }

        internal MeasuredEvaluator(IProgramExecutionEngine execution, Run run, string phase, int[] inputs, bool correctness, string binaryIdentity)
        {
            if (inputs.Length == 0 || inputs.Length > 9) throw new ArgumentException("Specify 1..9 workload inputs.");
            foreach (int input in inputs) _ = Reference(input);
            _execution = execution; _run = run; _inputs = (int[])inputs.Clone(); _correctness = correctness; Id = phase;
            VersionHash = EvolutionHash.Combine(new[] { Method, Unit, phase, correctness.ToString(), binaryIdentity,
                string.Join(",", inputs.Select(input => input.ToString(CultureInfo.InvariantCulture))) });
        }

        public async ValueTask<EvolutionTaskResult> EvaluateAsync(ProgramGenome candidate, EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        {
            RequireCatalog(candidate);
            cancellationToken.ThrowIfCancellationRequested();
            var elapsed = new List<double>();
            int dispatched = 0, passed = 0;
            foreach (int input in _inputs)
            {
                if (cancellationToken.IsCancellationRequested)
                    return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled, costUnits: dispatched);
                var sample = new Sample(Id, candidate.Id, context.EvaluationId, input);
                _run.Samples.Add(sample);
                dispatched++;
                var timer = Stopwatch.StartNew();
                try
                {
                    var response = await _execution.ExecuteAsync(new ProgramExecuteRequest
                    {
                        Language = candidate.Language,
                        SourceCode = candidate.Source,
                        StdIn = input.ToString(CultureInfo.InvariantCulture)
                    }, cancellationToken).ConfigureAwait(false);
                    timer.Stop();
                    sample.ElapsedMilliseconds = timer.Elapsed.TotalMilliseconds;
                    sample.ExitCode = response.ExitCode;
                    sample.ExecutionErrorCode = response.ErrorCode?.ToString();
                    sample.StdOutTruncated = response.StdOutTruncated;
                    sample.Status = response.Success && response.ExitCode == 0 && !response.StdOutTruncated &&
                        string.Equals(response.StdOut, Reference(input).ToString(CultureInfo.InvariantCulture), StringComparison.Ordinal) ? "passed" : "failed";
                    if (sample.Status == "passed") { passed++; elapsed.Add(sample.ElapsedMilliseconds.Value); }
                }
                catch (OperationCanceledException)
                {
                    sample.ElapsedMilliseconds = timer.Elapsed.TotalMilliseconds;
                    sample.Status = "cancelled";
                    return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled, costUnits: dispatched);
                }
                // Other exceptions leave a started/unknown sample; the enclosing ledger retains the reserved maximum.
            }
            if (cancellationToken.IsCancellationRequested)
                return new EvolutionTaskResult(EvolutionEvaluationStatus.Canceled, costUnits: dispatched);
            if (_correctness) return new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, (double)passed / _inputs.Length, costUnits: dispatched);
            return passed == _inputs.Length
                ? new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, Median(elapsed), EvolutionOptimizationDirection.Minimize, costUnits: dispatched)
                : new EvolutionTaskResult(EvolutionEvaluationStatus.Failed, costUnits: dispatched);
        }
    }

    internal static double Median(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(value => value).ToArray();
        if (sorted.Length == 0 || sorted.Any(value => !double.IsFinite(value) || value < 0)) throw new ArgumentException("Timing samples must be finite and nonnegative.");
        return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : sorted[sorted.Length / 2 - 1] / 2 + sorted[sorted.Length / 2] / 2;
    }

    private static bool SearchSamplesComplete(Run run, int measurements)
    {
        foreach (var genome in new[] { Baseline, Incorrect, Iterative })
        {
            var checks = run.Samples.Where(sample => sample.Phase == "correctness" && sample.GenomeId == genome.Id).ToArray();
            var timing = run.Samples.Where(sample => sample.Phase == "search" && sample.GenomeId == genome.Id).ToArray();
            bool incorrect = genome.Id == Incorrect.Id;
            if (checks.Length != 4 || checks.Any(sample => sample.Status != (incorrect ? "failed" : "passed")) ||
                timing.Length != (incorrect ? 0 : measurements) || timing.Any(sample => sample.Status != "passed")) return false;
        }
        return true;
    }

    internal sealed class CatalogVariation : IProgramVariationOperator
    {
        private int _calls;
        public string Id => "authored-catalog";
        public string VersionHash => EvolutionHash.Combine(new[] { Method, Incorrect.Id, Iterative.Id });
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new(_calls++ == 0 ? Incorrect : Iterative);
        }
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public ProgramEvolutionLlmUsage GetUsage() => new(proposals: _calls);
    }

    internal static void WriteNew(string path, object value)
    {
        using (var stream = new FileStream(path + ".pending", FileMode.CreateNew, FileAccess.Write, FileShare.None))
        {
            using var writer = new StreamWriter(stream, new System.Text.UTF8Encoding(false), 4096, leaveOpen: true);
            writer.Write(JsonConvert.SerializeObject(value, Json));
            writer.Flush(); stream.Flush(true);
        }
        File.Move(path + ".pending", path); // Never replace a prior plan or result.
    }
    private static string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)); }
    internal static bool MatchesHash(string path, string expected)
    {
        try { return Hash(path) == expected; }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    internal static string ReadWorkerTarget(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            if (stream.Length > 16384) throw new ArgumentException("Worker runtimeconfig exceeds 16 KiB.");
            using var reader = new StreamReader(stream, new System.Text.UTF8Encoding(false, true));
            var buffer = new char[16385];
            int count = reader.ReadBlock(buffer, 0, buffer.Length);
            if (count > 16384) throw new ArgumentException("Worker runtimeconfig exceeds 16 KiB.");
            using var json = new JsonTextReader(new StringReader(new string(buffer, 0, count))) { MaxDepth = 16 };
            var config = JObject.Load(json, new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error });
            if (json.Read()) throw new ArgumentException("Worker runtimeconfig contains trailing data.");
            var options = config["runtimeOptions"] as JObject;
            var framework = options?["framework"] as JObject;
            var cap = (options?["configProperties"] as JObject)?["System.GC.HeapHardLimit"];
            var declaredVersion = framework?["version"];
            string? version = declaredVersion?.Type == JTokenType.String ? (string?)declaredVersion : null;
            if (cap?.Type != JTokenType.Integer || cap.ToString() != "268435456" ||
                framework?["name"]?.Type != JTokenType.String || (string?)framework["name"] != "Microsoft.NETCore.App" ||
                string.IsNullOrEmpty(version) || version.Length > 64 || !Version.TryParse(version, out _))
                throw new ArgumentException("Worker runtimeconfig must declare Microsoft.NETCore.App and the expected 256 MiB GC heap limit.");
            return "Microsoft.NETCore.App/" + version;
        }
        catch (JsonException) { throw new ArgumentException("Worker runtimeconfig is invalid."); }
    }

    internal sealed class Report
    {
        public string Methodology => Method;
        public string Limitations => "Fixed authored catalog; no LLM, competitor, sealed tests or isolated algorithm timing. Stopwatch includes fresh process startup, compilation, execution and cleanup. Shared-host load is uncontrolled. Repetitions are not independent task families; no significance or superiority claim. Process supervision is not a security sandbox. Worker paths/binaries must be trusted. No paid services.";
        public string CostUnit => Unit;
        public string CostMeaning => "One dispatched worker request, including failed/cancelled calls; compiler setup, CPU, RAM and model work are not independently metered. Unknown receipts retain the reservation maximum.";
        public int TimeLimitSeconds => 30;
        public int AttemptedMemoryLimitMb => 4096;
        public int ConfiguredWorkerGcHeapHardLimitBytes => 268435456;
        public int[] CorrectnessInputs => new[] { 0, 1, 10, 20 };
        public int SearchInput => 39;
        public int ConfirmationInput => 40;
        public string SearchProtocol => "Seed=index; minimize median milliseconds; baseline seed; incorrect then iterative proposals; max evaluations/proposals=3 including seed; generations=2; proposal batch=1; fresh default EvolutionOptions otherwise.";
        public DateTimeOffset Started { get; set; }
        public DateTimeOffset? Finished { get; set; }
        public int Measurements { get; set; }
        public string Runtime { get; set; } = "";
        public string RuntimeScope => "Runtime identifies the controller. WorkerRequestedFramework is read from its validated runtimeconfig; the resolved child runtime and OS limit enforcement are not independently attested.";
        public string WorkerRequestedFramework { get; set; } = "";
        public string Platform { get; set; } = "";
        public string Architecture { get; set; } = "";
        public int ProcessorCount { get; set; }
        public Dictionary<string, string> BinaryHashes { get; set; } = new();
        public bool BinariesUnchanged { get; set; }
        public Dictionary<string, string> Catalog { get; set; } = new();
        public Run[] Runs { get; set; } = Array.Empty<Run>();
    }
    internal sealed class Run
    {
        public int Index { get; set; }
        public string Status { get; set; } = "not-started";
        public string? ErrorType { get; set; }
        public ProgramEvolutionResult? Search { get; set; }
        public object? Resources { get; set; }
        public List<Sample> Samples { get; } = new();
        public List<Confirmation> Confirmation { get; } = new();
    }
    internal sealed record Confirmation(string Role, string GenomeId, EvolutionEvaluationStatus Status, double? MedianMilliseconds);
    internal sealed record Sample(string Phase, string GenomeId, long EvaluationId, int Input)
    {
        public string Status { get; set; } = "started-unknown";
        public double? ElapsedMilliseconds { get; set; }
        public int? ExitCode { get; set; }
        public string? ExecutionErrorCode { get; set; }
        public bool? StdOutTruncated { get; set; }
    }
}
