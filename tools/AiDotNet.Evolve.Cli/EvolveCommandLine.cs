using System.Globalization;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

namespace AiDotNet.Evolve.Cli;

/// <summary>Every command the tool offers, with its output written to injected writers rather than to the console.</summary>
/// <remarks>
/// Keeping the console out of the command implementations is what makes them testable: a test calls
/// <see cref="ExecuteAsync"/> with two string writers and asserts on the exit code and the text, with no process to
/// spawn and no global console state to restore. <c>Program.Main</c> is then only the adapter that supplies the real
/// console and the real cancellation.
/// </remarks>
internal static class EvolveCommandLine
{
    public const int ExitSuccess = 0;
    public const int ExitUsage = 1;
    public const int ExitCancelled = 2;
    public const int ExitRunFailed = 3;

    /// <summary>Runs one command.</summary>
    /// <param name="args">The command name followed by its options.</param>
    /// <param name="output">Where normal output goes.</param>
    /// <param name="error">Where diagnostics go.</param>
    /// <param name="cancellationToken">Stops a run in progress.</param>
    /// <param name="control">Optional one-run graceful-stop control, distinct from cancellation.</param>
    /// <returns>The process exit code.</returns>
    public static async Task<int> ExecuteAsync(
        string[] args, TextWriter output, TextWriter error, CancellationToken cancellationToken = default,
        EvolutionRunControl? control = null)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp(output);
            return args.Length == 0 ? ExitUsage : ExitSuccess;
        }

        try
        {
            string command = args[0].ToLowerInvariant();
            string[] known = command switch
            {
                "run" => new[] { "config", "run-id", "seed", "max-evaluations", "output", "resume", "json", "show-best", "session", "preflight-max-tests" },
                "preflight" => new[] { "config", "run-id", "seed", "max-evaluations", "output", "resume", "preflight-max-tests" },
                "inspect" or "pause" or "cancel" => new[] { "session" },
                "validate" => new[] { "config", "run-id", "seed", "max-evaluations", "output", "resume" },
                "schema" or "docs" => new[] { "out" },
                "benchmark-program" => new[] { "worker", "output", "runs", "measurements" },
                _ => Array.Empty<string>()
            };

            Arguments rest = Arguments.Parse(args.Skip(1).ToArray(), known);
            return command switch
            {
                "run" => await RunAsync(rest, output, error, cancellationToken, control).ConfigureAwait(false),
                "validate" => Validate(rest, output, error),
                "preflight" => await PreflightAsync(rest, output, cancellationToken).ConfigureAwait(false),
                "inspect" or "pause" or "cancel" => await ControlAsync(command, rest, output, cancellationToken).ConfigureAwait(false),
                "benchmark-program" => await ProgramBenchmark.RunAsync(rest.Require("worker"), rest.Require("output"),
                    rest.TryGet("runs", out string runs) ? ParseInt32(runs, "runs") : 4,
                    rest.TryGet("measurements", out string measurements) ? ParseInt32(measurements, "measurements") : 3,
                    output, cancellationToken).ConfigureAwait(false),
                "schema" => WriteText(rest, output, YamlJsonSchema.Generate(), "aidotnet-config.schema.json"),
                "docs" => WriteText(rest, output, YamlDocsGenerator.Generate(), "yaml-config-reference.md"),
                _ => Fail(error, $"Unknown command '{args[0]}'. Run with --help for usage.")
            };
        }
        catch (OperationCanceledException)
        {
            error.WriteLine("Cancelled.");
            return ExitCancelled;
        }
        catch (RunFailedException failure)
        {
            // A run that started and then failed is a different thing from a file that was never usable, and a
            // script that cannot tell them apart cannot decide whether to retry or to fix the configuration.
            error.WriteLine(Flatten(failure.InnerException ?? failure));
            return ExitRunFailed;
        }
        catch (Exception exception)
        {
            // The message is the product here: a configuration mistake should read as one line a person can act on,
            // not as a stack trace. Causes are appended because YAML reports the reason on the inner exception.
            error.WriteLine(Flatten(exception));
            return ExitUsage;
        }
    }

    /// <summary>Marks an exception as having come from the run rather than from the configuration.</summary>
    private sealed class RunFailedException : Exception
    {
        public RunFailedException(Exception inner) : base(inner.Message, inner)
        {
        }
    }

    private static (YamlModelConfig Config, AiModelBuilder<double, Matrix<double>, Vector<double>> Builder) LoadProgramRun(Arguments arguments)
    {
        string configPath = arguments.Require("config");
        YamlModelConfig config = YamlConfigLoader.LoadFromFile(configPath);
        EvolutionOptions options = config.Evolution
            ?? throw new ArgumentException(
                $"'{configPath}' has no 'evolution:' section, so there is nothing to run. " +
                "Run 'aidotnet-evolve docs' for the available settings.");

        if (config.ProgramEvolution is null)
            throw new ArgumentException(
                $"'{configPath}' has no 'programEvolution:' section. A configuration file can describe a whole run " +
                "only for program evolution, because evolving any other kind of candidate needs an evaluation task " +
                "written in code. Use the library directly for that.");

        ApplyOverrides(options, arguments);
        if (options.OutputDirectory is not null) config.ProgramEvolution.Engine.OutputDirectory = options.OutputDirectory;
        var builder = AiModelBuilder<double, Matrix<double>, Vector<double>>.FromConfiguration(config);
        if (config.ProgramEvolution.TestCases.Count > 0) builder.WithProgramTestCaseCorrectness();
        return (config, builder);
    }

    private static int PreflightLimit(Arguments arguments) => arguments.TryGet("preflight-max-tests", out string maximum)
        ? ParseInt32(maximum, "preflight-max-tests") : 256;

    private static async Task<int> PreflightAsync(Arguments arguments, TextWriter output, CancellationToken token)
    {
        token.ThrowIfCancellationRequested();
        var (_, builder) = LoadProgramRun(arguments);
        var report = await builder.PreflightProgramEvolutionAsync(PreflightLimit(arguments), token).ConfigureAwait(false);
        output.WriteLine(JsonConvert.SerializeObject(report, Formatting.Indented));
        return report.IsReady ? ExitSuccess : ExitRunFailed;
    }

    /// <summary>Loads one configuration document, preflights its first seed, then runs and reports the search.</summary>
    private static async Task<int> RunAsync(
        Arguments arguments, TextWriter output, TextWriter error, CancellationToken cancellationToken, EvolutionRunControl? control)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var (_, builder) = LoadProgramRun(arguments);
        control ??= new EvolutionRunControl();
        using var runCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        var inspection = new RunInspection(control, runCancellation);
        builder.WithEvolutionControl(control).ObserveProgramEvolution(inspection);
        await using var service = arguments.TryGet("session", out string session)
            ? new LocalRunControl(session, inspection.Handle) : null;

        AiModelResult<double, Matrix<double>, Vector<double>> result;
        try
        {
            error.WriteLine("Preflight checks one seed before search. Its correctness and additional fitness costs are separate from the search evaluation budget.");
            var preflight = await builder.PreflightProgramEvolutionAsync(PreflightLimit(arguments), runCancellation.Token).ConfigureAwait(false);
            error.WriteLine(JsonConvert.SerializeObject(preflight));
            if (!preflight.IsReady)
            {
                await inspection.FinishAsync(null).ConfigureAwait(false);
                return ExitRunFailed;
            }
            result = await builder.BuildAsync(runCancellation.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            await inspection.FinishAsync(null).ConfigureAwait(false);
            throw;
        }
        catch (Exception exception)
        {
            await inspection.FinishAsync(null).ConfigureAwait(false);
            throw new RunFailedException(exception);
        }

        EvolutionRunSummary? summary = result.EvolutionSummary;
        await inspection.FinishAsync(summary).ConfigureAwait(false);
        if (summary is null)
        {
            error.WriteLine("The run produced no evolution summary.");
            return ExitRunFailed;
        }

        if (arguments.Has("json")) output.WriteLine(JsonConvert.SerializeObject(summary, Formatting.Indented));
        else PrintSummary(output, summary, result.ProgramEvolution?.BestProgram?.Source, arguments.Has("show-best"));
        if (service is not null) error.WriteLine(JsonConvert.SerializeObject(inspection.Read()));
        return RunExitCode(summary);
    }

    internal static int RunExitCode(EvolutionRunSummary summary) =>
        summary.StopReason is EvolutionStopReason.CandidateFailure or EvolutionStopReason.NoCandidates ||
        (summary.ArchiveCount == 0 && summary.StopReason != EvolutionStopReason.Canceled) ? ExitRunFailed : ExitSuccess;

    private static async Task<int> ControlAsync(string command, Arguments arguments, TextWriter output, CancellationToken token)
    {
        output.WriteLine(await LocalRunControl.SendAsync(arguments.Require("session"), command, token).ConfigureAwait(false));
        return ExitSuccess;
    }

    /// <summary>Loads and validates a configuration file without running anything.</summary>
    private static int Validate(Arguments arguments, TextWriter output, TextWriter error)
    {
        string configPath = arguments.Require("config");
        YamlModelConfig config = YamlConfigLoader.LoadFromFile(configPath);
        if (config.Evolution is null)
        {
            error.WriteLine($"'{configPath}' has no 'evolution:' section.");
            return ExitUsage;
        }

        ApplyOverrides(config.Evolution, arguments);
        EvolutionOptions validated = config.Evolution.SnapshotAndValidate();
        output.WriteLine($"{configPath} is valid.");
        output.WriteLine($"  run id            {validated.RunId}");
        output.WriteLine($"  seed              {validated.Seed.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  evaluation budget {validated.MaxEvaluationAttempts.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  islands           {validated.IslandCount.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  dispatch          {validated.Dispatch}");
        output.WriteLine($"  program section   {(config.ProgramEvolution is null ? "absent" : "present")}");
        foreach (EvolutionDescriptorDefinition descriptor in validated.Descriptors)
        {
            output.WriteLine(
                $"  descriptor        {descriptor.Name} " +
                $"[{descriptor.Minimum.ToString("R", CultureInfo.InvariantCulture)}, " +
                $"{descriptor.Maximum.ToString("R", CultureInfo.InvariantCulture)}] " +
                $"in {descriptor.BinCount.ToString(CultureInfo.InvariantCulture)} bins, {descriptor.OutOfRangePolicy}");
        }
        return ExitSuccess;
    }

    /// <summary>Applies the per-invocation overrides a configuration file should not have to carry.</summary>
    /// <remarks>
    /// These five are what differs between two runs of one configuration: which run this is, where its output goes,
    /// how much of it to do, and whether it continues an earlier one. Everything else belongs in the file, where it
    /// is reviewable and reproducible.
    /// </remarks>
    private static void ApplyOverrides(EvolutionOptions options, Arguments arguments)
    {
        if (arguments.TryGet("run-id", out string runId)) options.RunId = runId;
        if (arguments.TryGet("output", out string outputDirectory)) options.OutputDirectory = outputDirectory;
        if (arguments.TryGet("seed", out string seed)) options.Seed = ParseUInt64(seed, "seed");
        if (arguments.TryGet("max-evaluations", out string budget))
            options.MaxEvaluationAttempts = ParseInt32(budget, "max-evaluations");
        if (arguments.Has("resume")) options.Resume = true;
    }

    /// <summary>Writes generated text to the requested path, or to standard output when none is given.</summary>
    private static int WriteText(Arguments arguments, TextWriter output, string content, string defaultName)
    {
        if (!arguments.TryGet("out", out string path))
        {
            output.WriteLine(content);
            return ExitSuccess;
        }

        string target = Directory.Exists(path) ? Path.Combine(path, defaultName) : path;
        string? directory = Path.GetDirectoryName(Path.GetFullPath(target));
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
        File.WriteAllText(target, content);
        output.WriteLine($"Wrote {target}.");
        return ExitSuccess;
    }

    private static void PrintSummary(TextWriter output, EvolutionRunSummary summary, string? bestProgram, bool showBest)
    {
        output.WriteLine($"run {summary.RunId} stopped: {summary.StopReason}");
        output.WriteLine($"  proposals   {summary.Proposals.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  attempts    {summary.EvaluationAttempts.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  completed   {summary.CompletedEvaluations.ToString(CultureInfo.InvariantCulture)}");
        output.WriteLine($"  archive     {summary.ArchiveCount.ToString(CultureInfo.InvariantCulture)} cells across " +
                         $"{summary.IslandCount.ToString(CultureInfo.InvariantCulture)} islands");
        output.WriteLine($"  best        {Format(summary.BestQuality)} ({summary.BestGenomeId ?? "none"})");
        output.WriteLine($"  state hash  {summary.StateHash}");
        if (summary.CheckpointPath is not null) output.WriteLine($"  checkpoint  {summary.CheckpointPath}");
        if (summary.TracePath is not null) output.WriteLine($"  trace       {summary.TracePath}");
        if (summary.LlmUsage is not null)
        {
            output.WriteLine(
                $"  llm         {summary.LlmUsage.ChatCalls.ToString(CultureInfo.InvariantCulture)} calls, " +
                $"{summary.LlmUsage.Retries.ToString(CultureInfo.InvariantCulture)} retries, " +
                $"{summary.LlmUsage.InputTokens.ToString(CultureInfo.InvariantCulture)} in / " +
                $"{summary.LlmUsage.OutputTokens.ToString(CultureInfo.InvariantCulture)} out tokens");
        }

        foreach (EvolutionFailureSummary failure in summary.RetainedFailures.Take(5))
            output.WriteLine($"  failure     {failure.Code}: {failure.Message}");

        if (showBest && bestProgram is not null)
        {
            // Printed only on request: the text came from a language model and was run as untrusted input, so it is
            // something to review deliberately rather than something a routine command dumps into a terminal.
            output.WriteLine();
            output.WriteLine("--- best program ---");
            output.WriteLine(bestProgram);
        }
    }

    private static string Format(double? value) =>
        value.HasValue ? value.Value.ToString("R", CultureInfo.InvariantCulture) : "none";

    private static ulong ParseUInt64(string value, string option) =>
        ulong.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out ulong parsed)
            ? parsed
            : throw new ArgumentException($"--{option} expects a whole number, but got '{value}'.");

    private static int ParseInt32(string value, string option) =>
        int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
            ? parsed
            : throw new ArgumentException($"--{option} expects a whole number, but got '{value}'.");

    private static int Fail(TextWriter error, string message)
    {
        error.WriteLine(message);
        return ExitUsage;
    }

    private static string Flatten(Exception exception)
    {
        var text = new System.Text.StringBuilder();
        for (Exception? current = exception; current is not null; current = current.InnerException)
        {
            if (text.Length > 0) text.Append(" -> ");
            text.Append(current.Message);
        }
        return text.ToString();
    }

    private static void PrintHelp(TextWriter output)
    {
        output.WriteLine("aidotnet-evolve - run an evolution search described by a YAML configuration file.");
        output.WriteLine();
        output.WriteLine("  run       --config <file> [--run-id <id>] [--seed <n>] [--max-evaluations <n>]");
        output.WriteLine("            [--output <dir>] [--resume] [--json] [--show-best] [--session <name>]");
        output.WriteLine("            [--preflight-max-tests <1..4096>] first-seed preflight runs before search (default 256 public tests)");
        output.WriteLine("  preflight --config <file> [--preflight-max-tests <1..4096>] execute only the seed/setup checks");
        output.WriteLine("  inspect | pause | cancel --session <name>  current-user-only live control");
        output.WriteLine("            pause drains the batch and exits; resumability requires a verified checkpoint.");
        output.WriteLine("            Resume with run --config <same file> --resume and the same run id/output/budget.");
        output.WriteLine("  validate  --config <file>   load the file, validate it, and print what it resolved to");
        output.WriteLine("  schema    [--out <path>]    write the JSON schema an editor validates the file against");
        output.WriteLine("  docs      [--out <path>]    write the markdown reference for every setting");
        output.WriteLine("  benchmark-program --worker <absolute worker.dll> --output <new directory>");
        output.WriteLine("            [--runs <1..12>] [--measurements <1..9>] authored C# end-to-end timing pilot; no model calls");
        output.WriteLine();
        output.WriteLine("Any ${NAME} in the file is replaced by that environment variable, and ${NAME:-value}");
        output.WriteLine("supplies a default, so an API key stays out of a file you commit.");
        output.WriteLine();
        output.WriteLine("Exit codes: 0 success, 1 usage or configuration error, 2 cancelled, 3 preflight/run failed or no usable result.");
    }

    /// <summary>A minimal <c>--name value</c> and <c>--flag</c> parser.</summary>
    /// <remarks>
    /// Hand-rolled rather than taken from a package because the surface is a handful of options per command, and a
    /// tool that ships beside a library should not add a dependency to read them.
    /// </remarks>
    private sealed class Arguments
    {
        private readonly Dictionary<string, string?> _values = new(StringComparer.OrdinalIgnoreCase);

        private static readonly HashSet<string> Flags =
            new(new[] { "resume", "json", "show-best" }, StringComparer.OrdinalIgnoreCase);

        /// <summary>Parses the options of one command, refusing anything the command does not accept.</summary>
        /// <param name="args">The tokens after the command name.</param>
        /// <param name="known">Every option name the command accepts.</param>
        /// <returns>The parsed options.</returns>
        /// <exception cref="ArgumentException">
        /// A token is not an option, an option is not one this command accepts, or an option that takes a value was
        /// given none.
        /// </exception>
        /// <remarks>
        /// Both refusals matter. A misspelled option that is silently ignored means a run proceeds under settings the
        /// caller believes they changed, and an option whose value was swallowed by the next option means the same
        /// thing with no typo to notice.
        /// </remarks>
        public static Arguments Parse(string[] args, IReadOnlyCollection<string> known)
        {
            var accepted = new HashSet<string>(known, StringComparer.OrdinalIgnoreCase);
            var parsed = new Arguments();
            for (int index = 0; index < args.Length; index++)
            {
                string token = args[index];
                if (!token.StartsWith("--", StringComparison.Ordinal))
                    throw new ArgumentException($"Unexpected argument '{token}'. Options are written as --name value.");

                string name = token.Substring(2);
                if (accepted.Count > 0 && !accepted.Contains(name))
                {
                    throw new ArgumentException(
                        $"'--{name}' is not an option of this command. Accepted: " +
                        string.Join(", ", known.Select(option => "--" + option)) + ".");
                }

                if (parsed._values.ContainsKey(name))
                    throw new ArgumentException($"Option '--{name}' may be specified only once.");

                bool hasValue = index + 1 < args.Length && !args[index + 1].StartsWith("--", StringComparison.Ordinal);
                if (hasValue && Flags.Contains(name))
                    throw new ArgumentException($"Flag option '--{name}' does not accept a value; omit the flag to leave it disabled.");
                if (!hasValue && !Flags.Contains(name))
                    throw new ArgumentException($"'--{name}' expects a value.");

                parsed._values[name] = hasValue ? args[++index] : null;
            }
            return parsed;
        }

        public bool Has(string name) => _values.ContainsKey(name);

        public bool TryGet(string name, out string value)
        {
            value = string.Empty;
            if (!_values.TryGetValue(name, out string? stored) || stored is null) return false;
            value = stored;
            return true;
        }

        public string Require(string name) =>
            TryGet(name, out string value)
                ? value
                : throw new ArgumentException($"--{name} is required. Run with --help for usage.");
    }
}
