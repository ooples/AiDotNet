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
    /// <returns>The process exit code.</returns>
    public static async Task<int> ExecuteAsync(
        string[] args, TextWriter output, TextWriter error, CancellationToken cancellationToken = default)
    {
        if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
        {
            PrintHelp(output);
            return args.Length == 0 ? ExitUsage : ExitSuccess;
        }

        try
        {
            Arguments rest = Arguments.Parse(args.Skip(1).ToArray());
            return args[0].ToLowerInvariant() switch
            {
                "run" => await RunAsync(rest, output, error, cancellationToken).ConfigureAwait(false),
                "validate" => Validate(rest, output, error),
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
        catch (Exception exception)
        {
            // The message is the product here: a configuration mistake should read as one line a person can act on,
            // not as a stack trace. Causes are appended because YAML reports the reason on the inner exception.
            error.WriteLine(Flatten(exception));
            return ExitUsage;
        }
    }

    /// <summary>Loads a configuration file, runs the search it describes, and reports the outcome.</summary>
    private static async Task<int> RunAsync(
        Arguments arguments, TextWriter output, TextWriter error, CancellationToken cancellationToken)
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
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>(configPath);
        builder.ConfigureEvolution(options);
        AiModelResult<double, Matrix<double>, Vector<double>> result =
            await builder.BuildAsync(cancellationToken).ConfigureAwait(false);

        EvolutionRunSummary? summary = result.EvolutionSummary;
        if (summary is null)
        {
            error.WriteLine("The run produced no evolution summary.");
            return ExitRunFailed;
        }

        if (arguments.Has("json")) output.WriteLine(JsonConvert.SerializeObject(summary, Formatting.Indented));
        else PrintSummary(output, summary, result.ProgramEvolution?.BestProgram?.Source, arguments.Has("show-best"));
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
        output.WriteLine("            [--output <dir>] [--resume] [--json] [--show-best]");
        output.WriteLine("  validate  --config <file>   load the file, validate it, and print what it resolved to");
        output.WriteLine("  schema    [--out <path>]    write the JSON schema an editor validates the file against");
        output.WriteLine("  docs      [--out <path>]    write the markdown reference for every setting");
        output.WriteLine();
        output.WriteLine("Any ${NAME} in the file is replaced by that environment variable, and ${NAME:-value}");
        output.WriteLine("supplies a default, so an API key stays out of a file you commit.");
        output.WriteLine();
        output.WriteLine("Exit codes: 0 success, 1 usage or configuration error, 2 cancelled, 3 no result.");
    }

    /// <summary>A minimal <c>--name value</c> and <c>--flag</c> parser.</summary>
    /// <remarks>
    /// Hand-rolled rather than taken from a package because the surface is a handful of options per command, and a
    /// tool that ships beside a library should not add a dependency to read them.
    /// </remarks>
    private sealed class Arguments
    {
        private readonly Dictionary<string, string?> _values = new(StringComparer.OrdinalIgnoreCase);

        public static Arguments Parse(string[] args)
        {
            var parsed = new Arguments();
            for (int index = 0; index < args.Length; index++)
            {
                string token = args[index];
                if (!token.StartsWith("--", StringComparison.Ordinal))
                    throw new ArgumentException($"Unexpected argument '{token}'. Options are written as --name value.");

                string name = token.Substring(2);
                bool hasValue = index + 1 < args.Length && !args[index + 1].StartsWith("--", StringComparison.Ordinal);
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
