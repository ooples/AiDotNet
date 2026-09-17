using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Models.Results;
using Newtonsoft.Json;

namespace AiDotNet.Evolve.Cli;

/// <summary>Allowlisted invocation evidence captured from in-memory results, never a mutable best-file pair.</summary>
internal sealed class RunRecord : IDisposable
{
    private readonly string _destination;
    private readonly bool _includeSource;
    private readonly string[] _prohibited;
    private readonly byte[] _configuration, _environment;
    private readonly DateTimeOffset _started = DateTimeOffset.UtcNow;
    private readonly FileStream _lease;

    internal RunRecord(string destination, YamlModelConfig config, bool includeSource)
    {
        _destination = Path.GetFullPath(destination);
        if (Directory.Exists(_destination) || File.Exists(_destination))
            throw new IOException("Run record destination must not already exist.");
        _includeSource = includeSource;
        _ = config.Evolution ?? throw new ArgumentException("Missing evolution configuration.");
        _ = config.ProgramEvolution ?? throw new ArgumentException("Missing program configuration.");
        var exported = RunConfigurationTemplate.Create(config, includeSource);
        _prohibited = exported.ProhibitedValues;
        _configuration = exported.Json;
        RunEvidenceBundle.ValidateMetadata(_configuration, _prohibited);
        _environment = Json(new
        {
            SchemaVersion = 1, Runtime = RuntimeInformation.FrameworkDescription,
            Os = Environment.OSVersion.Platform.ToString(), Architecture = RuntimeInformation.ProcessArchitecture.ToString(),
            LogicalProcessors = Environment.ProcessorCount,
            Binaries = new[] { typeof(RunRecord).Assembly, typeof(AiModelBuilder<,,>).Assembly, typeof(EvolutionEvaluation).Assembly }
                .Select(Fingerprint).ToArray(),
            Scope = "loaded CLI/facade/core binaries; not a full environment, hardware, sandbox or dependency attestation"
        });
        Directory.CreateDirectory(Path.GetDirectoryName(_destination)!);
        string lockPath = _destination + ".aidotnet-record.lock";
        if (File.Exists(lockPath) && (File.GetAttributes(lockPath) & FileAttributes.ReparsePoint) != 0)
            throw new IOException("Linked record leases are not supported.");
        _lease = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        if (Directory.Exists(_destination) || File.Exists(_destination))
        {
            _lease.Dispose();
            throw new IOException("Run record destination must not already exist.");
        }
    }

    internal void Complete(ProgramEvolutionPreflightResult? preflight, EvolutionRunSummary? summary,
        ProgramEvolutionResult? programs, RunInspection.Snapshot inspection, int exitCode)
    {
        var best = programs?.BestProgram;
        if (best is not null && (inspection.BestFeasible?.GenomeId != best.Id ||
            inspection.BestFeasible.Quality != programs!.BestQuality ||
            inspection.BestFeasible.Direction != programs.Direction.ToString()))
            throw new InvalidDataException("Winner and archive inspection receipts disagree.");
        if (best is not null && Encoding.UTF8.GetByteCount(best.Source) > 4 * 1024 * 1024)
            throw new InvalidDataException("Winner source exceeds the evidence size bound.");
        byte[]? source = best is null ? null : new UTF8Encoding(false, true).GetBytes(best.Source);
        var files = new Dictionary<string, byte[]>
        {
            ["configuration.json"] = _configuration,
            ["environment.json"] = _environment,
            ["validation.json"] = Json(new
            {
                SchemaVersion = 1, Preflight = preflight,
                WinnerReceipt = inspection.BestFeasible,
                WinnerSourceSha256 = source is null ? null : Hash(source),
                Scope = "public seed preflight and archive receipt; no held-out winner revalidation or provider-honesty attestation"
            }),
            ["result.json"] = Json(new
            {
                SchemaVersion = 1, StartedUtc = _started, FinishedUtc = DateTimeOffset.UtcNow, ExitCode = exitCode,
                StopReason = summary?.StopReason.ToString(), StateHash = summary is null ? null : Hash(summary.StateHash),
                BestGenomeId = best?.Id, BestQuality = programs?.BestQuality, Direction = programs?.Direction.ToString(),
                WinnerSourceSha256 = source is null ? null : Hash(source), SourceIncluded = _includeSource && source is not null,
                Inspection = inspection,
                Usage = programs?.LlmUsage,
                UsageScope = "provider-reported final process-segment counters; zero tokens may mean unreported; no monetary estimate",
                SourcePolicy = "explicit opt-in; exact bytes; configured parameter/credential/environment strings checked; unknown or encoded secrets require human review"
            })
        };
        if (_includeSource && source is not null) files["program.txt"] = source;
        RunEvidenceBundle.Create(_destination, files, _prohibited);
    }

    private static object Fingerprint(Assembly assembly)
    {
        using var file = new FileStream(assembly.Location, FileMode.Open, FileAccess.Read, FileShare.Read);
        using (var pe = new PEReader(file, PEStreamOptions.LeaveOpen))
        {
            MetadataReader metadata = pe.GetMetadataReader();
            if (metadata.GetGuid(metadata.GetModuleDefinition().Mvid) != assembly.ManifestModule.ModuleVersionId)
                throw new IOException("A loaded assembly differs from its on-disk fingerprint; restart after the build completes.");
        }
        file.Position = 0;
        return new { Name = assembly.GetName().Name, Version = assembly.GetName().Version?.ToString(),
            Sha256 = Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant() };
    }

    internal static byte[] Json(object value) => Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(value, Formatting.Indented));
    internal static string Hash(string value) => Hash(Encoding.UTF8.GetBytes(value));
    internal static string Hash(byte[] value) => Convert.ToHexString(SHA256.HashData(value)).ToLowerInvariant();
    public void Dispose() => _lease.Dispose();
}
