using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Evolve.Cli;

internal static class EvidenceCommands
{
    internal static int Inspect(string record, TextWriter output)
    {
        var files = RunEvidenceBundle.ReadAndVerify(record);
        output.WriteLine(JsonConvert.SerializeObject(View(files), Formatting.Indented));
        return 0;
    }

    internal static int Export(string record, string destination, TextWriter output)
    {
        var files = RunEvidenceBundle.ReadAndVerify(record);
        _ = View(files); // Generic valid JSON is insufficient to be an invocation record.
        RunEvidenceBundle.Create(destination, files, Array.Empty<string>());
        output.WriteLine("Verified evidence copied without replacement. Hashes attest integrity, not authenticity or absence of unknown secrets.");
        return 0;
    }

    internal static int Compare(string left, string right, string? destination, TextWriter output)
    {
        var a = RunEvidenceBundle.ReadAndVerify(left);
        var b = RunEvidenceBundle.ReadAndVerify(right);
        var report = new
        {
            SchemaVersion = 1, Scope = "descriptive comparison of two invocation records; no significance, causal or competitor-superiority claim",
            SameConfigurationProjection = a["configuration.json"].SequenceEqual(b["configuration.json"]),
            SameEnvironmentProjection = a["environment.json"].SequenceEqual(b["environment.json"]),
            Left = View(a), Right = View(b)
        };
        byte[] json = RunRecord.Json(report);
        if (destination is not null)
        {
            string target = Path.GetFullPath(destination);
            if (Directory.Exists(target) || File.Exists(target)) throw new IOException("Comparison destination must not already exist.");
            string parent = Path.GetDirectoryName(target) ?? throw new ArgumentException("Missing comparison parent.");
            Directory.CreateDirectory(parent);
            string staging = Path.Combine(parent, ".aidotnet-comparison-" + Guid.NewGuid().ToString("N"));
            if (OperatingSystem.IsWindows()) Directory.CreateDirectory(staging);
            else Directory.CreateDirectory(staging, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            RunEvidenceBundle.Create(Path.Combine(staging, "left"), a, Array.Empty<string>());
            RunEvidenceBundle.Create(Path.Combine(staging, "right"), b, Array.Empty<string>());
            WriteNew(Path.Combine(staging, "comparison.json"), json);
            WriteNew(Path.Combine(staging, "manifest.json"), RunRecord.Json(new
            {
                SchemaVersion = 1,
                Files = new[] { "comparison.json", "left/manifest.json", "right/manifest.json" }.Select(name => new
                {
                    Name = name, Sha256 = RunRecord.Hash(File.ReadAllBytes(Path.Combine(staging, name)))
                }).ToArray()
            }));
            Directory.Move(staging, target); // Failed private staging is retained, never recursively deleted.
        }
        output.WriteLine(Encoding.UTF8.GetString(json));
        return 0;
    }

    private static object View(IReadOnlyDictionary<string, byte[]> files)
    {
        var result = JObject.Parse(Encoding.UTF8.GetString(files["result.json"]));
        var validation = JObject.Parse(Encoding.UTF8.GetString(files["validation.json"]));
        if ((int?)result["SchemaVersion"] != 1 || (int?)validation["SchemaVersion"] != 1 || result["ExitCode"]?.Type != JTokenType.Integer)
            throw new InvalidDataException("Unsupported invocation record schema.");
        int exitCode = result.Value<int>("ExitCode");
        if (exitCode is < 0 or > 3) throw new InvalidDataException("Invalid invocation exit code.");
        string? digest = Digest(result["WinnerSourceSha256"]);
        if (digest != Digest(validation["WinnerSourceSha256"])) throw new InvalidDataException("Winner receipts disagree.");
        if (files.TryGetValue("program.txt", out byte[]? source) && RunRecord.Hash(source) != digest)
            throw new InvalidDataException("Winner source does not match its receipt.");
        if (result.Value<bool>("SourceIncluded") != (source is not null))
            throw new InvalidDataException("Source inclusion receipt disagrees with the bundle.");
        double? quality = result.Value<double?>("BestQuality");
        if (quality.HasValue && !double.IsFinite(quality.Value)) throw new InvalidDataException("Non-finite winner quality.");
        return new
        {
            IntegrityVerified = true, AuthenticityVerified = false, ExitCode = exitCode,
            BestQuality = quality, BestGenomeId = Digest(result["BestGenomeId"]), WinnerSourceSha256 = digest,
            SourceIncluded = source is not null, ConfigurationSha256 = RunRecord.Hash(files["configuration.json"]),
            EnvironmentSha256 = RunRecord.Hash(files["environment.json"]), ValidationSha256 = RunRecord.Hash(files["validation.json"]),
            ValidationScope = "stored public-test/archive receipts; not independent held-out revalidation"
        };
    }

    private static string? Digest(JToken? value)
    {
        if (value is null || value.Type == JTokenType.Null) return null;
        string? text = value.Type == JTokenType.String ? value.Value<string>() : null;
        if (text?.Length != 64 || text.Any(c => !char.IsAsciiHexDigit(c)))
            throw new InvalidDataException("Malformed receipt identity.");
        return text.ToLowerInvariant();
    }

    private static void WriteNew(string path, byte[] data)
    {
        var options = new FileStreamOptions
        {
            Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None,
            Options = FileOptions.WriteThrough
        };
        if (!OperatingSystem.IsWindows()) options.UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
        using var stream = new FileStream(path, options);
        stream.Write(data);
        stream.Flush(flushToDisk: true);
    }
}
