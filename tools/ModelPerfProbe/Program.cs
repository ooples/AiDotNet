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
    private static readonly string[] RequiredMetrics =
    {
        "constructMs", "coldForwardMs", "steadyForwardMedianMs", "steadyForwardP95Ms",
        "tapeForwardMs", "backwardMs", "trainStepMs", "allocatedBytes", "wallMs",
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

            CompareBaseline(records, baseline, options, diagnostics);
            DetectCohortOutliers(records, diagnostics);
            ValidateAbsoluteCeilings(records, options, diagnostics);

            var summary = BuildSummary(records, diagnostics, options.ExpectedCount);
            WriteJson(options.OutputPath!, summary);
            if (options.WriteBaselinePath is not null)
                WriteJson(options.WriteBaselinePath, BuildBaseline(records));

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
                diagnostics.Add(Diagnostic.Error(record.Fixture, "status", $"status is '{record.Status}'"));

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

    private static void CompareBaseline(
        IReadOnlyList<CensusRecord> records,
        BaselineDocument? baseline,
        Options options,
        ICollection<Diagnostic> diagnostics)
    {
        if (baseline is null) return;
        var index = baseline.Entries.ToDictionary(
            entry => (entry.Fixture, entry.Environment),
            entry => entry,
            new FixtureEnvironmentComparer());

        foreach (CensusRecord record in records)
        {
            if (!index.TryGetValue((record.Fixture, record.Environment), out BaselineEntry? prior))
            {
                diagnostics.Add(Diagnostic.Warning(record.Fixture, "baseline",
                    $"no environment-qualified baseline for {record.Environment}"));
                continue;
            }

            foreach ((string metric, double current) in record.Metrics)
            {
                if (!prior.Metrics.TryGetValue(metric, out double previous) || previous <= 0.0) continue;
                double ratio = current / previous;
                double noiseFloor = metric == "allocatedBytes" ? 1_048_576.0 : 25.0;
                if (ratio > options.MaxRegressionRatio && current - previous > noiseFloor)
                {
                    diagnostics.Add(Diagnostic.Error(record.Fixture, metric,
                        $"regressed {ratio:F2}x ({previous:F2} -> {current:F2}); " +
                        $"limit is {options.MaxRegressionRatio:F2}x"));
                }
            }
        }
    }

    private static void DetectCohortOutliers(
        IReadOnlyList<CensusRecord> records,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (IGrouping<string, CensusRecord> cohort in records
                     .Where(r => r.ParameterCount > 0 && r.Metric("trainStepMs") > 0.0)
                     .GroupBy(r => r.Cohort))
        {
            double[] values = cohort.Select(r => Math.Log(1.0 + r.Metric("trainStepMs"))).OrderBy(v => v).ToArray();
            if (values.Length < 5) continue;
            double median = Median(values);
            double mad = Median(values.Select(v => Math.Abs(v - median)).OrderBy(v => v).ToArray());
            if (mad <= 1e-9) continue;

            foreach (CensusRecord record in cohort)
            {
                double robustZ = 0.6745 * (Math.Log(1.0 + record.Metric("trainStepMs")) - median) / mad;
                if (robustZ > 6.0)
                {
                    diagnostics.Add(Diagnostic.Warning(record.Fixture, "trainStepMs",
                        $"robust cohort outlier (z={robustZ:F1}, cohort={cohort.Key})"));
                }
            }
        }
    }

    private static void ValidateAbsoluteCeilings(
        IReadOnlyList<CensusRecord> records,
        Options options,
        ICollection<Diagnostic> diagnostics)
    {
        foreach (CensusRecord record in records)
        {
            double trainStep = record.Metric("trainStepMs");
            if (trainStep > options.MaxTrainStepMs)
            {
                diagnostics.Add(Diagnostic.Error(record.Fixture, "trainStepMs",
                    $"single train step {trainStep:F0} ms exceeds {options.MaxTrainStepMs:F0} ms ceiling"));
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
        CensusRecord[] slowest = records.OrderByDescending(r => r.Metric("trainStepMs")).Take(25).ToArray();
        return new SummaryDocument
        {
            SchemaVersion = 1,
            GeneratedUtc = DateTimeOffset.UtcNow,
            ExpectedFixtures = expectedCount,
            ObservedFixtures = records.Select(r => r.Fixture).Distinct().Count(),
            ErrorCount = diagnostics.Count(d => d.Severity == "error"),
            WarningCount = diagnostics.Count(d => d.Severity == "warning"),
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
            }).ToArray(),
        };
    }

    private static BaselineDocument BuildBaseline(IReadOnlyList<CensusRecord> records) => new()
    {
        SchemaVersion = 1,
        GeneratedUtc = DateTimeOffset.UtcNow,
        Entries = records.Select(record => new BaselineEntry
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
        Console.WriteLine($"summary: {outputPath}");
    }

    private static double Median(double[] sorted) => sorted.Length % 2 == 1
        ? sorted[sorted.Length / 2]
        : (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]) / 2.0;

    private static int RunSelfTest()
    {
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-perf-selftest-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            string record = """
            {"schemaVersion":1,"status":"ok","fixture":"F","model":"M","precision":"System.Single",
             "parameterCount":10,"engine":"Cpu","framework":".NET","os":"test","processArchitecture":"X64","processorCount":1,
             "constructMs":1,"coldForwardMs":2,"steadyForwardMedianMs":1,"steadyForwardP95Ms":1,
             "tapeForwardMs":1,"tapeEntries":2,"backwardMs":2,"trainStepMs":3,"allocatedBytes":4,"wallMs":5,
             "projectedTrainingReduceLossMs":90}
            """;
            File.WriteAllText(Path.Combine(directory, "record.json"), record);
            IReadOnlyList<CensusRecord> records = LoadRecords(directory);
            var diagnostics = new List<Diagnostic>();
            ValidateCoverage(records, 1, diagnostics);
            ValidateRecords(records, diagnostics);
            if (records.Count != 1 || diagnostics.Count != 0) return 1;
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
        Console.WriteLine("  [--baseline FILE] [--write-baseline FILE] [--max-regression-ratio 1.25]");
        Console.WriteLine("  [--max-train-step-ms 120000] [--max-correctness-probe-ms 120000]");
        Console.WriteLine("ModelPerfProbe --self-test");
        return 2;
    }

    private sealed class Options
    {
        public string? ResultsDirectory { get; private set; }
        public string? OutputPath { get; private set; }
        public string? BaselinePath { get; private set; }
        public string? WriteBaselinePath { get; private set; }
        public int? ExpectedCount { get; private set; }
        public double MaxRegressionRatio { get; private set; } = 1.25;
        public double MaxTrainStepMs { get; private set; } = 120_000.0;
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
                    case "--expected-count": options.ExpectedCount = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-regression-ratio": options.MaxRegressionRatio = double.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-train-step-ms": options.MaxTrainStepMs = double.Parse(Next(), CultureInfo.InvariantCulture); break;
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
            string environment = string.Join("|", frameworkKey, osKey, Text("processArchitecture"),
                Text("engine"), Text("processorCount"));
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
        [JsonPropertyName("diagnostics")] public Diagnostic[] Diagnostics { get; set; } = Array.Empty<Diagnostic>();
        [JsonPropertyName("slowestTrainSteps")] public RankedRecord[] SlowestTrainSteps { get; set; } = Array.Empty<RankedRecord>();
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
    }

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        WriteIndented = true,
    };
}
