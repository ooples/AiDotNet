using AttributionRuntime;
using AttributionSubject;
using System.Diagnostics;
using System.Text.Json;

if (args.Length != 1) throw new ArgumentException("Expected worker scenario.");
if (!Enum.TryParse(args[0], ignoreCase: true, out WorkerScenario scenario) || !Enum.IsDefined(scenario))
    throw new ArgumentException("Unknown worker scenario.");
if (scenario == WorkerScenario.Missing) return; // Simulate a child that never joins attribution.
if (scenario == WorkerScenario.Benchmark)
{
    bool collecting = !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("ATTRIBUTION_OUTPUT"));
    string? benchmarkOwner = collecting ? Tracker.BeginWorker() : null;
    const int count = 131072;
    long checksum = 0;
    for (int i = 0; i < count; i++) checksum += CodePaths.HotPath(i);
    var samples = new double[7];
    for (int sample = 0; sample < samples.Length; sample++)
    {
        checksum = 0;
        long start = Stopwatch.GetTimestamp();
        for (int i = 0; i < count; i++) checksum += CodePaths.HotPath(i);
        samples[sample] = Stopwatch.GetElapsedTime(start).TotalNanoseconds / count;
    }
    if (benchmarkOwner is not null) Tracker.End(benchmarkOwner);
    Console.WriteLine(JsonSerializer.Serialize(new { Count = count, Checksum = checksum, NanosecondsPerCall = samples }));
    return;
}
if (Environment.GetEnvironmentVariable("ATTRIBUTION_OUTPUT") is null)
{
    if (scenario != WorkerScenario.Complete) throw new ArgumentException("Uninstrumented control only supports complete.");
    Console.WriteLine(CodePaths.WorkerOnly(21));
    return;
}
string owner = Tracker.BeginWorker();
if (scenario == WorkerScenario.Unclosed) return; // Exit without completing the ownership scope.
Console.WriteLine(CodePaths.WorkerOnly(21));
Tracker.End(owner);

enum WorkerScenario { Complete, Missing, Unclosed, Benchmark }
