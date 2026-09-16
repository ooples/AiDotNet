using AttributionRuntime;
using AttributionSubject;

if (args.Length != 1) throw new ArgumentException("Expected worker scenario.");
if (args[0] == "missing") return; // Simulate a child that never joins attribution.
if (Environment.GetEnvironmentVariable("ATTRIBUTION_OUTPUT") is null)
{
    if (args[0] != "complete") throw new ArgumentException("Uninstrumented control only supports complete.");
    Console.WriteLine(CodePaths.WorkerOnly(21));
    return;
}
string owner = Tracker.BeginWorker();
if (args[0] == "unclosed") return; // Exit without completing the ownership scope.
if (args[0] != "complete") throw new ArgumentException("Unknown worker scenario.");
Console.WriteLine(CodePaths.WorkerOnly(21));
Tracker.End(owner);
