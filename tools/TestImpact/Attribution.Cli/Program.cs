using AiDotNet.TestImpact;
using AttributionRuntime;

if (args.Length == 0 || !Enum.TryParse(args[0], out Command command) || !Enum.IsDefined(command))
    throw new ArgumentException("Expected Prepare or Verify command.");
switch (command)
{
    case Command.Prepare:
    {
        if (args.Length < 4 || !Enum.TryParse(args[3], out ValidationScope scope) || !Enum.IsDefined(scope))
            throw new ArgumentException("Prepare inventory.json plan.json FullWorkload|SelectedMethods [method ...]");
        DiscoveryManifest manifest = Read<DiscoveryManifest>(args[1]);
        RunnerBinding.WriteNew(args[2], RunnerBinding.Prepare(manifest, args[4..], scope));
        break;
    }
    case Command.Verify:
    {
        if (args.Length != 10) throw new ArgumentException("Verify inventory.json plan.json report-directory results.trx collection-run repository workflow-run attempt output.json");
        string[] files = Directory.GetFileSystemEntries(args[3]);
        // A revoked/pending report or unexplained extra process is never ignored.
        if (files.Length != 1 || !File.Exists(files[0]) || Path.GetExtension(files[0]) != ".json")
            throw new EvidenceException(EvidenceFailure.Outcome, "Expected one complete, non-revoked single-bundle host report.");
        AttributionReport report = Read<AttributionReport>(files[0]);
        if (Path.GetFileNameWithoutExtension(files[0]) != report.Token)
            throw new EvidenceException(EvidenceFailure.Provenance, "Report filename differs from its process identity.");
        var origin = new RunIdentity(args[6], long.Parse(args[7], System.Globalization.CultureInfo.InvariantCulture),
            int.Parse(args[8], System.Globalization.CultureInfo.InvariantCulture));
        VerifiedExecution result = PlannedEvidence.Verify(Read<DiscoveryManifest>(args[1]), Read<ExecutionPlan>(args[2]),
            report, args[4], args[5], origin);
        RunnerBinding.WriteNew(args[9], new { result.Scope, result.PlanHash, result.InventoryHash, result.Context,
            result.Workload, result.Origin, result.Cases, result.CanReplaceFullBaseline,
            AuthenticatedWorkflowOrigin = false });
        break;
    }
}

static T Read<T>(string path) where T : class => ExecutionEvidence.ReadDocument<T>(File.ReadAllText(path));
enum Command { Prepare, Verify }
