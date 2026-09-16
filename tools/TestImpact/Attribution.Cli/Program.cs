using AiDotNet.TestImpact;
using AttributionRuntime;

if (args.Length == 0 || !Enum.TryParse(args[0], out Command command) || !Enum.IsDefined(command))
    throw new ArgumentException("Expected Prepare, Verify, SelectChanges, PrepareReuse, CompleteReuse or ImportWorkflow command.");
switch (command)
{
    case Command.ImportWorkflow:
    {
        if (args.Length != 4) throw new ArgumentException("ImportWorkflow request.json new-download-directory output.json");
        VerifiedObservedExecution observed = await GitHubEvidenceReader.VerifyObserved(Read<WorkflowImportRequest>(args[1]), args[2]);
        VerifiedExecution imported = observed.Execution;
        RunnerBinding.WriteNew(args[3], new { imported.Scope, imported.PlanHash, imported.InventoryHash, imported.Context,
            imported.Workload, imported.Origin, imported.Cases, imported.CanReplaceFullBaseline,
            observed.StandardCases, observed.RuntimeProfile, AuthenticatedWorkflowOrigin = true, ProductionSelectionEnabled = false });
        break;
    }
    case Command.PrepareReuse:
    {
        if (args.Length != 4) throw new ArgumentException("PrepareReuse request.json plan-output partition-output");
        ReusePartition partition = SourceReuseCommands.Prepare(args[1]);
        if (partition.Execution is ExecutionPlan execution) RunnerBinding.WriteNew(args[2], execution);
        RunnerBinding.WriteNew(args[3], new { partition.Context, partition.Workload, partition.InventoryHash,
            partition.Cases, partition.ReusedCases, ExecutionRequired = partition.Execution is not null,
            BaselineOrigin = partition.Baseline.Origin, AuthenticatedWorkflowOrigin = false });
        break;
    }
    case Command.CompleteReuse:
    {
        if (args.Length is < 3 or > 4) throw new ArgumentException("CompleteReuse request.json result-output [execution-input.json]");
        VerifiedReusePartition completed = SourceReuseCommands.Complete(args[1], args.Length == 4 ? args[3] : null);
        RunnerBinding.WriteNew(args[2], new { completed.Partition.Context, completed.Partition.Workload,
            completed.Partition.Cases, completed.Partition.ReusedCases, ExecutedCases = completed.Executed?.Cases,
            BaselineOrigin = completed.Partition.Baseline.Origin, ExecutionOrigin = completed.Executed?.Origin,
            BaselineContext = completed.Partition.Baseline.Context, BaselinePlanHash = completed.Partition.Baseline.PlanHash,
            completed.CanReplaceFullBaseline, AuthenticatedWorkflowOrigin = false, ProductionSelectionEnabled = false });
        break;
    }
    case Command.SelectChanges:
    {
        if (args.Length != 10) throw new ArgumentException("SelectChanges repository before-snapshot after-snapshot before-inventory after-inventory before-bundle after-bundle plan-output selection-output");
        SourceBundleSnapshot before = LocalEvidenceReader.ReadSource(args[2]);
        SourceBundleSnapshot after = LocalEvidenceReader.ReadSource(args[3]);
        DiscoveryManifest oldInventory = Read<DiscoveryManifest>(args[4]);
        DiscoveryManifest currentInventory = Read<DiscoveryManifest>(args[5]);
        LocalEvidenceReader.ValidateBundle(before, oldInventory, args[6]);
        LocalEvidenceReader.ValidateBundle(after, currentInventory, args[7]);
        if (oldInventory.Workload != currentInventory.Workload) throw new InvalidDataException("Workload identity changed.");
        SourceDelta delta = GitSourceDelta.Read(args[1], before.SourceTree, after.SourceTree);
        if (oldInventory.Context.ProfileFingerprint != currentInventory.Context.ProfileFingerprint) delta = delta with { Unmapped = true };
        SourceSelection selected = SourceImpact.Select(before, after, oldInventory.Cases, currentInventory.Cases, delta);
        // No fabricated successful zero-test execution. A reuse-only result needs
        // the separately verified baseline protocol, not an empty runner plan.
        if (selected.Methods.Length == 0) throw new EvidenceException(EvidenceFailure.Scope, "No execution required by this graph; verified baseline reuse is required before skipping.");
        ValidationScope scope = selected.Methods.Length == currentInventory.Cases.Select(test => test.MethodId).Distinct(StringComparer.Ordinal).Count()
            ? ValidationScope.FullWorkload : ValidationScope.SelectedMethods;
        ExecutionPlan plan = RunnerBinding.Prepare(currentInventory, scope == ValidationScope.FullWorkload ? [] :
            selected.Methods.Select(method => method.MethodId).ToArray(), scope);
        RunnerBinding.WriteNew(args[9], new { Before = before.SourceTree, After = after.SourceTree, Delta = delta,
            Selection = selected, ProductionSelectionEnabled = false, AuthenticatedWorkflowOrigin = false });
        RunnerBinding.WriteNew(args[8], plan);
        break;
    }
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
        if (args.Length is not (10 or 12)) throw new ArgumentException("Verify inventory.json plan.json report-directory results.trx collection-run repository workflow-run attempt output.json [bundle test-assembly-file]");
        var origin = new RunIdentity(args[6], long.Parse(args[7], System.Globalization.CultureInfo.InvariantCulture),
            int.Parse(args[8], System.Globalization.CultureInfo.InvariantCulture));
        VerifiedObservedExecution observed = LocalEvidenceReader.VerifyObserved(Read<DiscoveryManifest>(args[1]), new(args[2], args[3], args[4], args[5], origin));
        VerifiedExecution result = observed.Execution;
        var ownerCompletion = args.Length == 12 ? ReviewedOwnerCompletion.ReadAll(args[10], args[11],
            result.Cases.Select(item => item.MethodId).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(), observed) : [];
        RunnerBinding.WriteNew(args[9], new { result.Scope, result.PlanHash, result.InventoryHash, result.Context,
            result.Workload, result.Origin, result.Cases, result.CanReplaceFullBaseline,
            observed.StandardCases, observed.RuntimeProfile, OwnerCompletion = ownerCompletion, AuthenticatedWorkflowOrigin = false });
        break;
    }
}

static T Read<T>(string path) where T : class => ExecutionEvidence.ReadDocumentFile<T>(path);
enum Command { Prepare, Verify, SelectChanges, PrepareReuse, CompleteReuse, ImportWorkflow }
