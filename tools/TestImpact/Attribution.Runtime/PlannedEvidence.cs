using System.Xml;
using System.Xml.Linq;
using AiDotNet.TestImpact;

namespace AttributionRuntime;

public static class PlannedEvidence
{
    /// <summary>Largest TRX this reader will materialise: generous for a full shard, far short of harmful.</summary>
    internal const long MaximumTrxBytes = 64L * 1024 * 1024;

    /// <summary>
    /// Loads a TRX with a size bound. An imported TRX comes from a pull-request artifact, and the archive
    /// and extraction limits bound the archive rather than this one file, so without a local bound a large
    /// TRX is fully materialised as a DOM before any validation runs. Missing, empty, oversized and
    /// malformed input all fail as typed Format evidence rather than an untyped I/O or XML exception.
    /// </summary>
    internal static XDocument LoadTrx(string trxPath)
    {
        var file = new FileInfo(trxPath);
        if (!file.Exists || file.Length <= 0 || file.Length > MaximumTrxBytes)
            throw new EvidenceException(EvidenceFailure.Format, "Missing, empty or oversized TRX evidence.");
        try
        {
            using var reader = XmlReader.Create(trxPath, new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
                MaxCharactersInDocument = MaximumTrxBytes
            });
            return XDocument.Load(reader);
        }
        catch (XmlException exception)
        {
            throw new EvidenceException(EvidenceFailure.Format, $"Malformed or oversized TRX evidence: {exception.Message}");
        }
    }
    // Consistency validation only. Workflow origin must still be authenticated
    // by the caller. Worker-backed execution needs a multi-bundle binding and
    // is deliberately not reusable through this single-bundle path.
    public static VerifiedExecution Verify(DiscoveryManifest manifest, ExecutionPlan plan,
        AttributionReport report, string trxPath, string expectedCollectionRun, RunIdentity origin)
        => VerifyObserved(manifest, plan, report, trxPath, expectedCollectionRun, origin).Execution;

    public static VerifiedObservedExecution VerifyObserved(DiscoveryManifest manifest, ExecutionPlan plan,
        AttributionReport report, string trxPath, string expectedCollectionRun, RunIdentity origin)
    {
        RuntimeContractProfile? runtimeProfile = RuntimeProfileEvidence.Read(manifest);
        if (manifest.Schema != 1 || report.Schema != 4 || report.Kind != AttributionProcessKind.TestHost ||
            report.Run != expectedCollectionRun || !Guid.TryParseExact(report.Run, "N", out _) ||
            !Guid.TryParseExact(report.Token, "N", out _) || report.WorkerOwner is not null ||
            report.Faults is null || report.Faults.Length != 0 || report.Workers is null || report.Workers.Length != 0 ||
            report.Cases is null || report.CompletedOwners is null || report.Plan is null)
            throw new EvidenceException(EvidenceFailure.Outcome, "Incomplete, faulted, foreign or unsupported worker evidence.");
        ExecutionPlan expected = ExecutionEvidence.ValidatePlan(plan, manifest.Cases, manifest.Workload, manifest.Context);
        ExecutionPlan recorded = ExecutionEvidence.ValidatePlan(report.Plan, manifest.Cases, manifest.Workload, manifest.Context);
        if (recorded.PlanHash != expected.PlanHash)
            throw new EvidenceException(EvidenceFailure.Plan, "Runner used a different execution plan.");
        var required = expected.RequiredCases.ToDictionary(test => test.CaseId, StringComparer.Ordinal);
        string[] owners = expected.RequiredCases.Select(test => test.MethodId).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
        if (!owners.SequenceEqual(report.CompletedOwners.Order(StringComparer.Ordinal)))
            throw new EvidenceException(EvidenceFailure.CaseSet, "Completed methods differ from the plan.");
        var results = new List<TestCaseResult>();
        var names = new List<string>();
        var boundRows = new List<(string Name, string Marker)>();
        var standardCases = new List<string>();
        foreach (CaseExecutionReport execution in report.Cases)
        {
            if (execution is null || execution.Case is null || !execution.Finished || execution.Results is null ||
                execution.Results.Length == 0 || !Enum.IsDefined(execution.Case.Kind) ||
                !required.TryGetValue(execution.Case.Id, out TestCaseIdentity? identity) || identity.MethodId != execution.Case.Owner ||
                (execution.Case.Kind == DiscoveredCaseKind.Enumerated && execution.Results.Length != 1))
                throw new EvidenceException(EvidenceFailure.CaseSet, "Invalid discovered case or missing completion.");
            foreach (ObservedCaseResult result in execution.Results)
            {
                if (result is null || result.Outcome != ObservedOutcome.Passed || string.IsNullOrWhiteSpace(result.DisplayName) ||
                    (execution.Case.Kind == DiscoveredCaseKind.Enumerated && result.DisplayName != execution.Case.DisplayName))
                    throw new EvidenceException(EvidenceFailure.Outcome, "Nonpassing or inconsistent runtime row.");
                names.Add(result.DisplayName);
                boundRows.Add((result.DisplayName, CaseOutputIdentity.Format(report.Run, report.Token, execution.Case.Id)));
            }
            results.Add(new(execution.Case.Id, CaseOutcome.Passed));
            if (execution.Case.Kind == DiscoveredCaseKind.Enumerated) standardCases.Add(execution.Case.Id);
        }
        XDocument trx = LoadTrx(trxPath);
        XElement[] actual = trx.Descendants().Where(element => element.Name.LocalName == "UnitTestResult").ToArray();
        XElement[] summaries = trx.Descendants().Where(element => element.Name.LocalName == "ResultSummary").ToArray();
        if (summaries.Length != 1 || (string?)summaries[0].Attribute("outcome") != "Completed" || actual.Length == 0 ||
            actual.Any(element => (string?)element.Attribute("outcome") != "Passed") ||
            !names.Order(StringComparer.Ordinal).SequenceEqual(actual.Select(element => (string?)element.Attribute("testName") ?? "").Order(StringComparer.Ordinal)))
            throw new EvidenceException(EvidenceFailure.Outcome, "Independent TRX does not match the completed case ledger.");
        var actualRows = new List<(string Name, string Marker)>();
        foreach (XElement row in actual)
        {
            string[] markers = row.Descendants().Where(element => element.Name.LocalName == "StdOut")
                .SelectMany(element => element.Value.Split('\n')).Select(line => line.TrimEnd('\r'))
                .Where(line => line.StartsWith(CaseOutputIdentity.Prefix, StringComparison.Ordinal)).ToArray();
            if (markers.Length != 1)
                throw new EvidenceException(EvidenceFailure.Provenance, "TRX row lacks a unique execution identity.");
            actualRows.Add(((string?)row.Attribute("testName") ?? "", markers[0]));
        }
        if (!boundRows.OrderBy(row => row.Name, StringComparer.Ordinal).ThenBy(row => row.Marker, StringComparer.Ordinal)
            .SequenceEqual(actualRows.OrderBy(row => row.Name, StringComparer.Ordinal).ThenBy(row => row.Marker, StringComparer.Ordinal)))
            throw new EvidenceException(EvidenceFailure.Provenance, "TRX belongs to a different execution.");
        var receipt = new ExecutionReceipt(1, expected.Workload, expected.Scope, expected.Context,
            expected.InventoryHash, expected.PlanHash, origin, results.ToArray());
        VerifiedExecution verified = ExecutionEvidence.Verify(expected, manifest.Cases, receipt, origin);
        // Preserve the independently checked case kind. A generic passing receipt
        // or a custom case runner cannot prove that standard xUnit awaited a task.
        string[] standard = standardCases.Order(StringComparer.Ordinal).ToArray();
        return new(verified, standard, runtimeProfile, TrialScopeEvidence.Read(report.TrialScopes, verified, standard));
    }
}

// Consistency evidence, not workflow authentication or an assertion that the
// supplied runner binary implements the reviewed xUnit task-awaiting semantics.
// Those bindings must be checked separately before this can close a contract.
public sealed class VerifiedObservedExecution
{
    private readonly HashSet<string> standardCases;

    internal VerifiedObservedExecution(VerifiedExecution execution, string[] standard, RuntimeContractProfile? runtimeProfile,
        TrialScopeObservation[]? trialScopes = null)
    {
        Execution = execution;
        standardCases = standard.ToHashSet(StringComparer.Ordinal);
        StandardCases = Array.AsReadOnly(standard.ToArray());
        RuntimeProfile = runtimeProfile;
        TrialScopes = Array.AsReadOnly(trialScopes?.ToArray() ?? []);
    }

    public VerifiedExecution Execution { get; }
    public IReadOnlyList<string> StandardCases { get; }
    public RuntimeContractProfile? RuntimeProfile { get; }
    public IReadOnlyList<TrialScopeObservation> TrialScopes { get; }

    public bool HasStandardOwnerCompletion(string owner)
    {
        TestCaseIdentity[] cases = Execution.Cases.Where(item => item.MethodId == owner).ToArray();
        return cases.Length != 0 && cases.All(item => standardCases.Contains(item.CaseId));
    }
}
