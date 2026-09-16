using System.Xml;
using System.Xml.Linq;
using AiDotNet.TestImpact;

namespace AttributionRuntime;

public static class PlannedEvidence
{
    // Consistency validation only. Workflow origin must still be authenticated
    // by the caller. Worker-backed execution needs a multi-bundle binding and
    // is deliberately not reusable through this single-bundle path.
    public static VerifiedExecution Verify(DiscoveryManifest manifest, ExecutionPlan plan,
        AttributionReport report, string trxPath, string expectedCollectionRun, RunIdentity origin)
    {
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
        }
        using var reader = XmlReader.Create(trxPath, new XmlReaderSettings { DtdProcessing = DtdProcessing.Prohibit, XmlResolver = null });
        XDocument trx = XDocument.Load(reader);
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
        return ExecutionEvidence.Verify(expected, manifest.Cases, receipt, origin);
    }
}
