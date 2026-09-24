using AiDotNet.TestImpact;

namespace AttributionRuntime;

internal static class TrialScopeEvidence
{
    internal static TrialScopeObservation[] Read(TrialScopeReport? report, VerifiedExecution execution, string[] standardCases)
    {
        if (report is null) return []; // Legacy evidence cannot invent a scope observation.
        bool Hash(string value) => value is not null && value.Length == 64 &&
            value.All(character => character is >= '0' and <= '9' or >= 'a' and <= 'f');
        var owners = execution.Cases.GroupBy(test => test.MethodId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.ToArray(), StringComparer.Ordinal);
        if (!Enum.IsDefined(report.State) || report.Scopes is null || report.Scopes.Any(scope => scope is null ||
            string.IsNullOrWhiteSpace(scope.Owner) || !owners.ContainsKey(scope.Owner) || !Enum.IsDefined(scope.State) ||
            !Enum.IsDefined(scope.ObserversBefore) || !Enum.IsDefined(scope.ObserversAfter) ||
            !Hash(scope.PathHash) || !Hash(scope.PreviousHash) || !Hash(scope.PreviousPathHash) ||
            !(Hash(scope.RootHash) || scope.State == TrialScopeState.Rejected && scope.RootHash == "")) ||
            report.Scopes.Select(scope => scope.Owner).Distinct(StringComparer.Ordinal).Count() != report.Scopes.Length)
            throw new EvidenceException(EvidenceFailure.Format, "Malformed or foreign trial-scope evidence.");
        TrialScopeObservation[] completed = report.Scopes.Where(scope => scope.State == TrialScopeState.Complete).ToArray();
        var pathCounts = report.Scopes.GroupBy(scope => scope.PathHash, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
        if (completed.Any(scope => scope.PathHash == scope.PreviousPathHash || pathCounts[scope.PathHash] != 1))
            throw new EvidenceException(EvidenceFailure.Format, "A completed trial scope reused an existing path.");
        if (report.State != TrialScopeLedgerState.Recorded) return [];
        // This bounded contract supports one standard case per owner, not a
        // theory's aggregated rows or a custom runner's claimed completion.
        var standard = standardCases.ToHashSet(StringComparer.Ordinal);
        return completed.Where(scope => owners[scope.Owner].Length == 1 && standard.Contains(owners[scope.Owner][0].CaseId))
            .OrderBy(scope => scope.Owner, StringComparer.Ordinal).ToArray();
    }
}
