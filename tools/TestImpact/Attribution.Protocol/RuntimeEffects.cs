namespace AiDotNet.TestImpact;

public enum RuntimeEffectScope { FreshReturnedObject }
public sealed record OwnedBooleanWrite(string Field, int Instruction, bool Value);

// Describes a bounded change footprint, NOT purity of the whole method. Its
// unchanged allocation prefix may still perform native/ambient operations.
// Experiments using this summary require a separate full-workload control.
public sealed record OwnedResultEffect(int Schema, RuntimeEffectScope Scope, string ShapeHash,
    string[] OwnershipDependencies, OwnedBooleanWrite[] Writes);
public sealed record RuntimeEffectControlInput(string[] Inventory, string[] Selected, TestCaseResult[] BeforeFull,
    TestCaseResult[] AfterFull, TestCaseResult[] AfterSelected);

// An outcome comparison is experimental evidence, never an execution/reuse
// certificate. In particular, it cannot establish absence of latent side effects.
public static class RuntimeEffectControl
{
    public static void Compare(string[] inventory, string[] selected, TestCaseResult[] beforeFull,
        TestCaseResult[] afterFull, TestCaseResult[] afterSelected)
    {
        var expected = new HashSet<string>(inventory, StringComparer.Ordinal);
        var subset = new HashSet<string>(selected, StringComparer.Ordinal);
        if (expected.Count == 0 || expected.Count != inventory.Length || subset.Count == 0 ||
            subset.Count != selected.Length || !subset.IsProperSubsetOf(expected))
            throw new InvalidDataException("A nonempty proper subset of an exact inventory is required.");
        Dictionary<string, CaseOutcome> Rows(TestCaseResult[] rows, HashSet<string> cases)
        {
            if (rows.Length != cases.Count || rows.Select(row => row.CaseId).Distinct(StringComparer.Ordinal).Count() != rows.Length ||
                rows.Any(row => !cases.Contains(row.CaseId) || row.Outcome is not (CaseOutcome.Passed or CaseOutcome.Failed)))
                throw new InvalidDataException("Incomplete, duplicate, skipped or foreign control rows.");
            return rows.ToDictionary(row => row.CaseId, row => row.Outcome, StringComparer.Ordinal);
        }
        var before = Rows(beforeFull, expected);
        var after = Rows(afterFull, expected);
        var actual = Rows(afterSelected, subset);
        if (before.Values.Any(outcome => outcome != CaseOutcome.Passed)) throw new InvalidDataException("Baseline is not green.");
        if (expected.Except(subset).Any(id => before[id] != after[id]))
            throw new InvalidDataException("Candidate selection missed a changed outcome.");
        if (subset.Any(id => actual[id] != after[id]))
            throw new InvalidDataException("Selected execution disagrees with the independent full control.");
    }
}
