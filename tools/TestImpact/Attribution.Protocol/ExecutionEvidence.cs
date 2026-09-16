using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AiDotNet.TestImpact;

public enum ValidationScope { FullWorkload, SelectedMethods }
public enum CaseOutcome { Passed, Failed, Skipped, Cancelled }
public enum EvidenceFailure { Inventory, Context, Plan, Scope, Provenance, CaseSet, Outcome, Format }

public sealed class EvidenceException(EvidenceFailure reason, string message) : Exception(message)
{
    public EvidenceFailure Reason { get; } = reason;
}

public sealed record TestCaseIdentity(string CaseId, string MethodId);
public sealed record TestCaseResult(string CaseId, CaseOutcome Outcome);
public sealed record ExecutionContextIdentity(string SourceTree, string BuildFingerprint, string ProfileFingerprint);
public sealed record RunIdentity(string Repository, long RunId, int Attempt);
public sealed record ExecutionPlan(string Workload, ValidationScope Scope, ExecutionContextIdentity Context,
    string InventoryHash, string PlanHash, TestCaseIdentity[] RequiredCases);
public sealed record ExecutionReceipt(int Schema, string Workload, ValidationScope Scope,
    ExecutionContextIdentity Context, string InventoryHash, string PlanHash, RunIdentity Run,
    TestCaseResult[] Results);

// This validates consistency, not GitHub authenticity. The caller must independently
// authenticate the originating workflow/run and must generate the impact plan safely.
// It cannot turn coverage, a PR-provided "success" flag, or a partial run into a baseline.
public static class ExecutionEvidence
{
    public static ExecutionPlan CreatePlan(string workload, TestCaseIdentity[] inventory,
        string[] methods, ValidationScope scope, ExecutionContextIdentity context)
    {
        RequireText(workload);
        ValidateContext(context);
        if (!Enum.IsDefined(scope)) Fail(EvidenceFailure.Scope, "Undefined scope.");
        TestCaseIdentity[] ordered = ValidateInventory(inventory);
        var knownMethods = ordered.Select(test => test.MethodId).ToHashSet(StringComparer.Ordinal);
        ArgumentNullException.ThrowIfNull(methods);
        var selected = new HashSet<string>(StringComparer.Ordinal);
        foreach (string method in methods)
        {
            RequireText(method);
            if (!selected.Add(method)) Fail(EvidenceFailure.Plan, "Duplicate method selection.");
            if (!knownMethods.Contains(method)) Fail(EvidenceFailure.Plan, "Unknown selected method.");
        }
        if (scope == ValidationScope.FullWorkload && selected.Count != 0)
            Fail(EvidenceFailure.Plan, "Full-workload validation cannot contain a narrowing filter.");
        TestCaseIdentity[] required = scope == ValidationScope.FullWorkload
            ? ordered : ordered.Where(test => selected.Contains(test.MethodId)).ToArray();
        if (required.Length == 0) Fail(EvidenceFailure.Plan, "Runtime validation cannot certify zero selected cases.");
        string inventoryHash = Hash(ordered);
        string planHash = Hash(new { workload, scope, context, inventoryHash, required });
        return new(workload, scope, context, inventoryHash, planHash, required);
    }

    public static VerifiedExecution Verify(ExecutionPlan plan, TestCaseIdentity[] inventory,
        ExecutionReceipt receipt, RunIdentity independentlyResolvedRun)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(receipt);
        ValidateRun(independentlyResolvedRun);
        ValidateContext(receipt.Context);
        if (receipt.Schema != 1) Fail(EvidenceFailure.Format, "Unsupported receipt schema.");
        if (receipt.Run != independentlyResolvedRun) Fail(EvidenceFailure.Provenance, "Originating run/attempt differs.");
        if (!Enum.IsDefined(receipt.Scope) || receipt.Scope != plan.Scope)
            Fail(EvidenceFailure.Scope, "Partial and full-workload evidence are not interchangeable.");
        if (receipt.Context != plan.Context) Fail(EvidenceFailure.Context, "Source, binaries or execution profile differ.");
        if (receipt.Workload != plan.Workload || receipt.PlanHash != plan.PlanHash || receipt.InventoryHash != plan.InventoryHash)
            Fail(EvidenceFailure.Plan, "Receipt does not belong to this plan.");
        ExecutionPlan expected = ValidatePlan(plan, inventory, plan.Workload, plan.Context);
        if (receipt.Results is null || receipt.Results.Length != expected.RequiredCases.Length)
            Fail(EvidenceFailure.CaseSet, "Missing or extra execution results.");
        var cases = new HashSet<string>(StringComparer.Ordinal);
        var requiredIds = expected.RequiredCases.Select(test => test.CaseId).ToHashSet(StringComparer.Ordinal);
        foreach (TestCaseResult result in receipt.Results)
        {
            if (result is null || !cases.Add(result.CaseId) || !requiredIds.Contains(result.CaseId))
                Fail(EvidenceFailure.CaseSet, "Duplicate or unexpected case.");
            if (result.Outcome != CaseOutcome.Passed) Fail(EvidenceFailure.Outcome, "Failure, skip, cancellation or unknown outcome.");
        }
        return new VerifiedExecution(plan.Scope, plan.Context, plan.PlanHash, plan.InventoryHash,
            plan.Workload, independentlyResolvedRun, expected.RequiredCases.ToArray());
    }

    public static ExecutionPlan ValidatePlan(ExecutionPlan plan, TestCaseIdentity[] inventory,
        string expectedWorkload, ExecutionContextIdentity actualContext)
    {
        ArgumentNullException.ThrowIfNull(plan);
        ValidateContext(actualContext);
        if (plan.Context != actualContext) Fail(EvidenceFailure.Context, "Plan does not match the current source/build/profile.");
        if (plan.Workload != expectedWorkload) Fail(EvidenceFailure.Plan, "Plan is for a different workload.");
        TestCaseIdentity[] ordered = ValidateInventory(inventory);
        if (Hash(ordered) != plan.InventoryHash) Fail(EvidenceFailure.Inventory, "Test inventory changed.");
        if (plan.RequiredCases is null || plan.RequiredCases.Any(test => test is null))
            Fail(EvidenceFailure.Plan, "Missing required cases.");
        // Reconstruct the plan so mutating a deserialized RequiredCases array cannot
        // drop a theory row while retaining the original plan hash.
        string[] selectedMethods = plan.Scope == ValidationScope.FullWorkload ? [] :
            plan.RequiredCases.Select(test => test.MethodId).Distinct(StringComparer.Ordinal).ToArray();
        ExecutionPlan expected = CreatePlan(plan.Workload, ordered, selectedMethods, plan.Scope, plan.Context);
        if (expected.PlanHash != plan.PlanHash || !expected.RequiredCases.SequenceEqual(plan.RequiredCases))
            Fail(EvidenceFailure.Plan, "Plan was modified or has incomplete method rows.");
        return expected;
    }

    public static bool CanReuseIdenticalExecution(VerifiedExecution execution, ExecutionPlan requiredPlan) =>
        execution.Context == requiredPlan.Context && execution.PlanHash == requiredPlan.PlanHash &&
        execution.InventoryHash == requiredPlan.InventoryHash && execution.Workload == requiredPlan.Workload &&
        execution.Scope == requiredPlan.Scope && execution.Cases.SequenceEqual(requiredPlan.RequiredCases);

    public static ExecutionReceipt ReadReceipt(string json) => ReadDocument<ExecutionReceipt>(json);
    public static ExecutionPlan ReadPlan(string json) => ReadDocument<ExecutionPlan>(json);

    public static T ReadDocument<T>(string json) where T : class
    {
        if (string.IsNullOrWhiteSpace(json) || json.Length > 32 * 1024 * 1024)
            Fail(EvidenceFailure.Format, "Missing or oversized receipt.");
        try
        {
            using JsonDocument document = JsonDocument.Parse(json, new JsonDocumentOptions { MaxDepth = 32 });
            RejectDuplicateProperties(document.RootElement);
            var options = new JsonSerializerOptions { UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
                RespectRequiredConstructorParameters = true };
            options.Converters.Add(new JsonStringEnumConverter(allowIntegerValues: false));
            return document.RootElement.Deserialize<T>(options)
                ?? throw new JsonException("Null receipt.");
        }
        catch (JsonException exception) { throw new EvidenceException(EvidenceFailure.Format, exception.Message); }
    }

    private static void RejectDuplicateProperties(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (JsonProperty property in element.EnumerateObject())
            {
                if (!names.Add(property.Name)) throw new JsonException("Duplicate receipt property.");
                RejectDuplicateProperties(property.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
            foreach (JsonElement item in element.EnumerateArray()) RejectDuplicateProperties(item);
    }

    private static TestCaseIdentity[] ValidateInventory(TestCaseIdentity[] inventory)
    {
        if (inventory is null || inventory.Length == 0) Fail(EvidenceFailure.Inventory, "Missing independent test inventory.");
        var cases = new HashSet<string>(StringComparer.Ordinal);
        foreach (TestCaseIdentity test in inventory)
        {
            if (test is null) Fail(EvidenceFailure.Inventory, "Null test identity.");
            RequireText(test.CaseId);
            RequireText(test.MethodId);
            if (!cases.Add(test.CaseId)) Fail(EvidenceFailure.Inventory, "Duplicate inventory case.");
        }
        return inventory.OrderBy(test => test.CaseId, StringComparer.Ordinal).ToArray();
    }

    private static void ValidateContext(ExecutionContextIdentity context)
    {
        if (context is null || !IsHash(context.SourceTree, 40) || !IsHash(context.BuildFingerprint, 64) || !IsHash(context.ProfileFingerprint, 64))
            Fail(EvidenceFailure.Context, "Missing or invalid source/build/profile identity.");
    }

    private static void ValidateRun(RunIdentity run)
    {
        if (run is null || string.IsNullOrWhiteSpace(run.Repository) || run.RunId <= 0 || run.Attempt <= 0)
            Fail(EvidenceFailure.Provenance, "Missing resolved workflow origin.");
    }

    private static bool IsHash(string? text, int length) => text is not null && text.Length == length &&
        text.All(value => value is >= '0' and <= '9' or >= 'a' and <= 'f');
    private static void RequireText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) Fail(EvidenceFailure.Format, "Empty identity.");
    }
    private static string Hash<T>(T value) => Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(value))));
    [System.Diagnostics.CodeAnalysis.DoesNotReturn]
    private static void Fail(EvidenceFailure failure, string message) => throw new EvidenceException(failure, message);
}

public sealed class VerifiedExecution
{
    internal VerifiedExecution(ValidationScope scope, ExecutionContextIdentity context, string planHash,
        string inventoryHash, string workload, RunIdentity origin, TestCaseIdentity[] cases)
    {
        Scope = scope; Context = context; PlanHash = planHash; InventoryHash = inventoryHash;
        Workload = workload; Origin = origin; Cases = Array.AsReadOnly(cases);
    }
    public ValidationScope Scope { get; }
    public ExecutionContextIdentity Context { get; }
    public string PlanHash { get; }
    public string InventoryHash { get; }
    public string Workload { get; }
    public RunIdentity Origin { get; }
    public IReadOnlyList<TestCaseIdentity> Cases { get; }
    public bool CanReplaceFullBaseline => Scope == ValidationScope.FullWorkload;
}
