using AiDotNet.TestImpact;
using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "Protocol")]
public sealed class ProtocolTests
{
    private static TestCaseIdentity[] Inventory() =>
        [new("Theory/one", "Theory"), new("Theory/two", "Theory"), new("Other/one", "Other")];
    private static ExecutionContextIdentity Context() => new(new('a', 40), new('b', 64), new('c', 64));
    private static RunIdentity Origin() => new("ooples/AiDotNet", 123, 1);
    private static ExecutionPlan Plan(ValidationScope scope = ValidationScope.SelectedMethods) =>
        ExecutionEvidence.CreatePlan("unit-fixture/net10.0", Inventory(),
            scope == ValidationScope.FullWorkload ? [] : ["Theory"], scope, Context());
    private static ExecutionReceipt Receipt(ExecutionPlan plan) => new(1, plan.Workload, plan.Scope, plan.Context,
        plan.InventoryHash, plan.PlanHash, Origin(), plan.RequiredCases.Select(test => new TestCaseResult(test.CaseId, CaseOutcome.Passed)).ToArray());
    private static void Reject(EvidenceFailure expected, Action action) =>
        Assert.Equal(expected, Assert.Throws<EvidenceException>(action).Reason);

    [Fact]
    public void MethodSelectionRetainsEveryTheoryRowAndCannotReplaceBaseline()
    {
        ExecutionPlan plan = Plan();
        Assert.Equal(new[] { "Theory/one", "Theory/two" }, plan.RequiredCases.Select(test => test.CaseId));
        VerifiedExecution verified = ExecutionEvidence.Verify(plan, Inventory(), Receipt(plan), Origin());
        Assert.False(verified.CanReplaceFullBaseline);
        Assert.True(ExecutionEvidence.CanReuseIdenticalExecution(verified, plan));
    }

    [Fact]
    public void OnlyCompleteInventoryCanReplaceFullBaseline()
    {
        ExecutionPlan plan = Plan(ValidationScope.FullWorkload);
        Assert.True(ExecutionEvidence.Verify(plan, Inventory(), Receipt(plan), Origin()).CanReplaceFullBaseline);
        ExecutionReceipt partial = Receipt(plan) with { Results = Receipt(plan).Results[..2] };
        Reject(EvidenceFailure.CaseSet, () => ExecutionEvidence.Verify(plan, Inventory(), partial, Origin()));
    }

    [Fact]
    public void PartialReceiptCannotBecomeFullWorkloadEvidence()
    {
        ExecutionPlan plan = Plan();
        ExecutionReceipt receipt = Receipt(plan) with { Scope = ValidationScope.FullWorkload };
        Reject(EvidenceFailure.Scope, () => ExecutionEvidence.Verify(plan, Inventory(), receipt, Origin()));
    }

    [Theory]
    [InlineData(CaseOutcome.Failed)]
    [InlineData(CaseOutcome.Skipped)]
    [InlineData(CaseOutcome.Cancelled)]
    [InlineData((CaseOutcome)999)]
    public void NonPassingOutcomesCannotCertify(CaseOutcome outcome)
    {
        ExecutionPlan plan = Plan();
        ExecutionReceipt receipt = Receipt(plan);
        receipt.Results[0] = receipt.Results[0] with { Outcome = outcome };
        Reject(EvidenceFailure.Outcome, () => ExecutionEvidence.Verify(plan, Inventory(), receipt, Origin()));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DuplicateOrForeignCasesCannotCertify(bool duplicate)
    {
        ExecutionPlan plan = Plan();
        ExecutionReceipt receipt = Receipt(plan);
        receipt.Results[0] = new(duplicate ? receipt.Results[1].CaseId : "Other/one", CaseOutcome.Passed);
        Reject(EvidenceFailure.CaseSet, () => ExecutionEvidence.Verify(plan, Inventory(), receipt, Origin()));
    }

    [Fact]
    public void ChangedInventoryInvalidatesEvidence()
    {
        ExecutionPlan plan = Plan();
        TestCaseIdentity[] changed = [.. Inventory(), new("Theory/three", "Theory")];
        Reject(EvidenceFailure.Inventory, () => ExecutionEvidence.Verify(plan, changed, Receipt(plan), Origin()));
    }

    [Fact]
    public void MissingTheoryRowCannotHideBehindTheOriginalPlanHash()
    {
        ExecutionPlan original = Plan();
        ExecutionPlan changed = original with { RequiredCases = original.RequiredCases[..1] };
        Reject(EvidenceFailure.Plan, () => ExecutionEvidence.Verify(changed, Inventory(), Receipt(changed), Origin()));
        VerifiedExecution verified = ExecutionEvidence.Verify(original, Inventory(), Receipt(original), Origin());
        Assert.False(ExecutionEvidence.CanReuseIdenticalExecution(verified, changed));
    }

    [Fact]
    public void ChangingSourceBuildOrProfilePreventsReuse()
    {
        ExecutionPlan plan = Plan();
        VerifiedExecution verified = ExecutionEvidence.Verify(plan, Inventory(), Receipt(plan), Origin());
        ExecutionContextIdentity[] changes =
        [Context() with { SourceTree = new('d', 40) }, Context() with { BuildFingerprint = new('e', 64) },
            Context() with { ProfileFingerprint = new('f', 64) }];
        foreach (ExecutionContextIdentity context in changes)
        {
            Reject(EvidenceFailure.Context, () => ExecutionEvidence.Verify(plan, Inventory(), Receipt(plan) with { Context = context }, Origin()));
            Assert.False(ExecutionEvidence.CanReuseIdenticalExecution(verified, plan with { Context = context }));
        }
    }

    [Fact]
    public void WrongRunAttemptOrRepositoryCannotCertify()
    {
        ExecutionPlan plan = Plan();
        foreach (RunIdentity origin in new[] { Origin() with { RunId = 124 }, Origin() with { Attempt = 2 }, Origin() with { Repository = "other/repo" } })
            Reject(EvidenceFailure.Provenance, () => ExecutionEvidence.Verify(plan, Inventory(), Receipt(plan), origin));
    }

    [Fact]
    public void EmptyUnknownOrDuplicateSelectionCannotCertify()
    {
        foreach (string[] methods in new string[][] { [], ["Unknown"], ["Theory", "Theory"] })
            Reject(EvidenceFailure.Plan, () => ExecutionEvidence.CreatePlan("unit", Inventory(), methods, ValidationScope.SelectedMethods, Context()));
    }

    [Fact]
    public void ReceiptParsingRejectsDuplicateUnknownMissingAndNumericEnumFields()
    {
        var options = new JsonSerializerOptions();
        options.Converters.Add(new JsonStringEnumConverter());
        string json = JsonSerializer.Serialize(Receipt(Plan()), options);
        ExecutionReceipt roundTrip = ExecutionEvidence.ReadReceipt(json);
        ExecutionEvidence.Verify(Plan(), Inventory(), roundTrip, Origin());
        string[] invalid =
        [json.Replace("\"Schema\":1", "\"Schema\":1,\"Schema\":1", StringComparison.Ordinal),
            json.Replace("\"Schema\":1", "\"Unknown\":1,\"Schema\":1", StringComparison.Ordinal),
            json.Replace("\"Schema\":1,", "", StringComparison.Ordinal),
            json.Replace("\"SelectedMethods\"", "1", StringComparison.Ordinal), "null", "{truncated"];
        foreach (string input in invalid) Reject(EvidenceFailure.Format, () => ExecutionEvidence.ReadReceipt(input));
    }
}
