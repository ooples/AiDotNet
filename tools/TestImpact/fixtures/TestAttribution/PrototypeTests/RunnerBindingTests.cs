using AiDotNet.TestImpact;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;
using AiDotNet.TestImpact.Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class RunnerBindingTests
{
    private sealed class RecordingSink(Action<IMessageSinkMessage> record) : LongLivedMarshalByRefObject, IMessageSink
    {
        public bool OnMessage(IMessageSinkMessage message) { record(message); return true; }
    }

    private static TestAssemblyFinished Finished() => new([], new TestAssembly(new ReflectionAssemblyInfo(typeof(RunnerBindingTests).Assembly)), 0m, 0, 0, 0);

    [Fact]
    public void AssemblyCompletionCannotReachVstestBeforePublication()
    {
        bool published = false;
        var observed = new List<IMessageSinkMessage>();
        var sink = new AttributionExecutionSink(new RecordingSink(message => { Assert.True(published); observed.Add(message); }));
        TestAssemblyFinished completion = Finished();
        Assert.True(sink.OnMessage(completion));
        Assert.Empty(observed);
        published = true;
        sink.Complete();
        Assert.Same(completion, Assert.Single(observed));
    }

    [Fact]
    public void MissingOrDuplicateAssemblyCompletionIsRejected()
    {
        var sink = new AttributionExecutionSink(new RecordingSink(_ => { }));
        Assert.Throws<InvalidOperationException>(() => sink.Complete());
        sink.OnMessage(Finished());
        Assert.Throws<InvalidOperationException>(() => sink.OnMessage(Finished()));
        sink.Complete();
        Assert.Throws<InvalidOperationException>(() => sink.Complete());
    }

    private static TestCaseIdentity[] Cases() => [new("one", "Theory"), new("two", "Theory"), new("other", "Other")];
    private static ExecutionContextIdentity Context() => new(new('a', 40), new('b', 64), new('c', 64));

    [Fact]
    public void PlanValidationRetainsAllRowsAndMakesAnIndependentCopy()
    {
        ExecutionPlan plan = ExecutionEvidence.CreatePlan("work", Cases(), ["Theory"], ValidationScope.SelectedMethods, Context());
        ExecutionPlan validated = ExecutionEvidence.ValidatePlan(plan, Cases(), "work", Context());
        plan.RequiredCases[0] = new("other", "Other");
        Assert.Equal(new[] { "one", "two" }, validated.RequiredCases.Select(test => test.CaseId));
        Assert.Throws<EvidenceException>(() => ExecutionEvidence.ValidatePlan(plan, Cases(), "work", Context()));
    }

    [Fact]
    public void WrongWorkloadOrBuildCannotReachTheRunner()
    {
        ExecutionPlan plan = ExecutionEvidence.CreatePlan("work", Cases(), ["Theory"], ValidationScope.SelectedMethods, Context());
        Assert.Throws<EvidenceException>(() => ExecutionEvidence.ValidatePlan(plan, Cases(), "other", Context()));
        Assert.Throws<EvidenceException>(() => ExecutionEvidence.ValidatePlan(plan, Cases(), "work", Context() with { BuildFingerprint = new('d', 64) }));
        Assert.Throws<EvidenceException>(() => ExecutionEvidence.ValidatePlan(plan, Cases()[..2], "work", Context()));
    }

    [Fact]
    public void PlanParsingRejectsDuplicateMissingUnknownAndNumericFields()
    {
        ExecutionPlan plan = ExecutionEvidence.CreatePlan("work", Cases(), ["Theory"], ValidationScope.SelectedMethods, Context());
        string json = RunnerBinding.Serialize(plan);
        ExecutionPlan copy = ExecutionEvidence.ReadPlan(json);
        Assert.Equal(plan.PlanHash, ExecutionEvidence.ValidatePlan(copy, Cases(), "work", Context()).PlanHash);
        foreach (string invalid in new[] { json.Replace("\"Workload\":", "\"Workload\":\"duplicate\",\"Workload\":"),
            json.Replace("\"Workload\":", "\"Unknown\":"), json.Replace("\"SelectedMethods\"", "1"), "{}", "null" })
            Assert.Throws<EvidenceException>(() => ExecutionEvidence.ReadPlan(invalid));
    }

    [Fact]
    public void BundleFingerprintIncludesDependencyContentNamesAndData()
    {
        string root = Path.Combine(Path.GetTempPath(), "attribution-bundle-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            Assert.Throws<InvalidDataException>(() => RunnerBinding.HashBundle(root));
            File.WriteAllText(Path.Combine(root, "tests.dll"), "test bytes");
            string first = RunnerBinding.HashBundle(root);
            Assert.Equal(first, RunnerBinding.HashBundle(root));
            File.WriteAllText(Path.Combine(root, "dependency.dll"), "one");
            string second = RunnerBinding.HashBundle(root);
            Assert.NotEqual(first, second);
            File.WriteAllText(Path.Combine(root, "dependency.dll"), "two");
            string third = RunnerBinding.HashBundle(root);
            Assert.NotEqual(second, third);
            File.Move(Path.Combine(root, "dependency.dll"), Path.Combine(root, "renamed.dll"));
            string fourth = RunnerBinding.HashBundle(root);
            Assert.NotEqual(third, fourth);
            File.WriteAllText(Path.Combine(root, "test-data.json"), "{}");
            Assert.NotEqual(fourth, RunnerBinding.HashBundle(root));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void OutputCannotModifyTheBoundBundle()
    {
        string root = Path.Combine(Path.GetTempPath(), "bound-root");
        Assert.Throws<InvalidDataException>(() => RunnerBinding.RequireOutsideBundle(Path.Combine(root, "plan.json"), root));
        Assert.Throws<InvalidDataException>(() => RunnerBinding.RequireOutsideBundle(root, root));
        RunnerBinding.RequireOutsideBundle(root + "-sibling/plan.json", root);
    }

    [Fact]
    public void DiscoveryCannotProduceEmptyOrPartialFullPlan()
    {
        var discovery = new DiscoveryManifest(1, "work", Context(), Cases());
        Assert.Equal(3, RunnerBinding.Prepare(discovery, [], ValidationScope.FullWorkload).RequiredCases.Length);
        Assert.Throws<EvidenceException>(() => RunnerBinding.Prepare(discovery, ["Theory"], ValidationScope.FullWorkload));
        Assert.Throws<EvidenceException>(() => RunnerBinding.Prepare(discovery with { Schema = 99 }, [], ValidationScope.FullWorkload));
        Assert.Throws<EvidenceException>(() => RunnerBinding.Prepare(discovery with { Cases = [] }, [], ValidationScope.FullWorkload));
    }
}
