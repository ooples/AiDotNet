using System.Xml.Linq;
using AiDotNet.TestImpact;
using AttributionRuntime;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class ObservedOwnerTests
{
    public enum Mutation { UnknownKind, Failed, Skipped, Unfinished, MissingOwner, ForeignPlan, WrongMarker, DuplicateCase }

    [Fact]
    public void StandardCompletionRequiresEveryCaseAndIsAnIndependentSnapshot()
    {
        using var fixture = new Evidence();
        VerifiedObservedExecution observed = fixture.Verify();
        Assert.True(observed.HasStandardOwnerCompletion("Owner"));
        Assert.False(observed.HasStandardOwnerCompletion("NotExecuted"));
        Assert.Equal(new[] { "one", "two" }, observed.StandardCases);
        fixture.Report.Cases[0] = fixture.Report.Cases[0] with
        {
            Case = fixture.Report.Cases[0].Case with { Kind = DiscoveredCaseKind.DeferredOrCustom }
        };
        Assert.True(observed.HasStandardOwnerCompletion("Owner"));
        Assert.False(fixture.Verify().HasStandardOwnerCompletion("Owner"));
    }

    [Fact]
    public void PassingCustomCaseRemainsValidExecutionButNotStandardOwnerCompletion()
    {
        using var fixture = new Evidence();
        fixture.Report.Cases[1] = fixture.Report.Cases[1] with
        {
            Case = fixture.Report.Cases[1].Case with { Kind = DiscoveredCaseKind.DeferredOrCustom }
        };
        VerifiedObservedExecution observed = fixture.Verify();
        Assert.Equal(2, observed.Execution.Cases.Count);
        Assert.Single(observed.StandardCases);
        Assert.False(observed.HasStandardOwnerCompletion("Owner"));
    }

    [Theory]
    [InlineData(Mutation.UnknownKind)]
    [InlineData(Mutation.Failed)]
    [InlineData(Mutation.Skipped)]
    [InlineData(Mutation.Unfinished)]
    [InlineData(Mutation.MissingOwner)]
    [InlineData(Mutation.ForeignPlan)]
    [InlineData(Mutation.WrongMarker)]
    [InlineData(Mutation.DuplicateCase)]
    public void UnverifiedReportsCannotMintOwnerCompletion(Mutation mutation)
    {
        using var fixture = new Evidence();
        CaseExecutionReport first = fixture.Report.Cases[0];
        switch (mutation)
        {
            case Mutation.UnknownKind: fixture.Report.Cases[0] = first with { Case = first.Case with { Kind = (DiscoveredCaseKind)99 } }; break;
            case Mutation.Failed: first.Results[0] = first.Results[0] with { Outcome = ObservedOutcome.Failed }; break;
            case Mutation.Skipped: first.Results[0] = first.Results[0] with { Outcome = ObservedOutcome.Skipped }; break;
            case Mutation.Unfinished: fixture.Report.Cases[0] = first with { Finished = false }; break;
            case Mutation.MissingOwner: fixture.Report = fixture.Report with { CompletedOwners = [] }; break;
            case Mutation.ForeignPlan: fixture.Report = fixture.Report with { Plan = fixture.Plan with { Context = fixture.Plan.Context with { BuildFingerprint = new('d', 64) } } }; break;
            case Mutation.WrongMarker: fixture.WriteTrx(Guid.NewGuid().ToString("N")); break;
            case Mutation.DuplicateCase: fixture.Report.Cases[1] = first; break;
        }
        Assert.Throws<EvidenceException>(() => fixture.Verify());
    }

    internal sealed class Evidence : IDisposable
    {
        private readonly string directory = Directory.CreateTempSubdirectory("owner-evidence-").FullName;
        private readonly string run = Guid.NewGuid().ToString("N");
        private readonly string token = Guid.NewGuid().ToString("N");
        private readonly DiscoveryManifest inventory;
        internal ExecutionPlan Plan { get; }
        internal AttributionReport Report { get; set; }

        internal Evidence(TestCaseIdentity[]? discovered = null, string? fingerprint = null)
        {
            TestCaseIdentity[] cases = discovered ?? [new("one", "Owner"), new("two", "Owner")];
            inventory = new(1, "work", new(new('a', 40), fingerprint ?? new('b', 64), new('c', 64)), cases);
            Plan = RunnerBinding.Prepare(inventory, [], ValidationScope.FullWorkload);
            Report = new(4, run, token, AttributionProcessKind.TestHost, Environment.ProcessId, null, 1,
                HitCollectionMode.Cached, [], cases.Select(item => item.MethodId).Distinct(StringComparer.Ordinal).ToArray(), [], [], cases.Select(item => new CaseExecutionReport(
                    new(item.CaseId, item.MethodId, item.CaseId, DiscoveredCaseKind.Enumerated), true,
                    [new(item.CaseId, ObservedOutcome.Passed)])).ToArray(), Plan);
            WriteTrx(run);
        }

        internal void WriteTrx(string markerRun) => new XDocument(new XElement("TestRun",
            new XElement("ResultSummary", new XAttribute("outcome", "Completed")),
            new XElement("Results", inventory.Cases.Select(item => new XElement("UnitTestResult",
                new XAttribute("outcome", "Passed"), new XAttribute("testName", item.CaseId),
                new XElement("Output", new XElement("StdOut", CaseOutputIdentity.Format(markerRun, token, item.CaseId))))))))
            .Save(Path.Combine(directory, "results.trx"));

        internal VerifiedObservedExecution Verify() => PlannedEvidence.VerifyObserved(inventory, Plan, Report,
            Path.Combine(directory, "results.trx"), run, new("local/test", 1, 1));

        public void Dispose() => Directory.Delete(directory, recursive: true);
    }
}
