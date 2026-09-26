using System.Runtime.ExceptionServices;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using AttributionRuntime;
using Xunit;

namespace PrototypeTests;

[CollectionDefinition(nameof(ExceptionObserverProbeTests), DisableParallelization = true)]
public sealed class ExceptionObserverProbeCollection { }

[Collection(nameof(ExceptionObserverProbeTests))]
[Trait("Scenario", "RuntimeEffects")]
public sealed class ExceptionObserverProbeTests
{
    [Fact]
    public void ActualCallbackPresenceIsObservedAndRestored()
    {
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ExceptionObserverProbe.RuntimeHash;
        ExceptionObserverState before = ExceptionObserverProbe.Capture();
        if (!supported)
        {
            Assert.Equal(ExceptionObserverState.Unknown, before);
            Assert.Null(ExceptionObserverProbe.Bind(typeof(object).Assembly));
            return;
        }
        Assert.NotEqual(ExceptionObserverState.Unknown, before);
        EventHandler<FirstChanceExceptionEventArgs> handler = static (_, _) => { };
        AppDomain.CurrentDomain.FirstChanceException += handler;
        try { Assert.Equal(ExceptionObserverState.Present, ExceptionObserverProbe.Capture()); }
        finally { AppDomain.CurrentDomain.FirstChanceException -= handler; }
        Assert.Equal(before, ExceptionObserverProbe.Capture());
    }

    [Fact]
    public void AnotherAssemblyCannotSupplyTheRuntimeBinding()
        => Assert.Null(ExceptionObserverProbe.Bind(typeof(ExceptionObserverProbeTests).Assembly));

    [Theory]
    [InlineData(ExceptionObserverState.NoneObserved, ExceptionObserverState.NoneObserved)]
    [InlineData(ExceptionObserverState.Present, ExceptionObserverState.NoneObserved)]
    [InlineData(ExceptionObserverState.NoneObserved, ExceptionObserverState.Present)]
    [InlineData(ExceptionObserverState.Unknown, ExceptionObserverState.Unknown)]
    public void ScopeAndVerifiedEvidenceRetainBothBoundaryStates(ExceptionObserverState before, ExceptionObserverState after)
    {
        var ledger = new TrialPathScopes();
        var sample = new TrialPathSample(new('a', 64), new('b', 64), TrialPathState.Absent);
        ledger.Begin("Owner", sample, new('c', 64), sample.PathHash, observers: before);
        TrialScopeObservation initial = Assert.Single(ledger.Snapshot().Scopes);
        ledger.End("Owner", sample, new('c', 64), after);
        Assert.Equal(ExceptionObserverState.Unknown, initial.ObserversAfter);
        using var fixture = new ObservedOwnerTests.Evidence([new("one", "Owner")]);
        TrialScopeReport report = ledger.Snapshot();
        fixture.Report = fixture.Report with { TrialScopes = report };
        VerifiedObservedExecution verified = fixture.Verify();
        TrialScopeObservation observed = Assert.Single(verified.TrialScopes);
        Assert.Equal(TrialScopeState.Complete, observed.State);
        Assert.Equal(before, observed.ObserversBefore);
        Assert.Equal(after, observed.ObserversAfter);
        report.Scopes[0] = observed with { ObserversBefore = ExceptionObserverState.Unknown, ObserversAfter = ExceptionObserverState.Unknown };
        Assert.Equal(before, Assert.Single(verified.TrialScopes).ObserversBefore);
        Assert.Equal(after, Assert.Single(verified.TrialScopes).ObserversAfter);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void InvalidBoundaryEnumsCannotSupplyEvidence(bool before)
    {
        using var fixture = new ObservedOwnerTests.Evidence([new("one", "Owner")]);
        var verified = fixture.Verify();
        var sample = new TrialScopeObservation("Owner", new('a', 64), new('b', 64), new('c', 64), new('d', 64), TrialScopeState.Complete,
            before ? (ExceptionObserverState)99 : ExceptionObserverState.NoneObserved,
            before ? ExceptionObserverState.NoneObserved : (ExceptionObserverState)99);
        Assert.Throws<EvidenceException>(() => TrialScopeEvidence.Read(new(TrialScopeLedgerState.Recorded, [sample]), verified.Execution, verified.StandardCases.ToArray()));
    }

    [Fact]
    public void MissingObservationsAreUnknownNotAbsent()
    {
        var sample = new TrialScopeObservation("Owner", new('a', 64), new('b', 64), new('c', 64), new('d', 64), TrialScopeState.Complete);
        Assert.Equal(ExceptionObserverState.Unknown, sample.ObserversBefore);
        Assert.Equal(ExceptionObserverState.Unknown, sample.ObserversAfter);
    }
}
