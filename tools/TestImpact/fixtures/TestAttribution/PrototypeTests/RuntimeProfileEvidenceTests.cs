using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class RuntimeProfileEvidenceTests
{
    public enum Startup { Missing, NoCompletion, Conflicting, OtherEngine, InvalidDop, InitialObserver, EffectiveObserver, MissingInputs }
    public enum Malformed { Hash, Schema, Status, NumericEnum, Duplicate, UnknownMember, ResetMode, ResetLogging, ResetWithoutStartup, GpuStartup, NumericGpuStartup,
        GpuDiagnostics, NumericGpuDiagnostics }
    private static RuntimeContractProfile Profile() => new(new(1, new('a', 64), RuntimeObserverSignals.NoneReported),
        new(RuntimeInitializationStatus.Recorded, new(1, new('b', 64), RuntimeObserverSignals.NoneReported, RuntimeGpuStartupPolicy.Disabled,
            RuntimeGpuDiagnosticsPolicy.NoDumpRequested), new(RuntimeCpuMode.Cpu, 1),
            new(RuntimeCpuEntryMode.PlainCpu, RuntimeCpuLogging.Suppressed)));
    private static DiscoveryManifest Manifest(string profile) => new(1, "work",
        new(new('a', 40), new('b', 64), Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(profile)))),
        [new("case", "Owner")], profile);
    private static string Json(RuntimeContractProfile profile) => RunnerBinding.Serialize(new { RuntimeContracts = profile });

    [Fact]
    public void BoundProfileRetainsInputsAndActualCpuCompletion()
    {
        RuntimeContractProfile expected = Profile();
        RuntimeContractProfile? actual = RuntimeProfileEvidence.Read(Manifest(Json(expected)));
        Assert.Equal(expected, actual);
        Assert.True(RuntimeProfileEvidence.HasObservedCpuStartup(actual));
        Assert.True(RuntimeProfileEvidence.HasObservedCpuResetPreconditions(actual));
    }

    [Fact]
    public void OldOpaqueProfileDoesNotInventStartupEvidence()
    {
        DiscoveryManifest legacy = Manifest(Json(Profile())) with { ProfileJson = null };
        Assert.Null(RuntimeProfileEvidence.Read(legacy));
        Assert.False(RuntimeProfileEvidence.HasObservedCpuStartup(null));
    }

    [Fact]
    public void CompletionWithoutInputsOrConflictingCompletionCannotBecomeReady()
    {
        var missing = new RuntimeInitializationLedger();
        missing.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.Equal(RuntimeInitializationStatus.Conflicting, missing.Snapshot.Status);
        var ledger = new RuntimeInitializationLedger();
        ledger.Record(Profile().Effective);
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.True(RuntimeProfileEvidence.HasObservedCpuStartup(new(Profile().Effective, ledger.Snapshot)));
        ledger.Complete(new(RuntimeCpuMode.Other, 1));
        ledger.Complete(new(RuntimeCpuMode.Cpu, 1));
        Assert.False(RuntimeProfileEvidence.HasObservedCpuStartup(new(Profile().Effective, ledger.Snapshot)));
        Assert.Equal(RuntimeInitializationStatus.Conflicting, ledger.Snapshot.Status);
    }

    [Theory]
    [InlineData(Startup.Missing)]
    [InlineData(Startup.NoCompletion)]
    [InlineData(Startup.Conflicting)]
    [InlineData(Startup.OtherEngine)]
    [InlineData(Startup.InvalidDop)]
    [InlineData(Startup.InitialObserver)]
    [InlineData(Startup.EffectiveObserver)]
    [InlineData(Startup.MissingInputs)]
    public void IncompleteOrObservedStartupCannotSatisfyContract(Startup mutation)
    {
        RuntimeContractProfile profile = Profile();
        profile = mutation switch
        {
            Startup.Missing => profile with { Initialization = new(RuntimeInitializationStatus.Missing, null) },
            Startup.NoCompletion => profile with { Initialization = profile.Initialization with { Completion = null } },
            Startup.Conflicting => profile with { Initialization = profile.Initialization with { Status = RuntimeInitializationStatus.Conflicting } },
            Startup.OtherEngine => profile with { Initialization = profile.Initialization with { Completion = new(RuntimeCpuMode.Other, 1) } },
            Startup.InvalidDop => profile with { Initialization = profile.Initialization with { Completion = new(RuntimeCpuMode.Cpu, 0) } },
            Startup.InitialObserver => profile with { Initialization = profile.Initialization with { Inputs = new(1, new('b', 64), RuntimeObserverSignals.Present) } },
            Startup.EffectiveObserver => profile with { Effective = profile.Effective with { ObserverSignals = RuntimeObserverSignals.Present } },
            Startup.MissingInputs => profile with { Initialization = profile.Initialization with { Inputs = null } },
            _ => throw new ArgumentOutOfRangeException(nameof(mutation))
        };
        Assert.False(RuntimeProfileEvidence.HasObservedCpuStartup(profile));
        if (mutation == Startup.MissingInputs)
            Assert.Throws<EvidenceException>(() => RuntimeProfileEvidence.Read(Manifest(Json(profile))));
        else Assert.False(RuntimeProfileEvidence.HasObservedCpuStartup(RuntimeProfileEvidence.Read(Manifest(Json(profile)))));
    }

    [Theory]
    [InlineData(Malformed.Hash)]
    [InlineData(Malformed.Schema)]
    [InlineData(Malformed.Status)]
    [InlineData(Malformed.NumericEnum)]
    [InlineData(Malformed.Duplicate)]
    [InlineData(Malformed.UnknownMember)]
    [InlineData(Malformed.ResetMode)]
    [InlineData(Malformed.ResetLogging)]
    [InlineData(Malformed.ResetWithoutStartup)]
    [InlineData(Malformed.GpuStartup)]
    [InlineData(Malformed.NumericGpuStartup)]
    [InlineData(Malformed.GpuDiagnostics)]
    [InlineData(Malformed.NumericGpuDiagnostics)]
    public void ProfileClaimsMustBeWellFormedAndMatchTheExecutedHash(Malformed mutation)
    {
        string original = Json(Profile());
        var document = JsonNode.Parse(original)?.AsObject() ?? throw new InvalidOperationException("Missing test profile.");
        JsonObject contract = document["RuntimeContracts"]?.AsObject() ?? throw new InvalidOperationException("Missing contract.");
        JsonObject effective = contract["Effective"]?.AsObject() ?? throw new InvalidOperationException("Missing settings.");
        JsonObject initialization = contract["Initialization"]?.AsObject() ?? throw new InvalidOperationException("Missing startup.");
        switch (mutation)
        {
            case Malformed.Schema: effective["Schema"] = 9; break;
            case Malformed.Status: initialization["Status"] = "ApprovedByCaller"; break;
            case Malformed.NumericEnum: initialization["Status"] = 1; break;
            case Malformed.GpuStartup: effective["GpuStartup"] = "TrustedGpu"; break;
            case Malformed.NumericGpuStartup: effective["GpuStartup"] = 2; break;
            case Malformed.GpuDiagnostics: effective["GpuDiagnostics"] = "ApprovedDump"; break;
            case Malformed.NumericGpuDiagnostics: effective["GpuDiagnostics"] = 1; break;
            case Malformed.UnknownMember: contract["TrustMe"] = true; break;
            case Malformed.ResetMode:
                initialization["ResetInput"] = new JsonObject { ["Mode"] = "TrustedGpu", ["Logging"] = "Suppressed" }; break;
            case Malformed.ResetLogging:
                initialization["ResetInput"] = new JsonObject { ["Mode"] = "PlainCpu", ["Logging"] = 1 }; break;
            case Malformed.ResetWithoutStartup:
                initialization["Status"] = "Missing"; initialization["Inputs"] = null; initialization["Completion"] = null; break;
        }
        string json = mutation == Malformed.Duplicate ? original.Replace("\"RuntimeContracts\":", "\"RuntimeContracts\":null,\"RuntimeContracts\":", StringComparison.Ordinal) : document.ToJsonString();
        DiscoveryManifest manifest = Manifest(json);
        if (mutation == Malformed.Hash) manifest = manifest with { ProfileJson = json + " " };
        Assert.Throws<EvidenceException>(() => RuntimeProfileEvidence.Read(manifest));
    }
}
