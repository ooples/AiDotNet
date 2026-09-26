using System.Security.Cryptography;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class WorkloadBodyTests
{
    private const string NullOwner = "PrototypeTests:PrototypeTests.NullBackendBodyTests+BodyFixture.Run";

    [Fact]
    public void BodyProofNeverDischargesStartupAndLifetimeRequirements()
    {
        using var fixture = new Fixture();
        var result = WorkloadBodyReader.Read(fixture.Assembly, [NullOwner]);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported, result.AllBodiesRecognized);
        Assert.Equal(Enum.GetValues<WorkloadBodyRequirement>(), result.RemainingRequirements);
        Assert.Equal(NullOwner, Assert.Single(result.Bodies).Owner);
    }

    [Theory]
    [InlineData("PrototypeTests:Missing.Type.Run")]
    [InlineData("OtherAssembly:PrototypeTests.NullBackendBodyTests+BodyFixture.Run")]
    [InlineData("PrototypeTests:PrototypeTests.OwnedConfigurationBodyTests.Direct")]
    public void OneUnknownOrUntrustedOwnerInvalidatesWholeInventory(string unknown)
    {
        using var fixture = new Fixture();
        var result = WorkloadBodyReader.Read(fixture.Assembly, [NullOwner, unknown]);
        Assert.False(result.AllBodiesRecognized);
        Assert.Equal(2, result.Bodies.Length);
        Assert.Equal(WorkloadBodyKind.Unresolved, result.Bodies[1].Kind);
    }

    [Fact]
    public void EmptyDuplicateOrBlankInventoriesCannotProveIsolation()
    {
        using var fixture = new Fixture();
        Assert.Throws<ArgumentException>(() => WorkloadBodyReader.Read(fixture.Assembly, []));
        Assert.Throws<ArgumentException>(() => WorkloadBodyReader.Read(fixture.Assembly, [NullOwner, NullOwner]));
        Assert.Throws<ArgumentException>(() => WorkloadBodyReader.Read(fixture.Assembly, [NullOwner, " "]));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(WorkloadBodyTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(WorkloadBodyTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
