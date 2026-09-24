using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class BuildKeyOverrideTests
{
    [Fact]
    public void RecognizesCompleteLockedCopyAndInitializer()
    {
        using var fixture = new Fixture();
        Assert.NotNull(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    public static IEnumerable<object[]> Instructions() => Enumerable.Range(0, 29).Select(index => new object[] { index });

    [Theory]
    [MemberData(nameof(Instructions))]
    public void RejectsEveryMissingInstruction(int index)
    {
        using var fixture = new Fixture();
        fixture.Method.Body.Instructions[index].OpCode = OpCodes.Nop;
        Assert.Null(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    [Theory]
    [InlineData(8)] [InlineData(13)] [InlineData(15)] [InlineData(22)] [InlineData(24)]
    public void RejectsAlteredBranchTargets(int index)
    {
        using var fixture = new Fixture();
        fixture.Method.Body.Instructions[index].Operand = fixture.Method.Body.Instructions[0];
        Assert.Null(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    [Fact]
    public void RejectsHiddenInitializerCall()
    {
        using var fixture = new Fixture();
        var init = fixture.Method.DeclaringType.Methods.Single(method => method.IsConstructor);
        init.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Call, fixture.Method.Module.ImportReference(
            typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException())));
        Assert.Null(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    [Fact]
    public void RejectsReleasingDifferentGate()
    {
        using var fixture = new Fixture();
        fixture.Method.Body.Instructions[25].OpCode = OpCodes.Ldnull;
        Assert.Null(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    [Fact]
    public void RejectsChangedFinallyBoundary()
    {
        using var fixture = new Fixture();
        fixture.Method.Body.ExceptionHandlers[0].HandlerEnd = fixture.Method.Body.Instructions[27];
        Assert.Null(BuildKeyOverrideReader.ReadShape(fixture.Method));
    }

    private static class Provider
    {
        private static byte[]? cached;
        private static bool loaded;
        private static readonly object gate = new();
        internal static void Override(byte[]? key)
        {
            lock (gate)
            {
                cached = key is { Length: > 0 } ? (byte[])key.Clone() : null;
                loaded = true;
            }
        }
        internal static bool Loaded => loaded;
        internal static byte[]? Cached => cached;
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Method { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(BuildKeyOverrideTests).Assembly.Location));
            assembly = AssemblyDefinition.ReadAssembly(typeof(BuildKeyOverrideTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            Method = assembly.MainModule.GetType(typeof(Provider).FullName?.Replace('+', '/')).Methods.Single(method => method.Name == "Override");
        }
        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
