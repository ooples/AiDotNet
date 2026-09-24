using System.Security.Cryptography;
using AiDotNet.DistributedTraining;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class NullBackendBodyTests
{
    public enum Mutation { ExtraCall, EscapedException, WrongBranch, DifferentCallback, WrongException, ExtraClosureState }

    [Fact]
    public void ExpectedNullFailuresHaveBoundedCallbacksButStillNeedObserverIsolation()
    {
        using var fixture = new Fixture();
        var result = NullBackendBodyReader.ReadShape(fixture.Entry, fixture.Body);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        if (!supported) { Assert.Null(result); return; }
        Assert.NotNull(result);
        Assert.Equal(2, result.Factories.Length);
        Assert.Equal(2, result.DelegateCaches.Length);
    }

    [Theory]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.EscapedException)]
    [InlineData(Mutation.WrongBranch)]
    [InlineData(Mutation.DifferentCallback)]
    [InlineData(Mutation.WrongException)]
    [InlineData(Mutation.ExtraClosureState)]
    public void UnreviewedExceptionCallbacksCannotInheritTheShape(Mutation mutation)
    {
        using var fixture = new Fixture();
        var il = fixture.Body.Body.Instructions;
        var callback = ((MethodReference)il[45].Operand).Resolve();
        switch (mutation)
        {
            case Mutation.ExtraCall: il.Insert(40, Instruction.Create(OpCodes.Call, fixture.Assembly.MainModule.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.EscapedException: il[50].OpCode = OpCodes.Dup; break;
            case Mutation.WrongBranch: il[42].Operand = il[50]; break;
            case Mutation.DifferentCallback: callback.Body.Instructions[0].OpCode = OpCodes.Ldarg_0; break;
            case Mutation.WrongException: ((GenericInstanceMethod)il[49].Operand).GenericArguments[0] = fixture.Assembly.MainModule.ImportReference(typeof(InvalidOperationException)); break;
            case Mutation.ExtraClosureState: callback.DeclaringType.Fields.Add(new("state", FieldAttributes.Private, fixture.Assembly.MainModule.TypeSystem.Object)); break;
        }
        Assert.Null(NullBackendBodyReader.ReadShape(fixture.Entry, fixture.Body));
    }

    private sealed class BodyFixture
    {
        public static async Task Run()
        {
            await Task.Yield();
            Assert.Throws<ArgumentNullException>(() => ShapeConfiguration<double>.CreateForZeROOffload(null));
            Assert.Throws<ArgumentNullException>(() => ShapeConfiguration<double>.CreateForZeROOffload(null));
        }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal MethodDefinition Entry { get; }
        internal MethodDefinition Body { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(NullBackendBodyTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(NullBackendBodyTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            var type = Assembly.MainModule.GetType(typeof(BodyFixture).FullName?.Replace('+', '/'));
            Entry = type.Methods.Single(method => method.Name == nameof(BodyFixture.Run));
            var marker = Entry.CustomAttributes.Single(attribute => attribute.AttributeType.Name == "AsyncStateMachineAttribute");
            Body = ((TypeReference)marker.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
