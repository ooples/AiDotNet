using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class AsyncOwnerReaderTests
{
    public enum Mutation { ExtraCall, OtherTask, OtherBuilder, OtherState, NoMarker, Handler, SharedBuilder, FaultsOtherBuilder, ForeignDispatch, UninitializedLocal }

    [Fact]
    public void CompilerKickoffReturnsTheSameTaskWhoseBuilderReceivesFailures()
    {
        using var fixture = new Fixture();
        Assert.Equal(AsyncOwnerBinding.ReturnsStateMachineTask, AsyncOwnerReader.Read(fixture.Entry, fixture.MoveNext));
    }

    [Theory]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.OtherTask)]
    [InlineData(Mutation.OtherBuilder)]
    [InlineData(Mutation.OtherState)]
    [InlineData(Mutation.NoMarker)]
    [InlineData(Mutation.Handler)]
    [InlineData(Mutation.SharedBuilder)]
    [InlineData(Mutation.FaultsOtherBuilder)]
    [InlineData(Mutation.ForeignDispatch)]
    [InlineData(Mutation.UninitializedLocal)]
    public void UnprovenTaskOwnershipIsRejected(Mutation mutation)
    {
        using var fixture = new Fixture();
        var entry = fixture.Entry;
        var moveNext = fixture.MoveNext;
        var body = entry.Body.Instructions;
        var builder = ((FieldReference)body[2].Operand).Resolve();
        var other = new FieldDefinition("otherBuilder", FieldAttributes.Public, builder.FieldType);
        switch (mutation)
        {
            case Mutation.ExtraCall: body.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.OtherTask:
                body[12].Operand = entry.Module.ImportReference(typeof(Task).GetProperty(nameof(Task.CompletedTask))?.GetMethod
                    ?? throw new InvalidOperationException("Missing Task.CompletedTask")); break;
            case Mutation.OtherBuilder: builder.DeclaringType.Fields.Add(other); body[11].Operand = other; break;
            case Mutation.OtherState:
                ((GenericInstanceMethod)body[9].Operand).GenericArguments[0] = entry.Module.TypeSystem.Int32; break;
            case Mutation.NoMarker: entry.CustomAttributes.Clear(); break;
            case Mutation.Handler: entry.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.SharedBuilder: builder.IsStatic = true; break;
            case Mutation.FaultsOtherBuilder:
                builder.DeclaringType.Fields.Add(other);
                int fault = moveNext.Body.Instructions.Select((instruction, index) => (instruction, index)).Single(site =>
                    site.instruction.Operand is MethodReference call && call.Name == "SetException").index;
                moveNext.Body.Instructions[fault - 2].Operand = other; break;
            case Mutation.ForeignDispatch: moveNext.IsFinal = false; break;
            case Mutation.UninitializedLocal: entry.Body.InitLocals = false; break;
        }
        Assert.Equal(AsyncOwnerBinding.Unresolved, AsyncOwnerReader.Read(entry, moveNext));
    }

    private static async Task EntryPoint() { await Task.Yield(); Assert.True(true, "owner binding"); }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Entry { get; }
        internal MethodDefinition MoveNext { get; }

        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(AsyncOwnerReaderTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            assembly = AssemblyDefinition.ReadAssembly(typeof(AsyncOwnerReaderTests).Assembly.Location,
                new ReaderParameters { AssemblyResolver = resolver });
            Entry = assembly.MainModule.Types.Single(type => type.FullName == typeof(AsyncOwnerReaderTests).FullName)
                .Methods.Single(method => method.Name == nameof(EntryPoint));
            var marker = Entry.CustomAttributes.Single(attribute => attribute.AttributeType.Name == "AsyncStateMachineAttribute");
            MoveNext = ((TypeReference)marker.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
        }

        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
