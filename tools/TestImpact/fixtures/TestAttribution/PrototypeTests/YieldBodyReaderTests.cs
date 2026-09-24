using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class YieldBodyReaderTests
{
    public enum Mutation { ResumeBranch, ReadyBranch, ScheduledExit, ForeignAwaiter, ForeignState, SharedAwaiter,
        AwaiterLocal, ResetLocal, PrefixCall, GetResultCall, ExtraHandler, TryStart, SwallowedFailure,
        WrongCompletion, BodyReentry, EarlyReturn, FrameEscape, FrameWrite, FrameLocal, ExtraField, SynchronizedEntry,
        ForeignAwaitParameterOwner, TypeAwaitParameter }

    [Fact]
    public void ImportedGenericParameterNamesAreNotTheirIdentity()
    {
        using var fixture = new Fixture(nameof(Entry));
        var call = (GenericInstanceMethod)fixture.Body.Body.Instructions[25].Operand;
        foreach (var parameter in call.ElementMethod.GenericParameters) parameter.Name = "renamed" + parameter.Position;
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? YieldBodyContract.SingleYieldTaskBody : YieldBodyContract.Unresolved,
            YieldBodyReader.Read(fixture.Entry, fixture.Body).Contract);
    }

    [Fact]
    public void CanonicalYieldExposesTheEntireUnreviewedUserBody()
    {
        using var fixture = new Fixture(nameof(Entry));
        YieldBodyWindow result = YieldBodyReader.Read(fixture.Entry, fixture.Body);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? YieldBodyContract.SingleYieldTaskBody : YieldBodyContract.Unresolved, result.Contract);
        if (supported)
        {
            Assert.Equal(40, result.Start);
            Assert.True(result.Length > 0);
            Assert.Contains(fixture.Body.Body.Instructions.Skip(result.Start).Take(result.Length),
                instruction => instruction.Operand is MethodReference call && call.Name == nameof(UnreviewedBody));
        }
    }

    [Theory]
    [InlineData(nameof(TwoYields))]
    [InlineData(nameof(OtherAwait))]
    [InlineData(nameof(CapturedInstance))]
    public void UnsupportedAsyncShapesCannotBorrowTheSingleYieldFrame(string name)
    {
        using var fixture = new Fixture(name);
        Assert.Equal(YieldBodyContract.Unresolved, YieldBodyReader.Read(fixture.Entry, fixture.Body).Contract);
    }

    [Theory]
    [InlineData(Mutation.ResumeBranch)]
    [InlineData(Mutation.ReadyBranch)]
    [InlineData(Mutation.ScheduledExit)]
    [InlineData(Mutation.ForeignAwaiter)]
    [InlineData(Mutation.ForeignState)]
    [InlineData(Mutation.SharedAwaiter)]
    [InlineData(Mutation.AwaiterLocal)]
    [InlineData(Mutation.ResetLocal)]
    [InlineData(Mutation.PrefixCall)]
    [InlineData(Mutation.GetResultCall)]
    [InlineData(Mutation.ExtraHandler)]
    [InlineData(Mutation.TryStart)]
    [InlineData(Mutation.SwallowedFailure)]
    [InlineData(Mutation.WrongCompletion)]
    [InlineData(Mutation.BodyReentry)]
    [InlineData(Mutation.EarlyReturn)]
    [InlineData(Mutation.FrameEscape)]
    [InlineData(Mutation.FrameWrite)]
    [InlineData(Mutation.FrameLocal)]
    [InlineData(Mutation.ExtraField)]
    [InlineData(Mutation.SynchronizedEntry)]
    [InlineData(Mutation.ForeignAwaitParameterOwner)]
    [InlineData(Mutation.TypeAwaitParameter)]
    public void ChangedFrameCannotHideBehaviorOutsideTheBody(Mutation mutation)
    {
        using var fixture = new Fixture(nameof(Entry));
        var body = fixture.Body;
        var il = body.Body.Instructions;
        var handler = Assert.Single(body.Body.ExceptionHandlers);
        int failure = il.IndexOf(handler.HandlerStart);
        int completion = il.IndexOf(handler.HandlerEnd);
        switch (mutation)
        {
            case Mutation.ForeignAwaitParameterOwner:
            case Mutation.TypeAwaitParameter:
                var foreignType = new TypeDefinition("Synthetic", "Foreign", TypeAttributes.Class, body.Module.TypeSystem.Object);
                var foreignMethod = new MethodDefinition("Foreign", MethodAttributes.Public, body.Module.TypeSystem.Void);
                foreignType.Methods.Add(foreignMethod);
                var foreign = mutation == Mutation.TypeAwaitParameter ? new GenericParameter("T", foreignType) : new GenericParameter("T", foreignMethod);
                if (mutation == Mutation.TypeAwaitParameter) foreignType.GenericParameters.Add(foreign); else foreignMethod.GenericParameters.Add(foreign);
                ((GenericInstanceMethod)il[25].Operand).ElementMethod.Parameters[0].ParameterType = new ByReferenceType(foreign);
                break;
            case Mutation.ResumeBranch: il[4].Operand = il[40]; break;
            case Mutation.ReadyBranch: il[12].Operand = il[40]; break;
            case Mutation.ScheduledExit: il[26].Operand = il[40]; break;
            case Mutation.ForeignAwaiter: ((GenericInstanceMethod)il[25].Operand).GenericArguments[0] = body.Module.TypeSystem.Int32; break;
            case Mutation.ForeignState: ((GenericInstanceMethod)il[25].Operand).GenericArguments[1] = body.Module.TypeSystem.Int32; break;
            case Mutation.SharedAwaiter: ((FieldReference)il[20].Operand).Resolve().IsStatic = true; break;
            case Mutation.AwaiterLocal: il[23].Operand = body.Body.Variables[0]; break;
            case Mutation.ResetLocal: il[36].OpCode = OpCodes.Stloc_1; break;
            case Mutation.PrefixCall: il[5].Operand = body.Module.ImportReference(typeof(Task).GetProperty(nameof(Task.CompletedTask))?.GetMethod
                ?? throw new InvalidOperationException("Missing task property.")); break;
            case Mutation.GetResultCall: il[39].Operand = il[11].Operand; break;
            case Mutation.ExtraHandler: body.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.TryStart: handler.TryStart = il[40]; break;
            case Mutation.SwallowedFailure: il[failure + 7].Operand = il[completion + 5].Operand; break;
            case Mutation.WrongCompletion: il[completion + 1].Operand = (sbyte)0; break;
            case Mutation.BodyReentry: il[40].OpCode = OpCodes.Br; il[40].Operand = il[5]; break;
            case Mutation.EarlyReturn: il[40].OpCode = OpCodes.Ret; il[40].Operand = null; break;
            case Mutation.FrameEscape: il[40].OpCode = OpCodes.Ldarg_0; il[40].Operand = null; break;
            case Mutation.FrameWrite: il[40].OpCode = OpCodes.Stfld; il[40].Operand = il[1].Operand; break;
            case Mutation.FrameLocal: il[40].OpCode = OpCodes.Stloc_0; il[40].Operand = null; break;
            case Mutation.ExtraField: body.DeclaringType.Fields.Add(new("other", FieldAttributes.Public, body.Module.TypeSystem.Int32)); break;
            case Mutation.SynchronizedEntry: fixture.Entry.ImplAttributes |= MethodImplAttributes.Synchronized; break;
        }
        YieldBodyWindow result = YieldBodyReader.Read(fixture.Entry, body);
        Assert.Equal(YieldBodyContract.Unresolved, result.Contract);
        Assert.Equal(0, result.Length);
    }

    private static async Task Entry() { await Task.Yield(); UnreviewedBody(); }
    private static async Task TwoYields() { await Task.Yield(); await Task.Yield(); UnreviewedBody(); }
    private static async Task OtherAwait() { await Task.Delay(1); UnreviewedBody(); }
    private async Task CapturedInstance() { await Task.Yield(); GC.KeepAlive(this); }
    private static void UnreviewedBody() { }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Entry { get; }
        internal MethodDefinition Body { get; }
        internal Fixture(string name)
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(YieldBodyReaderTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            assembly = AssemblyDefinition.ReadAssembly(typeof(YieldBodyReaderTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            Entry = assembly.MainModule.Types.Single(type => type.FullName == typeof(YieldBodyReaderTests).FullName).Methods.Single(method => method.Name == name);
            var marker = Entry.CustomAttributes.Single(attribute => attribute.AttributeType.Name == "AsyncStateMachineAttribute");
            Body = ((TypeReference)marker.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
        }
        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
