using System.Security.Cryptography;
using System.Text;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class SignedLicenseTests
{
    public enum Mutation { Branch, Leave, DisposeBranch, HandlerStart, HandlerEnd, TryStart, TryEnd, Filter, Catch,
        WrongLocal, WrongArgument, MutableKey, ThreadKey, Character, Prefix, Separator, StoreLocal, LoadLocal, ExtraEffect, Generic }

    [Fact]
    public void CompleteSynchronousSignerHasExactRuntimeBindings()
    {
        using var fixture = new Fixture();
        Assert.NotNull(SignedLicenseReader.ReadShape(fixture.Method));
    }

    public static IEnumerable<object[]> Instructions() => Enumerable.Range(0, 41).Select(index => new object[] { index });

    [Theory]
    [MemberData(nameof(Instructions))]
    public void NoInstructionCanBeSkipped(int index)
    {
        using var fixture = new Fixture();
        fixture.Method.Body.Instructions[index].OpCode = OpCodes.Nop;
        Assert.Null(SignedLicenseReader.ReadShape(fixture.Method));
    }

    [Theory]
    [InlineData(Mutation.Branch)] [InlineData(Mutation.Leave)] [InlineData(Mutation.DisposeBranch)]
    [InlineData(Mutation.HandlerStart)] [InlineData(Mutation.HandlerEnd)] [InlineData(Mutation.TryStart)] [InlineData(Mutation.TryEnd)]
    [InlineData(Mutation.Filter)] [InlineData(Mutation.Catch)] [InlineData(Mutation.WrongLocal)] [InlineData(Mutation.WrongArgument)]
    [InlineData(Mutation.MutableKey)] [InlineData(Mutation.ThreadKey)] [InlineData(Mutation.Character)] [InlineData(Mutation.Prefix)]
    [InlineData(Mutation.Separator)] [InlineData(Mutation.StoreLocal)] [InlineData(Mutation.LoadLocal)] [InlineData(Mutation.ExtraEffect)]
    [InlineData(Mutation.Generic)]
    public void MatchingOpcodesDoNotHideDifferentEffects(Mutation mutation)
    {
        using var fixture = new Fixture();
        var method = fixture.Method;
        var il = method.Body.Instructions;
        var handler = method.Body.ExceptionHandlers.Single();
        switch (mutation)
        {
            case Mutation.Branch: il[6].Operand = il[10]; break;
            case Mutation.Leave: il[33].Operand = il[40]; break;
            case Mutation.DisposeBranch: il[35].Operand = il[39]; break;
            case Mutation.HandlerStart: handler.HandlerStart = il[35]; break;
            case Mutation.HandlerEnd: handler.HandlerEnd = il[40]; break;
            case Mutation.TryStart: handler.TryStart = il[12]; break;
            case Mutation.TryEnd: handler.TryEnd = il[33]; break;
            case Mutation.Filter: handler.FilterStart = il[0]; break;
            case Mutation.Catch: handler.HandlerType = ExceptionHandlerType.Catch; break;
            case Mutation.WrongLocal: method.Body.Variables[1].VariableType = method.Module.ImportReference(typeof(HashAlgorithm)); break;
            case Mutation.WrongArgument: method.Parameters[1].ParameterType = method.Module.TypeSystem.Object; break;
            case Mutation.MutableKey: ((FieldReference)il[8].Operand).Resolve().IsInitOnly = false; break;
            case Mutation.ThreadKey: ((FieldReference)il[8].Operand).Resolve().CustomAttributes.Add(new(method.Module.ImportReference(typeof(ThreadStaticAttribute).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.Character: il[19].Operand = (sbyte)42; break;
            case Mutation.Prefix: il[0].Operand = "different"; break;
            case Mutation.Separator: il[29].Operand = "different"; break;
            case Mutation.StoreLocal: il[32].Operand = method.Body.Variables[3]; break;
            case Mutation.LoadLocal: il[39].Operand = method.Body.Variables[3]; break;
            case Mutation.ExtraEffect: il.Insert(0, Instruction.Create(OpCodes.Call, method.Module.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.Generic: method.GenericParameters.Add(new("T", method)); break;
        }
        Assert.Null(SignedLicenseReader.ReadShape(method));
    }

    private static readonly byte[] Key = Encoding.UTF8.GetBytes("public-fixture-key");
    private static string Sign(string id, byte[]? supplied = null)
    {
        string payload = "aidn." + id;
        using var hmac = new HMACSHA256(supplied ?? Key);
        byte[] sig = hmac.ComputeHash(Encoding.UTF8.GetBytes(payload));
        string encoded = Convert.ToBase64String(sig).Replace('+', '-').Replace('/', '_').TrimEnd('=');
        return payload + "." + encoded;
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        private readonly AssemblyDefinition assembly;
        internal MethodDefinition Method { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(SignedLicenseTests).Assembly.Location));
            assembly = AssemblyDefinition.ReadAssembly(typeof(SignedLicenseTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            Method = assembly.MainModule.GetType(typeof(SignedLicenseTests).FullName).Methods.Single(method => method.Name == nameof(Sign));
        }
        public void Dispose() { assembly.Dispose(); resolver.Dispose(); }
    }
}
