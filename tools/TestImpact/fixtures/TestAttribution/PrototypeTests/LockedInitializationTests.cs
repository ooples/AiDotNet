using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class LockedInitializationTests
{
    public enum Mutation { PublicMap, MutableMap, InstanceMap, ForeignKey, MutableKey, PublicLock, WrongMapType,
        WrongLockType, WrongContains, WrongSetter, WrongMonitor, WrongFinally, WrongLeave, WrongSkip,
        WrongTakenLocal, InitiallyTaken, EnterTailDirectly, ThreadStaticMap, ForeignGenericParameter, ForeignClosedMap }

    [Fact]
    public void ConstantInsertionRetainsEveryExternalRequirement()
    {
        using var fixture = new Fixture();
        var result = LockedInitializationReader.Read(fixture.Method);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? LockedInitializationContract.ConstantInsertIfAbsent : LockedInitializationContract.Unresolved, result.Contract);
        if (!supported) return;
        Assert.Equal(0, result.Value);
        Assert.Equal(Enum.GetValues<LockedInitializationRequirement>(), result.Requirements);
        Assert.True(result.Start > 0); // The prefix still requires its own proof.
    }

    [Theory]
    [InlineData(Mutation.PublicMap)]
    [InlineData(Mutation.MutableMap)]
    [InlineData(Mutation.InstanceMap)]
    [InlineData(Mutation.ForeignKey)]
    [InlineData(Mutation.MutableKey)]
    [InlineData(Mutation.PublicLock)]
    [InlineData(Mutation.WrongMapType)]
    [InlineData(Mutation.WrongLockType)]
    [InlineData(Mutation.WrongContains)]
    [InlineData(Mutation.WrongSetter)]
    [InlineData(Mutation.WrongMonitor)]
    [InlineData(Mutation.WrongFinally)]
    [InlineData(Mutation.WrongLeave)]
    [InlineData(Mutation.WrongSkip)]
    [InlineData(Mutation.WrongTakenLocal)]
    [InlineData(Mutation.InitiallyTaken)]
    [InlineData(Mutation.EnterTailDirectly)]
    [InlineData(Mutation.ThreadStaticMap)]
    [InlineData(Mutation.ForeignGenericParameter)]
    [InlineData(Mutation.ForeignClosedMap)]
    public void SimilarLookingTailsDoNotInheritTheContract(Mutation mutation)
    {
        using var fixture = new Fixture();
        var il = fixture.Method.Body.Instructions;
        var tail = il.Skip(il.Count - 24).ToArray();
        var map = ((FieldReference)tail[7].Operand).Resolve();
        var gate = ((FieldReference)tail[0].Operand).Resolve();
        var key = ((FieldReference)tail[9].Operand).Resolve();
        switch (mutation)
        {
            case Mutation.PublicMap: map.IsPublic = true; break;
            case Mutation.MutableMap: map.IsInitOnly = false; break;
            case Mutation.InstanceMap: map.IsStatic = false; break;
            case Mutation.ForeignKey: tail[14].Operand = gate; break;
            case Mutation.MutableKey: key.IsInitOnly = false; break;
            case Mutation.PublicLock: gate.IsPublic = true; break;
            case Mutation.WrongMapType: map.FieldType = fixture.Assembly.MainModule.TypeSystem.Object; break;
            case Mutation.WrongLockType: gate.FieldType = fixture.Assembly.MainModule.TypeSystem.String; break;
            case Mutation.WrongContains: ((MethodReference)tail[10].Operand).Name = "Remove"; break;
            case Mutation.WrongSetter: ((MethodReference)tail[16].Operand).Name = "Add"; break;
            case Mutation.WrongMonitor: ((MethodReference)tail[6].Operand).Name = "TryEnter"; break;
            case Mutation.WrongFinally: fixture.Method.Body.ExceptionHandlers[0].TryStart = tail[7]; break;
            case Mutation.WrongLeave: tail[17].Operand = tail[22]; break;
            case Mutation.WrongSkip: tail[11].Operand = tail[23]; break;
            case Mutation.WrongTakenLocal: tail[18].OpCode = tail[20].OpCode; tail[18].Operand = tail[20].Operand; break;
            case Mutation.InitiallyTaken: tail[2].OpCode = OpCodes.Ldc_I4_1; break;
            case Mutation.EnterTailDirectly: il.Insert(0, Instruction.Create(OpCodes.Br, tail[7])); break;
            case Mutation.ThreadStaticMap:
                map.CustomAttributes.Add(new(fixture.Assembly.MainModule.ImportReference(typeof(ThreadStaticAttribute).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.ForeignGenericParameter:
                ((MethodReference)tail[10].Operand).Parameters[0].ParameterType = new GenericParameter("T", fixture.Method); break;
            case Mutation.ForeignClosedMap:
                var foreign = new GenericInstanceType(fixture.Method.DeclaringType);
                foreign.GenericArguments.Add(fixture.Assembly.MainModule.TypeSystem.Single);
                tail[7].Operand = new FieldReference(map.Name, map.FieldType, foreign);
                tail[12].Operand = tail[7].Operand;
                break;
        }
        Assert.Equal(LockedInitializationContract.Unresolved, LockedInitializationReader.Read(fixture.Method).Contract);
    }

    private sealed class Backend<T>
    {
        private static readonly Dictionary<string, int> Map = new();
        private static readonly object Gate = new();
        private readonly string key;
        public Backend(string key)
        {
            this.key = key;
            lock (Gate) { if (!Map.ContainsKey(this.key)) Map[this.key] = 0; }
        }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal MethodDefinition Method { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(LockedInitializationTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(LockedInitializationTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            Method = Assembly.MainModule.Types.Single(type => type.FullName == typeof(LockedInitializationTests).FullName)
                .NestedTypes.Single(type => type.Name == "Backend`1").Methods.Single(method => method.IsConstructor && !method.IsStatic);
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
