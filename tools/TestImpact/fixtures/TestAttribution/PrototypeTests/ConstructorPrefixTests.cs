using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ConstructorPrefixTests
{
    private static string Key() => Guid.NewGuid().ToString("N");
    private static Backend Build() => new(0, 1, Key());

    [Fact]
    public void ActualCallOperandsSupplyThePrefixInputs()
    {
        using var fixture = new Fixture();
        var caller = fixture.Assembly.MainModule.Types.Single(type => type.FullName == typeof(ConstructorPrefixTests).FullName)
            .Methods.Single(method => method.Name == nameof(Build));
        int index = caller.Body.Instructions.Select((instruction, index) => (instruction, index)).Single(item => item.instruction.OpCode.Code == Code.Newobj).index;
        var result = ConstructorCallReader.Read(caller, index);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        if (!supported) { Assert.Null(result); return; }
        Assert.NotNull(result);
        Assert.Equal(ConstructorPrefixContract.ReceiverFieldsAndBaseConstructor, result.Prefix.Contract);
        Assert.Equal(LockedInitializationContract.ConstantInsertIfAbsent, result.LockedTail.Contract);
    }

    public enum CallMutation { InvalidRank, UnknownRank, WrongKeyFormat, ExtraKeyEffect, NotConstruction, BranchIntoArguments, WrongParameterType }
    [Theory]
    [InlineData(CallMutation.InvalidRank)]
    [InlineData(CallMutation.UnknownRank)]
    [InlineData(CallMutation.WrongKeyFormat)]
    [InlineData(CallMutation.ExtraKeyEffect)]
    [InlineData(CallMutation.NotConstruction)]
    [InlineData(CallMutation.BranchIntoArguments)]
    [InlineData(CallMutation.WrongParameterType)]
    public void CallsiteFactsCannotBeAssumedFromMethodNames(CallMutation mutation)
    {
        using var fixture = new Fixture();
        var type = fixture.Assembly.MainModule.Types.Single(type => type.FullName == typeof(ConstructorPrefixTests).FullName);
        var caller = type.Methods.Single(method => method.Name == nameof(Build));
        var key = type.Methods.Single(method => method.Name == nameof(Key));
        var il = caller.Body.Instructions;
        var construction = il.Single(instruction => instruction.OpCode.Code == Code.Newobj);
        switch (mutation)
        {
            case CallMutation.InvalidRank: il[0].OpCode = OpCodes.Ldc_I4_1; break;
            case CallMutation.UnknownRank: il[0].OpCode = OpCodes.Ldarg_0; break;
            case CallMutation.WrongKeyFormat: key.Body.Instructions.Single(instruction => instruction.OpCode.Code == Code.Ldstr).Operand = "X"; break;
            case CallMutation.ExtraKeyEffect: key.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case CallMutation.NotConstruction: construction.OpCode = OpCodes.Call; break;
            case CallMutation.BranchIntoArguments: il.Insert(0, Instruction.Create(OpCodes.Br, il[1])); break;
            case CallMutation.WrongParameterType: ((MethodReference)construction.Operand).Parameters[0].ParameterType = fixture.Assembly.MainModule.TypeSystem.Object; break;
        }
        Assert.Null(ConstructorCallReader.Read(caller, il.IndexOf(construction)));
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 2)]
    [InlineData(0, int.MaxValue)]
    public void KnownNormalArgumentsLeaveOnlyTheBaseConstructorRequirement(int rank, int worldSize)
    {
        using var fixture = new Fixture();
        var result = ConstructorPrefixReader.Read(fixture.Method, rank, worldSize, ConstructorKeyFact.NonWhitespaceString);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? ConstructorPrefixContract.ReceiverFieldsAndBaseConstructor : ConstructorPrefixContract.Unresolved, result.Contract);
        if (!supported) return;
        Assert.Equal("System.Void System.Object::.ctor()", result.BaseConstructor);
        Assert.Equal(3, result.WrittenFields.Length);
    }

    [Theory]
    [InlineData(-1, 1, true)]
    [InlineData(1, 1, true)]
    [InlineData(0, 0, true)]
    [InlineData(0, -1, true)]
    [InlineData(int.MaxValue, int.MaxValue, true)]
    [InlineData(0, 1, false)]
    public void UnknownOrInvalidArgumentsCannotSkipGuardEffects(int rank, int worldSize, bool knownKey)
    {
        using var fixture = new Fixture();
        Assert.Equal(ConstructorPrefixContract.Unresolved, ConstructorPrefixReader.Read(fixture.Method, rank, worldSize,
            knownKey ? ConstructorKeyFact.NonWhitespaceString : ConstructorKeyFact.Unknown).Contract);
    }

    public enum Mutation { ExtraCall, MissingBase, BackwardJump, EscapingReceiver, MissingKeyAssignment, GuardReturnsDifferentType }
    [Theory]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.MissingBase)]
    [InlineData(Mutation.BackwardJump)]
    [InlineData(Mutation.EscapingReceiver)]
    [InlineData(Mutation.MissingKeyAssignment)]
    [InlineData(Mutation.GuardReturnsDifferentType)]
    public void PrefixEffectsAreNotHiddenByTheRecognizedLockTail(Mutation mutation)
    {
        using var fixture = new Fixture();
        var il = fixture.Method.Body.Instructions;
        switch (mutation)
        {
            case Mutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Call, fixture.Assembly.MainModule.ImportReference(typeof(Environment).GetMethod(nameof(Environment.GetEnvironmentVariable), [typeof(string)]) ?? throw new InvalidOperationException()))); break;
            case Mutation.MissingBase: il[1].OpCode = OpCodes.Pop; il[1].Operand = null; break;
            case Mutation.BackwardJump: il.Insert(2, Instruction.Create(OpCodes.Br, il[0])); break;
            case Mutation.EscapingReceiver: il.Insert(2, Instruction.Create(OpCodes.Ret)); break;
            case Mutation.MissingKeyAssignment:
                var keyWrite = il.Single(instruction => instruction.OpCode.Code == Code.Stfld && instruction.Operand is FieldReference field && field.Name == "key");
                keyWrite.OpCode = OpCodes.Pop; keyWrite.Operand = null; break;
            case Mutation.GuardReturnsDifferentType:
                var call = il.Select(instruction => instruction.Operand).OfType<MethodReference>().Single(method => method.Name == "IsNullOrWhiteSpace");
                call.ReturnType = fixture.Assembly.MainModule.TypeSystem.Object; break;
        }
        Assert.Equal(ConstructorPrefixContract.Unresolved, ConstructorPrefixReader.Read(fixture.Method, 0, 1, ConstructorKeyFact.NonWhitespaceString).Contract);
    }

    private sealed class Backend
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<string, int> Map = new();
        private readonly int rank;
        private readonly int worldSize;
        private readonly string key;
        public Backend(int rank, int worldSize, string key)
        {
            if (rank < 0 || rank >= worldSize) throw new ArgumentException(nameof(rank));
            if (worldSize <= 0) throw new ArgumentException(nameof(worldSize));
            if (string.IsNullOrWhiteSpace(key)) throw new ArgumentException(nameof(key));
            this.rank = rank; this.worldSize = worldSize; this.key = key;
            lock (Gate) { if (!Map.ContainsKey(this.key)) Map[this.key] = 0; }
        }
        public int Rank => rank;
        public int WorldSize => worldSize;
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal MethodDefinition Method { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(ConstructorPrefixTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(ConstructorPrefixTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            Method = Assembly.MainModule.Types.Single(type => type.FullName == typeof(ConstructorPrefixTests).FullName)
                .NestedTypes.Single(type => type.Name == nameof(Backend)).Methods.Single(method => method.IsConstructor && !method.IsStatic);
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
