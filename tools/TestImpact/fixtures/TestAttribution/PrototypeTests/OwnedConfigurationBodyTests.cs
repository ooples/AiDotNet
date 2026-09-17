using System.Security.Cryptography;
using AiDotNet.DistributedTraining;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class OwnedConfigurationBodyTests
{
    public enum BodyKind { Direct, Factory, Setters }
    public enum Mutation { Discard, UnknownCall, StaticWrite, ForeignBackend, InvalidRate, ChangedGetter, WrongAssertion, Branch, LocalAddress, UninitializedLocal, WrongLocalType }

    [Theory]
    [InlineData(BodyKind.Direct)]
    [InlineData(BodyKind.Factory)]
    [InlineData(BodyKind.Setters)]
    public void EveryBodyOperationMustBelongToTheOwnedProtocol(BodyKind kind)
    {
        using var fixture = new Fixture(kind);
        var shape = OwnedConfigurationBodyReader.ReadShape(fixture.Entry, fixture.Body);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        if (!supported) { Assert.Null(shape); return; }
        Assert.NotNull(shape);
        Assert.Single(shape.Configurations);
        Assert.NotEmpty(shape.Assertions);
        Assert.Equal(NumericBaseConstructorContract.Unresolved, shape.Backend.NumericBase.Contract);
        Assert.Equal(ConfigurationConstructorContract.Unresolved,
            ConfigurationConstructorReader.Read(shape.Configurations[0].Constructor, true, shape.Configurations[0].LearningRate).Contract);
    }

    [Theory]
    [InlineData(Mutation.Discard)]
    [InlineData(Mutation.UnknownCall)]
    [InlineData(Mutation.StaticWrite)]
    [InlineData(Mutation.ForeignBackend)]
    [InlineData(Mutation.InvalidRate)]
    [InlineData(Mutation.ChangedGetter)]
    [InlineData(Mutation.WrongAssertion)]
    [InlineData(Mutation.Branch)]
    [InlineData(Mutation.LocalAddress)]
    [InlineData(Mutation.UninitializedLocal)]
    [InlineData(Mutation.WrongLocalType)]
    public void UnreviewedOperationsCannotHideBetweenRecognizedCalls(Mutation mutation)
    {
        using var fixture = new Fixture(BodyKind.Direct);
        var il = fixture.Body.Body.Instructions;
        switch (mutation)
        {
            case Mutation.Discard: il.Insert(44, Instruction.Create(OpCodes.Pop)); break;
            case Mutation.UnknownCall: il.Insert(44, Instruction.Create(OpCodes.Call, fixture.Assembly.MainModule.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.StaticWrite:
                var slot = new FieldDefinition("escaped", FieldAttributes.Static | FieldAttributes.Private, fixture.Assembly.MainModule.TypeSystem.Object);
                fixture.Entry.DeclaringType.Fields.Add(slot); il.Insert(44, Instruction.Create(OpCodes.Stsfld, slot)); break;
            case Mutation.ForeignBackend:
                ((GenericInstanceType)((MethodReference)il[43].Operand).DeclaringType).GenericArguments[0] = fixture.Assembly.MainModule.ImportReference(typeof(float)); break;
            case Mutation.InvalidRate: il.Single(instruction => instruction.OpCode.Code == Code.Ldc_R8).Operand = double.NaN; break;
            case Mutation.ChangedGetter:
                var getter = il.Select(instruction => instruction.Operand).OfType<MethodReference>().Single(call => call.Name == "get_CpuOffloadOptimizer").Resolve();
                getter.Body.Instructions[1].OpCode = OpCodes.Ldsfld; break;
            case Mutation.WrongAssertion:
                var assertion = il.Single(instruction => instruction.Operand is MethodReference call && call.DeclaringType.FullName == "Xunit.Assert");
                assertion.Operand = fixture.Assembly.MainModule.ImportReference(typeof(GC).GetMethod(nameof(GC.KeepAlive)) ?? throw new InvalidOperationException()); break;
            case Mutation.Branch: il.Insert(44, Instruction.Create(OpCodes.Br, il[45])); break;
            case Mutation.LocalAddress:
                il[45].OpCode = OpCodes.Ldloca; il[45].Operand = fixture.Body.Body.Variables[1]; break;
            case Mutation.UninitializedLocal:
                il[45].OpCode = OpCodes.Ldloc; il[45].Operand = fixture.Body.Body.Variables[2]; break;
            case Mutation.WrongLocalType:
                fixture.Body.Body.Variables[1].VariableType = fixture.Assembly.MainModule.TypeSystem.Object; break;
        }
        Assert.Null(OwnedConfigurationBodyReader.ReadShape(fixture.Entry, fixture.Body));
    }

    private static string Key() => Guid.NewGuid().ToString("N");
    private static async Task Direct()
    {
        await Task.Yield();
        var backend = new Backend<double>(0, 1, Key());
        var config = new ShapeConfiguration<double>(backend, 0.01);
        Assert.False(config.CpuOffloadOptimizer);
    }
    private static async Task Factory()
    {
        await Task.Yield();
        var backend = new Backend<double>(0, 1, Key());
        var config = ShapeConfiguration<double>.CreateForZeROOffload(backend);
        Assert.True(config.CpuOffloadOptimizer);
    }
    private static async Task Setters()
    {
        await Task.Yield();
        var backend = new Backend<double>(0, 1, Key());
        var config = new ShapeConfiguration<double>(backend, 0.01) { CpuOffloadOptimizer = true, CpuOffloadParams = false };
        Assert.True(config.CpuOffloadOptimizer);
        Assert.False(config.CpuOffloadParams);
    }
    private abstract class Base<T> : ICommunicationBackend<T>
    {
        protected readonly AiDotNet.Tensors.Interfaces.INumericOperations<T> NumOps;
        private bool initialized;
        protected Base() { NumOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>(); initialized = false; }
        public bool Initialized => initialized;
    }
    private sealed class Backend<T> : Base<T>
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
        internal MethodDefinition Entry { get; }
        internal MethodDefinition Body { get; }
        internal Fixture(BodyKind kind)
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(OwnedConfigurationBodyTests).Assembly.Location));
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.ReadAssembly(typeof(OwnedConfigurationBodyTests).Assembly.Location, new ReaderParameters { AssemblyResolver = resolver });
            var type = Assembly.MainModule.GetType(typeof(OwnedConfigurationBodyTests).FullName);
            Entry = type.Methods.Single(method => method.Name == kind.ToString());
            var marker = Entry.CustomAttributes.Single(attribute => attribute.AttributeType.Name == "AsyncStateMachineAttribute");
            Body = ((TypeReference)marker.ConstructorArguments[0].Value).Resolve().Methods.Single(method => method.Name == "MoveNext");
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
