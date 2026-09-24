using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class PrivateMapInitializerTests
{
    public enum Mutation { ExtraCall, PublishedField, MutableField, DuplicateStore, MissingStore, ForeignKey, CustomComparer, MissingInitializer, ExtraInitializer, WrongAllocatedType }

    [Fact]
    public void DefaultPrivateMapsAreRecognizedWithoutProvingLaterIsolation()
    {
        using var fixture = new Fixture();
        var result = PrivateMapInitializerReader.Read(fixture.Owner);
        bool supported = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(typeof(object).Assembly.Location))) == ReviewedOwnerCompletion.RuntimeHash;
        Assert.Equal(supported ? PrivateMapInitializerContract.FreshPrivateDefaultStringMaps : PrivateMapInitializerContract.Unresolved, result.Contract);
        Assert.Equal(supported ? 2 : 0, result.Fields.Length);
    }

    [Theory]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.PublishedField)]
    [InlineData(Mutation.MutableField)]
    [InlineData(Mutation.DuplicateStore)]
    [InlineData(Mutation.MissingStore)]
    [InlineData(Mutation.ForeignKey)]
    [InlineData(Mutation.CustomComparer)]
    [InlineData(Mutation.MissingInitializer)]
    [InlineData(Mutation.ExtraInitializer)]
    [InlineData(Mutation.WrongAllocatedType)]
    public void UnknownInitializerEffectsAreRejected(Mutation mutation)
    {
        using var fixture = new Fixture();
        var il = fixture.Initializer.Body.Instructions;
        switch (mutation)
        {
            case Mutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Call, fixture.Assembly.MainModule.ImportReference(typeof(GC).GetMethod(nameof(GC.Collect), Type.EmptyTypes) ?? throw new InvalidOperationException()))); break;
            case Mutation.PublishedField: fixture.Map.IsPublic = true; break;
            case Mutation.MutableField: fixture.Map.IsInitOnly = false; break;
            case Mutation.DuplicateStore: il[3].Operand = il[1].Operand; break;
            case Mutation.MissingStore: fixture.Owner.Fields.Add(new("extra", FieldAttributes.Private | FieldAttributes.Static, fixture.Assembly.MainModule.TypeSystem.Object)); break;
            case Mutation.ForeignKey: ((GenericInstanceType)((MethodReference)il[2].Operand).DeclaringType).GenericArguments[0] = fixture.Assembly.MainModule.TypeSystem.Object; break;
            case Mutation.CustomComparer: il[2].Operand = fixture.Assembly.MainModule.ImportReference(typeof(Dictionary<string, int>).GetConstructor([typeof(IEqualityComparer<string>)]) ?? throw new InvalidOperationException()); break;
            case Mutation.MissingInitializer: fixture.Owner.Methods.Remove(fixture.Initializer); break;
            case Mutation.ExtraInitializer: fixture.Owner.Methods.Add(new(".cctor", fixture.Initializer.Attributes, fixture.Assembly.MainModule.TypeSystem.Void)); break;
            case Mutation.WrongAllocatedType: il[2].Operand = il[0].Operand; break;
        }
        Assert.Equal(PrivateMapInitializerContract.Unresolved, PrivateMapInitializerReader.Read(fixture.Owner).Contract);
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal AssemblyDefinition Assembly { get; }
        internal TypeDefinition Owner { get; }
        internal MethodDefinition Initializer { get; }
        internal FieldDefinition Map { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Assembly = AssemblyDefinition.CreateAssembly(new("InitializerFixture", new Version(1, 0)), "InitializerFixture", new ModuleParameters { Kind = ModuleKind.Dll, AssemblyResolver = resolver });
            var module = Assembly.MainModule;
            Owner = new("Fixtures", "Owner", TypeAttributes.Class, module.ImportReference(typeof(object)));
            module.Types.Add(Owner);
            var gate = new FieldDefinition("gate", FieldAttributes.Static | FieldAttributes.Private | FieldAttributes.InitOnly, module.ImportReference(typeof(object)));
            Map = new("map", gate.Attributes, module.ImportReference(typeof(Dictionary<string, int>)));
            Owner.Fields.Add(gate); Owner.Fields.Add(Map);
            Initializer = new(".cctor", MethodAttributes.Static | MethodAttributes.Private | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, module.ImportReference(typeof(void)));
            Owner.Methods.Add(Initializer);
            var il = Initializer.Body.Instructions;
            il.Add(Instruction.Create(OpCodes.Newobj, module.ImportReference(typeof(object).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException())));
            il.Add(Instruction.Create(OpCodes.Stsfld, gate));
            il.Add(Instruction.Create(OpCodes.Newobj, module.ImportReference(typeof(Dictionary<string, int>).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException())));
            il.Add(Instruction.Create(OpCodes.Stsfld, Map));
            il.Add(Instruction.Create(OpCodes.Ret));
        }
        public void Dispose() { Assembly.Dispose(); resolver.Dispose(); }
    }
}
