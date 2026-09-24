using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class NumericBaseConstructorTests
{
    public enum Mutation { MutableNumeric, PublicNumeric, StaticNumeric, StaticFlag, ForeignParameter, WrongFieldType,
        ExtraCall, Handler, NonzeroFlag, Synchronized, InstanceProvider, RootNotConstructor }

    [Fact]
    public void StructuralProviderCallDoesNotAuthorizeAnUnreviewedPackage()
    {
        using var fixture = new Fixture();
        MethodReference? call = NumericBaseConstructorReader.ProviderCall(fixture.Constructor);
        Assert.NotNull(call);
        Assert.Equal(NumericProviderContract.Unresolved, ReviewedNumericProvider.Read(call).Contract);
    }

    [Theory]
    [InlineData(Mutation.MutableNumeric)]
    [InlineData(Mutation.PublicNumeric)]
    [InlineData(Mutation.StaticNumeric)]
    [InlineData(Mutation.StaticFlag)]
    [InlineData(Mutation.ForeignParameter)]
    [InlineData(Mutation.WrongFieldType)]
    [InlineData(Mutation.ExtraCall)]
    [InlineData(Mutation.Handler)]
    [InlineData(Mutation.NonzeroFlag)]
    [InlineData(Mutation.Synchronized)]
    [InlineData(Mutation.InstanceProvider)]
    [InlineData(Mutation.RootNotConstructor)]
    public void ReceiverInitializationRejectsUnreviewedEffects(Mutation mutation)
    {
        using var fixture = new Fixture();
        var il = fixture.Constructor.Body.Instructions;
        switch (mutation)
        {
            case Mutation.MutableNumeric: fixture.Numeric.IsInitOnly = false; break;
            case Mutation.PublicNumeric: fixture.Numeric.IsPublic = true; break;
            case Mutation.StaticNumeric: fixture.Numeric.IsStatic = true; break;
            case Mutation.StaticFlag: fixture.Flag.IsStatic = true; break;
            case Mutation.ForeignParameter:
                ((GenericInstanceMethod)il[3].Operand).GenericArguments[0] = new GenericParameter("T", fixture.Constructor); break;
            case Mutation.WrongFieldType: fixture.Numeric.FieldType = fixture.Module.ImportReference(typeof(object)); break;
            case Mutation.ExtraCall: il.Insert(0, Instruction.Create(OpCodes.Nop)); break;
            case Mutation.Handler: fixture.Constructor.Body.ExceptionHandlers.Add(new(ExceptionHandlerType.Finally)); break;
            case Mutation.NonzeroFlag: il[6].OpCode = OpCodes.Ldc_I4_1; break;
            case Mutation.Synchronized: fixture.Constructor.ImplAttributes |= MethodImplAttributes.Synchronized; break;
            case Mutation.InstanceProvider: ((GenericInstanceMethod)il[3].Operand).ElementMethod.HasThis = true; break;
            case Mutation.RootNotConstructor: ((MethodReference)il[1].Operand).Name = "Other"; break;
        }
        Assert.Null(NumericBaseConstructorReader.ProviderCall(fixture.Constructor));
    }

    private sealed class Fixture : IDisposable
    {
        private readonly DefaultAssemblyResolver resolver = new();
        internal ModuleDefinition Module { get; }
        internal MethodDefinition Constructor { get; }
        internal FieldDefinition Numeric { get; }
        internal FieldDefinition Flag { get; }
        internal Fixture()
        {
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            Module = ModuleDefinition.CreateModule("numeric-base-shape", new ModuleParameters { Kind = ModuleKind.Dll, AssemblyResolver = resolver });
            var owner = new TypeDefinition("Fixtures", "Base`1", TypeAttributes.Public, Module.ImportReference(typeof(object)));
            owner.GenericParameters.Add(new("T", owner)); Module.Types.Add(owner);
            var operations = new TypeDefinition("AiDotNet.Tensors.Interfaces", "INumericOperations`1", TypeAttributes.Public | TypeAttributes.Interface | TypeAttributes.Abstract);
            operations.GenericParameters.Add(new("T", operations)); Module.Types.Add(operations);
            var helper = new TypeDefinition("AiDotNet.Tensors.Helpers", "MathHelper", TypeAttributes.Public, Module.ImportReference(typeof(object)));
            Module.Types.Add(helper);
            var provider = new MethodDefinition("GetNumericOperations", MethodAttributes.Public | MethodAttributes.Static, Module.TypeSystem.Void);
            provider.GenericParameters.Add(new("T", provider)); helper.Methods.Add(provider);
            var returned = new GenericInstanceType(operations); returned.GenericArguments.Add(provider.GenericParameters[0]); provider.ReturnType = returned;
            var fieldType = new GenericInstanceType(operations); fieldType.GenericArguments.Add(owner.GenericParameters[0]);
            Numeric = new("NumOps", FieldAttributes.Family | FieldAttributes.InitOnly, fieldType); owner.Fields.Add(Numeric);
            Flag = new("initialized", FieldAttributes.Private, Module.ImportReference(typeof(bool))); owner.Fields.Add(Flag);
            Constructor = new(".ctor", MethodAttributes.Public | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName, Module.ImportReference(typeof(void)));
            owner.Methods.Add(Constructor);
            var call = new GenericInstanceMethod(provider); call.GenericArguments.Add(owner.GenericParameters[0]);
            Instruction[] instructions = [Instruction.Create(OpCodes.Ldarg_0),
                Instruction.Create(OpCodes.Call, Module.ImportReference(typeof(object).GetConstructor(Type.EmptyTypes) ?? throw new InvalidOperationException())),
                Instruction.Create(OpCodes.Ldarg_0), Instruction.Create(OpCodes.Call, call), Instruction.Create(OpCodes.Stfld, Numeric),
                Instruction.Create(OpCodes.Ldarg_0), Instruction.Create(OpCodes.Ldc_I4_0), Instruction.Create(OpCodes.Stfld, Flag), Instruction.Create(OpCodes.Ret)];
            foreach (var instruction in instructions) Constructor.Body.Instructions.Add(instruction);
        }
        public void Dispose() { Module.Dispose(); resolver.Dispose(); }
    }
}
