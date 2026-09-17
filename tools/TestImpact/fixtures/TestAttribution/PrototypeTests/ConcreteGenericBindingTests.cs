using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ConcreteGenericBindingTests
{
    public enum TypeArgument { Double, Single, Open, Array, ByReference, Pointer }
    private enum BinaryInput { Replaced, Missing, Corrupt }

    [Theory]
    [InlineData(TypeArgument.Double)]
    [InlineData(TypeArgument.Single)]
    [InlineData(TypeArgument.Open)]
    [InlineData(TypeArgument.Array)]
    [InlineData(TypeArgument.ByReference)]
    [InlineData(TypeArgument.Pointer)]
    public void TypeArgumentsAreBoundWithoutErasingTypeOrIndirection(TypeArgument kind)
    {
        using var module = ModuleDefinition.CreateModule("binding", ModuleKind.Dll);
        TypeDefinition owner = Owner(module, "Owner");
        TypeReference argument = Argument(module, owner, kind);
        var type = new GenericInstanceType(owner);
        type.GenericArguments.Add(argument);
        var call = new MethodReference("Invoke", module.TypeSystem.Void, type);
        TypeReference? result = ConcreteGenericBinding.Read(owner.GenericParameters[0], call);
        if (kind is TypeArgument.Double or TypeArgument.Single) Assert.Same(argument, result);
        else Assert.Null(result);
    }

    [Fact]
    public void SameNamedParameterInAnotherTypeCannotBorrowTheBinding()
    {
        using var module = ModuleDefinition.CreateModule("binding", ModuleKind.Dll);
        TypeDefinition owner = Owner(module, "Owner"), unrelated = Owner(module, "Unrelated");
        var type = new GenericInstanceType(owner);
        type.GenericArguments.Add(module.TypeSystem.Double);
        Assert.Null(ConcreteGenericBinding.Read(unrelated.GenericParameters[0],
            new MethodReference("Invoke", module.TypeSystem.Void, type)));
    }

    [Fact]
    public void SameNamedTypeInAnotherAssemblyCannotBorrowTheBinding()
    {
        using var first = ModuleDefinition.CreateModule("first", ModuleKind.Dll);
        using var second = ModuleDefinition.CreateModule("second", ModuleKind.Dll);
        TypeDefinition owner = Owner(first, "Owner"), unrelated = Owner(second, "Owner");
        var type = new GenericInstanceType(owner);
        type.GenericArguments.Add(first.TypeSystem.Double);
        Assert.Null(ConcreteGenericBinding.Read(unrelated.GenericParameters[0],
            new MethodReference("Invoke", first.TypeSystem.Void, type)));
    }

    [Fact]
    public void MethodArgumentsRequireTheSameMethodNotJustTheSamePosition()
    {
        using var module = ModuleDefinition.CreateModule("binding", ModuleKind.Dll);
        TypeDefinition owner = Owner(module, "Owner");
        MethodDefinition method = Method(owner, "First"), other = Method(owner, "Second");
        var call = new GenericInstanceMethod(method);
        call.GenericArguments.Add(module.TypeSystem.Double);
        Assert.Same(module.TypeSystem.Double, ConcreteGenericBinding.Read(method.GenericParameters[0], call));
        Assert.Null(ConcreteGenericBinding.Read(other.GenericParameters[0], call));
        Assert.Null(ConcreteGenericBinding.Read(owner.GenericParameters[0], call));
    }

    [Fact]
    public void OpenAndMissingContextsStayUnresolved()
    {
        using var module = ModuleDefinition.CreateModule("binding", ModuleKind.Dll);
        TypeDefinition owner = Owner(module, "Owner");
        Assert.Null(ConcreteGenericBinding.Read(owner.GenericParameters[0], null));
        Assert.Null(ConcreteGenericBinding.Read(owner.GenericParameters[0], Method(owner, "Invoke")));
        Assert.Null(ConcreteGenericBinding.Read(owner, null));
        Assert.Null(ConcreteGenericBinding.Read(new ArrayType(module.TypeSystem.Double), null));
    }

    [Fact]
    public void MalformedGenericArityCannotBorrowItsFirstArgument()
    {
        using var module = ModuleDefinition.CreateModule("binding", ModuleKind.Dll);
        TypeDefinition owner = Owner(module, "Owner");
        var type = new GenericInstanceType(owner);
        type.GenericArguments.Add(module.TypeSystem.Double);
        type.GenericArguments.Add(module.TypeSystem.Single);
        Assert.Null(ConcreteGenericBinding.Read(owner.GenericParameters[0],
            new MethodReference("Invoke", module.TypeSystem.Void, type)));
        MethodDefinition method = Method(owner, "First");
        var call = new GenericInstanceMethod(method);
        call.GenericArguments.Add(module.TypeSystem.Double);
        call.GenericArguments.Add(module.TypeSystem.Single);
        Assert.Null(ConcreteGenericBinding.Read(method.GenericParameters[0], call));
    }

    [Fact]
    public void ImportedProviderSignatureBindsParameterIdentityNotItsDisplayName()
    {
        using var module = ModuleDefinition.CreateModule("provider", ModuleKind.Dll);
        var helper = new TypeDefinition("AiDotNet.Tensors.Helpers", "MathHelper", TypeAttributes.Public, module.TypeSystem.Object);
        module.Types.Add(helper);
        MethodDefinition method = Method(helper, "GetNumericOperations");
        var operations = new TypeDefinition("AiDotNet.Tensors.Interfaces", "INumericOperations`1", TypeAttributes.Public | TypeAttributes.Interface | TypeAttributes.Abstract);
        operations.GenericParameters.Add(new("T", operations));
        module.Types.Add(operations);
        var result = new GenericInstanceType(operations);
        result.GenericArguments.Add(method.GenericParameters[0]);
        method.ReturnType = result;
        using var caller = ModuleDefinition.CreateModule("caller", ModuleKind.Dll);
        MethodReference imported = caller.ImportReference(method);
        Assert.True(ReviewedNumericProvider.HasProviderSignature(method));
        Assert.True(ReviewedNumericProvider.HasProviderSignature(imported));
        MethodCallingConvention original = imported.CallingConvention;
        imported.CallingConvention = MethodCallingConvention.VarArg;
        Assert.False(ReviewedNumericProvider.HasProviderSignature(imported));
        imported.CallingConvention = original;
        imported.ExplicitThis = true;
        Assert.False(ReviewedNumericProvider.HasProviderSignature(imported));
        method.GenericParameters[0].Name = "Renamed";
        Assert.True(ReviewedNumericProvider.HasProviderSignature(method));
        result.GenericArguments[0] = operations.GenericParameters[0];
        Assert.False(ReviewedNumericProvider.HasProviderSignature(method));
        result.GenericArguments[0] = module.TypeSystem.Double;
        Assert.False(ReviewedNumericProvider.HasProviderSignature(method));
    }

    [Fact]
    public void SameNamedNumericMethodWithoutPinnedBytesStaysUnknown()
    {
        using var module = ModuleDefinition.CreateModule("AiDotNet.Tensors", ModuleKind.Dll);
        var helper = new TypeDefinition("AiDotNet.Tensors.Helpers", "MathHelper", TypeAttributes.Public, module.TypeSystem.Object);
        module.Types.Add(helper);
        MethodDefinition method = Method(helper, "GetNumericOperations");
        var operations = new TypeDefinition("AiDotNet.Tensors.Interfaces", "INumericOperations`1", TypeAttributes.Public | TypeAttributes.Interface | TypeAttributes.Abstract);
        operations.GenericParameters.Add(new("T", operations));
        module.Types.Add(operations);
        var resultType = new GenericInstanceType(operations);
        resultType.GenericArguments.Add(method.GenericParameters[0]);
        method.ReturnType = resultType;
        method.Body.Instructions.Add(Mono.Cecil.Cil.Instruction.Create(Mono.Cecil.Cil.OpCodes.Ldnull));
        method.Body.Instructions.Add(Mono.Cecil.Cil.Instruction.Create(Mono.Cecil.Cil.OpCodes.Ret));
        Assert.True(ReviewedNumericProvider.HasProviderSignature(method));
        string root = Directory.CreateTempSubdirectory("numeric-contract-").FullName;
        try
        {
            using var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
            foreach (BinaryInput input in Enum.GetValues<BinaryInput>())
            {
                string path = Path.Combine(root, input + ".dll");
                module.Write(path);
                using var candidate = AssemblyDefinition.ReadAssembly(path, new ReaderParameters { InMemory = true, AssemblyResolver = resolver });
                MethodDefinition loaded = candidate.MainModule.Types.Single(type => type.Name == "MathHelper").Methods.Single();
                var call = new GenericInstanceMethod(loaded);
                call.GenericArguments.Add(candidate.MainModule.ImportReference(typeof(double)));
                Assert.True(ReviewedNumericProvider.HasProviderSignature(loaded));
                if (input == BinaryInput.Missing) File.Delete(path);
                if (input == BinaryInput.Corrupt) File.WriteAllBytes(path, [0, 1, 2]);
                NumericProviderAssessment result = ReviewedNumericProvider.Read(call);
                Assert.Equal(NumericProviderContract.Unresolved, result.Contract);
                Assert.Empty(result.Requirements);
            }
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    private static TypeDefinition Owner(ModuleDefinition module, string name)
    {
        var type = new TypeDefinition("Fixtures", name + "`1", TypeAttributes.Public, module.TypeSystem.Object);
        type.GenericParameters.Add(new("T", type));
        module.Types.Add(type);
        return type;
    }

    private static MethodDefinition Method(TypeDefinition owner, string name)
    {
        var method = new MethodDefinition(name, MethodAttributes.Public | MethodAttributes.Static, owner.Module.TypeSystem.Void);
        method.GenericParameters.Add(new("T", method));
        owner.Methods.Add(method);
        return method;
    }

    private static TypeReference Argument(ModuleDefinition module, TypeDefinition owner, TypeArgument kind) => kind switch
    {
        TypeArgument.Double => module.TypeSystem.Double,
        TypeArgument.Single => module.TypeSystem.Single,
        TypeArgument.Open => owner.GenericParameters[0],
        TypeArgument.Array => new ArrayType(module.TypeSystem.Double),
        TypeArgument.ByReference => new ByReferenceType(module.TypeSystem.Double),
        TypeArgument.Pointer => new PointerType(module.TypeSystem.Double),
        _ => throw new ArgumentOutOfRangeException(nameof(kind))
    };
}
