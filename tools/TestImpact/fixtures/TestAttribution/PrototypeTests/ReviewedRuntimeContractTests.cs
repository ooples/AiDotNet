using AiDotNet.TestImpact;
using Mono.Cecil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RuntimeEffects")]
public sealed class ReviewedRuntimeContractTests
{
    public enum BinaryMutation { Implementation, Identity, Missing, NonAssembly }
    public enum RuntimeMutation { ExtraInstruction, StaticField, WrongGetter, WrongReturn, WrongParameter, Handler, Initializer }

    [Fact]
    public void AssertionTypeInitializationIsNotHiddenByTheLeafContract()
    {
        using var resolver = Resolver();
        using var assembly = Read(typeof(Assert).Assembly.Location, resolver);
        MethodDefinition method = BooleanAssertion(assembly, true);
        Assert.Contains(method.DeclaringType.Methods, candidate => candidate.IsConstructor && candidate.IsStatic && candidate.HasBody);
        RuntimeContractAssessment result = ReviewedRuntimeContracts.Assess(method);
        Assert.Equal(RuntimeContractStatus.ReviewedConditional, result.Status);
        Assert.Contains(RuntimeContractRequirement.InitializationEffectsProven, result.Requirements);
    }

    [Fact]
    public void ActualNullableRuntimeMatchesTheReviewedPassingPath()
    {
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(bool).Assembly.Location);
        Assert.True(ReviewedRuntimeContracts.IsReviewedNullableImplementation(
            assembly.MainModule.Types.Single(type => type.FullName == "System.Nullable`1")));
    }

    [Theory]
    [InlineData(RuntimeMutation.ExtraInstruction)]
    [InlineData(RuntimeMutation.StaticField)]
    [InlineData(RuntimeMutation.WrongGetter)]
    [InlineData(RuntimeMutation.WrongReturn)]
    [InlineData(RuntimeMutation.WrongParameter)]
    [InlineData(RuntimeMutation.Handler)]
    [InlineData(RuntimeMutation.Initializer)]
    public void UnreviewedRuntimeBehaviorIsRejected(RuntimeMutation mutation)
    {
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(bool).Assembly.Location);
        TypeDefinition type = assembly.MainModule.Types.Single(type => type.FullName == "System.Nullable`1");
        MethodDefinition constructor = type.Methods.Single(method => method.IsConstructor && !method.IsStatic);
        MethodDefinition getter = type.Methods.Single(method => method.Name == "get_HasValue");
        switch (mutation)
        {
            case RuntimeMutation.ExtraInstruction:
                getter.Body.Instructions.Insert(0, Mono.Cecil.Cil.Instruction.Create(Mono.Cecil.Cil.OpCodes.Nop)); break;
            case RuntimeMutation.StaticField:
                ((FieldReference)constructor.Body.Instructions[5].Operand).Resolve().IsStatic = true; break;
            case RuntimeMutation.WrongGetter:
                getter.Body.Instructions[1].Operand = constructor.Body.Instructions[2].Operand; break;
            case RuntimeMutation.WrongReturn: getter.ReturnType = assembly.MainModule.TypeSystem.Int32; break;
            case RuntimeMutation.WrongParameter: constructor.Parameters[0].ParameterType = assembly.MainModule.TypeSystem.Object; break;
            case RuntimeMutation.Handler:
                getter.Body.ExceptionHandlers.Add(new(Mono.Cecil.Cil.ExceptionHandlerType.Finally)); break;
            case RuntimeMutation.Initializer:
                type.Methods.Add(new(".cctor", MethodAttributes.Static | MethodAttributes.SpecialName | MethodAttributes.RTSpecialName,
                    assembly.MainModule.TypeSystem.Void)); break;
        }
        Assert.False(ReviewedRuntimeContracts.IsReviewedNullableImplementation(type));
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ReviewedBooleanAssertionsRemainConditional(bool assertion)
    {
        using var resolver = Resolver();
        using var assembly = Read(typeof(Assert).Assembly.Location, resolver);
        MethodDefinition method = BooleanAssertion(assembly, assertion);
        RuntimeContractAssessment result = ReviewedRuntimeContracts.Assess(method);
        Assert.Equal(RuntimeContractStatus.ReviewedConditional, result.Status);
        Assert.Equal(RuntimeContractId.XunitBooleanAssertion293, result.Contract);
        Assert.Equal(ReviewedRuntimeContracts.AssertionHash, result.AssemblyHash);
        Assert.Equal(64, result.RuntimeHash.Length);
        Assert.Equal([RuntimeContractEffect.ReadScalarArguments, RuntimeContractEffect.ThrowOnFailure], result.Effects);
        Assert.Equal(Enum.GetValues<RuntimeContractRequirement>(), result.Requirements);
        // A known leaf's conditional review must not close an unresolved caller,
        // group initializer, callback, or per-test fixture in the normal planner.
        var graph = new DependencySnapshot([new("assertion", [], [], DependencyBoundary.External)],
            [new("a", ["assertion"]), new("b", ["assertion"])], []);
        Assert.Equal(2, DependencySelection.Select(graph, graph, ["assertion"], false).Count);
    }

    [Theory]
    [InlineData("System.Void Xunit.Assert::True(System.Boolean)")]
    [InlineData("System.Void Xunit.Assert::True(System.Nullable`1<System.Boolean>,System.String)")]
    [InlineData("System.Void Xunit.Assert::False(System.Boolean)")]
    public void UnreviewedOverloadsDoNotInheritAContract(string signature)
    {
        using var resolver = Resolver();
        using var assembly = Read(typeof(Assert).Assembly.Location, resolver);
        var method = assembly.MainModule.Types.Single(type => type.FullName == "Xunit.Assert")
            .Methods.Single(candidate => candidate.FullName == signature);
        AssertUnknown(ReviewedRuntimeContracts.Assess(method));
    }

    [Theory]
    [InlineData(BinaryMutation.Implementation)]
    [InlineData(BinaryMutation.Identity)]
    [InlineData(BinaryMutation.Missing)]
    [InlineData(BinaryMutation.NonAssembly)]
    public void SameMethodNameCannotAuthorizeUnreviewedBytes(BinaryMutation mutation)
    {
        string root = Directory.CreateTempSubdirectory("runtime-contract-").FullName;
        try
        {
            string path = Path.Combine(root, "xunit.assert.dll");
            using var resolver = Resolver();
            using (var copy = Read(typeof(Assert).Assembly.Location, resolver))
            {
                if (mutation == BinaryMutation.Implementation)
                    BooleanAssertion(copy, true).Body.Instructions.Insert(0, Mono.Cecil.Cil.Instruction.Create(Mono.Cecil.Cil.OpCodes.Nop));
                if (mutation == BinaryMutation.Identity) copy.Name.Version = new Version(2, 9, 4, 0);
                copy.Write(path);
            }
            using var changed = Read(path, resolver);
            MethodDefinition method = BooleanAssertion(changed, true);
            if (mutation == BinaryMutation.Missing) File.Delete(path);
            if (mutation == BinaryMutation.NonAssembly) File.WriteAllBytes(path, [0, 1, 2]);
            AssertUnknown(ReviewedRuntimeContracts.Assess(method));
        }
        finally { Directory.Delete(root, recursive: true); }
    }

    [Fact]
    public void UnresolvableSameNamedMethodIsUnknown()
    {
        using var assembly = AssemblyDefinition.CreateAssembly(new("Imposter", new Version(1, 0)), "Imposter", ModuleKind.Dll);
        var type = new TypeReference("Xunit", "Assert", assembly.MainModule,
            new AssemblyNameReference("Missing.Runtime.Contract.Assembly", new Version(2, 9, 3, 0)));
        var method = new MethodReference("True", assembly.MainModule.TypeSystem.Void, type);
        method.Parameters.Add(new(assembly.MainModule.TypeSystem.Boolean));
        method.Parameters.Add(new(assembly.MainModule.TypeSystem.String));
        AssertUnknown(ReviewedRuntimeContracts.Assess(method));
    }

    private static void AssertUnknown(RuntimeContractAssessment result)
    {
        Assert.Equal(RuntimeContractStatus.Unknown, result.Status);
        Assert.Null(result.Contract);
        Assert.Empty(result.AssemblyHash);
        Assert.Empty(result.RuntimeHash);
        Assert.Empty(result.Effects);
        Assert.Empty(result.Requirements);
    }

    private static DefaultAssemblyResolver Resolver()
    {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(Assert).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        return resolver;
    }

    private static AssemblyDefinition Read(string path, IAssemblyResolver resolver) => AssemblyDefinition.ReadAssembly(path,
        new ReaderParameters { InMemory = true, AssemblyResolver = resolver });

    private static MethodDefinition BooleanAssertion(AssemblyDefinition assembly, bool assertion) =>
        assembly.MainModule.Types.Single(type => type.FullName == "Xunit.Assert").Methods.Single(method =>
            method.FullName == $"System.Void Xunit.Assert::{(assertion ? "True" : "False")}(System.Boolean,System.String)");
}
