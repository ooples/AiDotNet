using Mono.Cecil;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class GraphExtractionTests
{
    private static AssemblyDefinition Assembly() => AssemblyDefinition.CreateAssembly(
        new AssemblyNameDefinition("GraphFixture", new Version(1, 0)), "GraphFixture", ModuleKind.Dll);

    private static MethodDefinition Method(TypeDefinition type, string name)
    {
        var method = new MethodDefinition(name, MethodAttributes.Public | MethodAttributes.Static, type.Module.TypeSystem.Void);
        type.Methods.Add(method);
        method.MetadataToken = new MetadataToken(TokenType.Method, (uint)type.Module.Types.Sum(owner => owner.Methods.Count));
        method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
        return method;
    }

    [Fact]
    public void GenericArityDisambiguatesIdenticalCecilDisplayNames()
    {
        using var assembly = Assembly();
        TypeDefinition type = assembly.MainModule.Types[0];
        MethodDefinition first = Method(type, "Identity");
        first.GenericParameters.Add(new GenericParameter("T", first));
        MethodDefinition second = Method(type, "Identity");
        second.GenericParameters.Add(new GenericParameter("T", second));
        second.GenericParameters.Add(new GenericParameter("TOther", second));
        Assert.Equal(first.FullName, second.FullName);
        Assert.NotEqual(DependencyGraph.Stable(first), DependencyGraph.Stable(second));
    }

    [Fact]
    public void EveryMethodRetainsItsModuleInitializerDependency()
    {
        using var assembly = Assembly();
        TypeDefinition module = assembly.MainModule.Types[0];
        MethodDefinition initializer = Method(module, ".cctor");
        initializer.Attributes |= MethodAttributes.SpecialName | MethodAttributes.RTSpecialName;
        var type = new TypeDefinition("GraphFixture", "Subject", TypeAttributes.Public, assembly.MainModule.TypeSystem.Object);
        assembly.MainModule.Types.Add(type);
        MethodDefinition method = Method(type, "Run");
        const string hash = "fixture";
        MethodDependencyNode node = Assert.Single(DependencyGraph.Read(assembly, hash, selectedMethods: [method]).Methods);
        Assert.Contains($"{hash}:{initializer.MetadataToken.ToInt32():X8}", node.LocalCalls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void OnlyVirtualDispatchMakesAnExplicitCallOpen(bool virtualCall)
    {
        using var assembly = Assembly();
        var type = new TypeDefinition("GraphFixture", "Subject", TypeAttributes.Public, assembly.MainModule.TypeSystem.Object);
        assembly.MainModule.Types.Add(type);
        MethodDefinition target = Method(type, "Target");
        target.Attributes = MethodAttributes.Public | MethodAttributes.Virtual;
        MethodDefinition caller = Method(type, "Caller");
        caller.Body.Instructions.Insert(0, Instruction.Create(virtualCall ? OpCodes.Callvirt : OpCodes.Call, target));
        caller.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Ldnull));
        MethodDependencyNode node = Assert.Single(DependencyGraph.Read(assembly, "fixture", selectedMethods: [caller]).Methods);
        Assert.Equal(virtualCall, node.OpenDependencies.Any(boundary => boundary.Kind == OpenDependencyKind.VirtualDispatch));
    }
}
