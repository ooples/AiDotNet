using Mono.Cecil;
using AiDotNet.TestImpact;
using Mono.Cecil.Cil;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "SourceImpact")]
public sealed class GraphExtractionTests
{
    public enum ConstraintCase { Missing, TaskWins, Cycle }

    [Theory]
    [InlineData(ConstraintCase.Missing)]
    [InlineData(ConstraintCase.TaskWins)]
    [InlineData(ConstraintCase.Cycle)]
    public void UnknownGenericConstraintsRemainUnresolved(ConstraintCase kind)
    {
        using var assembly = Assembly();
        MethodDefinition method = Method(assembly.MainModule.Types[0], "Generic");
        var parameter = new GenericParameter("T", method);
        method.GenericParameters.Add(parameter);
        var missing = new TypeReference("Missing", "Base", assembly.MainModule, new AssemblyNameReference("Missing", new Version(1, 0)));
        parameter.Constraints.Add(new(kind == ConstraintCase.Cycle ? parameter : missing));
        if (kind == ConstraintCase.TaskWins)
            parameter.Constraints.Add(new(new TypeReference("System.Threading.Tasks", "Task", assembly.MainModule, assembly.MainModule.TypeSystem.CoreLibrary)));
        Assert.Equal(kind == ConstraintCase.TaskWins ? TaskReturnKind.Task : TaskReturnKind.Unresolved, new TaskTypeInspector().Classify(parameter));
    }
    [Fact]
    public void ExternHashTracksItsDeclarationNotOtherMethodBodies()
    {
        using var assembly = Assembly();
        TypeDefinition type = assembly.MainModule.Types[0];
        var nativeModule = new ModuleReference("native-library");
        assembly.MainModule.ModuleReferences.Add(nativeModule);
        var external = new MethodDefinition("Native", MethodAttributes.Public | MethodAttributes.Static | MethodAttributes.PInvokeImpl,
            assembly.MainModule.TypeSystem.Void)
        {
            PInvokeInfo = new PInvokeInfo(PInvokeAttributes.CallConvCdecl, "native_entry", nativeModule)
        };
        type.Methods.Add(external);
        MethodDefinition ordinary = Method(type, "Managed");
        string DeclarationHash()
        {
            using var stream = new MemoryStream();
            assembly.Write(stream);
            stream.Position = 0;
            using var pe = new System.Reflection.PortableExecutable.PEReader(stream);
            return SourceSnapshotReader.BodyHash(pe, external);
        }
        string original = DeclarationHash();
        ordinary.Body.Instructions.Insert(0, Instruction.Create(OpCodes.Nop));
        Assert.Equal(original, DeclarationHash());
        external.PInvokeInfo.EntryPoint = "different_native_entry";
        Assert.NotEqual(original, DeclarationHash());
    }

    [Fact, BeforeAfterProbe]
    public void BeforeAfterRootsExcludeUnrelatedAttributeHelpers()
    {
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(GraphExtractionTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(GraphExtractionTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        SourceTestLifecycle test = XunitLifecycleReader.Read(assembly).Map.Tests.Single(test =>
            test.Owner.EndsWith("GraphExtractionTests.BeforeAfterRootsExcludeUnrelatedAttributeHelpers", StringComparison.Ordinal));
        Assert.Contains(test.Roots, root => root.Contains("BeforeAfterProbeAttribute::Before", StringComparison.Ordinal));
        Assert.DoesNotContain(test.Roots, root => root.Contains("UnusedHelper", StringComparison.Ordinal) ||
            root.Contains("System.Attribute::GetCustomAttribute", StringComparison.Ordinal));
    }

    private static AssemblyDefinition Assembly() => AssemblyDefinition.CreateAssembly(
        new AssemblyNameDefinition("GraphFixture", new Version(1, 0)), "GraphFixture", ModuleKind.Dll);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AssemblyHooksRunPerOwnerWithoutLosingSharedAttributeState(bool sharedField)
    {
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(GraphExtractionTests).Assembly.Location));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        using var assembly = AssemblyDefinition.ReadAssembly(typeof(GraphExtractionTests).Assembly.Location,
            new ReaderParameters { AssemblyResolver = resolver });
        TypeDefinition hook = assembly.MainModule.Types.Single(type => type.FullName == typeof(BeforeAfterProbeAttribute).FullName);
        assembly.CustomAttributes.Add(new CustomAttribute(hook.Methods.Single(method => method.IsConstructor && !method.IsStatic)));
        if (sharedField) hook.Fields.Add(new("state", FieldAttributes.Public, assembly.MainModule.TypeSystem.Int32));
        XunitLifecycleResult result = XunitLifecycleReader.Read(assembly);
        string callback = DependencyGraph.Stable(hook.Methods.Single(method => method.Name == "Before"));
        Assert.DoesNotContain(callback, result.Map.GroupRoots);
        Assert.All(result.Map.Tests, test => Assert.Contains(callback, test.Roots));
        Assert.Contains(DependencyGraph.Stable(hook.Methods.Single(method => method.IsConstructor)), result.Map.GroupRoots);
        SourceMethod[] shared = result.SyntheticMethods.Where(method => method.Dependency.Id.StartsWith("xunit-assembly-attribute-state:", StringComparison.Ordinal)).ToArray();
        if (sharedField)
        {
            string id = Assert.Single(shared).Dependency.Id;
            Assert.Contains(id, result.Map.GroupRoots);
            Assert.All(result.Map.Tests, test => Assert.Contains(id, test.Roots));
        }
        else Assert.Empty(shared);
    }

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

public sealed class BeforeAfterProbeAttribute : Xunit.Sdk.BeforeAfterTestAttribute
{
    public override void Before(System.Reflection.MethodInfo methodUnderTest) { }
    public static void UnusedHelper() => throw new InvalidOperationException("The runner never invokes this helper.");
}
