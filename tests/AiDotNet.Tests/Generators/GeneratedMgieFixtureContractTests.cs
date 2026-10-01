using System.Collections.Immutable;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage.Editing;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class GeneratedMgieFixtureContractTests
{
    public GeneratedMgieFixtureContractTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TypedGuidanceSurface_EmitsBoundedThreeStackFactories_FromSourceAndMetadata(bool metadata)
    {
        var fixtures = GenerateFixtures(metadata);
        Assert.Equal(2, fixtures.Length);
        foreach (var fixture in fixtures)
        {
            var factory = Factory(fixture);
            var options = Assert.Single(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>(),
                creation => creation.Type.ToString().EndsWith("MGIEOptions", StringComparison.Ordinal));
            AssertAssignment(options, nameof(MGIEOptions.ImageSize), 8);
            AssertAssignment(options, nameof(MGIEOptions.VisionPatchSize), 2);
            AssertAssignment(options, nameof(MGIEOptions.DecoderDim), 8);
            AssertAssignment(options, nameof(MGIEOptions.EditHiddenDim), 8);
            AssertAssignment(options, nameof(MGIEOptions.EditQueryCount), 3);
            Assert.Contains(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>(),
                creation => creation.Type.ToString().Contains("UNetNoisePredictor<double>"));
            Assert.Contains(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>(),
                creation => creation.Type.ToString().Contains("StandardVAE<double>"));
            var shape = Assert.Single(fixture.Members.OfType<PropertyDeclarationSyntax>(), property => property.Identifier.ValueText == "InputShape");
            Assert.Equal(new[] { 1, 4, 8, 8 }, shape.DescendantNodes().OfType<LiteralExpressionSyntax>()
                .Select(literal => Assert.IsType<int>(literal.Token.Value)).ToArray());
        }
    }

    [Fact]
    public void EmittedFactory_SemanticallyCompilesAndExecutesActualJointEditingModel()
    {
        var fixture = GenerateActualFixture();
        var factory = Factory(fixture).WithModifiers(SyntaxFactory.TokenList(
            SyntaxFactory.Token(SyntaxKind.PublicKeyword), SyntaxFactory.Token(SyntaxKind.StaticKeyword)));
        string source = "using AiDotNet.Interfaces; using AiDotNet.NeuralNetworks; public static class GeneratedFactory { " +
            factory.NormalizeWhitespace().ToFullString() + " }";
        var compilation = CSharpCompilation.Create("MgieFactory_" + Guid.NewGuid().ToString("N"),
            new[] { CSharpSyntaxTree.ParseText(source) }, References(includeModels: true),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var bytes = new MemoryStream();
        var emitted = compilation.Emit(bytes);
        Assert.True(emitted.Success, string.Join(Environment.NewLine, emitted.Diagnostics));
        var assembly = Assembly.Load(bytes.ToArray());
        var create = assembly.GetType("GeneratedFactory")?.GetMethod(factory.Identifier.ValueText)
            ?? throw new InvalidOperationException("The emitted MGIE factory is missing.");
        using var model = Assert.IsType<MGIE<double>>(create.Invoke(null, null));
        var options = Assert.IsType<MGIEOptions>(model.GetOptions());
        Assert.Equal(8, options.DecoderDim);
        Assert.Equal(8, options.EditHiddenDim);
        Assert.Equal(3, options.EditQueryCount);
        var image = new Tensor<double>(new[] { 1, 3, 8, 8 });
        for (int i = 0; i < image.Length; i++) image[i] = Math.Sin(i * 0.37) * 0.5;
        var context = model.EncodeEditGuidance(image, new[] { 1, 2, 3 });
        Assert.Equal(new[] { 1, 3, 768 }, context.Shape.ToArray());
        var output = model.EditImage(image, "bright image", 42);
        Assert.Equal(image.Shape.ToArray(), output.Shape.ToArray());
        Assert.InRange(model.ParameterCount, 1, 5_000_000);
#if NETFRAMEWORK
        Assert.True(model.GetParameters().Length > 0);
#else
        Assert.Contains(model.GetParameterChunks(), chunk => chunk.Length > 0);
#endif
        Assert.All(output.ToArray(), value => Assert.False(double.IsNaN(value) || double.IsInfinity(value)));
    }

    private static MethodDeclarationSyntax Factory(ClassDeclarationSyntax fixture)
        => Assert.Single(fixture.Members.OfType<MethodDeclarationSyntax>(), method =>
            method.Modifiers.Any(SyntaxKind.OverrideKeyword) && method.Identifier.ValueText.StartsWith("Create", StringComparison.Ordinal));

    private static ClassDeclarationSyntax GenerateActualFixture()
    {
        var compilation = CSharpCompilation.Create("AiDotNetTests", references: References(includeModels: true),
            options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var symbol = compilation.GetTypeByMetadataName("AiDotNet.VisionLanguage.Editing.MGIE`1");
        Assert.NotNull(symbol);
        Assert.All(symbol.Locations, location => Assert.True(location.IsInMetadata));
        // Isolate this model using the generator's ordinary existing-test inventory. These
        // declarations are not runtime stand-ins or evidence for any unrelated model. No
        // diagnostic is filtered, and MGIE must still be discovered from its real metadata.
        var otherTestNames = Types(symbol.ContainingAssembly.GlobalNamespace)
            .Where(type => type.TypeKind == TypeKind.Class && type.DeclaredAccessibility == Accessibility.Public && type.Name != "MGIE")
            .Select(type => type.Name + "Tests").Distinct(StringComparer.Ordinal).ToArray();
        Assert.True(otherTestNames.Length > 100);
        Assert.DoesNotContain("MGIETests", otherTestNames);
        string inventory = "namespace UnrelatedModelInventory { " +
            string.Join(" ", otherTestNames.Select(name => "internal class " + name + " {}")) + " }";
        compilation = compilation.AddSyntaxTrees(CSharpSyntaxTree.ParseText(inventory));
        Assert.DoesNotContain(compilation.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.TestScaffoldGenerator());
        var result = driver.RunGenerators(compilation).GetRunResult();
        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.All(result.Results, generator => Assert.Null(generator.Exception));
        return Assert.Single(result.GeneratedTrees
            .SelectMany(tree => tree.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>()),
            type => type.Identifier.ValueText == "MGIETests");
    }

    private static IEnumerable<INamedTypeSymbol> Types(INamespaceSymbol scope)
    {
        foreach (var type in scope.GetTypeMembers()) yield return type;
        foreach (var child in scope.GetNamespaceMembers())
        foreach (var type in Types(child)) yield return type;
    }

    private static ClassDeclarationSyntax[] GenerateFixtures(bool metadata = false)
    {
        // Discovery-only types exercise the generator's structural mapper classification. They are
        // never used by the semantic compilation/runtime test, which references the shipping DLL.
        const string discovery = """
            using System;
            namespace AiDotNet.Attributes
            {
                [AttributeUsage(AttributeTargets.Class)]
                public sealed class ModelDomainAttribute : Attribute { public ModelDomainAttribute(int value) {} }
            }
            namespace AiDotNet.Tensors.LinearAlgebra { public class Tensor<T> {} }
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<T,TInput,TOutput> {}
                public interface IDiffusionModel<T> : IFullModel<T,
                    AiDotNet.Tensors.LinearAlgebra.Tensor<T>,AiDotNet.Tensors.LinearAlgebra.Tensor<T>> {}
            }
            namespace AiDotNet.NeuralNetworks { public class NeuralNetworkArchitecture<T> {} }
            namespace AiDotNet.NeuralNetworks.Layers { public class MultimodalEditMapperLayer<T> {} }
            namespace AiDotNet.Models.Options { public class ModelOptions {} }
            namespace AiDotNet.Diffusion { public abstract class LatentDiffusionModelBase<T> : AiDotNet.Interfaces.IDiffusionModel<T> {} }
            namespace AiDotNet.VisionLanguage.Editing
            {
                public class MGIEOptions : AiDotNet.Models.Options.ModelOptions { public MGIEOptions() {} }
                [AiDotNet.Attributes.ModelDomain(0)]
                public class MGIE<T> : AiDotNet.Diffusion.LatentDiffusionModelBase<T>
                {
                    private AiDotNet.NeuralNetworks.Layers.MultimodalEditMapperLayer<T> _mapper = new();
                    public MGIE(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture = null,
                        MGIEOptions options = null) {}
                    public AiDotNet.Tensors.LinearAlgebra.Tensor<T> EncodeEditGuidance(
                        AiDotNet.Tensors.LinearAlgebra.Tensor<T> image, System.Collections.Generic.IReadOnlyList<int> tokens) => image;
                }
                [AiDotNet.Attributes.ModelDomain(0)]
                public class OtherJointEditor<T> : AiDotNet.Diffusion.LatentDiffusionModelBase<T>
                {
                    private AiDotNet.NeuralNetworks.Layers.MultimodalEditMapperLayer<T> _mapper = new();
                    public OtherJointEditor(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture = null,
                        MGIEOptions options = null) {}
                    public AiDotNet.Tensors.LinearAlgebra.Tensor<T> EncodeEditGuidance(
                        AiDotNet.Tensors.LinearAlgebra.Tensor<T> image, System.Collections.Generic.IReadOnlyList<int> tokens) => image;
                }
            }
            """;
        var compilation = CSharpCompilation.Create(metadata ? "AiDotNet" : "AiDotNetTests", new[] { CSharpSyntaxTree.ParseText(discovery) },
            References(includeModels: false), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.DoesNotContain(compilation.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        if (metadata)
        {
            using var discoveryBytes = new MemoryStream();
            var emitted = compilation.Emit(discoveryBytes);
            Assert.True(emitted.Success, string.Join(Environment.NewLine, emitted.Diagnostics));
            compilation = CSharpCompilation.Create("AiDotNetTests", references: References(includeModels: false)
                .Add(MetadataReference.CreateFromImage(discoveryBytes.ToArray())),
                options: new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        }
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.TestScaffoldGenerator());
        var result = driver.RunGenerators(compilation).GetRunResult();
        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.All(result.Results, generator => Assert.Null(generator.Exception));
        var fixtures = result.GeneratedTrees.SelectMany(tree => tree.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>())
            .Where(type => type.Identifier.ValueText == "MGIETests" || type.Identifier.ValueText == "OtherJointEditorTests").ToArray();
        Assert.Equal(2, fixtures.Length); // A missing/empty discovery census is never evidence.
        return fixtures;
    }

    private static ImmutableArray<MetadataReference> References(bool includeModels)
    {
        var assemblies = AppDomain.CurrentDomain.GetAssemblies().Concat(new[] { typeof(MGIE<>).Assembly, typeof(Tensor<>).Assembly });
        return assemblies.Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
            .Where(assembly => assembly.GetName().Name is string name &&
                (name == "mscorlib" || name == "netstandard" || name == "System" || name.StartsWith("System.", StringComparison.Ordinal) ||
                 (includeModels && (name == "AiDotNet" || name == "AiDotNet.Tensors"))))
            .Select(assembly => assembly.Location).Distinct(StringComparer.Ordinal)
            .Select(path => MetadataReference.CreateFromFile(path)).ToImmutableArray<MetadataReference>();
    }

    private static void AssertAssignment(ObjectCreationExpressionSyntax options, string name, int expected)
    {
        Assert.NotNull(options.Initializer);
        var assignment = Assert.Single(options.Initializer.Expressions.OfType<AssignmentExpressionSyntax>(),
            expression => expression.Left.ToString() == name);
        Assert.Equal(expected, Assert.IsType<LiteralExpressionSyntax>(assignment.Right).Token.Value);
    }
}
