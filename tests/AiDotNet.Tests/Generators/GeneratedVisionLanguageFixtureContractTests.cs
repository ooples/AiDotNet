using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Executes the effective generated Flamingo factory against the real model assembly.</summary>
public sealed class GeneratedVisionLanguageFixtureContractTests
{
    public GeneratedVisionLanguageFixtureContractTests() => TestModuleInitializer.EnsureInitialized();

    [Fact]
    public void Flamingo_EmittedOptionsContainBoundedPatchesAndAnActualLanguageGate()
    {
        MethodDeclarationSyntax factory = GenerateFactory();
        ObjectCreationExpressionSyntax options = Assert.Single(factory.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), creation =>
                creation.Type.ToString().EndsWith(".FlamingoOptions", StringComparison.Ordinal));
        AssertAssignment(options, "ImageSize", 32);
        AssertAssignment(options, "PatchSize", 8);
        AssertAssignment(options, "NumLmLayers", 4);
        AssertAssignment(options, "NumPerceiverLayers", 1);
        AssertAssignment(options, "NumPerceiverTokens", 4);
    }

    [Fact]
    public void Flamingo_ActualGeneratedFactoryCompilesConstructsAndProducesConfiguredPatchTokens()
    {
        // Only accessibility changes: the emitted factory body, types, options and seed are untouched.
        MethodDeclarationSyntax factory = GenerateFactory().WithModifiers(
            SyntaxFactory.TokenList(SyntaxFactory.Token(SyntaxKind.PublicKeyword),
                SyntaxFactory.Token(SyntaxKind.StaticKeyword)));
        string source = "using AiDotNet.Interfaces; using AiDotNet.NeuralNetworks; public static class GeneratedFactory { "
            + factory.NormalizeWhitespace().ToFullString() + " }";
        var compilation = CSharpCompilation.Create("FlamingoFixtureExecution_" + Guid.NewGuid().ToString("N"),
            new[] { CSharpSyntaxTree.ParseText(source) }, RuntimeReferences(includeModels: true),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        using var assemblyBytes = new MemoryStream();
        var emit = compilation.Emit(assemblyBytes);
        Assert.True(emit.Success, string.Join(Environment.NewLine, emit.Diagnostics));
        Assembly assembly = Assembly.Load(assemblyBytes.ToArray());
        MethodInfo create = assembly.GetType("GeneratedFactory")?.GetMethod("CreateNetwork")
            ?? throw new InvalidOperationException("The emitted factory method was not compiled.");
        object? result = create.Invoke(null, null);
        switch (result)
        {
            case FlamingoNeuralNetwork<float> single:
                using (single) AssertActualFixture(single);
                break;
            case FlamingoNeuralNetwork<double> precise:
                using (precise) AssertActualFixture(precise);
                break;
            default:
                throw new InvalidOperationException("The generated factory did not return a real Flamingo model.");
        }
    }

    private static void AssertActualFixture<T>(FlamingoNeuralNetwork<T> model)
    {
        var activations = model.GetNamedLayerActivations(new Tensor<T>(new[] { 3, 32, 32 }));
        Assert.Equal(new[] { 16, 64 }, activations["vision_features"].Shape.ToArray());
        Assert.Equal(new[] { 4, 64 }, activations["perceiver_features"].Shape.ToArray());
        // One Perceiver gate plus a distinct LM gate. A one-layer language stack had only the former.
        Assert.Equal(2, model.Layers.OfType<CrossAttentionLayer<T>>().Count());
    }

    private static MethodDeclarationSyntax GenerateFactory()
    {
        // Minimal discovery symbols isolate one fixture; they are never used to compile or execute it.
        const string discoverySource = """
            using System;
            namespace AiDotNet.Attributes
            {
                [AttributeUsage(AttributeTargets.Class)]
                public sealed class ModelDomainAttribute : Attribute
                { public ModelDomainAttribute(int domain) { } }
            }
            namespace AiDotNet.Tensors.LinearAlgebra { public class Tensor<T> { } }
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<T, TInput, TOutput> { }
                public interface INeuralNetworkModel<T> : IFullModel<T,
                    AiDotNet.Tensors.LinearAlgebra.Tensor<T>, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> { }
            }
            namespace AiDotNet.NeuralNetworks
            {
                public class NeuralNetworkArchitecture<T> { }
                public abstract class NeuralNetworkBase<T> : AiDotNet.Interfaces.INeuralNetworkModel<T> { }
                [AiDotNet.Attributes.ModelDomain(0)]
                public class FlamingoNeuralNetwork<T> : NeuralNetworkBase<T>
                { public FlamingoNeuralNetwork(NeuralNetworkArchitecture<T> architecture) { } }
            }
            """;
        var compilation = CSharpCompilation.Create("AiDotNetTests",
            new[] { CSharpSyntaxTree.ParseText(discoverySource) }, RuntimeReferences(includeModels: false),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.DoesNotContain(compilation.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.TestScaffoldGenerator());
        var result = driver.RunGenerators(compilation).GetRunResult();
        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.All(result.Results, generator => Assert.Null(generator.Exception));
        ClassDeclarationSyntax fixture = Assert.Single(result.GeneratedTrees.SelectMany(tree => tree.GetRoot()
            .DescendantNodes().OfType<ClassDeclarationSyntax>()),
            declaration => declaration.Identifier.ValueText == "FlamingoNeuralNetworkTests");
        Assert.DoesNotContain(fixture.SyntaxTree.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        return Assert.Single(fixture.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "CreateNetwork");
    }

    private static ImmutableArray<MetadataReference> RuntimeReferences(bool includeModels)
    {
        var assemblies = AppDomain.CurrentDomain.GetAssemblies()
            .Concat(new[] { typeof(FlamingoNeuralNetwork<>).Assembly, typeof(Tensor<>).Assembly });
        return assemblies.Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
            .Where(assembly => assembly.GetName().Name is string name
                && (name == "mscorlib" || name == "netstandard" || name == "System"
                    || name.StartsWith("System.", StringComparison.Ordinal)
                    || (includeModels && (name == "AiDotNet" || name == "AiDotNet.Tensors"))))
            .Select(assembly => assembly.Location).Distinct(StringComparer.Ordinal)
            .Select(path => MetadataReference.CreateFromFile(path)).ToImmutableArray<MetadataReference>();
    }

    private static void AssertAssignment(ObjectCreationExpressionSyntax options, string property, int expected)
    {
        Assert.NotNull(options.Initializer);
        AssignmentExpressionSyntax assignment = Assert.Single(options.Initializer.Expressions
            .OfType<AssignmentExpressionSyntax>(), expression => expression.Left.ToString() == property);
        Assert.Equal(expected, Assert.IsType<LiteralExpressionSyntax>(assignment.Right).Token.Value);
    }
}
