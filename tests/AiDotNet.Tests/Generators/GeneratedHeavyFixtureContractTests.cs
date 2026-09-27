using System;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Exercises constructor selection through the scaffold generator, including fallback controls.</summary>
public sealed class GeneratedHeavyFixtureContractTests
{
    // These compiler inputs deliberately expose both constructors. A fixture that omits the
    // parameterless overload cannot reproduce the fallback shadowing the model-specific rule.
    private const string Models = """
        using System;
        namespace AiDotNet.Attributes
        {
            [AttributeUsage(AttributeTargets.Class)]
            public sealed class ModelDomainAttribute : Attribute
            {
                public ModelDomainAttribute(int domain) { }
            }
        }
        namespace AiDotNet.Tensors.LinearAlgebra { public class Tensor<T> { } }
        namespace AiDotNet.Interfaces
        {
            public interface IFullModel<T, TInput, TOutput> { }
            public interface INeuralNetworkModel<T> : IFullModel<T,
                AiDotNet.Tensors.LinearAlgebra.Tensor<T>, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> { }
            public interface IVocoder<T> { }
        }
        namespace AiDotNet.NeuralNetworks
        {
            public class NeuralNetworkArchitecture<T> { }
            public abstract class NeuralNetworkBase<T> : AiDotNet.Interfaces.INeuralNetworkModel<T> { }
        }
        namespace AiDotNet.Video.Motion
        {
            public abstract class OpticalFlowBase<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T> { }
            [AiDotNet.Attributes.ModelDomain(0)]
            public class MemFlow<T> : OpticalFlowBase<T>
            {
                public MemFlow() { }
                public MemFlow(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture,
                    int numFeatures = 64, int numLayers = 8) { }
            }
            [AiDotNet.Attributes.ModelDomain(0)]
            public class PlainFlow<T> : OpticalFlowBase<T>
            {
                public PlainFlow() { }
                public PlainFlow(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture) { }
            }
            [AiDotNet.Attributes.ModelDomain(0)]
            public class ArchitectureOnlyFlow<T> : OpticalFlowBase<T>
            {
                public ArchitectureOnlyFlow(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture) { }
            }
        }
        namespace AiDotNet.TextToSpeech.Vocoders
        {
            public class MelGANOptions { public int NgfBase { get; set; } }
            public abstract class VocoderBase<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>,
                AiDotNet.Interfaces.IVocoder<T> { }
            [AiDotNet.Attributes.ModelDomain(0)]
            public class MelGAN<T> : VocoderBase<T>
            {
                public MelGAN() { }
                public MelGAN(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture,
                    MelGANOptions? options = null) { }
            }
        }
        """;

    [Fact]
    public void MemFlow_ExplicitSmokeConstructorWinsOverAvailableParameterlessConstructor()
    {
        ClassDeclarationSyntax fixture = GenerateFixture("MemFlowTests");
        ObjectCreationExpressionSyntax creation = ModelConstructor(fixture, "MemFlow");

        AssertNamedInteger(creation, "numFeatures", 8);
        AssertNamedInteger(creation, "numLayers", 2);
        ObjectCreationExpressionSyntax architecture = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(),
            node => node.Type.ToString().Contains("NeuralNetworkArchitecture"));
        AssertNamedInteger(architecture, "inputHeight", 64);
        AssertNamedInteger(architecture, "inputWidth", 64);
        AssertNamedInteger(architecture, "inputDepth", 6);
        AssertNamedInteger(architecture, "outputSize", 2);
        Assert.Contains("OpticalFlowTestBase<float>", fixture.BaseList?.ToString() ?? string.Empty);
    }

    [Fact]
    public void UnconfiguredModel_RetainsItsParameterlessFallback()
    {
        ObjectCreationExpressionSyntax creation = ModelConstructor(GenerateFixture("PlainFlowTests"), "PlainFlow");
        Assert.NotNull(creation.ArgumentList);
        Assert.Empty(creation.ArgumentList.Arguments);
    }

    [Fact]
    public void ArchitectureOnlyModel_RetainsItsArchitectureFallback()
    {
        ObjectCreationExpressionSyntax creation = ModelConstructor(
            GenerateFixture("ArchitectureOnlyFlowTests"), "ArchitectureOnlyFlow");
        ObjectCreationExpressionSyntax architecture = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(),
            node => node.Type.ToString().Contains("NeuralNetworkArchitecture"));
        AssertNamedInteger(architecture, "inputDepth", 6);
    }

    [Fact]
    public void MelGAN_EmitsOneMelFrameAndItsCorrespondingWaveformWithoutChangingChannels()
    {
        ClassDeclarationSyntax fixture = GenerateFixture("MelGANTests");
        ObjectCreationExpressionSyntax creation = ModelConstructor(fixture, "MelGAN");
        ObjectCreationExpressionSyntax options = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), node => node.Type.ToString().EndsWith("MelGANOptions"));
        AssignmentExpressionSyntax width = Assert.Single(options.DescendantNodes()
            .OfType<AssignmentExpressionSyntax>());
        Assert.Equal("NgfBase", width.Left.ToString());
        Assert.Equal(32, Assert.IsType<LiteralExpressionSyntax>(width.Right).Token.Value);
        AssertShape(fixture, "InputShape", 1, 80, 1);
        AssertShape(fixture, "OutputShape", 1, 1, 256);
    }

    public enum TransformerFixture
    {
        DistilBERTNER, ELECTRANER, FinBERTNER, LegalBERTNER, XLMRoBERTaNER, TemplateNER,
        DeBERTaNER, SECBertNER, SpanBERTNER, RELNER, RoBERTaNER, SciBERTNER, BLINKNER,
        ClinicalBERTNER, InstructionNER, ONNXNER, PubMedBERTNER, BioBERTNER
    }

    [Theory]
    [InlineData(TransformerFixture.DistilBERTNER)]
    [InlineData(TransformerFixture.ELECTRANER)]
    [InlineData(TransformerFixture.FinBERTNER)]
    [InlineData(TransformerFixture.LegalBERTNER)]
    [InlineData(TransformerFixture.XLMRoBERTaNER)]
    [InlineData(TransformerFixture.TemplateNER)]
    [InlineData(TransformerFixture.DeBERTaNER)]
    [InlineData(TransformerFixture.SECBertNER)]
    [InlineData(TransformerFixture.SpanBERTNER)]
    [InlineData(TransformerFixture.RELNER)]
    [InlineData(TransformerFixture.RoBERTaNER)]
    [InlineData(TransformerFixture.SciBERTNER)]
    [InlineData(TransformerFixture.BLINKNER)]
    [InlineData(TransformerFixture.ClinicalBERTNER)]
    [InlineData(TransformerFixture.InstructionNER)]
    [InlineData(TransformerFixture.ONNXNER)]
    [InlineData(TransformerFixture.PubMedBERTNER)]
    [InlineData(TransformerFixture.BioBERTNER)]
    public void TransformerSmokeFactory_ExplicitlyConfiguresItsPositiveWarmupBeforeConstruction(
        TransformerFixture fixtureKind)
    {
        string modelName = fixtureKind.ToString();
        ClassDeclarationSyntax fixture = GenerateTransformerFixture(fixtureKind);
        ObjectCreationExpressionSyntax creation = ModelConstructor(fixture, modelName);
        ObjectCreationExpressionSyntax options = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), node => node.Type.ToString().EndsWith("TransformerNEROptions"));
        var argument = Assert.IsType<ArgumentSyntax>(options.Parent);
        var arguments = Assert.IsType<ArgumentListSyntax>(argument.Parent);
        var configuration = Assert.IsType<InvocationExpressionSyntax>(arguments.Parent);
        Assert.Equal("WithPositiveSmokeWarmup", configuration.Expression.ToString());
        Assert.Single(arguments.Arguments);
        Assert.Contains("TransformerNERTestBase", fixture.BaseList?.ToString() ?? string.Empty);
    }

    [Theory]
    [InlineData(TransformerFixture.TemplateNER)]
    [InlineData(TransformerFixture.XLMRoBERTaNER)]
    public void LargeTransformerSmokeFactory_BoundsItsEncoderAndKeepsTheInputShapeInSync(
        TransformerFixture fixtureKind)
    {
        ClassDeclarationSyntax fixture = GenerateTransformerFixture(fixtureKind);
        ObjectCreationExpressionSyntax creation = ModelConstructor(fixture, fixtureKind.ToString());
        ObjectCreationExpressionSyntax architecture = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), node => node.Type.ToString().Contains("NeuralNetworkArchitecture"));
        AssertNamedInteger(architecture, "inputSize", 32);
        AssertNamedInteger(architecture, "outputSize", 9);
        ObjectCreationExpressionSyntax options = Assert.Single(creation.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), node => node.Type.ToString().EndsWith("TransformerNEROptions"));
        AssertAssignedInteger(options, "HiddenDimension", 32);
        AssertAssignedInteger(options, "NumAttentionHeads", 4);
        AssertAssignedInteger(options, "NumTransformerLayers", 2);
        AssertAssignedInteger(options, "IntermediateDimension", 64);
        AssertAssignedInteger(options, "MaxSequenceLength", 16);
        AssertAssignedInteger(options, "NumLabels", 9);
        AssertShape(fixture, "InputShape", 8, 32);
    }

    private static ClassDeclarationSyntax GenerateTransformerFixture(TransformerFixture fixtureKind)
    {
        string modelName = fixtureKind.ToString();
        string modelSource = Models + $$"""
            namespace AiDotNet.NER.Options { public class TransformerNEROptions { } }
            namespace AiDotNet.NER.TransformerBased
            {
                public abstract class TransformerNERBase<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T> { }
                [AiDotNet.Attributes.ModelDomain(0)]
                public class {{modelName}}<T> : TransformerNERBase<T>
                {
                    public {{modelName}}(AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<T> architecture,
                        AiDotNet.NER.Options.TransformerNEROptions? options = null) { }
                }
            }
            """;
        return GenerateFixture(modelName + "Tests", modelSource);
    }

    private static ClassDeclarationSyntax GenerateFixture(string name, string models = Models)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
            .Where(assembly => assembly.GetName().Name is string assemblyName
                && (assemblyName == "mscorlib" || assemblyName == "netstandard"
                    || assemblyName == "System" || assemblyName.StartsWith("System.", StringComparison.Ordinal)))
            .Select(assembly => assembly.Location).Distinct(StringComparer.Ordinal)
            .Select(path => MetadataReference.CreateFromFile(path)).ToImmutableArray<MetadataReference>();
        CSharpCompilation compilation = CSharpCompilation.Create("AiDotNetTests",
            new[] { CSharpSyntaxTree.ParseText(models) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.DoesNotContain(compilation.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);

        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.TestScaffoldGenerator());
        driver = driver.RunGenerators(compilation);
        GeneratorDriverRunResult result = driver.GetRunResult();
        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.All(result.Results, generator => Assert.Null(generator.Exception));
        ClassDeclarationSyntax fixture = Assert.Single(result.GeneratedTrees
            .SelectMany(tree => tree.GetRoot().DescendantNodes().OfType<ClassDeclarationSyntax>()),
            declaration => declaration.Identifier.ValueText == name);
        Assert.DoesNotContain(fixture.SyntaxTree.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        return fixture;
    }

    private static ObjectCreationExpressionSyntax ModelConstructor(ClassDeclarationSyntax fixture, string modelName)
    {
        MethodDeclarationSyntax factory = Assert.Single(fixture.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "CreateNetwork");
        return Assert.Single(factory.DescendantNodes().OfType<ObjectCreationExpressionSyntax>(),
            creation => creation.Type.DescendantNodesAndSelf().OfType<GenericNameSyntax>()
                .Any(type => type.Identifier.ValueText == modelName));
    }

    private static void AssertNamedInteger(ObjectCreationExpressionSyntax creation, string name, int expected)
    {
        Assert.NotNull(creation.ArgumentList);
        ArgumentSyntax argument = Assert.Single(creation.ArgumentList.Arguments,
            candidate => candidate.NameColon?.Name.Identifier.ValueText == name);
        Assert.Equal(expected, Assert.IsType<LiteralExpressionSyntax>(argument.Expression).Token.Value);
    }

    private static void AssertAssignedInteger(ObjectCreationExpressionSyntax creation, string name, int expected)
    {
        AssignmentExpressionSyntax assignment = Assert.Single(creation.DescendantNodes()
            .OfType<AssignmentExpressionSyntax>(), candidate => candidate.Left.ToString() == name);
        Assert.Equal(expected, Assert.IsType<LiteralExpressionSyntax>(assignment.Right).Token.Value);
    }

    private static void AssertShape(ClassDeclarationSyntax fixture, string name, params int[] expected)
    {
        PropertyDeclarationSyntax property = Assert.Single(fixture.Members.OfType<PropertyDeclarationSyntax>(),
            candidate => candidate.Identifier.ValueText == name);
        ImplicitArrayCreationExpressionSyntax shape = Assert.Single(property.DescendantNodes()
            .OfType<ImplicitArrayCreationExpressionSyntax>());
        Assert.Equal(expected, shape.Initializer.Expressions.Select(expression =>
            Assert.IsType<int>(Assert.IsType<LiteralExpressionSyntax>(expression).Token.Value)));
    }
}
