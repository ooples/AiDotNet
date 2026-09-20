using System;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Checks the effective sequence fixtures, including rules that would otherwise be shadowed.</summary>
public sealed class GeneratedSequenceFixtureContractTests
{
    public enum SequenceFixture
    {
        XLSTMLanguageModel,
        GriffinLanguageModel,
        HawkLanguageModel,
        GLALanguageModel,
        GatedDeltaNetLanguageModel,
        RecurrentGemmaLanguageModel,
    }

    [Theory]
    [InlineData(SequenceFixture.XLSTMLanguageModel, 64, 128, 4)]
    [InlineData(SequenceFixture.GriffinLanguageModel, 128, 32, 128)]
    [InlineData(SequenceFixture.HawkLanguageModel, 128, 32, 128)]
    [InlineData(SequenceFixture.GLALanguageModel, 128, 32, 128)]
    [InlineData(SequenceFixture.GatedDeltaNetLanguageModel, 128, 32, 128)]
    [InlineData(SequenceFixture.RecurrentGemmaLanguageModel, 256, 128, 4)]
    public void EffectiveFixturePreservesItsBoundedOptions(
        SequenceFixture model, int vocabulary, int context, int outputSize)
    {
        ClassDeclarationSyntax fixture = GenerateFixture(model);
        MethodDeclarationSyntax factory = Assert.Single(fixture.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "CreateNetwork");
        ObjectCreationExpressionSyntax options = Assert.Single(factory.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(), node => node.Initializer != null);
        AssertAssignment(options, "VocabSize", vocabulary);
        AssertAssignment(options, "ModelDimension", 32);
        AssertAssignment(options, "NumLayers", 1);
        AssertAssignment(options, "MaxSequenceLength", context);
        if (model is SequenceFixture.GriffinLanguageModel or SequenceFixture.HawkLanguageModel)
        {
            AssertAssignment(options, "RecurrenceDimension", 40);
            Assert.DoesNotContain(options.Initializer?.Expressions ?? default,
                expression => expression is AssignmentExpressionSyntax assignment
                    && assignment.Left.ToString() == "NumHeads");
        }
        else if (model != SequenceFixture.RecurrentGemmaLanguageModel)
        {
            AssertAssignment(options, "NumHeads", 4);
        }

        ObjectCreationExpressionSyntax architecture = Assert.Single(factory.DescendantNodes()
            .OfType<ObjectCreationExpressionSyntax>(),
            node => node.Type.ToString().Contains("NeuralNetworkArchitecture"));
        Assert.NotNull(architecture.ArgumentList);
        ArgumentSyntax output = Assert.Single(architecture.ArgumentList.Arguments,
            argument => argument.NameColon?.Name.Identifier.ValueText == "outputSize");
        Assert.Equal(outputSize, Assert.IsType<LiteralExpressionSyntax>(output.Expression).Token.Value);
    }

    [Theory]
    [InlineData(SequenceFixture.XLSTMLanguageModel)]
    [InlineData(SequenceFixture.GriffinLanguageModel)]
    [InlineData(SequenceFixture.HawkLanguageModel)]
    [InlineData(SequenceFixture.GLALanguageModel)]
    [InlineData(SequenceFixture.GatedDeltaNetLanguageModel)]
    public void SpecializedFixtureHasOneEffectiveConstructorRule(SequenceFixture model)
    {
        SyntaxNode generator = ReadGenerator();
        Assert.Single(generator.DescendantNodes().OfType<IfStatementSyntax>(), branch =>
            branch.Condition.DescendantNodesAndSelf().OfType<MemberAccessExpressionSyntax>()
                .Any(access => access.Name.Identifier.ValueText == "ClassName")
            && branch.Condition.DescendantNodesAndSelf().OfType<LiteralExpressionSyntax>()
                .Any(literal => Equals(literal.Token.Value, model.ToString()))
            && branch.Statement is BlockSyntax block
            && block.Statements.OfType<ExpressionStatementSyntax>().Any(statement =>
                statement.Expression is AssignmentExpressionSyntax assignment
                && assignment.Left.ToString() == "constructorExpr"));
    }

    [Theory]
    [InlineData(SequenceFixture.GriffinLanguageModel)]
    [InlineData(SequenceFixture.HawkLanguageModel)]
    public void SpecializedFixtureHasNoUnreachableScaleFallback(SequenceFixture model)
    {
        VariableDeclaratorSyntax scale = Assert.Single(ReadGenerator().DescendantNodes()
            .OfType<VariableDeclaratorSyntax>(), variable => variable.Identifier.ValueText == "scaleArgs");
        Assert.DoesNotContain(scale.DescendantNodes().OfType<SwitchExpressionArmSyntax>(), arm =>
            arm.Pattern is ConstantPatternSyntax pattern
            && pattern.Expression is LiteralExpressionSyntax literal
            && Equals(literal.Token.Value, model.ToString()));
    }

    private static SyntaxNode ReadGenerator()
    {
        // Embedded by both test projects: inspect the actual generator source, not a copied rule.
        using Stream stream = typeof(GeneratedSequenceFixtureContractTests).Assembly
            .GetManifestResourceStream("Review.TestScaffoldGenerator.cs")
            ?? throw new InvalidOperationException("The generator source contract resource is missing.");
        using var reader = new StreamReader(stream);
        return CSharpSyntaxTree.ParseText(reader.ReadToEnd()).GetRoot();
    }

    private static ClassDeclarationSyntax GenerateFixture(SequenceFixture model)
    {
        string models = """
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
            }
            """;
        // The enum is the closed fixture policy; its names become compiler symbols only here.
        models += $"namespace AiDotNet.NeuralNetworks {{ [AiDotNet.Attributes.ModelDomain(0)] "
            + $"public class {model}<T> : NeuralNetworkBase<T> {{ "
            + $"public {model}(NeuralNetworkArchitecture<T> architecture) {{ }} }} }}";
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
            .Where(assembly => assembly.GetName().Name is string name
                && (name == "mscorlib" || name == "netstandard" || name == "System"
                    || name.StartsWith("System.", StringComparison.Ordinal)))
            .Select(assembly => assembly.Location).Distinct(StringComparer.Ordinal)
            .Select(path => MetadataReference.CreateFromFile(path)).ToImmutableArray<MetadataReference>();
        var compilation = CSharpCompilation.Create("AiDotNetTests",
            new[] { CSharpSyntaxTree.ParseText(models) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.DoesNotContain(compilation.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.TestScaffoldGenerator());
        var result = driver.RunGenerators(compilation).GetRunResult();
        Assert.DoesNotContain(result.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.All(result.Results, generator => Assert.Null(generator.Exception));
        var fixture = Assert.Single(result.GeneratedTrees.SelectMany(tree => tree.GetRoot()
            .DescendantNodes().OfType<ClassDeclarationSyntax>()),
            declaration => declaration.Identifier.ValueText == $"{model}Tests");
        Assert.DoesNotContain(fixture.SyntaxTree.GetDiagnostics(), diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        return fixture;
    }

    private static void AssertAssignment(ObjectCreationExpressionSyntax options, string property, int expected)
    {
        Assert.NotNull(options.Initializer);
        AssignmentExpressionSyntax assignment = Assert.Single(options.Initializer.Expressions
            .OfType<AssignmentExpressionSyntax>(), expression => expression.Left.ToString() == property);
        Assert.Equal(expected, Assert.IsType<LiteralExpressionSyntax>(assignment.Right).Token.Value);
    }
}
