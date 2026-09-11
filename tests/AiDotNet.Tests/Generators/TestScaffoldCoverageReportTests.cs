using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Regression tests for the model test-coverage report (#2091).
///
/// The report published a coverage figure from the AiDotNet compilation, where no test class
/// exists: a source generator's syntax provider only observes the compilation it runs in, and the
/// model-family tests live in AiDotNetTests. It therefore reported 2 of 1816 models covered
/// (0.1%), and both "covered" entries were models rather than tests - NeuralStressTest matched
/// itself, and TEST matched the tail of NeuralStressTest under a case-insensitive substring search.
///
/// The dependency direction makes measuring it there impossible rather than merely unimplemented:
/// AiDotNetTests references AiDotNet, never the reverse. So the fix is to publish the number only
/// from the compilation that can measure it, and to say so plainly in the one that cannot.
/// </summary>
public class TestScaffoldCoverageReportTests
{
    // The generator resolves these attribute symbols from the compilation before doing anything,
    // so a synthetic compilation has to declare them or nothing is emitted at all.
    private const string Infrastructure = @"
using System;
namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ModelDomainAttribute : Attribute { public ModelDomainAttribute(int domain) { } }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ModelCategoryAttribute : Attribute { public ModelCategoryAttribute(int category) { } }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ModelTaskAttribute : Attribute { public ModelTaskAttribute(int task) { } }
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class ModelMetadataExemptAttribute : Attribute { }
}
namespace AiDotNet.Interfaces
{
    public interface IFullModel<T, TInput, TOutput> { }
}";

    private const string ModelSource = @"
namespace Probe
{
    // Discovery requires the full-model interface, not just metadata attributes. Deliberately
    // use scalar I/O with no supported family so no generated scaffold can supply real coverage.
    public abstract class Base : AiDotNet.Interfaces.IFullModel<float, float, float> { }

    // A MODEL whose name ends in ""Test"". This is the shape that produced the phantom coverage:
    // IsTestCandidate admits it to the test-name set purely on its name. The attributes are what
    // make the generator track this full model - without the interface and attributes the census
    // is empty and any assertion about the report's contents passes vacuously.
    [AiDotNet.Attributes.ModelDomain(0)]
    [AiDotNet.Attributes.ModelCategory(0)]
    public class WidgetTest : Base { }

    // A second model whose name is a case-insensitive suffix of the first, which is how the model
    // TEST was scored as covered by NeuralStressTest.
    [AiDotNet.Attributes.ModelDomain(0)]
    [AiDotNet.Attributes.ModelCategory(0)]
    public class TEST : Base { }
}";

    [Fact]
    public void NonTestCompilation_ReportsCoverageAsUnmeasurable()
    {
        var report = Run("AiDotNet");

        Assert.Contains("IsMeasurable = false", report);
        Assert.Contains("CoveragePercent = -1.0", report);
        Assert.Contains("TestedCount = -1", report);
    }

    [Fact]
    public void NonTestCompilation_AttributesCoverageToNoModelAtAll()
    {
        var report = Run("AiDotNet");

        // The defect was a POPULATED TestedModelNames in a compilation containing no tests. The
        // old emitter always wrote `new string[] { ... }` and filled it from a match that a model
        // could satisfy against itself; the unmeasurable report hands back a fixed empty list, so
        // there is no array for a phantom name to appear in.
        Assert.Contains("TestedModelNames { get; } = Array.Empty<string>()", report);
        Assert.DoesNotContain("TestedModelNames { get; } = new string[]", report);
    }

    [Fact]
    public void TestCompilation_WithModelsButNoTests_PublishesMeasuredZero()
    {
        var report = Run("AiDotNetTests");

        Assert.Contains("IsMeasurable = true", report);
        Assert.DoesNotContain("CoveragePercent = -1.0", report);

        // The test compilation contains two models but no actual test classes. Measurability
        // alone must not let WidgetTest vouch for itself or its TEST suffix vouch for that model.
        Assert.Contains("TotalModels = 2", report);
        Assert.Contains("TestedCount = 0", report);
        Assert.Contains("UntestedCount = 2", report);

        var reportSyntax = CSharpSyntaxTree.ParseText(report).GetRoot();
        var testedNames = Assert.Single(
            reportSyntax.DescendantNodes().OfType<PropertyDeclarationSyntax>(),
            property => property.Identifier.ValueText == "TestedModelNames");
        var initializer = Assert.IsType<EqualsValueClauseSyntax>(testedNames.Initializer);
        var names = Assert.IsType<ArrayCreationExpressionSyntax>(initializer.Value);
        Assert.Empty(Assert.IsType<InitializerExpressionSyntax>(names.Initializer).Expressions);
    }

    // The assembly names are exact on purpose: RegisterSourceOutput refuses to run this generator
    // for any compilation not named "AiDotNet" or "AiDotNetTests", so that repository-only fixtures
    // never leak into a PackageReference consumer's build. Those are therefore the only two names
    // whose behaviour is worth pinning.
    private static string Run(string assemblyName)
    {
        var compilation = CSharpCompilation.Create(
            assemblyName,
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(ModelSource) },
            new[] { MetadataReference.CreateFromFile(typeof(object).Assembly.Location) },
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        Assert.Empty(compilation.GetDiagnostics().Where(diagnostic => diagnostic.Severity == DiagnosticSeverity.Error));

        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.TestScaffoldGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out _);

        var result = driver.GetRunResult();
        var source = result.GeneratedTrees
            .FirstOrDefault(tree => tree.FilePath.EndsWith("TestCoverage.g.cs", System.StringComparison.Ordinal));

        if (source is null)
        {
            Assert.Fail($"TestScaffoldGenerator emitted no TestCoverage.g.cs for assembly '{assemblyName}'.");
        }

        return source.GetText().ToString();
    }
}
