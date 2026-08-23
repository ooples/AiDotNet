using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public class CompatibilityMatrixGeneratorSuppressionTests
{
    private const string Infrastructure = @"
using System;
namespace AiDotNet.Enums
{
    public enum ModelCategory { NeuralNetwork = 0, SVM = 27 }
}
namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ModelCategoryAttribute : Attribute
    {
        public ModelCategoryAttribute(AiDotNet.Enums.ModelCategory category) { }
    }
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class ModelMetadataExemptAttribute : Attribute { }
}
namespace AiDotNet.Interfaces
{
    public interface IFullModel<T> { }
}";

    [Fact]
    public void ConflictDiagnostic_IsReportedWithoutPragma()
    {
        var diagnostics = Run(CreateSource(suppress: false));

        Assert.Single(diagnostics.Where(diagnostic => diagnostic.Id == "AIDN030"));
    }

    [Fact]
    public void ConflictDiagnostic_RehydratesPragmaSuppressibleSourceLocation()
    {
        var diagnostics = Run(CreateSource(suppress: true));

        // GeneratorDriver exposes raw diagnostics before the compiler host filters
        // pragma suppressions. Verify the important contract directly: the diagnostic
        // points into the original syntax tree at a position where AIDN030 is disabled.
        // An external Location (the regression this guards) has no SourceTree and cannot
        // participate in either #pragma or SuppressMessage processing.
        var diagnostic = Assert.Single(diagnostics.Where(item => item.Id == "AIDN030"));
        var sourceTree = Assert.IsAssignableFrom<SyntaxTree>(diagnostic.Location.SourceTree);
        string source = sourceTree.GetText().ToString();
        int disable = source.IndexOf("#pragma warning disable AIDN030", StringComparison.Ordinal);
        int restore = source.IndexOf("#pragma warning restore AIDN030", StringComparison.Ordinal);
        Assert.True(disable >= 0);
        Assert.True(restore > disable);
        Assert.InRange(diagnostic.Location.SourceSpan.Start, disable, restore);
    }

    private static string CreateSource(bool suppress)
    {
        string pragmaBefore = suppress ? "#pragma warning disable AIDN030" : string.Empty;
        string pragmaAfter = suppress ? "#pragma warning restore AIDN030" : string.Empty;
        return $@"
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
{pragmaBefore}
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory((ModelCategory)999)]
public sealed class ConflictingModel : IFullModel<double> {{ }}
{pragmaAfter}";
    }

    private static ImmutableArray<Diagnostic> Run(string source)
    {
        var compilation = CSharpCompilation.Create(
            "CompatibilitySuppressionRegression",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.CompatibilityMatrixGenerator());
        driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        return diagnostics;
    }

    private static ImmutableArray<MetadataReference> BaseReferences()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location))
                continue;
            references.Add(MetadataReference.CreateFromFile(assembly.Location));
        }

        return references.ToImmutableArray();
    }
}
