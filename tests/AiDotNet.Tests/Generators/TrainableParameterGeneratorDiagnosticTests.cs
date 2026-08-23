using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public class TrainableParameterGeneratorDiagnosticTests
{
    [Fact]
    public void NonPartialDiagnostic_RehydratesSourceLocationWithoutCachingSyntax()
    {
        const string source = @"
using System;
namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class TrainableParameterAttribute : Attribute { }
}

#pragma warning disable AIDN099
public sealed class NonPartialLayer
{
    [AiDotNet.Attributes.TrainableParameter]
    private double weight;
}
#pragma warning restore AIDN099
";

        var diagnostics = Run(source);
        var diagnostic = Assert.Single(diagnostics.Where(item => item.Id == "AIDN099"));
        var sourceTree = Assert.IsAssignableFrom<SyntaxTree>(diagnostic.Location.SourceTree);
        string sourceText = sourceTree.GetText().ToString();
        int disable = sourceText.IndexOf("#pragma warning disable AIDN099", StringComparison.Ordinal);
        int restore = sourceText.IndexOf("#pragma warning restore AIDN099", StringComparison.Ordinal);

        Assert.Contains("NonPartialLayer", diagnostic.GetMessage());
        Assert.Contains("weight", diagnostic.GetMessage());
        Assert.InRange(diagnostic.Location.SourceSpan.Start, disable, restore);
    }

    private static ImmutableArray<Diagnostic> Run(string source)
    {
        var compilation = CSharpCompilation.Create(
            "TrainableParameterDiagnosticRegression",
            new[] { CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.TrainableParameterGenerator());
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
