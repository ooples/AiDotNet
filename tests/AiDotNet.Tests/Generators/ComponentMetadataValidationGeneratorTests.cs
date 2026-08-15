using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks the compile-time gradient-contract validation consumed by generated layer tests.</summary>
public class ComponentMetadataValidationGeneratorTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class LayerPropertyAttribute : Attribute
    {
        public bool IsTrainable { get; set; } = true;
        public bool SupportsBackpropagation { get; set; } = true;
        public bool UsesSurrogateGradient { get; set; }
        public bool TrainsViaCustomLoss { get; set; }
    }
    [AttributeUsage(AttributeTargets.Class)] public sealed class LayerCategoryAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Class)] public sealed class LayerTaskAttribute : Attribute { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T> { }
}";

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

    private static ImmutableArray<Diagnostic> Run(string layerSource)
    {
        var compilation = CSharpCompilation.Create(
            "AiDotNet",
            new[]
            {
                CSharpSyntaxTree.ParseText(Infrastructure),
                CSharpSyntaxTree.ParseText(layerSource)
            },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.ComponentMetadataValidationGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        return diagnostics;
    }

    [Fact]
    public async Task CustomLoss_WithGenericBackpropContract_IsCompileError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
[LayerProperty(TrainsViaCustomLoss = true), LayerCategory, LayerTask]
public sealed class InvalidLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double> { }";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN086"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task CustomLoss_WithExplicitNonGenericBackpropContract_IsValid()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
[LayerProperty(TrainsViaCustomLoss = true, SupportsBackpropagation = false), LayerCategory, LayerTask]
public sealed class ValidLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double> { }";

        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN086");
    }
}
