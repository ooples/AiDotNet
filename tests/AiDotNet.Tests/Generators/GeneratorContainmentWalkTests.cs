using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Pins the containing-type walks in <c>LayerStateGenerator.Analyze</c> (a layer the generated
/// factory table cannot name is declined) and in <c>YamlConfigSourceGenerator</c>'s
/// <c>IsEffectivelyPublicForGeneratedCode</c> / <c>HasOnlyResolvableTypeParametersForRegistry</c>
/// (a [YamlConfigurable] type the registry cannot name is not registered).
/// </summary>
/// <remarks>
/// Those walks were rewritten from <c>for (x = type; x is not null; x = x.ContainingType)</c> to a
/// do/while so CodeQL's nullness analysis stops reporting the already-dereferenced start symbol as
/// always null (cs/dereferenced-value-is-always-null). The rewrite must keep visiting the type itself
/// AND every containing type; these tests fail if either end of the chain is dropped.
/// </remarks>
public class GeneratorContainmentWalkTests
{
    private const string LayerInfrastructure = @"
namespace AiDotNet.Attributes
{
    [System.AttributeUsage(System.AttributeTargets.Parameter)]
    public sealed class LayerStateAttribute : System.Attribute { public string? Key { get; set; } }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T> { }
}";

    private const string YamlInfrastructure = @"
namespace AiDotNet.Configuration
{
    [System.AttributeUsage(System.AttributeTargets.Class | System.AttributeTargets.Interface)]
    public sealed class YamlConfigurableAttribute : System.Attribute
    {
        public YamlConfigurableAttribute(string sectionName) { }
    }
}
namespace AiDotNet
{
    public class AiModelBuilder
    {
        public AiModelBuilder ConfigureSeed(int seed) => this;
    }
}";

    private static ImmutableArray<MetadataReference> References()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location)
                || !seen.Add(assembly.Location)) continue;
            references.Add(MetadataReference.CreateFromFile(assembly.Location));
        }
        return references.ToImmutableArray();
    }

    private static (string Generated, ImmutableArray<Diagnostic> Diagnostics) Run(
        IIncrementalGenerator generator, string infrastructure, string source)
    {
        var compilation = CSharpCompilation.Create(
            "GeneratorContainmentWalk",
            new[] { CSharpSyntaxTree.ParseText(infrastructure), CSharpSyntaxTree.ParseText(source) },
            References(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(generator);
        driver = driver.RunGenerators(compilation);
        var result = driver.GetRunResult();
        return (string.Join("\n", result.GeneratedTrees.Select(t => t.GetText().ToString())), result.Diagnostics);
    }

    private const string LayerSource = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;

namespace Walk
{
    public partial class TopLevelLayer<T> : LayerBase<T>
    {
        private readonly int _units;
        public TopLevelLayer([LayerState] int units) { _units = units; }
    }

    public partial class PublicOuter
    {
        public partial class PublicNestedLayer<T> : LayerBase<T>
        {
            private readonly int _units;
            public PublicNestedLayer([LayerState] int units) { _units = units; }
        }

        // The type ITSELF is unreachable: the first link of the walk.
        private partial class PrivateNestedLayer<T> : LayerBase<T>
        {
            private readonly int _units;
            public PrivateNestedLayer([LayerState] int units) { _units = units; }
        }

        private protected partial class PrivateProtectedNestedLayer<T> : LayerBase<T>
        {
            private readonly int _units;
            public PrivateProtectedNestedLayer([LayerState] int units) { _units = units; }
        }

        // The type is public but a CONTAINING type is not: only a walk that continues past the
        // first link declines it.
        private partial class PrivateMiddle
        {
            public partial class PublicLayerInsidePrivateMiddle<T> : LayerBase<T>
            {
                private readonly int _units;
                public PublicLayerInsidePrivateMiddle([LayerState] int units) { _units = units; }
            }
        }
    }
}";

    [Fact]
    public void LayerStateGenerator_EmitsFactoriesOnlyForLayersNameableFromTheFactoryTable()
    {
        var (generated, diagnostics) = Run(new AiDotNet.Generators.LayerStateGenerator(), LayerInfrastructure, LayerSource);

        Assert.DoesNotContain(diagnostics, d => d.Severity == DiagnosticSeverity.Error);

        // Reachable layers, including one nested in a public type, get a factory entry.
        Assert.Contains("TopLevelLayer<>", generated, StringComparison.Ordinal);
        Assert.Contains("PublicOuter.PublicNestedLayer<>", generated, StringComparison.Ordinal);

        // Unreachable ones are declined silently: the type itself private/private protected, or a
        // public type inside a private containing type.
        Assert.DoesNotContain("PrivateNestedLayer", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("PrivateProtectedNestedLayer", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("PublicLayerInsidePrivateMiddle", generated, StringComparison.Ordinal);
    }

    private const string YamlSource = @"
using AiDotNet.Configuration;

namespace Walk
{
    [YamlConfigurable(""TopLevelSection"")]
    public class TopLevelConcrete { }

    public class PublicOuter
    {
        [YamlConfigurable(""PublicNestedSection"")]
        public class PublicNestedConcrete { }
    }

    internal class InternalOuter
    {
        // Public itself, unreachable through its containing type.
        [YamlConfigurable(""HiddenByOuterSection"")]
        public class HiddenByOuterConcrete { }
    }

    public class OuterWithForeignTypeParameter<TKey>
    {
        // The registry can only close T / TInput / TOutput; TKey lives on the CONTAINING type.
        [YamlConfigurable(""ForeignTypeParameterSection"")]
        public class ForeignTypeParameterConcrete { }
    }

    [YamlConfigurable(""OwnForeignTypeParameterSection"")]
    public class OwnForeignTypeParameterConcrete<TKey> { }
}";

    [Fact]
    public void YamlConfigSourceGenerator_RegistersOnlyTypesTheRegistryCanName()
    {
        var (generated, diagnostics) = Run(new AiDotNet.Generators.YamlConfigSourceGenerator(), YamlInfrastructure, YamlSource);

        Assert.DoesNotContain(diagnostics, d => d.Severity == DiagnosticSeverity.Error);

        Assert.Contains("Walk.TopLevelConcrete", generated, StringComparison.Ordinal);
        Assert.Contains("Walk.PublicOuter.PublicNestedConcrete", generated, StringComparison.Ordinal);

        // IsEffectivelyPublicForGeneratedCode must check the containing chain, not just the type.
        Assert.DoesNotContain("HiddenByOuterConcrete", generated, StringComparison.Ordinal);

        // HasOnlyResolvableTypeParametersForRegistry must collect the type's own type parameters
        // AND those of every containing type.
        Assert.DoesNotContain("ForeignTypeParameterConcrete", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("OwnForeignTypeParameterConcrete", generated, StringComparison.Ordinal);
    }
}
