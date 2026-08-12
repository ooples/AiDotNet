using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks the generated value-domain and model-input contracts.</summary>
public class TensorPortContractGeneratorTests
{
    private const string Infrastructure = @"
using System;
using System.Collections.Generic;
namespace AiDotNet.NeuralNetworks.Layers
{
    public enum LayerInputDomainKind { Unspecified, Continuous, IntegerIndices, BooleanMask }
    public enum TensorPortRole { Unspecified, Features, TokenIds, PositionIds, TokenTypeIds, Mask, EncoderInput, EncoderMemory, DecoderIds, AudioCodes, Output }
    public sealed class LayerInputDomain
    {
        public static LayerInputDomain Unspecified { get; } = new LayerInputDomain();
        public static LayerInputDomain Continuous { get; } = new LayerInputDomain();
        public static LayerInputDomain BooleanMask { get; } = new LayerInputDomain();
        public static LayerInputDomain Indices(int maximum) => new LayerInputDomain();
    }
    public sealed class LayerPort
    {
        public LayerPort(string name, int[] shape, bool required, LayerInputDomain domain, TensorPortRole role) { }
    }
    public abstract class LayerBase<T>
    {
        public virtual IReadOnlyList<LayerPort> InputPorts => Array.Empty<LayerPort>();
        public virtual IReadOnlyList<LayerPort> OutputPorts => Array.Empty<LayerPort>();
        public virtual LayerInputDomain GetInputDomain(int[] inputShape) => LayerInputDomain.Continuous;
        public virtual bool PropagatesInputDomain => false;
        public int[] GetInputShape() => new[] { 1 };
        public int[] GetOutputShape() => new[] { 1 };
    }
}
namespace AiDotNet.NeuralNetworks
{
    using AiDotNet.NeuralNetworks.Layers;
    public readonly struct ModelInputShapeConstraint
    {
        public ModelInputShapeConstraint(int minimumRank, int minimumElements, int exactRank = 0) { }
    }
    public abstract class NeuralNetworkBase<T>
    {
        protected List<LayerBase<T>> Layers { get; } = new List<LayerBase<T>>();
        public virtual LayerInputDomain GetInputDomain(int[] inputShape) => LayerInputDomain.Continuous;
        public virtual ModelInputShapeConstraint GetInputShapeConstraint() => new ModelInputShapeConstraint();
    }
}
namespace AiDotNet.Attributes
{
    using AiDotNet.NeuralNetworks.Layers;
    public enum TensorPortDirection { Input, Output }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class TensorPortAttribute : Attribute
    {
        public TensorPortAttribute(string name, TensorPortDirection direction, LayerInputDomainKind domain = LayerInputDomainKind.Continuous) { }
        public TensorPortRole Role { get; set; }
        public bool Required { get; set; } = true;
        public string MaxExclusiveMember { get; set; }
        public string MaxExclusiveResolver { get; set; }
        public string ShapeMember { get; set; }
        public bool PropagatesInputDomain { get; set; }
    }
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class RankRoutedInputDomainAttribute : Attribute
    {
        public RankRoutedInputDomainAttribute(int maximumRank, int layerIndex) { }
    }
    [AttributeUsage(AttributeTargets.Class)]
    public sealed class ModelInputShapeConstraintAttribute : Attribute
    {
        public int ExactRank { get; set; }
        public int MinimumRank { get; set; }
        public int MinimumElementCount { get; set; }
        public string MinimumElementCountMember { get; set; }
    }
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class ValidateSequentialLayerDomainsAttribute : Attribute { }
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

    private static (ImmutableArray<Diagnostic> Diagnostics, string Generated) Run(string source)
    {
        var compilation = CSharpCompilation.Create(
            "TensorPortContracts",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.TensorPortContractGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        var result = driver.GetRunResult();
        string generated = string.Join("\n", result.GeneratedTrees.Select(tree => tree.GetText().ToString()));
        return (diagnostics, generated);
    }

    [Fact]
    public async Task FlatContinuousToIndexFactory_IsCompilerError()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""output"", TensorPortDirection.Output, LayerInputDomainKind.Continuous)]
public partial class Dense<T> : LayerBase<T> { }
[TensorPort(""ids"", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, MaxExclusiveMember = ""_size"")]
public partial class Lookup<T> : LayerBase<T> { private int _size = 10; }
public static class Factory
{
    [ValidateSequentialLayerDomains]
    public static IEnumerable<LayerBase<T>> Bad<T>()
    {
        yield return new Dense<T>();
        yield return new Lookup<T>();
    }
}";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT004"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("generated composite/named branch", diagnostic.GetMessage());
    }

    [Fact]
    public async Task BranchAwareComposite_DoesNotCreateFalseSequentialEdge()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""ids"", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices, MaxExclusiveMember = ""_size"")]
[TensorPort(""output"", TensorPortDirection.Output, LayerInputDomainKind.Continuous)]
public partial class Composite<T> : LayerBase<T> { private int _size = 10; }
public static class Factory
{
    [ValidateSequentialLayerDomains]
    public static IEnumerable<LayerBase<T>> Valid<T>() { yield return new Composite<T>(); }
}";

        Assert.DoesNotContain(Run(source).Diagnostics, item => item.Id == "ADNPORT004");
    }

    [Fact]
    public async Task PortAndModelConstraint_AreGeneratedFromDeclarations()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""token_ids"", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Role = TensorPortRole.TokenIds, MaxExclusiveMember = ""_vocabularySize"")]
public partial class Lookup<T> : LayerBase<T> { private int _vocabularySize = 10; }
[ModelInputShapeConstraint(MinimumRank = 2, MinimumElementCountMember = ""MinimumSize"", ExactRank = 2)]
public partial class Model<T> : NeuralNetworkBase<T> { private int MinimumSize() => 4096; }
";

        var run = Run(source);
        Assert.Contains("override global::System.Collections.Generic.IReadOnlyList", run.Generated);
        Assert.Contains("LayerInputDomain.Indices(_vocabularySize)", run.Generated);
        Assert.Contains("GetInputShapeConstraint() => new(2, MinimumSize(), 2)", run.Generated);
    }

    [Fact]
    public async Task MissingConstraintMember_IsCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
[ModelInputShapeConstraint(MinimumElementCountMember = ""Missing"")]
public partial class Model<T> : NeuralNetworkBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT005"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task GeneratedContractOnNonPartialType_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""input"", TensorPortDirection.Input, LayerInputDomainKind.Continuous)]
public class Layer<T> : LayerBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT006"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("must be declared partial", diagnostic.GetMessage());
    }

    [Fact]
    public async Task ContradictoryShapeConstraint_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
[ModelInputShapeConstraint(MinimumRank = 3, ExactRank = 2)]
public partial class Model<T> : NeuralNetworkBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT007"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("MinimumRank 3 exceeds ExactRank 2", diagnostic.GetMessage());
        Assert.Contains("Correct the attribute values", diagnostic.GetMessage());
    }

    [Fact]
    public async Task InvalidRankRoute_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
[RankRoutedInputDomain(2, -1)]
public partial class Model<T> : NeuralNetworkBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT007"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("LayerIndex is -1", diagnostic.GetMessage());
        Assert.Contains("Correct the attribute values", diagnostic.GetMessage());
    }
}
