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
    public enum LayerInputDomainKind { Unspecified, Continuous, IntegerIndices, BooleanMask, AdditiveMask, Deferred, Custom }
    public enum TensorPortRole { Unspecified, Features, TokenIds, PositionIds, TokenTypeIds, Mask, EncoderInput, EncoderMemory, DecoderIds, AudioCodes, Output }
    public enum TensorPortSource { External, Derived, Defaulted, Internal }
    public sealed class PortShapeConstraint
    {
        public static PortShapeConstraint None { get; } = new PortShapeConstraint();
        public int ExactRank { get; set; }
        public int MinimumRank { get; set; }
        public int MaximumRank { get; set; }
        public int MinimumElementCount { get; set; }
        public string SameShapeAs { get; set; }
        public IReadOnlyList<int> MinimumAxisSizes { get; set; }
        public IReadOnlyList<int> AxisDivisors { get; set; }
    }
    public sealed class LayerInputDomain
    {
        public static LayerInputDomain Unspecified { get; } = new LayerInputDomain();
        public static LayerInputDomain Continuous { get; } = new LayerInputDomain();
        public static LayerInputDomain BooleanMask { get; } = new LayerInputDomain();
        public static LayerInputDomain AdditiveMask { get; } = new LayerInputDomain();
        public static LayerInputDomain Deferred(string reason) => new LayerInputDomain();
        public static LayerInputDomain Indices(int maximum) => new LayerInputDomain();
        public static LayerInputDomain Custom(string key) => new LayerInputDomain();
    }
    public sealed class LayerPort
    {
        public LayerPort(string name, int[] shape, bool required, LayerInputDomain domain,
            TensorPortRole role, string stableId = null, TensorPortSource source = TensorPortSource.External,
            string variant = ""default"", PortShapeConstraint shapeConstraint = null) { }
    }
    public abstract class LayerBase<T>
    {
        public virtual IReadOnlyList<LayerPort> InputPorts => Array.Empty<LayerPort>();
        public virtual IReadOnlyList<LayerPort> OutputPorts => Array.Empty<LayerPort>();
        public virtual LayerInputDomain GetInputDomain(int[] inputShape) => LayerInputDomain.Continuous;
        public virtual bool PropagatesInputDomain => false;
        public int[] GetInputShape() => new[] { 1 };
        public int[] GetOutputShape() => new[] { 1 };
        public AiDotNet.Tensors.LinearAlgebra.Tensor<T> Forward(
            IReadOnlyDictionary<string, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> inputs) => null;
        protected virtual AiDotNet.Tensors.LinearAlgebra.Tensor<T> ForwardTraced(
            AiDotNet.Tensors.LinearAlgebra.Tensor<T> input) => input;
        protected virtual AiDotNet.Tensors.LinearAlgebra.Tensor<T> ForwardTracedPorts(
            IReadOnlyDictionary<string, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> inputs) => null;
    }
}
namespace AiDotNet.Tensors.LinearAlgebra
{
    public sealed class Tensor<T> { }
}
namespace AiDotNet.NeuralNetworks
{
    using AiDotNet.NeuralNetworks.Layers;
    public readonly struct ModelInputShapeConstraint
    {
        public ModelInputShapeConstraint(int minimumRank, int minimumElements, int exactRank = 0,
            int maximumRank = 0, IReadOnlyList<int> minimumAxes = null,
            IReadOnlyList<int> divisors = null) { }
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
        public string CustomProviderKey { get; set; }
        public string ShapeMember { get; set; }
        public bool PropagatesInputDomain { get; set; }
        public string StableId { get; set; }
        public TensorPortSource Source { get; set; }
        public string Variant { get; set; } = ""default"";
        public int ExactRank { get; set; }
        public int MinimumRank { get; set; }
        public int MaximumRank { get; set; }
        public int MinimumElementCount { get; set; }
        public string SameShapeAs { get; set; }
        public int[] MinimumAxisSizes { get; set; }
        public int[] AxisDivisors { get; set; }
    }
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class GenerateInputContractAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Parameter)]
    public sealed class TensorInputAttribute : Attribute
    {
        public TensorInputAttribute(LayerInputDomainKind domain = LayerInputDomainKind.Continuous) { }
        public string Name { get; set; }
        public TensorPortRole Role { get; set; }
        public string MaxExclusiveMember { get; set; }
        public string MaxExclusiveResolver { get; set; }
        public string CustomProviderKey { get; set; }
        public TensorPortSource Source { get; set; }
        public string Variant { get; set; }
        public int ExactRank { get; set; }
        public int MinimumRank { get; set; }
        public int MaximumRank { get; set; }
        public int MinimumElementCount { get; set; }
        public string SameShapeAs { get; set; }
        public int[] MinimumAxisSizes { get; set; }
        public int[] AxisDivisors { get; set; }
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
        public int MaximumRank { get; set; }
        public int MinimumElementCount { get; set; }
        public string MinimumElementCountMember { get; set; }
        public int[] MinimumAxisSizes { get; set; }
        public int[] AxisDivisors { get; set; }
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
        Assert.Contains("GetInputShapeConstraint() => new(2, MinimumSize(), 2, 0", run.Generated);
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

    [Fact]
    public async Task AnnotatedForwardCore_GeneratesBeginnerDefaultAndBridge()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
public partial class Projection<T> : LayerBase<T>
{
    [GenerateInputContract]
    private Tensor<T> Compute(Tensor<T> features) => features;
}";

        var run = Run(source);

        Assert.DoesNotContain(run.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.Contains("LayerInputDomain.Continuous", run.Generated);
        Assert.Contains("ForwardTraced", run.Generated);
        Assert.Contains("=> Compute(input)", run.Generated);
    }

    [Fact]
    public async Task AnnotatedMultiInputForward_GeneratesTypedFacadeAndSemanticPorts()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
public partial class TokenMixer<T> : LayerBase<T>
{
    private int _vocabularySize = 128;
    [GenerateInputContract]
    private Tensor<T> Compute(
        [TensorInput(LayerInputDomainKind.IntegerIndices, Name = ""token_ids"",
            Role = TensorPortRole.TokenIds, MaxExclusiveMember = ""_vocabularySize"")] Tensor<T> ids,
        [TensorInput(LayerInputDomainKind.BooleanMask, Name = ""mask"", Role = TensorPortRole.Mask)] Tensor<T> mask)
        => ids;
}";

        var run = Run(source);

        Assert.DoesNotContain(run.Diagnostics, diagnostic => diagnostic.Severity == DiagnosticSeverity.Error);
        Assert.Contains("public readonly struct Inputs", run.Generated);
        Assert.Contains("TokenIds { get; }", run.Generated);
        Assert.Contains("LayerInputDomain.BooleanMask", run.Generated);
        Assert.Contains("ForwardTracedPorts", run.Generated);
    }

    [Fact]
    public async Task RangeMemberWithWrongType_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""ids"", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    MaxExclusiveMember = ""_size"")]
public partial class Lookup<T> : LayerBase<T> { private string _size = ""ten""; }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT008"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("int field/property", diagnostic.GetMessage());
    }

    [Fact]
    public async Task AlternativeVariants_MayReuseAStablePortName()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""input"", TensorPortDirection.Input, Variant = ""features"")]
[TensorPort(""input"", TensorPortDirection.Input, LayerInputDomainKind.IntegerIndices,
    Variant = ""tokens"", MaxExclusiveMember = ""_size"")]
public partial class Either<T> : LayerBase<T> { private int _size = 10; }";

        Assert.DoesNotContain(Run(source).Diagnostics, item => item.Id is "ADNPORT003" or "ADNPORT010");
    }

    [Fact]
    public async Task GeneratedForwardWithWrongReturnType_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
public partial class Invalid<T> : LayerBase<T>
{
    [GenerateInputContract]
    private int Compute(Tensor<T> input) => 0;
}";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT009"));
        Assert.Contains("return Tensor<T>", diagnostic.GetMessage());
    }

    [Fact]
    public async Task GeneratedDefaultedPortWithoutNullDefault_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
public partial class Invalid<T> : LayerBase<T>
{
    [GenerateInputContract]
    private Tensor<T> Compute(
        Tensor<T> input,
        [TensorInput(Source = TensorPortSource.Defaulted)] Tensor<T> context) => input;
}";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT009"));
        Assert.Contains("must declare '= null'", diagnostic.GetMessage());
    }

    [Fact]
    public async Task MissingSameShapePort_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""mask"", TensorPortDirection.Input, SameShapeAs = ""missing"")]
public partial class Invalid<T> : LayerBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT007"));
        Assert.Contains("does not exist in variant", diagnostic.GetMessage());
    }

    [Fact]
    public async Task ModelGeometryMemberWithWrongType_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks;
[ModelInputShapeConstraint(MinimumElementCountMember = ""MinimumSize"")]
public partial class Invalid<T> : NeuralNetworkBase<T>
{
    private string MinimumSize => ""large"";
}";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT008"));
        Assert.Contains("int field/property", diagnostic.GetMessage());
    }

    [Fact]
    public async Task CustomDomain_GeneratesExplicitProviderBinding()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""input"", TensorPortDirection.Input, LayerInputDomainKind.Custom,
    CustomProviderKey = ""sparse-probability"")]
public partial class SparseProbability<T> : LayerBase<T> { }";

        var run = Run(source);

        Assert.DoesNotContain(run.Diagnostics, item => item.Severity == DiagnosticSeverity.Error);
        Assert.Contains("LayerInputDomain.Custom(\"sparse-probability\")", run.Generated);
    }

    [Fact]
    public async Task CustomDomainWithoutProvider_IsFriendlyCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.NeuralNetworks.Layers;
[TensorPort(""input"", TensorPortDirection.Input, LayerInputDomainKind.Custom)]
public partial class Invalid<T> : LayerBase<T> { }";

        var diagnostic = Assert.Single(Run(source).Diagnostics.Where(item => item.Id == "ADNPORT007"));
        Assert.Contains("without a CustomProviderKey", diagnostic.GetMessage());
    }
}
