using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Proves both B1 generators consume declarations rather than tensor conventions.</summary>
public class ParameterGeneratorSemanticTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Class)] public sealed class AutoParametersAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class TrainableParameterAttribute : Attribute
    {
        public int Role { get; set; }
        public int Order { get; set; }
        public bool Optional { get; set; }
        public string? Condition { get; set; }
        public string? Shape { get; set; }
        public int Availability { get; set; }
    }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class FittedParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class FrozenParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class BufferAttribute : Attribute
    {
        public string? Name { get; set; }
        public int Role { get; set; }
        public int Availability { get; set; }
    }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ScratchAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ExternalStateAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ParameterAliasAttribute : Attribute
    {
        public ParameterAliasAttribute(string target) { }
    }
}
namespace AiDotNet.Tensors.LinearAlgebra
{
    public class Tensor<T> { public int Length => 1; }
    public class Matrix<T> { }
    public class Vector<T> { }
}
namespace AiDotNet.Interfaces
{
    public interface ILayer<T> { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T>
    {
        protected void RegisterTrainableParameter(
            AiDotNet.Tensors.LinearAlgebra.Tensor<T> tensor,
            AiDotNet.Tensors.Engines.PersistentTensorRole role) { }
    }
}
namespace AiDotNet.NeuralNetworks
{
    using System.Collections.Generic;
    using AiDotNet.Interfaces;
    using AiDotNet.NeuralNetworks.Layers;
    using AiDotNet.Tensors.LinearAlgebra;

    public abstract class NeuralNetworkBase<T>
    {
        public List<ILayer<T>> Layers { get; } = new();
        protected virtual IEnumerable<Tensor<T>> GetExtraTrainableTensors() => new List<Tensor<T>>();
        protected virtual IEnumerable<LayerBase<T>?> GetExtraTrainableLayers() => new List<LayerBase<T>?>();
        protected virtual void RebindLayerAliases(
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers) { }
        protected static TLayer? RebindLayerAlias<TLayer>(
            TLayer? alias,
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers,
            string memberName) where TLayer : class, ILayer<T> => alias;
        protected static TLayer RebindRequiredLayerAlias<TLayer>(
            TLayer alias,
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers,
            string memberName) where TLayer : class, ILayer<T> => alias;
        protected static void RebindLayerAliasCollection<TLayer>(
            IEnumerable<TLayer>? aliases,
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers,
            string memberName) where TLayer : class, ILayer<T> { }
        protected static void ValidateReadonlyLayerAlias<TLayer>(
            TLayer? alias,
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers,
            string memberName) where TLayer : class, ILayer<T> { }
    }
}
namespace AiDotNet.Tensors.Engines
{
    public enum PersistentTensorRole { Weights, Biases }
}
namespace AiDotNet.Models
{
    public abstract class ModelBase<T, TInput, TOutput>
    {
        protected void RegisterParameterComponent(object value) { }
        protected virtual void RegisterComponents() { }
        protected virtual void RegisterGeneratedParameterComponents(object registry) { }
    }
}";

    private static ImmutableArray<MetadataReference> BaseReferences()
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

    private static string Run(IIncrementalGenerator generator, string source)
    {
        var compilation = CSharpCompilation.Create(
            "GeneratorContract",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(generator);
        driver = driver.RunGenerators(compilation);
        return string.Join("\n", driver.GetRunResult().GeneratedTrees.Select(tree => tree.GetText().ToString()));
    }

    private static ImmutableArray<Diagnostic> RunDiagnostics(IIncrementalGenerator generator, string source)
    {
        var compilation = CSharpCompilation.Create(
            "GeneratorContract",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(generator);
        driver = driver.RunGenerators(compilation);
        return driver.GetRunResult().Diagnostics;
    }

    [Fact]
    public async Task LayerGenerator_AutoParametersDoesNotPromotePlainTensor()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
[AutoParameters]
public partial class CacheLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _lastInput = new();
}";

        Assert.DoesNotContain("_lastInput", Run(new AiDotNet.Generators.TrainableParameterGenerator(), source));
    }

    [Fact]
    public async Task LayerGenerator_ProvesMigratedStatelessLayerIsParameterFree()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
[AutoParameters]
public partial class StatelessLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [Scratch] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _lastInput = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("IsDeclaredParameterFree => true", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_DoesNotDeclareInheritedParameterGraphFree()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public abstract class StatefulAdapterBase<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    protected AiDotNet.Interfaces.ILayer<T> _child = default!;
}
[AutoParameters]
public partial class DerivedAdapter<T> : StatefulAdapterBase<T>
{
    private readonly System.Collections.Generic.Dictionary<string, AiDotNet.Interfaces.ILayer<T>>
        _taskAdapters = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.DoesNotContain("IsDeclaredParameterFree => true", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_DoesNotDeclareInheritedRuntimeRegistryFree()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public abstract class RuntimeRegisteredHeadBase<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    protected void AddProjection()
    {
        var weight = new AiDotNet.Tensors.LinearAlgebra.Tensor<T>();
        RegisterTrainableParameter(
            weight, AiDotNet.Tensors.Engines.PersistentTensorRole.Weights);
    }
}
[AutoParameters]
public partial class GaussianHead<T> : RuntimeRegisteredHeadBase<T>
{
    private readonly AiDotNet.Tensors.LinearAlgebra.Tensor<T> _mean = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.DoesNotContain("IsDeclaredParameterFree => true", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_DeclaresChildStructureForAllocationFreeManifest()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class Child<T> : AiDotNet.Interfaces.ILayer<T> { }
[AutoParameters]
public partial class CompositeLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private Child<T> _child = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("HasDeclaredSubLayerStructure => true", generated, StringComparison.Ordinal);
        Assert.Contains("EnsureSubLayersRegistered", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_BoundAdaptiveAxisSeparatesValidationFromManifestSizing()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class AdaptiveLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private int _inputSize = 4;
    private int _outputSize = 8;
    [TrainableParameter(Shape = ""*(_inputSize), _outputSize"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weights = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("ShapeOf(-2, _outputSize)", generated, StringComparison.Ordinal);
        Assert.Contains("DeclaredParameterCountShapes", generated, StringComparison.Ordinal);
        Assert.Contains("ShapeOf(_inputSize, _outputSize)", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_EmitsOnlyDeclaredTrainableAndPersistentRoles()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class DeclaredLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [TrainableParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weight = new();
    [FittedParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _fitted = new();
    [FrozenParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _frozen = new();
    [Buffer] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _buffer = new();
    [Scratch] private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _cache = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("_weight", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.LearnedState", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.Frozen", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.Buffer", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("_cache", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_MergesAttributesAndRegistrationsAcrossPartialDeclarations()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class MixedLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _declared = new();
}
public partial class MixedLayer<T>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _registered = new();
    private void Configure() => RegisterTrainableParameter(
        _registered, AiDotNet.Tensors.Engines.PersistentTensorRole.Biases);
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("_declared", generated, StringComparison.Ordinal);
        Assert.Contains("_registered", generated, StringComparison.Ordinal);
        // Fixed generated surfaces use one cached backing array rather than allocating an inline
        // array on every read. Verify that partial declarations still merge into that stable view
        // in declaration/registration order.
        Assert.Contains("__storage[0] = _declared;", generated, StringComparison.Ordinal);
        Assert.Contains("__storage[1] = _registered;", generated, StringComparison.Ordinal);
        Assert.True(
            generated.IndexOf("__storage[0] = _declared;", StringComparison.Ordinal)
            < generated.IndexOf("__storage[1] = _registered;", StringComparison.Ordinal));
    }

    [Fact]
    public async Task LayerGenerator_FlattensMutableTensorCollectionsDeterministically()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
using System.Collections.Generic;
public partial class CollectionLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T>[] _ordered = new AiDotNet.Tensors.LinearAlgebra.Tensor<T>[0];

    [TrainableParameter]
    private Dictionary<string, AiDotNet.Tensors.LinearAlgebra.Tensor<T>> _keyed = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("ParameterCollectionOrdering.PresentNonNull(_ordered)", generated, StringComparison.Ordinal);
        Assert.Contains("_ordered[__slot] = parameters[__i++]", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterCollectionOrdering.OrderedValues(_keyed)", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterCollectionOrdering.OrderedKeys(_keyed)", generated, StringComparison.Ordinal);
        Assert.Contains("AppendTrainableParameter(__parameter", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelGenerator_DoesNotPromoteUnclassifiedTensor()
    {
        await Task.Yield();
        const string source = @"
public partial class CacheModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _lastOutput = new();
}";

        Assert.DoesNotContain("_lastOutput", Run(new AiDotNet.Generators.ModelParameterGenerator(), source));
    }

    [Fact]
    public async Task ModelGenerator_EmitsDeclaredRoleAndAvailability()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class DeclaredModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [TrainableParameter(Availability = 1)]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weight = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("_weight", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.Trainable", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterAvailability)1", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelGenerator_FitProducedVectorUsesOneTimeReplacingRestoreSource()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class DeferredBufferModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [Buffer(Availability = 2)]
    private AiDotNet.Tensors.LinearAlgebra.Vector<T> _history = new();
    [FittedParameter]
    private T _rho = default!;
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains(
            "new VectorFieldParameterSource<T>(() => _history, value => _history = value)",
            generated,
            StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.Buffer", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterAvailability)2", generated, StringComparison.Ordinal);
        Assert.Contains(
            "new ScalarParameterSource<T>(() => _rho, value => _rho = value)",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelGenerator_FittedObjectGraphUsesGeneratedSerializedStateSource()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
public partial class TreeModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private sealed class Node { public double Value { get; set; } }

    [FittedParameter]
    private List<Node>? _trees;
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("new SerializedObjectParameterSource<T>(() => _trees", generated,
            StringComparison.Ordinal);
        Assert.Contains("ParameterSlotRole.LearnedState", generated, StringComparison.Ordinal);
        Assert.Contains("ParameterAvailability.Fit", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelGenerator_EmitsTypeSafeCanonicalLayerAliasRebinding()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class AliasNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
    private AiDotNet.Interfaces.ILayer<T>? _head;
    private AiDotNet.Interfaces.ILayer<T> _required = null!;
    private readonly List<AiDotNet.Interfaces.ILayer<T>> _encoder = new();
    private readonly AiDotNet.Interfaces.ILayer<T>? _readonlyAlias;
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("protected override void RebindLayerAliases(", generated, StringComparison.Ordinal);
        Assert.Contains(
            "_head = RebindLayerAlias(_head, previousLayers, replacementLayers, nameof(_head));",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "_required = RebindRequiredLayerAlias(_required, previousLayers, replacementLayers, nameof(_required));",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "RebindLayerAliasCollection(_encoder, previousLayers, replacementLayers, nameof(_encoder));",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "ValidateReadonlyLayerAlias(_readonlyAlias, previousLayers, replacementLayers, nameof(_readonlyAlias));",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelGenerator_NestedNetworksUseReadinessAwareLayerAndTensorTraversal()
    {
        await Task.Yield();
        const string source = @"
public partial class ChildNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
}
public partial class CompositeNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
    private readonly ChildNetwork<T> _child = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("EnumerateNestedNetworkLayers(_child)", generated, StringComparison.Ordinal);
        Assert.Contains("EnumerateNestedNetworkTensors(_child)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("_child?.Layers", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_DuplicateBufferIdentity_IsCompilerError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class InvalidLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [Buffer(Name = ""running_state"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _first = new();
    [Buffer(Name = ""running_state"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _second = new();
}";

        var diagnostic = Assert.Single(RunDiagnostics(
            new AiDotNet.Generators.TrainableParameterGenerator(),
            source).Where(item => item.Id == "ADNBUF001"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("running_state", diagnostic.GetMessage(), StringComparison.Ordinal);
    }
    [Fact]
    public async Task ModelGenerator_AbstractBaseEmitsResizableFitTensorSource()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public abstract partial class DeferredTensorModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [TrainableParameter(Availability = 2)]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weights = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("partial class DeferredTensorModel<T>", generated, StringComparison.Ordinal);
        Assert.Contains(
            "new ResizableTensorFieldParameterSource<T>(() => _weights, value => _weights = value)",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_ConditionControlsEveryGeneratedParameterSurface()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class ConditionalLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    public bool Affine { get; }
    [TrainableParameter(Condition = nameof(Affine), Shape = ""4"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _gamma = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("if (Affine)", generated, StringComparison.Ordinal);
        Assert.Contains("(Affine) && (_gamma.Length > 0)", generated, StringComparison.Ordinal);
        Assert.Contains("if (Affine) __withAllOptional++;", generated, StringComparison.Ordinal);
        Assert.Contains("HasActiveDeclaredParameterShapes => (Affine)", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_ConditionComposesWithGeneratedCollectionOwnership()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class ConditionalExperts<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    public bool UseExperts { get; }
    [TrainableParameter(Condition = nameof(UseExperts))]
    private readonly AiDotNet.Tensors.LinearAlgebra.Tensor<T>[] _experts =
        System.Array.Empty<AiDotNet.Tensors.LinearAlgebra.Tensor<T>>();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("if (UseExperts)", generated, StringComparison.Ordinal);
        Assert.Contains(
            "ParameterCollectionOrdering.PresentNonNull(_experts)",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "if (UseExperts) foreach (var __parameter in global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull(_experts)) __expected++;",
            generated,
            StringComparison.Ordinal);
    }
}
