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
        Assert.Contains("new Tensor<T>[] { _declared, _registered }", generated, StringComparison.Ordinal);
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
}
