using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
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
        public string? LowPrecisionBacking { get; set; }
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
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class TensorLayoutAttribute : Attribute
    {
        public TensorLayoutAttribute(params AiDotNet.Enums.TensorAxis[] axes) { }
        public AiDotNet.Enums.TensorLayoutDirection Direction { get; set; }
        public bool BatchOptional { get; set; }
    }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ParameterAliasAttribute : Attribute
    {
        public ParameterAliasAttribute(string target) { }
    }
}
namespace AiDotNet.Enums
{
    public enum TensorAxis { Batch, Channels, Depth, Height, Width, Features }
    public enum TensorLayoutDirection { Input, Output }
}
namespace AiDotNet.Tensors.LinearAlgebra
{
    public class Tensor<T> : AiDotNet.Interfaces.IParameterSource<T> { public int Length => 1; }
    public class Matrix<T> : AiDotNet.Interfaces.IParameterSource<T> { }
    public class Vector<T> : AiDotNet.Interfaces.IParameterSource<T> { }
}
namespace AiDotNet.Interfaces
{
    public interface IParameterSource<T> { }
    public interface IModelSerializer { }
    public interface ILayer<T> { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T>
    {
        protected AiDotNet.Tensors.LinearAlgebra.Vector<T> Parameters = new();
        public virtual AiDotNet.Tensors.LinearAlgebra.Vector<T> GetParameters() => Parameters;
        protected virtual bool LegacyParametersAreDerivedSnapshot => false;
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
        protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state) { }
        protected virtual IEnumerable<Tensor<T>> GetExtraTrainableTensors() => new List<Tensor<T>>();
        protected virtual IEnumerable<LayerBase<T>?> GetExtraTrainableLayers() => new List<LayerBase<T>?>();
        protected virtual void RebindLayerAliases(
            IReadOnlyList<ILayer<T>> previousLayers,
            IReadOnlyList<ILayer<T>> replacementLayers) { }
        protected virtual void CopyGeneratedLayerAliasesTo(NeuralNetworkBase<T> destination) { }
        protected virtual void CopyGeneratedTrainableTensorsTo(NeuralNetworkBase<T> destination) { }
        protected static Tensor<T>? CloneGeneratedTrainableTensor(Tensor<T>? source) => source;
        protected static Tensor<T> CloneRequiredGeneratedTrainableTensor(Tensor<T> source) => source;
        protected static void CopyGeneratedTrainableTensorValues(
            Tensor<T>? source, Tensor<T>? destination, string memberName) { }
        protected static Vector<T>? CloneGeneratedTrainableVector(Vector<T>? source) => source;
        protected static Vector<T> CloneRequiredGeneratedTrainableVector(Vector<T> source) => source;
        protected static void CopyGeneratedTrainableVectorValues(
            Vector<T>? source, Vector<T>? destination, string memberName) { }
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
        protected static TLayer? CopyLayerAlias<TLayer>(
            TLayer? sourceAlias,
            TLayer? destinationAlias,
            IReadOnlyList<ILayer<T>> sourceLayers,
            IReadOnlyList<ILayer<T>> destinationLayers,
            string memberName) where TLayer : class, ILayer<T> => destinationAlias;
        protected static TLayer CopyRequiredLayerAlias<TLayer>(
            TLayer sourceAlias,
            TLayer destinationAlias,
            IReadOnlyList<ILayer<T>> sourceLayers,
            IReadOnlyList<ILayer<T>> destinationLayers,
            string memberName) where TLayer : class, ILayer<T> => destinationAlias;
        protected static void CopyLayerAliasCollection<TLayer>(
            IEnumerable<TLayer>? sourceAliases,
            IEnumerable<TLayer>? destinationAliases,
            IReadOnlyList<ILayer<T>> sourceLayers,
            IReadOnlyList<ILayer<T>> destinationLayers,
            string memberName) where TLayer : class, ILayer<T> { }
        protected static void ValidateCopiedReadonlyLayerAlias<TLayer>(
            TLayer? sourceAlias,
            TLayer? destinationAlias,
            IReadOnlyList<ILayer<T>> sourceLayers,
            IReadOnlyList<ILayer<T>> destinationLayers,
            string memberName) where TLayer : class, ILayer<T> { }
    }
}
namespace AiDotNet.Tensors.Engines
{
    public enum PersistentTensorRole { Weights, Biases }
}
namespace AiDotNet.Models
{
    public sealed class ModelStateRegistry<T>
    {
        public void DeclareBoolean(string name, System.Func<bool> get, System.Action<bool> set) { }
    }
    public abstract class ModelBase<T, TInput, TOutput>
    {
        protected void RegisterParameterComponent(object value) { }
        protected virtual void RegisterComponents() { }
        protected virtual void RegisterGeneratedParameterComponents(object registry) { }
        protected virtual void RegisterGeneratedState(ModelStateRegistry<T> state) { }
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

    private static void AssertGeneratedExecutableMembersAreMarked(string generated, string generatorName)
    {
        SyntaxNode root = CSharpSyntaxTree.ParseText(generated).GetRoot();
        var members = root.DescendantNodes()
            .OfType<MemberDeclarationSyntax>()
            .Where(member => member is MethodDeclarationSyntax or PropertyDeclarationSyntax)
            .ToList();

        Assert.NotEmpty(members);
        foreach (MemberDeclarationSyntax member in members)
        {
            Assert.Contains(
                $"GeneratedCode(\"AiDotNet.Generators.{generatorName}\"",
                member.AttributeLists.ToFullString(),
                StringComparison.Ordinal);
        }
    }

    [Fact]
    public void LayerGenerator_MarksAllGeneratedExecutableMembersAsGeneratedCode()
    {
        const string source = @"
using AiDotNet.Attributes;
[AutoParameters]
public partial class GeneratedCoverageLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [TrainableParameter(Shape = ""4, 4"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weight = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        AssertGeneratedExecutableMembersAreMarked(generated, "TrainableParameterGenerator");
    }

    [Fact]
    public void ModelGenerator_MarksAllGeneratedExecutableMembersAsGeneratedCode()
    {
        const string source = @"
using AiDotNet.Attributes;
public partial class GeneratedCoverageModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weight = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        AssertGeneratedExecutableMembersAreMarked(generated, "ModelParameterGenerator");
    }

    [Fact]
    public void ModelGenerator_SurfacesTrainableVectorWithoutConcreteOverride()
    {
        const string source = @"
using AiDotNet.Attributes;
public partial class VectorBackedNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Vector<T> _bias = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("new Tensor<T>([_bias.Length], _bias)", generated, StringComparison.Ordinal);
        Assert.Contains(
            "__destination._bias = CloneRequiredGeneratedTrainableVector(_bias);",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public void ClonePlanGenerator_MarksItsGeneratedRegistryAsGeneratedCode()
    {
        const string source = @"
public class GeneratedCoverageClone<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    public int Width { get; set; }
}
public class SecondGeneratedCoverageClone<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    public int Height { get; set; }
}";

        string generated = Run(new AiDotNet.Generators.ClonePlanGenerator(), source);
        Assert.Contains(
            "[global::System.CodeDom.Compiler.GeneratedCode(\"AiDotNet.Generators.ClonePlanGenerator\", \"1.0.0\")]\ninternal static class CloneRegistrations",
            generated.Replace("\r\n", "\n"),
            StringComparison.Ordinal);

        SyntaxNode root = CSharpSyntaxTree.ParseText(generated).GetRoot();
        ClassDeclarationSyntax registry = Assert.Single(
            root.DescendantNodes().OfType<ClassDeclarationSyntax>(),
            declaration => declaration.Identifier.ValueText == "CloneRegistrations");
        MethodDeclarationSyntax dispatcher = Assert.Single(
            registry.Members.OfType<MethodDeclarationSyntax>(),
            method => method.Identifier.ValueText == "RegisterAll");
        var registrationMethods = registry.Members
            .OfType<MethodDeclarationSyntax>()
            .Where(method => method.Identifier.ValueText.StartsWith("Register_", StringComparison.Ordinal))
            .ToList();

        Assert.Equal(2, registrationMethods.Count);
        Assert.DoesNotContain("new List<ClonePlanEntry>", dispatcher.Body!.ToFullString(), StringComparison.Ordinal);
        Assert.All(registrationMethods, method =>
            Assert.Contains("new List<ClonePlanEntry>", method.Body!.ToFullString(), StringComparison.Ordinal));
    }

    [Fact]
    public void ClonePlanGenerator_EmitsToolingSafeMethodBodies()
    {
        Type registry = typeof(AiDotNet.Models.CloneRegistry).Assembly.GetType(
            "AiDotNet.Generated.CloneRegistrations",
            throwOnError: true)!;
        var generatedMethods = registry
            .GetMethods(System.Reflection.BindingFlags.Static |
                        System.Reflection.BindingFlags.Public |
                        System.Reflection.BindingFlags.NonPublic |
                        System.Reflection.BindingFlags.DeclaredOnly)
            .Select(method => (method.Name, Size: method.GetMethodBody()?.GetILAsByteArray()?.Length ?? 0))
            .ToList();

        Assert.NotEmpty(generatedMethods);
        Assert.All(generatedMethods, method =>
            Assert.True(
                method.Size < 64 * 1024,
                $"Generated clone method {method.Name} is {method.Size:N0} bytes of IL; " +
                "large monolithic methods make coverage control-flow analysis pathological."));
    }

    [Fact]
    public void ClonePlanGenerator_UsesDirectConstructorAssignmentWhenMemberNameDiffers()
    {
        const string source = """
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<TInput, TOutput> { }
            }

            namespace Example
            {
                public sealed class WidthModel : AiDotNet.Interfaces.IFullModel<int, int>
                {
                    public int ImageSize { get; private set; }

                    public WidthModel(int imageWidth = 128)
                    {
                        ImageSize = imageWidth;
                    }
                }
            }
            """;

        string generated = Run(new AiDotNet.Generators.ClonePlanGenerator(), source);

        Assert.Contains("new[] { \"ImageSize\" }", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("new[] { \"=default\" }", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("Add(e, t, \"ImageSize\"", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ClonePlanGenerator_UsesEffectiveOptionalConfigurationStoredThroughCoalesce()
    {
        const string source = """
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<TInput, TOutput> { }
            }

            namespace Example
            {
                public sealed class Settings { }
                public sealed class ConfiguredModel : AiDotNet.Interfaces.IFullModel<int, int>
                {
                    private Settings Options { get; }
                    public ConfiguredModel(Settings? options = null)
                    {
                        Options = options ?? new Settings();
                    }
                }
            }
            """;

        string generated = Run(new AiDotNet.Generators.ClonePlanGenerator(), source);

        Assert.Contains("new[] { \"Options\" }", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("new[] { \"=default\" }", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ClonePlanGenerator_MapsNamedNestedArchitectureBeforeGenericParentArchitecture()
    {
        const string source = """
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<TInput, TOutput> { }
            }

            namespace Example
            {
                public sealed class Architecture { }
                public sealed class Network
                {
                    public Architecture Architecture { get; } = new Architecture();
                }

                public sealed class CompositeModel : AiDotNet.Interfaces.IFullModel<int, int>
                {
                    public Architecture Architecture { get; } = new Architecture();
                    public Network Generator { get; private set; } = new Network();
                    public Network Critic { get; private set; } = new Network();

                    public CompositeModel(
                        Architecture generatorArchitecture,
                        Architecture criticArchitecture)
                    {
                    }
                }
            }
            """;

        string generated = Run(new AiDotNet.Generators.ClonePlanGenerator(), source);

        Assert.Contains(
            "new[] { \"Generator.Architecture\", \"Critic.Architecture\" }",
            generated,
            StringComparison.Ordinal);
        Assert.DoesNotContain(
            "new[] { \"Architecture\", \"Architecture\" }",
            generated,
            StringComparison.Ordinal);
        Assert.DoesNotContain("Add(e, t, \"Generator\"", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("Add(e, t, \"Critic\"", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ClonePlanGenerator_DoesNotUseScratchGraphAsOptionalConfiguration()
    {
        const string source = """
            namespace AiDotNet.Interfaces
            {
                public interface IFullModel<TInput, TOutput> { }
            }

            namespace Example
            {
                public sealed class LazyGraphModel : AiDotNet.Interfaces.IFullModel<int, int>
                {
                    [AiDotNet.Attributes.Scratch]
                    private System.Collections.Generic.List<int> _layers = new();

                    public LazyGraphModel(System.Collections.Generic.List<int>? layers = null)
                    {
                        _layers = layers is null
                            ? new System.Collections.Generic.List<int>()
                            : new System.Collections.Generic.List<int>(layers);
                    }
                }
            }
            """;

        string generated = Run(new AiDotNet.Generators.ClonePlanGenerator(), source);

        Assert.Contains("new[] { \"=default\" }", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("new[] { \"_layers\" }", generated, StringComparison.Ordinal);
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
    public async Task LayerGenerator_MaterializesCountableDeclarationsForOptimizerDiscovery()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
[AutoParameters]
public partial class ChannelPinnedLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private int _channels = 4;
    [TrainableParameter(Shape = ""_channels, 3"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weights = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains(
            "OwnParameterReadiness == AiDotNet.Models.Parameters.ParameterReadiness.ShapeResolvedUnmaterialized",
            generated,
            StringComparison.Ordinal);
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
    public void LayerGenerator_MarksLegacyFlatParameterSnapshotsAsDerived()
    {
        const string source = @"
using AiDotNet.Attributes;
public sealed class Child<T> : AiDotNet.Interfaces.ILayer<T> { }
[AutoParameters]
public partial class LegacyComposite<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private Child<T> _child = new();

    public LegacyComposite()
    {
        Parameters = GetParameters();
    }
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);

        Assert.Contains("LegacyParametersAreDerivedSnapshot => true", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_ExcludesAliasedChildFromOwnedStructure()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class Child<T> : AiDotNet.Interfaces.ILayer<T> { }
[AutoParameters]
public partial class CompositeLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private Child<T> _owned = new();
    [ParameterAlias(nameof(_owned))] private Child<T> _alias;
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("RegisterSubLayer(_owned)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("RegisterSubLayer(_alias)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("DeclareParameterSubLayer(components, _alias", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_ExcludesNonOwningChildViewsFromOwnedStructure()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class Child<T> : AiDotNet.Interfaces.ILayer<T> { }
[AutoParameters]
public partial class CompositeLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private Child<T> _owned = new();
    [Scratch] private Child<T>[] _traversalView = [];
    [ExternalState] private Child<T>? _external;
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("RegisterSubLayer(_owned)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("RegisterSubLayer(_traversalView)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("RegisterSubLayer(_external)", generated, StringComparison.Ordinal);
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
    public async Task LayerGenerator_FiltersNonSemanticAxesBeforeSentinelAnalysis()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class SparseShapeLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private int _channels = -1;
    [TrainableParameter(Shape = ""*, , _channels / 2"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weights = new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);

        // Empty and fully-adaptive axes carry no sentinel source. The concrete expression must
        // still be walked and guarded before integer arithmetic can launder -1 into zero.
        Assert.Contains("if (_channels < 0) return", generated, StringComparison.Ordinal);
        Assert.Contains("ShapeOf(-2, _channels / 2)", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("if (*", generated, StringComparison.Ordinal);
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
        // array on every read. Runtime registration is the optimizer/tape order, so registrations
        // discovered in another partial declaration lead declaration-only attributed storage.
        Assert.Contains("__storage[0] = _registered;", generated, StringComparison.Ordinal);
        Assert.Contains("__storage[1] = _declared;", generated, StringComparison.Ordinal);
        Assert.True(
            generated.IndexOf("__storage[0] = _registered;", StringComparison.Ordinal)
            < generated.IndexOf("__storage[1] = _declared;", StringComparison.Ordinal));
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
    public async Task ModelGenerator_DoesNotDuplicateManuallyRegisteredMember()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class MigratingModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [FittedParameter]
    private AiDotNet.Tensors.LinearAlgebra.Vector<T> _fitted = new();

    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_fitted);
    }
}";

        Assert.DoesNotContain("_fitted", Run(new AiDotNet.Generators.ModelParameterGenerator(), source));
    }

    [Fact]
    public async Task ModelGenerator_StillDiscoversUnclassifiedNestedComponent()
    {
        await Task.Yield();
        const string source = @"
public sealed class Component<T> : AiDotNet.Interfaces.IParameterSource<T> { }
public partial class CompositeModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private Component<T> _child = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains("ComponentAccessorParameterSource<T>(() => _child)", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DoesNotPersistRegistryLifecycleLatch()
    {
        await Task.Yield();
        const string source = @"
public partial class LifecycleModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private bool _componentsRegistered;
    private bool _trained;
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.DoesNotContain("_componentsRegistered", generated, StringComparison.Ordinal);
        Assert.Contains("LifecycleModel._trained", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ModelStateGenerator_DoesNotPersistReconstructibleFeatureServicesAsFittedState()
    {
        const string source = @"
namespace AiDotNet.Interfaces
{
    public interface IAudioFeatureExtractor<T> { int FeatureDimension { get; } }
}
public sealed class FeatureExtractor<T> : AiDotNet.Interfaces.IAudioFeatureExtractor<T>
{
    public int FeatureDimension => 13;
}
public partial class AudioModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    public FeatureExtractor<T>? Extractor { get; protected set; } = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);

        Assert.DoesNotContain("AudioModel.Extractor", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_RestoresReadonlyCollectionsInPlace()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class OnlineModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private readonly List<T> _knownClasses = new();
    private readonly Dictionary<int, ClassStats> _stats = new();
    public long SamplesSeen { get; private set; }
    private sealed class ClassStats { public long Count { get; set; } }
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains(
            "state.DeclareObjectInPlace(\"OnlineModel._knownClasses\", () => _knownClasses);",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "state.DeclareObjectInPlace(\"OnlineModel._stats\", () => _stats);",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "state.DeclareInt64(\"OnlineModel.SamplesSeen\"",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DeclaresNestedObjectGraphsWithoutOverrides()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class ForestModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private Node? _root;
    private List<TreeRecord>? _trees;
    private double[][]? _boundaries;
    private sealed class Node { public Node? Left { get; set; } public double Value { get; set; } }
    private sealed class TreeRecord { public Node? Root { get; set; } }
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains("ForestModel._root", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareObject(\"ForestModel._trees\"", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareObject(\"ForestModel._boundaries\"", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("RegisterState", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DoesNotForceLegacyCollectionsIntoPartialMigration()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public class LegacyModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private readonly List<string> _configuration = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.DoesNotContain("LegacyModel", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DoesNotJsonSerializeUnreconstructableCollections()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class NetworkState<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private readonly List<AiDotNet.Interfaces.ILayer<T>> _layers = new();
    private Dictionary<AiDotNet.Tensors.LinearAlgebra.Tensor<T>, AiDotNet.Tensors.LinearAlgebra.Vector<T>>
        _gradients = new();
    private readonly List<Record> _records = new();
    private sealed class Record
    {
        public AiDotNet.Tensors.LinearAlgebra.Matrix<T> Covariance { get; set; } = new();
    }
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.DoesNotContain("_layers", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("_gradients", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("_records", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_RestoresReadonlyNumericCollectionsThroughBinaryState()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class NumericCollectionState<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private readonly List<AiDotNet.Tensors.LinearAlgebra.Vector<T>> _vectors = new();
    private readonly List<AiDotNet.Tensors.LinearAlgebra.Matrix<T>> _matrices = new();
    private readonly List<AiDotNet.Tensors.LinearAlgebra.Tensor<T>> _tensors = new();
    private readonly Dictionary<string, AiDotNet.Tensors.LinearAlgebra.Vector<T>> _byName = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains("state.DeclareInPlace(\"NumericCollectionState._vectors\"", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareInPlace(\"NumericCollectionState._matrices\"", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareInPlace(\"NumericCollectionState._tensors\"", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareInPlace(\"NumericCollectionState._byName\"", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("DeclareObjectInPlace", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_PersistsTrainableStorageOnLegacyStateOnlyTrunk()
    {
        await Task.Yield();
        const string source = @"
public abstract partial class LegacyStateBase<T>
{
    protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<T> state)
        => RegisterGeneratedStateCore(state);
}
public partial class LegacyTrainable<T> : LegacyStateBase<T>
{
    [AiDotNet.Attributes.TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Vector<T> _weights = new();
    [AiDotNet.Attributes.Buffer]
    private AiDotNet.Tensors.LinearAlgebra.Vector<byte>? _quantized;
    [AiDotNet.Attributes.Buffer]
    private AiDotNet.Tensors.LinearAlgebra.Vector<double>? _scales;
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains(
            "state.Declare(\"LegacyTrainable._weights\", () => _weights",
            generated,
            StringComparison.Ordinal);
        Assert.Contains("state.DeclareByteVector(\"LegacyTrainable._quantized\"", generated, StringComparison.Ordinal);
        Assert.Contains("state.DeclareDoubleVector(\"LegacyTrainable._scales\"", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_PreservesNativeDoublePrecisionAfterFlatVectorRestore()
    {
        await Task.Yield();
        const string source = @"
public partial class WideWorkingState<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    [AiDotNet.Attributes.TrainableParameter]
    private readonly double[] _weights = new double[4];
    [AiDotNet.Attributes.TrainableParameter]
    private readonly double[][] _matrix = new[] { new double[2] };
    [AiDotNet.Attributes.TrainableParameter]
    private double _bias;
    [AiDotNet.Attributes.Buffer]
    private readonly double[] _statistics = new double[2];
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains(
            "state.DeclareExactInPlace(\"WideWorkingState._weights\"",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "state.DeclareExactInPlace(\"WideWorkingState._matrix\"",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "state.DeclareExactDouble(\"WideWorkingState._bias\"",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "state.DeclareExactInPlace(\"WideWorkingState._statistics\"",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DeclaresRecursiveGraphListsWithoutSerializationHelpers()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public partial class ForestState<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private List<Node> _trees = new();
    private sealed class Node
    {
        public T Value { get; set; }
        public Node? Left { get; set; }
        public Node? Right { get; set; }
        public Node(T zero) { Value = zero; }
    }
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains("state.DeclareGraphList<global::ForestState<T>.Node>", generated, StringComparison.Ordinal);
        Assert.Contains("new global::ForestState<T>.Node(default!)", generated, StringComparison.Ordinal);
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
    private sealed class OwnedLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<T>, AiDotNet.Interfaces.ILayer<T> { }
    [AiDotNet.Attributes.TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T>? _runtimeWeight;
    private AiDotNet.Interfaces.ILayer<T>? _head;
    private AiDotNet.Interfaces.ILayer<T> _required = null!;
    private readonly List<AiDotNet.Interfaces.ILayer<T>> _encoder = new();
    private readonly List<OwnedLayer> _owned = new();
    private readonly AiDotNet.Interfaces.ILayer<T>? _readonlyAlias;
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);
        Assert.Contains(
            "protected override global::System.Collections.Generic.IEnumerable<global::AiDotNet.NeuralNetworks.Layers.LayerBase<T>?> GetExtraTrainableLayers()",
            generated,
            StringComparison.Ordinal);
        Assert.Contains("foreach (var __layer in _owned ??", generated, StringComparison.Ordinal);
        Assert.Contains("protected override global::System.Collections.Generic.IEnumerable<GeneratedAdditionalLayerGroup> GetGeneratedAdditionalLayerGroups()", generated,
            StringComparison.Ordinal);
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
        Assert.Contains("protected override void CopyGeneratedLayerAliasesTo(", generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "__destination._head = CopyLayerAlias(_head, __destination._head, Layers, __destination.Layers, nameof(_head));",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "CopyLayerAliasCollection(_encoder, __destination._encoder, Layers, __destination.Layers, nameof(_encoder));",
            generated,
            StringComparison.Ordinal);
        Assert.Contains("protected override void CopyGeneratedTrainableTensorsTo(", generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "__destination._runtimeWeight = CloneGeneratedTrainableTensor(_runtimeWeight);",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DoesNotPersistCanonicalNetworkLayerViewsTwice()
    {
        await Task.Yield();
        const string source = @"
using System.Collections.Generic;
public sealed class ConcreteLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T> { }
public partial class StateNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
    private ConcreteLayer<T>? _head;
    private readonly List<ConcreteLayer<T>> _blocks = new();
    private ConcreteLayer<T>[] _stages = System.Array.Empty<ConcreteLayer<T>>();
    private bool _trained;
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains("StateNetwork._trained", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("StateNetwork._head", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("StateNetwork._blocks", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("StateNetwork._stages", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("DeclareLayerList", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("DeclareParameterSource", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_DoesNotPersistRegisteredParameterChildTwice()
    {
        await Task.Yield();
        const string source = @"
public sealed class SerializableChild : AiDotNet.Interfaces.IModelSerializer { }
public partial class CompositeModel<T> : AiDotNet.Models.ModelBase<T, object, object>
{
    private SerializableChild _child = new();
    private bool _trained;

    protected override void RegisterComponents()
    {
        RegisterParameterComponent(_child);
    }
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains("CompositeModel._trained", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("CompositeModel._child", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("DeclareChild", generated, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ModelStateGenerator_ImplementsCommonAbstractSerializationSurface()
    {
        await Task.Yield();
        const string source = @"
public abstract partial class GeneratedSerializationBase<T>
    : AiDotNet.Models.ModelBase<T, object, object>
{
    public abstract byte[] Serialize();
    public abstract void Deserialize(byte[] data);
    protected byte[] SerializeGeneratedModelState() => System.Array.Empty<byte>();
    protected void DeserializeGeneratedModelState(byte[] data) { }
}
public partial class GeneratedSerializationModel<T> : GeneratedSerializationBase<T>
{
    private bool _trained;
}";

        string generated = Run(new AiDotNet.Generators.ModelStateGenerator(), source);
        Assert.Contains(
            "public override byte[] Serialize() => SerializeGeneratedModelState();",
            generated,
            StringComparison.Ordinal);
        Assert.Contains(
            "public override void Deserialize(byte[] data) => DeserializeGeneratedModelState(data);",
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
        Assert.Contains("GetGeneratedNestedNetworkLayerViews", generated, StringComparison.Ordinal);
        Assert.Contains("RebindNestedNetworkCanonicalLayerAliases(_child", generated, StringComparison.Ordinal);
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
        Assert.Contains(
            "if (Affine) DeclareTrainableParameter(components, _gamma);",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerGenerator_LowPrecisionBackingFlowsThroughLogicalParameterSurfaces()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public partial class ResidentLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    [TrainableParameter(
        Shape = ""4, 4"",
        LowPrecisionBacking = nameof(_weightHalf))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weight = new();
    private AiDotNet.Tensors.LinearAlgebra.Tensor<System.Half>? _weightHalf;
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains(
            "DeclareTrainableParameter(components, _weight, _weightHalf);",
            generated,
            StringComparison.Ordinal);
        Assert.Contains("_weightHalf = null;", generated, StringComparison.Ordinal);
        Assert.Contains("MarkTrainableParametersRebound();", generated, StringComparison.Ordinal);
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
        Assert.Contains(
            $"if (UseExperts){System.Environment.NewLine}            foreach (var __componentTensor in global::AiDotNet.Models.Parameters.ParameterCollectionOrdering.PresentNonNull(_experts))",
            generated,
            StringComparison.Ordinal);
    }

    [Fact]
    public void LayerGenerator_InfersInputDepthFromDeclaredChannelAxis()
    {
        const string source = @"
using AiDotNet.Attributes;
using AiDotNet.Enums;
[TensorLayout(TensorAxis.Batch, TensorAxis.Height, TensorAxis.Width, TensorAxis.Channels,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Height, TensorAxis.Width, TensorAxis.Channels,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ChannelsLastLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
{
    private int _inputDepth = -1;
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> _weights = new();

    private void Allocate()
    {
        _weights = Create([_inputDepth, 3]);
    }

    private AiDotNet.Tensors.LinearAlgebra.Tensor<T> Create(int[] shape) => new();
}";

        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);

        Assert.Contains("InputShape[InputShape.Length - 1], 3", generated, StringComparison.Ordinal);
        Assert.DoesNotContain("InputShape[0], 3", generated, StringComparison.Ordinal);
    }

    [Fact]
    public void ModelGenerator_DiscoversConventionalNestedLayerOwners()
    {
        const string source = @"
using System.Collections.Generic;
internal sealed class OwnedLayer<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>, AiDotNet.Interfaces.ILayer<T> { }
internal sealed class LayerBlock<T>
{
    private readonly OwnedLayer<T> _layer = new();
    internal IEnumerable<OwnedLayer<T>> EnumerateLayers() { yield return _layer; }
}
public partial class EncapsulatedNetwork<T> : AiDotNet.NeuralNetworks.NeuralNetworkBase<T>
{
    private readonly LayerBlock<T> _stem = new();
    private readonly List<LayerBlock<T>> _stages = new();
}";

        string generated = Run(new AiDotNet.Generators.ModelParameterGenerator(), source);

        Assert.Contains("_stem.EnumerateLayers()", generated, StringComparison.Ordinal);
        Assert.Contains("SelectMany(__owner => __owner.EnumerateLayers())", generated,
            StringComparison.Ordinal);
    }
}
