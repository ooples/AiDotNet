using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks the B1 rule that raw numeric storage never invents its own semantics.</summary>
public class ParameterAutomationAnalyzerTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class TrainableParameterAttribute : Attribute
    {
        public bool Optional { get; set; }
        public string? Condition { get; set; }
        public string? Shape { get; set; }
        public string? LowPrecisionBacking { get; set; }
        public int Availability { get; set; }
    }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class FittedParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class FrozenParameterAttribute : Attribute { public int Availability { get; set; } }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class BufferAttribute : Attribute { public int Availability { get; set; } }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ScratchAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ExternalStateAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)] public sealed class ParameterAliasAttribute : Attribute
    {
        public ParameterAliasAttribute(string target) { }
    }
}
namespace AiDotNet.Tensors.LinearAlgebra
{
    public sealed class Tensor<T> { }
    public sealed class Matrix<T> { }
    public sealed class Vector<T> { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T>
    {
        public virtual long ParameterCount => 0;
        public virtual AiDotNet.Tensors.LinearAlgebra.Vector<T> GetParameters() => null!;
        public virtual void SetParameters(AiDotNet.Tensors.LinearAlgebra.Vector<T> parameters) { }
    }
}
namespace AiDotNet.Models
{
    public abstract class ModelBase<T>
    {
        public virtual long ParameterCount => 0;
        public virtual AiDotNet.Tensors.LinearAlgebra.Vector<T> GetParameters() => null!;
        public virtual void SetParameters(AiDotNet.Tensors.LinearAlgebra.Vector<T> parameters) { }
    }
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

    private static ImmutableArray<Diagnostic> Run(string source, string assemblyName = "AiDotNet")
    {
        var compilation = CSharpCompilation.Create(
            assemblyName,
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.ParameterAutomationAnalyzer());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        return diagnostics;
    }

    [Fact]
    public async Task NonNullableTensor_DoesNotImplyTrainable()
    {
        await Task.Yield();
        const string source = @"
public sealed class CacheLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _lastInput = new();
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN088"));
        Assert.Contains("_lastInput", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ExistingRegistrationApi_IsAnExplicitSemanticDeclaration()
    {
        await Task.Yield();
        const string source = @"
public sealed class RegisteredLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _running = new();

    private void Configure()
    {
        RegisterTrainableParameter(_weight);
        RegisterBuffer(_running);
    }
}";

        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN088");
    }

    [Fact]
    public async Task RegistrationAndAttributeWithDifferentRoles_AreCompileError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ConflictingRegistrationLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [Buffer] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    private void Configure() => RegisterTrainableParameter(_weight);
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN089"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task NullableTensor_DoesNotImplyScratchOrOptional()
    {
        await Task.Yield();
        const string source = @"
public sealed class MaybeLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _maybe;
}";

        Assert.Single(Run(source).Where(item => item.Id == "AIDN088"));
    }

    [Fact]
    public async Task EachExplicitSemanticRole_IsAccepted()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ClassifiedLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    [FittedParameter] private AiDotNet.Tensors.LinearAlgebra.Vector<double>? _fit;
    [FrozenParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _frozen = new();
    [Buffer] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _running = new();
    [Scratch] private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _cache;
    [ExternalState] private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _external;
    [ParameterAlias(nameof(_weight))] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _alias = new();
}";

        Assert.DoesNotContain(Run(source), item => item.Id is "AIDN088" or "AIDN089");
    }

    [Fact]
    public async Task ConflictingRoles_AreCompileError()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class InvalidLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter, Scratch]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _state = new();
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN089"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task AbstractBase_IsClassifiedBecauseItsStateIsInherited()
    {
        await Task.Yield();
        const string source = @"
public abstract class SharedLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _shared = new();
}";

        Assert.Single(Run(source).Where(item => item.Id == "AIDN088"));
    }

    [Fact]
    public async Task ConventionGradient_IsAcceptedOnlyWhenItMatchesDeclaredParameter()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class GradientLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _weightGradient;
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _orphanGradient;
}";

        var diagnostics = Run(source).Where(item => item.Id == "AIDN088").ToArray();
        var diagnostic = Assert.Single(diagnostics);
        Assert.Contains("_orphanGradient", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task NumericProperty_UsesTheSameExhaustiveClassification()
    {
        await Task.Yield();
        const string source = @"
public sealed class PropertyLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    public AiDotNet.Tensors.LinearAlgebra.Tensor<double> State { get; } = new();
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN088"));
        Assert.Contains("State", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task NullableTrainable_RequiresDeclaredLifecycle()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class DeferredLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _ambiguous;

    [TrainableParameter(Optional = true)]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _conditional;

    [TrainableParameter(Availability = 1)]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>? _shapeDeferred;
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN090"));
        Assert.Contains("_ambiguous", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Alias_MustNameOneCompatibleOwnedMember()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class AliasLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    [ParameterAlias(""_missing"")] private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _alias = new();
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN091"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("_missing", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task Alias_MayNameTheOwnedLayerField()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ChildLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double> { }
public sealed class CompositeLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private ChildLayer _owned = new();
    [ParameterAlias(nameof(_owned))] private ChildLayer _alias = null!;
}";

        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN091");
    }

    [Theory]
    [InlineData("Missing", "no such field")]
    [InlineData("Rank", "not a readable Boolean")]
    [InlineData("Shared", "instance-specific")]
    public async Task TrainableCondition_MustNameOneInstanceBoolean(
        string condition, string expectedReason)
    {
        await Task.Yield();
        string source = $@"
using AiDotNet.Attributes;
public sealed class ConditionalLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{{
    private int Rank => 2;
    private static bool Shared => true;
    [TrainableParameter(Condition = ""{condition}"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
}}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN092"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains(expectedReason, diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TrainableCondition_AcceptsReadableInstanceBoolean()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ConditionalLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private bool Enabled => true;
    [TrainableParameter(Condition = nameof(Enabled))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
}";

        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN092");
    }

    [Theory]
    [InlineData("*(Missing)", "no such field")]
    [InlineData("*(Enabled)", "not a readable Int32")]
    [InlineData("*(Input + 1)", "one member name")]
    [InlineData("*Input", "must be '*' or")]
    public async Task AdaptiveShapeBinding_MustNameOneInstanceInt32(
        string axis, string expectedReason)
    {
        await Task.Yield();
        string source = $@"
using AiDotNet.Attributes;
public sealed class AdaptiveLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{{
    private bool Enabled = true;
    private int Input = 4;
    [TrainableParameter(Shape = ""{axis}"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
}}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN093"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains(expectedReason, diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AdaptiveShapeBinding_AcceptsReadableInstanceInt32AndBareWildcard()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class AdaptiveLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private int Input = 4;
    [TrainableParameter(Shape = ""*(Input), *"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
}";

        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN093");
    }

    [Fact]
    public async Task LowPrecisionBacking_IsOneExplicitLogicalParameterSlot()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ResidentLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    [TrainableParameter(LowPrecisionBacking = nameof(_weightHalf))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
    private AiDotNet.Tensors.LinearAlgebra.Tensor<System.Half>? _weightHalf;
}";

        var diagnostics = Run(source);
        Assert.DoesNotContain(diagnostics, item => item.Id == "AIDN094");
        Assert.DoesNotContain(diagnostics, item =>
            item.Id == "AIDN088" && item.GetMessage().Contains("_weightHalf", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("_missing", "no such field")]
    [InlineData("_wrongType", "not Tensor<Half>")]
    [InlineData("Shared", "instance-specific")]
    public async Task LowPrecisionBacking_MustNameOneInstanceHalfTensor(
        string backing, string expectedReason)
    {
        await Task.Yield();
        string source = $@"
using AiDotNet.Attributes;
public sealed class ResidentLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<float>? _wrongType;
    private static AiDotNet.Tensors.LinearAlgebra.Tensor<System.Half>? Shared;
    [TrainableParameter(LowPrecisionBacking = ""{backing}"")]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _weight = new();
}}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN094"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains(expectedReason, diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task LowPrecisionBacking_CannotRepresentTwoLogicalParameters()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ResidentLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<System.Half>? _shared;
    [TrainableParameter(LowPrecisionBacking = nameof(_shared))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _first = new();
    [TrainableParameter(LowPrecisionBacking = nameof(_shared))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double> _second = new();
}";

        var diagnostics = Run(source).Where(item => item.Id == "AIDN094").ToArray();
        Assert.Equal(2, diagnostics.Length);
        Assert.All(diagnostics, diagnostic =>
            Assert.Contains("more than one trainable parameter", diagnostic.GetMessage(), StringComparison.Ordinal));
    }

    [Fact]
    public async Task LowPrecisionBacking_CannotBeAttachedToAParameterCollection()
    {
        await Task.Yield();
        const string source = @"
using AiDotNet.Attributes;
public sealed class ResidentLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    private AiDotNet.Tensors.LinearAlgebra.Tensor<System.Half>? _shared;
    [TrainableParameter(LowPrecisionBacking = nameof(_shared))]
    private AiDotNet.Tensors.LinearAlgebra.Tensor<double>[] _weights =
        System.Array.Empty<AiDotNet.Tensors.LinearAlgebra.Tensor<double>>();
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN094"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
        Assert.Contains("only one tensor field", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayerParameterSurfaceOverride_IsCompileError()
    {
        await Task.Yield();
        const string source = @"
public sealed class ManualLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    public override long ParameterCount => 1;
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN081"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task ModelParameterSurfaceOverride_IsCompileError()
    {
        await Task.Yield();
        const string source = @"
public sealed class ManualModel : AiDotNet.Models.ModelBase<double>
{
    public override long ParameterCount => 1;
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN082"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }

    [Fact]
    public async Task AssemblyNameCannotDowngradeParameterSurfaceCompilerError()
    {
        await Task.Yield();
        const string source = @"
public sealed class FailureProbeLayer : AiDotNet.NeuralNetworks.Layers.LayerBase<double>
{
    public override long ParameterCount => throw new System.InvalidOperationException();
}";

        var diagnostic = Assert.Single(Run(source, "Example.Tests")
            .Where(item => item.Id == "AIDN081"));
        Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
    }
}
