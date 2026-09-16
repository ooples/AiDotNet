using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks AIDN100 to current-instance trainable storage, including container slots.</summary>
public sealed class ParameterUpdateInPlaceAnalyzerTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Field)] public sealed class TrainableParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)] public sealed class FittedParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)] public sealed class FrozenParameterAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)] public sealed class BufferAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)] public sealed class ScratchAttribute : Attribute { }
    [AttributeUsage(AttributeTargets.Field)] public sealed class ExternalStateAttribute : Attribute { }
}
public enum ParameterSlotRole { Trainable, LearnedState, Frozen, Buffer, Scratch, Alias, External }
public sealed class Tensor<T> { }
public sealed class TensorEngine
{
    public Tensor<T> TensorSubtract<T>(Tensor<T> left, Tensor<T> right) => left;
}
public abstract class LayerBase
{
    protected TensorEngine Engine { get; } = new TensorEngine();
    protected void RegisterTrainableParameter(object value) { }
    protected void RegisterBuffer(object value) { }
    protected void RegisterParameterComponent(object value, ParameterSlotRole role) { }
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

    private static async Task<ImmutableArray<Diagnostic>> RunAsync(string source)
    {
        var compilation = CSharpCompilation.Create(
            "AiDotNet",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        return await compilation.WithAnalyzers(
                ImmutableArray.Create<DiagnosticAnalyzer>(
                    new AiDotNet.Generators.ParameterUpdateInPlaceAnalyzer()))
            .GetAnalyzerDiagnosticsAsync();
    }

    [Fact]
    public async Task CurrentInstanceTrainableField_IsRejected()
    {
        const string source = @"
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{
    [TrainableParameter] private Tensor<double> _weight = new Tensor<double>();
    private Tensor<double> _gradient = new Tensor<double>();
    public void UpdateParameters() => _weight = Engine.TensorSubtract(_weight, _gradient);
}";

        var diagnostic = Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
        Assert.Contains("_weight", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TrainableArrayListAndDictionaryElements_AreRejected()
    {
        const string source = @"
using System.Collections.Generic;
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{
    [TrainableParameter] private Tensor<double>[] _array = { new Tensor<double>() };
    [TrainableParameter] private List<Tensor<double>> _list = new List<Tensor<double>> { new Tensor<double>() };
    [TrainableParameter] private Dictionary<string, Tensor<double>> _map =
        new Dictionary<string, Tensor<double>> { [""x""] = new Tensor<double>() };
    private Tensor<double> _gradient = new Tensor<double>();

    public void UpdateParameters()
    {
        _array[0] = Engine.TensorSubtract(_array[0], _gradient);
        this._list[0] = Engine.TensorSubtract(_list[0], _gradient);
        _map[""x""] = Engine.TensorSubtract(_map[""x""], _gradient);
    }
}";

        Assert.Equal(3, (await RunAsync(source)).Count(item => item.Id == "AIDN100"));
    }

    [Fact]
    public async Task ManuallyRegisteredTrainableContainer_IsRejected()
    {
        const string source = @"
using System.Collections.Generic;
public sealed class Subject : LayerBase
{
    private Dictionary<string, Tensor<double>> _map =
        new Dictionary<string, Tensor<double>> { [""x""] = new Tensor<double>() };
    private Tensor<double> _gradient = new Tensor<double>();
    private void Configure() => RegisterTrainableParameter(_map);
    public void UpdateParameters() =>
        _map[""x""] = Engine.TensorSubtract(_map[""x""], _gradient);
}";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
    }

    [Fact]
    public async Task UnrelatedNullGuard_DoesNotHideTrainableRebinding()
    {
        const string source = @"
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{
    [TrainableParameter] private Tensor<double> _weight = new Tensor<double>();
    private Tensor<double> _gradient = new Tensor<double>();
    private object _cache;
    public void UpdateParameters()
    {
        if (_cache == null)
            _weight = Engine.TensorSubtract(_weight, _gradient);
    }
}";

        Assert.Single((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
    }

    [Theory]
    [InlineData("FittedParameter")]
    [InlineData("FrozenParameter")]
    [InlineData("Buffer")]
    [InlineData("Scratch")]
    [InlineData("ExternalState")]
    public async Task NonTrainablePersistentOrScratchField_IsAllowed(string attribute)
    {
        string source = $@"
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{{
    [{attribute}] private Tensor<double> _state = new Tensor<double>();
    private Tensor<double> _gradient = new Tensor<double>();
    public void UpdateParameters() => _state = Engine.TensorSubtract(_state, _gradient);
}}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
    }

    [Fact]
    public async Task AnotherInstancesTrainableField_IsAllowed()
    {
        const string source = @"
using AiDotNet.Attributes;
public sealed class Subject : LayerBase
{
    [TrainableParameter] private Tensor<double> _weight = new Tensor<double>();
    private Tensor<double> _gradient = new Tensor<double>();
    public void UpdateParameters(Subject other) =>
        other._weight = Engine.TensorSubtract(other._weight, _gradient);
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
    }

    [Fact]
    public async Task UnclassifiedField_IsAllowed()
    {
        const string source = @"
public sealed class Subject : LayerBase
{
    private Tensor<double> _state = new Tensor<double>();
    private Tensor<double> _gradient = new Tensor<double>();
    public void UpdateParameters() => _state = Engine.TensorSubtract(_state, _gradient);
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "AIDN100"));
    }
}
