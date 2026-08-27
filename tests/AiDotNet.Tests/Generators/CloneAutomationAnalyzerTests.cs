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

/// <summary>Locks the rule that concrete models and layers cannot regain lifecycle plumbing.</summary>
public sealed class CloneAutomationAnalyzerTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Interfaces
{
    public interface IFullModel { }
    public interface IModelSerializer { }
    public interface IModelShape { }
    public interface IOptimizer : IModelSerializer { }
}
namespace AiDotNet.NeuralNetworks.Layers
{
    public abstract class LayerBase<T>
    {
        public virtual byte[] Serialize() => new byte[0];
    }
}
public abstract class ModelBase : AiDotNet.Interfaces.IFullModel
{
    public virtual object Clone() => new object();
}
public abstract class ClassifierBase : AiDotNet.Interfaces.IModelSerializer, AiDotNet.Interfaces.IModelShape
{
    public abstract byte[] Serialize();
}
public abstract class OptimizerBase : AiDotNet.Interfaces.IOptimizer, AiDotNet.Interfaces.IModelShape
{
    public virtual byte[] Serialize() => new byte[0];
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
            ImmutableArray.Create<DiagnosticAnalyzer>(new AiDotNet.Generators.CloneAutomationAnalyzer()))
            .GetAnalyzerDiagnosticsAsync();
    }

    [Theory]
    [InlineData("public sealed class Bad : ModelBase { public override object Clone() => new object(); }")]
    [InlineData("public sealed class Bad : AiDotNet.NeuralNetworks.Layers.LayerBase<double> { public override byte[] Serialize() => new byte[0]; }")]
    [InlineData("public sealed class Bad : ClassifierBase { public override byte[] Serialize() => new byte[0]; }")]
    public async Task ConcreteLifecycleOverride_IsRejected(string source)
    {
        var diagnostic = Assert.Single((await RunAsync(source)).Where(item => item.Id == "ADN0063"));
        Assert.Contains("Bad", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AbstractFamilyBase_MayOwnSharedLifecyclePolicy()
    {
        const string source = @"
public abstract class SharedFamilyBase : ModelBase
{
    public override object Clone() => new object();
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "ADN0063"));
    }

    [Fact]
    public async Task OptimizerSerializer_IsOutsideModelLifecycleRule()
    {
        const string source = @"
public sealed class DistributedOptimizer : OptimizerBase
{
    public override byte[] Serialize() => new byte[0];
}";

        Assert.Empty((await RunAsync(source)).Where(item => item.Id == "ADN0063"));
    }
}
