using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>Locks the scope and write/read classification of facade configuration diagnostics.</summary>
public class FacadeConfigurationValidationGeneratorTests
{
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

    private static ImmutableArray<Diagnostic> Run(params string[] sources)
    {
        var trees = sources
            .Select((source, index) => CSharpSyntaxTree.ParseText(source, path: $"FacadePart{index}.cs"));
        var compilation = CSharpCompilation.Create(
            "AiDotNet",
            trees,
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.FacadeConfigurationValidationGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out var diagnostics);
        return diagnostics;
    }

    [Fact]
    public async Task ImplicitPrivateThisQualifiedWrite_IsDiagnosed()
    {
        await Task.Yield();
        const string source = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    object _configured;
    public AiModelBuilder<T, TInput, TOutput> ConfigureValue(object value)
    {
        this._configured = value;
        return this;
    }
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN096"));
        Assert.Contains("ConfigureValue", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.Contains("_configured", diagnostic.GetMessage(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThisQualifiedAccessorWithoutCaller_IsDiagnosedSeparately()
    {
        await Task.Yield();
        const string source = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private object _configured;
    internal object Configured => this._configured;
    public AiModelBuilder<T, TInput, TOutput> ConfigureValue(object value)
    {
        this._configured = value;
        return this;
    }
}";

        var diagnostic = Assert.Single(Run(source).Where(item => item.Id == "AIDN097"));
        Assert.Contains("Configured", diagnostic.GetMessage(), StringComparison.Ordinal);
        Assert.DoesNotContain(Run(source), item => item.Id == "AIDN096");
    }

    [Fact]
    public async Task ReadAcrossPartialDeclarations_SatisfiesTheContract()
    {
        await Task.Yield();
        const string first = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private object _configured;
    public AiModelBuilder<T, TInput, TOutput> ConfigureValue(object value)
    {
        _configured = value;
        return this;
    }
}";
        const string second = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    public object BuildValue() => _configured;
}";

        Assert.DoesNotContain(Run(first, second), item => item.Id is "AIDN096" or "AIDN097");
    }

    [Fact]
    public async Task SameNamedForeignType_CannotMakeFacadeFieldLookLive()
    {
        await Task.Yield();
        const string facade = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private object _shared;
    public AiModelBuilder<T, TInput, TOutput> ConfigureValue(object value)
    {
        _shared = value;
        return this;
    }
}";
        const string foreign = @"
namespace Consumer;
public class AiModelBuilder<T, TInput, TOutput>
{
    private object _shared;
    public object Read() => _shared;
}";

        Assert.Single(Run(facade, foreign).Where(item => item.Id == "AIDN096"));
    }

    [Fact]
    public async Task AccessorCalledAcrossPartialDeclarations_IsARealConsumer()
    {
        await Task.Yield();
        const string first = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private object _configured;
    internal object Configured => this._configured;
    public AiModelBuilder<T, TInput, TOutput> ConfigureValue(object value)
    {
        this._configured = value;
        return this;
    }
}";
        const string second = @"
namespace AiDotNet;
public partial class AiModelBuilder<T, TInput, TOutput>
{
    public object BuildValue() => Configured;
}";

        Assert.DoesNotContain(Run(first, second), item => item.Id is "AIDN096" or "AIDN097");
    }
}
