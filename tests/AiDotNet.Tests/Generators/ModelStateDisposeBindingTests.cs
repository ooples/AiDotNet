using System;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public sealed class ModelStateDisposeBindingTests
{
    private const string Infrastructure = """
        namespace AiDotNet.Attributes
        {
            public sealed class FittedParameterAttribute : System.Attribute { }
            public sealed class ScratchAttribute : System.Attribute { }
        }
        namespace AiDotNet.Models
        {
            public sealed class ModelStateRegistry<T>
            {
                public void DeclareBoolean(string name, System.Func<bool> get, System.Action<bool> set) { }
            }
        }
        public abstract class StateBase
        {
            protected virtual void RegisterGeneratedState(AiDotNet.Models.ModelStateRegistry<float> state) { }
        }
        """;

    [Theory]
    [InlineData("System.Action<bool> action = value => value = true;", true)]
    [InlineData("System.Action<bool> action = (value) => value = true;", true)]
    [InlineData("void Local(bool value) { value = true; } Local(false);", true)]
    [InlineData("{ bool value = false; value = true; } value = true;", false)]
    [InlineData("{ object item = false; if (item is bool value) value = true; } value = true;", false)]
    [InlineData("this.value = true;", false)]
    public void OnlyAssignmentsBoundToTheMemberExcludeItsState(string body, bool persisted)
    {
        var result = Run(CreateCompilation($$"""
            public partial class Probe : StateBase
            {
                private bool value;
                public void Dispose() { {{body}} }
            }
            """));
        Assert.Empty(result.Diagnostics);
        Assert.Equal(persisted, Generated(result).IndexOf("Probe.value", StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void OtherTypesAndNestedDisposeMethodsDoNotExcludeOwnerState()
    {
        var result = Run(CreateCompilation("""
            public class Other { public bool value; }
            public partial class Probe : StateBase
            {
                private bool value;
                public void Dispose() { new Other().value = true; }
                private class Nested
                {
                    private bool value;
                    public void Dispose() { value = true; }
                }
            }
            """));
        Assert.Empty(result.Diagnostics);
        Assert.Contains("Probe.value", Generated(result), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("FittedParameter", "value = true;", true)]
    [InlineData("Scratch", "value = true;", false)]
    [InlineData("FittedParameter", "System.Action<bool> action = value => value = true;", false)]
    public void DisposeDiagnosticUsesBoundPropertyAssignmentsAcrossPartialFiles(
        string attribute, string body, bool diagnosticExpected)
    {
        var result = Run(CreateCompilation($$"""
            public partial class Probe : StateBase
            {
                [AiDotNet.Attributes.{{attribute}}] private bool value { get; set; }
            }
            """, $$"""
            public partial class Probe
            {
                public void DisposeAsync() { {{body}} }
            }
            """));
        if (diagnosticExpected)
        {
            var diagnostic = Assert.Single(result.Diagnostics);
            Assert.Equal("ADN0067", diagnostic.Id);
            Assert.Equal(DiagnosticSeverity.Error, diagnostic.Severity);
            Assert.Contains("Probe.value", diagnostic.GetMessage(), StringComparison.Ordinal);
        }
        else
        {
            Assert.Empty(result.Diagnostics);
        }
    }

    [Fact]
    public void EditingOnlyTheDisposePartialUpdatesStateClassification()
    {
        var compilation = CreateCompilation("""
            public partial class Probe : StateBase { private bool value; }
            """, "public partial class Probe { public void Dispose() { } }");
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.ModelStateGenerator());
        driver = driver.RunGenerators(compilation);
        Assert.Contains("Probe.value", Generated(driver.GetRunResult()), StringComparison.Ordinal);

        var changed = compilation.ReplaceSyntaxTree(compilation.SyntaxTrees.Last(), CSharpSyntaxTree.ParseText(
            "public partial class Probe { public void Dispose() { value = true; } }"));
        driver = driver.RunGenerators(changed);
        Assert.Empty(driver.GetRunResult().Diagnostics);
        Assert.DoesNotContain("Probe.value", Generated(driver.GetRunResult()), StringComparison.Ordinal);
    }

    private static CSharpCompilation CreateCompilation(params string[] sources)
    {
        var paths = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string platformAssemblies
            ? platformAssemblies.Split(System.IO.Path.PathSeparator)
            : AppDomain.CurrentDomain.GetAssemblies()
                .Where(assembly => !assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location))
                .Select(assembly => assembly.Location).ToArray();
        var references = paths
            .Select(path => CachedMetadataReference.FromFile(path));
        var compilation = CSharpCompilation.Create("DisposeBinding",
            new[] { Infrastructure }.Concat(sources).Select(source => CSharpSyntaxTree.ParseText(source)),
            references, new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        return compilation;
    }

    private static GeneratorDriverRunResult Run(CSharpCompilation compilation)
    {
        GeneratorDriver driver = CSharpGeneratorDriver.Create(new AiDotNet.Generators.ModelStateGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out var output, out _);
        Assert.Empty(output.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        return driver.GetRunResult();
    }

    private static string Generated(GeneratorDriverRunResult result)
        => string.Join("\n", result.GeneratedTrees.Select(tree => tree.ToString()));
}
