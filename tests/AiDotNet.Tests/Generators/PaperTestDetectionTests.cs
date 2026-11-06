using AiDotNet.Generators;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Which hand-written <c>*PaperTests</c> classes clear ADNGEN003: only those xUnit will actually run a test from.
/// </summary>
public class PaperTestDetectionTests
{
    [Theory]
    [InlineData("public class FooPaperTests { [Fact] public void Check() { } }", true)]
    [InlineData("public class FooPaperTests { [Theory, InlineData(1)] public void Check(int value) { } }", true)]
    [InlineData("public class FooPaperTests { public static TheoryData<int> Data => new() { 1 }; [Theory, MemberData(nameof(Data))] public void Check(int value) { } }", true)]
    [InlineData("public static class FooPaperTests { [Fact] public static void Check() { } }", true)]
    // xUnit runs no theory without data (xUnit1003), so it proves nothing about the paper.
    [InlineData("public class FooPaperTests { [Theory] public void Check(int value) { } }", false)]
    [InlineData("public class FooPaperTests { [Fact(Skip = \"later\")] public void Check() { } }", false)]
    [InlineData("public abstract class FooPaperTests { [Fact] public void Check() { } }", false)]
    [InlineData("internal class FooPaperTests { [Fact] public void Check() { } }", false)]
    [InlineData("public class FooPaperTests { public void Check() { } }", false)]
    public void Only_a_class_xUnit_runs_a_test_from_counts(string declaration, bool runnable)
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string trustedAssemblies)
            paths.UnionWith(trustedAssemblies.Split(Path.PathSeparator));
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
            if (!assembly.IsDynamic && !string.IsNullOrEmpty(assembly.Location)) paths.Add(assembly.Location);
        paths.Add(typeof(FactAttribute).Assembly.Location);
        var tree = CSharpSyntaxTree.ParseText("using Xunit;\nnamespace Review\n{\n" + declaration + "\n}");
        var compilation = CSharpCompilation.Create(
            "PaperTestDetection_" + Guid.NewGuid().ToString("N"),
            new[] { tree },
            paths.Select(path => CachedMetadataReference.FromFile(path)),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var type = compilation.GetTypeByMetadataName("Review.FooPaperTests");
        Assert.NotNull(type);
        Assert.Equal(runnable, PaperTestDetection.HasRunnableTest(type));
    }
}
