using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Xunit;

namespace AiDotNet.Tests.Generators;

public class AgentToolSchemaGeneratorTests
{
    private const string Infrastructure = @"
using System;
namespace AiDotNet.Agentic.Tools
{
    [AttributeUsage(AttributeTargets.Method)]
    public sealed class AgentToolAttribute : Attribute
    {
        public AgentToolAttribute(string description = """") { }
        public string Name { get; set; }
    }

    [AttributeUsage(AttributeTargets.Parameter)]
    public sealed class ToolParameterAttribute : Attribute
    {
        public ToolParameterAttribute(string description = """") { }
        public bool Required { get; set; }
    }
}";

    [Fact]
    public void InterleavedOverloads_PreserveDeclarationOrderAndOverloadSpecificNames()
    {
        const string source = @"
using AiDotNet.Agentic.Tools;
public sealed class InterleavedTools
{
    [AgentTool(Name = ""lookup_text"")]
    public string Lookup(string value) => value;

    [AgentTool(Name = ""count_items"")]
    public int Count(int value) => value;

    [AgentTool(Name = ""lookup_id"")]
    public string Lookup(int value) => value.ToString();
}";

        string generated = Run(source);

        int textOverload = generated.IndexOf("@\"lookup_text\"", StringComparison.Ordinal);
        int interleavedTool = generated.IndexOf("@\"count_items\"", StringComparison.Ordinal);
        int idOverload = generated.IndexOf("@\"lookup_id\"", StringComparison.Ordinal);

        Assert.True(textOverload >= 0, "The string overload's explicit tool name was not emitted.");
        Assert.True(interleavedTool > textOverload, "The interleaved tool moved ahead of the first overload.");
        Assert.True(idOverload > interleavedTool, "The second overload did not retain declaration order.");
        Assert.Contains("instance.Lookup(__p_value)", generated);
        Assert.Equal(2, CountOccurrences(generated, "instance.Lookup(__p_value)"));
    }

    private static string Run(string source)
    {
        var compilation = CSharpCompilation.Create(
            "AgentToolGeneratorRegression",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            BaseReferences(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        GeneratorDriver driver = CSharpGeneratorDriver.Create(
            new AiDotNet.Generators.AgentToolSchemaGenerator());
        driver = driver.RunGeneratorsAndUpdateCompilation(compilation, out _, out _);
        return string.Join("\n", driver.GetRunResult().GeneratedTrees.Select(tree => tree.GetText().ToString()));
    }

    private static int CountOccurrences(string text, string value)
    {
        int count = 0;
        int start = 0;
        while ((start = text.IndexOf(value, start, StringComparison.Ordinal)) >= 0)
        {
            count++;
            start += value.Length;
        }

        return count;
    }

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
}
