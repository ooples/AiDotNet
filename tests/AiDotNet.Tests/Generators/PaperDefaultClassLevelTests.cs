using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Attributes;
using AiDotNet.Generators;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// The class-level form of <c>[PaperDefault(nameof(Property), value)]</c>, for a property an options class
/// inherits from a shared base and sets in its constructor: ADNOPT003 checks it as it checks the property form,
/// and the paper-defaults harness reads both, the nearest declaration winning.
/// </summary>
public sealed class PaperDefaultClassLevelTests
{
    private const string Infrastructure = @"
namespace AiDotNet.Attributes
{
    using System;
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    public sealed class PaperDefaultAttribute : Attribute
    {
        public PaperDefaultAttribute(object value) { Value = value; }
        public PaperDefaultAttribute(string property, object value) { Property = property; Value = value; }
        public string Property { get; }
        public object Value { get; }
        public string Source { get; set; } = string.Empty;
    }
}
namespace AiDotNet.Models.Options
{
    public class ModelOptions { }
    public class SequenceOptions : ModelOptions
    {
        public int VocabSize { get; set; }
        public int NumLayers { get; set; }
    }
}";

    private static ImmutableArray<MetadataReference> References()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location)) continue;
            references.Add(CachedMetadataReference.FromFile(assembly.Location));
        }

        return references.ToImmutableArray();
    }

    // Only ADNOPT003: the analyzer's baselines name types this small compilation does not have.
    private static async Task<string[]> Adnopt003(string source)
    {
        var compilation = CSharpCompilation.Create("AiDotNet",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            References(), new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        var compileErrors = compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(compileErrors.Length == 0, string.Join("\n", compileErrors.Select(d => d.ToString())));
        var diagnostics = await compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new OptionsFidelityAnalyzer()))
            .GetAnalyzerDiagnosticsAsync();
        return diagnostics.Where(d => d.Id == "ADNOPT003").Select(d => d.GetMessage()).ToArray();
    }

    private static string Options(string attributes, string members = "") => $@"
using AiDotNet.Attributes;
using AiDotNet.Models.Options;
{attributes}
public class SubjectOptions : SequenceOptions
{{
    public SubjectOptions() {{ VocabSize = 65024; }}
    {members}
}}";

    [Fact(Timeout = 60000)]
    public async Task ClassLevel_NamingAnInheritedProperty_IsValid()
    {
        var messages = await Adnopt003(Options(@"[PaperDefault(nameof(VocabSize), 65024, Source = ""Table 1"")]"));
        Assert.Empty(messages);
    }

    [Fact(Timeout = 60000)]
    public async Task ClassLevel_WithoutSource_IsReported()
    {
        var messages = await Adnopt003(Options("[PaperDefault(nameof(VocabSize), 65024)]"));
        Assert.Contains(messages, m => m.Contains("SubjectOptions.VocabSize") && m.Contains("has no Source"));
    }

    [Fact(Timeout = 60000)]
    public async Task ClassLevel_NamingNoProperty_IsReported()
    {
        var messages = await Adnopt003(Options(@"[PaperDefault(""VocabularySize"", 65024, Source = ""Table 1"")]"));
        Assert.Contains(messages, m => m.Contains("VocabularySize") && m.Contains("names no public instance property"));
    }

    [Fact(Timeout = 60000)]
    public async Task ClassLevel_WithAValueThePropertyCannotHold_IsReported()
    {
        var messages = await Adnopt003(Options(@"[PaperDefault(nameof(VocabSize), ""64K"", Source = ""Sec. 2"")]"));
        Assert.Contains(messages, m => m.Contains("SubjectOptions.VocabSize") && m.Contains("numeric property type"));
    }

    [Fact(Timeout = 60000)]
    public async Task SamePropertyDeclaredTwiceOnOneClass_IsReported()
    {
        var messages = await Adnopt003(Options(
            @"[PaperDefault(nameof(MaxLength), 256, Source = ""Sec. 4"")]",
            @"[PaperDefault(256, Source = ""Sec. 4"")] public int MaxLength { get; set; } = 256;"));
        Assert.Contains(messages, m => m.Contains("SubjectOptions.MaxLength") && m.Contains("more than once"));
    }

    [Fact(Timeout = 60000)]
    public async Task PropertyNamingForm_OnAProperty_IsReported()
    {
        var messages = await Adnopt003(Options("",
            @"[PaperDefault(""MaxLength"", 256, Source = ""Sec. 4"")] public int MaxLength { get; set; } = 256;"));
        Assert.Contains(messages, m => m.Contains("SubjectOptions.MaxLength") && m.Contains("takes only the value"));
    }

    // ---- the harness ----

    public class BaseOptions
    {
        public int VocabSize { get; set; } = 100;

        [PaperDefault(12, Source = "base: Table 1")]
        public int NumLayers { get; set; } = 12;
    }

    [PaperDefault(nameof(VocabSize), 65024, Source = "derived: Table 1")]
    [PaperDefault(nameof(NumLayers), 4, Source = "derived: Table 2")]
    public class DerivedOptions : BaseOptions
    {
        public DerivedOptions() { VocabSize = 65024; NumLayers = 4; }
    }

    [PaperDefault(nameof(VocabSize), 50277, Source = "drifted: Table 1")]
    public class DriftedOptions : BaseOptions
    {
        public DriftedOptions() { VocabSize = 65536; }
    }

    [Fact(Timeout = 60000)]
    public async Task Harness_ReadsClassLevelDeclarations_TheNearestWinning()
    {
        await Task.Yield();
        var declarations = PaperDefaultsHarness.Declarations(typeof(DerivedOptions))
            .ToDictionary(d => d.Property.Name, d => d.Declaration);
        Assert.Equal(2, declarations.Count);
        Assert.Equal(65024, declarations["VocabSize"].Value);
        // The derived class restates the base's property-level 12 as 4; the nearer declaration is the one checked.
        Assert.Equal(4, declarations["NumLayers"].Value);
        Assert.Equal("derived: Table 2", declarations["NumLayers"].Source);
        PaperDefaultsHarness.AssertConstructorDefaultsMatchPaper(typeof(PaperDefaultClassLevelTests), typeof(DerivedOptions));
    }

    [Fact(Timeout = 60000)]
    public async Task Harness_ReportsAClassLevelValueTheConstructorDoesNotShip()
    {
        await Task.Yield();
        var declarations = PaperDefaultsHarness.Declarations(typeof(DriftedOptions)).Select(d => d.Property.Name).ToArray();
        Assert.Equal(new[] { "VocabSize", "NumLayers" }, declarations);
        var failure = Assert.ThrowsAny<Exception>(() =>
            PaperDefaultsHarness.AssertConstructorDefaultsMatchPaper(typeof(PaperDefaultClassLevelTests), typeof(DriftedOptions)));
        Assert.Contains("VocabSize: paper value 50277 (drifted: Table 1), constructor default 65536", failure.Message);
    }
}
