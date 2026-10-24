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

/// <summary>
/// ADNDEF001 and ADNDEF005 of <see cref="AiDotNet.Generators.ModelDefectClassAnalyzer"/>: two models of one paper under
/// one name are a duplicate unless a declaration the code bears out joins them.
/// </summary>
public sealed class ModelDefectDuplicateTests
{
    private const string Infrastructure = @"
using System;
namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class ResearchPaperAttribute : Attribute { public ResearchPaperAttribute(string title, string url) { } }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class TaskVariantOfAttribute : Attribute { public TaskVariantOfAttribute(Type canonical, string task) { } }
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
    public sealed class DistinctImplementationOfAttribute : Attribute { public DistinctImplementationOfAttribute(Type other, string reason) { } }
}
namespace Lib
{
    public class SharedLayer<T> { }
    public class NetOptions { }
    public class Network<T> { public sealed class Output { } }
    public class OtherNetwork<T> { }
}";

    private const string Paper = @"[AiDotNet.Attributes.ResearchPaper(""P"", ""https://arxiv.org/abs/2001.00001"")]";
    private const string OtherPaper = @"[AiDotNet.Attributes.ResearchPaper(""Q"", ""https://arxiv.org/abs/2001.00002"")]";

    private static ImmutableArray<MetadataReference> References()
    {
        var references = new List<MetadataReference>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var assembly in AppDomain.CurrentDomain.GetAssemblies())
        {
            if (assembly.IsDynamic || string.IsNullOrEmpty(assembly.Location) || !seen.Add(assembly.Location))
                continue;
            references.Add(CachedMetadataReference.FromFile(assembly.Location));
        }
        return references.ToImmutableArray();
    }

    /// <summary>The ADNDEF001 and ADNDEF005 reports, as "id:type" (the stale-baseline reports about the real library are
    /// not about this source).</summary>
    private static async Task<string[]> RunAsync(string source)
    {
        var compilation = CSharpCompilation.Create(
            "AiDotNet",
            new[] { CSharpSyntaxTree.ParseText(Infrastructure), CSharpSyntaxTree.ParseText(source) },
            References(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var diagnostics = await compilation.WithAnalyzers(
            ImmutableArray.Create<DiagnosticAnalyzer>(new AiDotNet.Generators.ModelDefectClassAnalyzer()))
            .GetAnalyzerDiagnosticsAsync();
        return diagnostics.Where(d => d.Id is "ADNDEF001" or "ADNDEF005")
            .Select(d => d.Id + ":" + d.GetMessage().Split('\'')[1])
            .OrderBy(s => s, StringComparer.Ordinal).ToArray();
    }

    [Fact]
    public async Task SameNameSamePaper_IsADuplicate()
    {
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class Vim<T> {{ }} }}
namespace B {{ {Paper} public class Vim<T> {{ }} }}");
        Assert.Equal(new[] { "ADNDEF001:A.Vim<T>", "ADNDEF001:B.Vim<T>" }, reports);
    }

    [Fact]
    public async Task NamesDifferingOnlyByASuffix_AreADuplicate()
    {
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class ResNet<T> {{ }} }}
namespace B {{ {Paper} public class ResNetNetwork<T> {{ }} }}");
        Assert.Equal(new[] { "ADNDEF001:A.ResNet<T>", "ADNDEF001:B.ResNetNetwork<T>" }, reports);
    }

    [Fact]
    public async Task SameNameDifferentPaper_IsNotADuplicate()
    {
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class Vim<T> {{ }} }}
namespace B {{ {OtherPaper} public class Vim<T> {{ }} }}"));
    }

    [Fact]
    public async Task TaskVariant_OnASharedNetwork_IsAccepted()
    {
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class Sam<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Sam<>), ""encoder"")]
  public class Sam<T> {{ Lib.Network<T> n = new(); }} }}"));
    }

    [Fact]
    public async Task TaskVariant_OfASubclassFacade_IsAccepted()
    {
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class BarkModel<T> {{ }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.BarkModel<>), ""facade"")]
  public class Bark<T> : A.BarkModel<T> {{ }} }}"));
    }

    [Fact]
    public async Task TaskVariant_SharingOnlyAGenericPiece_IsRejected()
    {
        // The shared layer is also used by a model of another paper, so it proves nothing.
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class Sam<T> {{ object n = new Lib.SharedLayer<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Sam<>), ""encoder"")]
  public class Sam<T> {{ object n = new Lib.SharedLayer<T>(); }} }}
namespace C {{ {OtherPaper} public class Unrelated<T> {{ object n = new Lib.SharedLayer<T>(); }} }}");
        Assert.Equal(new[] { "ADNDEF001:A.Sam<T>", "ADNDEF001:B.Sam<T>", "ADNDEF005:B.Sam<T>" }, reports);
    }

    [Fact]
    public async Task TaskVariant_SharingOnlyOptions_IsRejected()
    {
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class Sam<T> {{ object o = new Lib.NetOptions(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Sam<>), ""encoder"")]
  public class Sam<T> {{ object o = new Lib.NetOptions(); }} }}");
        Assert.Contains("ADNDEF005:B.Sam<T>", reports);
    }

    [Fact]
    public async Task TaskVariant_SharingOnlyANestedRecord_IsRejected()
    {
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class Sam<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Sam<>), ""encoder"")]
  public class Sam<T> {{ object o = new Lib.Network<T>.Output(); }} }}");
        Assert.Contains("ADNDEF005:B.Sam<T>", reports);
    }

    [Fact]
    public async Task OtherPaperUsingAGroupMember_DoesNotBreakTheVariant()
    {
        // A detector of another paper built on the backbone uses the backbone; it is not a second network.
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class ResNetNetwork<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.ResNetNetwork<>), ""backbone"")]
  public class ResNet<T> {{ object n = new Lib.Network<T>(); }} }}
namespace C {{ {OtherPaper} public class Detector<T> {{ object b = new B.ResNet<T>(); }} }}"));
    }

    [Fact]
    public async Task SamePaperThirdTask_DoesNotBreakTheVariant()
    {
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class Wav2Vec2<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Wav2Vec2<>), ""asr"")]
  public class Wav2Vec2Model<T> {{ object n = new Lib.Network<T>(); }} }}
namespace C {{ {Paper} public class LanguageIdentifier<T> {{ object n = new Lib.Network<T>(); }} }}"));
    }

    [Fact]
    public async Task Distinct_WithoutASharedNetwork_IsAccepted()
    {
        Assert.Empty(await RunAsync($@"
namespace A {{ {Paper} public class Lumina<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.DistinctImplementationOf(typeof(A.Lumina<>), ""video"")]
  public class Lumina<T> {{ object n = new Lib.OtherNetwork<T>(); }} }}"));
    }

    [Fact]
    public async Task Distinct_OnASharedNetwork_IsRejected()
    {
        var reports = await RunAsync($@"
namespace A {{ {Paper} public class Lumina<T> {{ object n = new Lib.Network<T>(); }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.DistinctImplementationOf(typeof(A.Lumina<>), ""video"")]
  public class Lumina<T> {{ object n = new Lib.Network<T>(); }} }}");
        Assert.Contains("ADNDEF005:B.Lumina<T>", reports);
        Assert.Contains("ADNDEF001:A.Lumina<T>", reports);
    }

    [Fact]
    public async Task Declaration_NamingANonNamesake_IsRejected()
    {
        var reports = await RunAsync($@"
namespace A {{ {OtherPaper} public class Other<T> {{ }} }}
namespace B {{ {Paper} [AiDotNet.Attributes.TaskVariantOf(typeof(A.Other<>), ""x"")] public class Sam<T> {{ }} }}");
        Assert.Equal(new[] { "ADNDEF005:B.Sam<T>" }, reports);
    }
}
