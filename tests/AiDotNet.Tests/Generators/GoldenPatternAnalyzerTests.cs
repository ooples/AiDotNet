using System;
using System.Collections.Immutable;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// The golden-pattern rules (AIDN070-AIDN077). The analyzer binds only invocations and creations whose names a rule
/// can match, so these cover the names reaching a rule by every spelling (qualified, aliased in a file or globally,
/// target-typed <c>new()</c>) and a same-named member of an unrelated type.
/// </summary>
public class GoldenPatternAnalyzerTests
{
    private static async Task<string[]> Findings(params string[] sources)
    {
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && !string.IsNullOrEmpty(a.Location))
            .Select(a => a.Location)
            // The rules bind Console and Regex calls; a host that has not loaded those assemblies yet still needs them.
            .Concat(new[] { typeof(Console).Assembly.Location, typeof(System.Text.RegularExpressions.Regex).Assembly.Location })
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(CachedMetadataReference.FromFile);
        var compilation = CSharpCompilation.Create("AiDotNet",
            sources.Select(s => CSharpSyntaxTree.ParseText(s, new CSharpParseOptions(LanguageVersion.Latest))), references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, nullableContextOptions: NullableContextOptions.Enable));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var diagnostics = await compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(
            new AiDotNet.Generators.GoldenPatternAnalyzer())).GetAnalyzerDiagnosticsAsync();
        return diagnostics.Select(d => d.Id + ":" + d.Location.GetLineSpan().StartLinePosition.Line)
            .OrderBy(s => s, StringComparer.Ordinal).ToArray();
    }

    [Fact]
    public async Task EachRuleReportsItsViolation()
    {
        string source = @"
using System;
using System.Text.RegularExpressions;
namespace Example {
    public class Model {
        public string Bang(string? s) => s!;
        public Random Raw() => new Random(1);
        public bool Untimed(string s) => Regex.IsMatch(s, ""a+"");
        public Regex UntimedInstance() => new Regex(""a+"");
        public void Stub() { throw new NotImplementedException(); }
        public void Log() { Console.WriteLine(""x""); Console.Error.WriteLine(""y""); }
        public void Swallow() { try { Log(); } catch { } }
    }
    public class ModelOptions {
        public int Kept { get; set; }
        public int Missed { get; set; }
        public ModelOptions() {}
        public ModelOptions(ModelOptions other) { Kept = other.Kept; }
    }
}";
        Assert.Equal(new[]
        {
            "AIDN070:15", "AIDN071:5", "AIDN072:6", "AIDN073:7", "AIDN073:8", "AIDN074:9", "AIDN075:10", "AIDN075:10",
            "AIDN076:11",
        }, await Findings(source));
    }

    [Fact]
    public async Task ExemptFormsReportNothing()
    {
        string source = @"
using System;
using System.Text.RegularExpressions;
namespace Example {
    public class Writer { public void Write(string s) {} public void WriteLine(string s) {} public string Replace(string s) => s; }
    public class Model {
        public bool Timed(string s) => Regex.IsMatch(s, ""a+"", RegexOptions.None, TimeSpan.FromSeconds(1));
        public Regex TimedInstance() => new Regex(""a+"", RegexOptions.None, TimeSpan.FromSeconds(1));
        public virtual void Hook() => throw new NotImplementedException();
        public void Other(Writer w) { w.Write(""x""); w.WriteLine(""y""); w.Replace(""z""); }
        public void Filtered() { try { Other(new Writer()); } catch (ArgumentException) { } }
        public void Explained() { try { Other(new Writer()); } catch { /* best effort */ } }
    }
}";
        Assert.Empty(await Findings(source));
    }

    [Fact]
    public async Task BoundNamesAreFoundThroughEverySpelling()
    {
        string aliases = @"
global using GlobalRandom = System.Random;";
        string source = @"
using System;
using LocalRandom = System.Random;
using Rx = System.Text.RegularExpressions.Regex;
namespace Example {
    public class Model {
        public Random Qualified() => new global::System.Random();
        public Random Local() => new LocalRandom();
        public Random Global() => new GlobalRandom();
        public Random TargetTyped() { Random r = new(); return r; }
        public Rx AliasedRegex() => new Rx(""a"");
    }
}";
        Assert.Equal(new[] { "AIDN072:6", "AIDN072:7", "AIDN072:8", "AIDN072:9", "AIDN073:10" },
            await Findings(aliases, source));
    }

    [Fact]
    public async Task OptimizersDrawFromTheirSeededGenerator()
    {
        string source = @"
namespace AiDotNet.Tensors.Helpers { public static class RandomHelper {
    public static System.Random CreateSecureRandom() => new System.Random(0);
    public static System.Random CreateSeededRandom(int seed) => new System.Random(seed); } }
namespace AiDotNet.Optimizers {
    public abstract class OptimizerBase<T, TIn, TOut> { protected System.Random Random = new System.Random(0); }
    public class Swarm : OptimizerBase<double, int, int> {
        private readonly System.Random _own = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1);
        private readonly System.Random _raw = new System.Random(2);
    }
}";
        Assert.Equal(new[] { "AIDN072:2", "AIDN072:3", "AIDN072:5", "AIDN077:7", "AIDN077:8" }, await Findings(source));
    }
}
