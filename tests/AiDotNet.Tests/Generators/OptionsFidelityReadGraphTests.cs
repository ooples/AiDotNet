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

public class OptionsFidelityReadGraphTests
{
    [Theory]
    [InlineData("typeof(Example.Asr<>)", true)]
    [InlineData("typeof(Example.Asr<int>)", true)]
    [InlineData("typeof(Example.Tts<>)", false)]
    [InlineData("null", true)]
    public void ModelScopedExemptionKeepsOtherTaskVariantsActive(string scope, bool skipsAsr)
    {
        string source = @"
namespace AiDotNet.Models.Options { public abstract class ModelOptions {} }
namespace AiDotNet.Attributes {
    [System.AttributeUsage(System.AttributeTargets.Property, AllowMultiple = true)]
    public sealed class OptionBehaviourExemptAttribute : System.Attribute {
        public OptionBehaviourExemptAttribute(string reason) {}
        public System.Type ModelType { get; set; }
    }
}
namespace Example {
    public class Options : AiDotNet.Models.Options.ModelOptions {
        [AiDotNet.Attributes.OptionBehaviourExempt(""scope"", ModelType = " + scope + @")]
        public int Scoped { get; set; }
        [AiDotNet.Attributes.OptionBehaviourExempt(""all models"")]
        public int Global { get; set; }
        [AiDotNet.Attributes.OptionBehaviourExempt(""asr only"", ModelType = typeof(Example.Asr<>))]
        [AiDotNet.Attributes.OptionBehaviourExempt(""tts only"", ModelType = typeof(Example.Tts<>))]
        public int Repeated { get; set; }
        public int Live { get; set; }
    }
    public class Asr<T> { public Asr(Options options) {} }
    public class DerivedAsr<T> : Asr<T> { public DerivedAsr(Options options) : base(options) {} }
    public class Tts<T> { public Tts(Options options) {} }
    public class Other<T> { public Other(Options options) {} }
}";
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && !string.IsNullOrEmpty(a.Location))
            .Select(a => a.Location).Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(CachedMetadataReference.FromFile);
        var compilation = CSharpCompilation.Create("ScopedOptions", new[] { CSharpSyntaxTree.ParseText(source) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var compute = typeof(AiDotNet.Generators.TestScaffoldGenerator).GetMethod("ComputeOptionSensitivityCases",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        foreach (string name in new[] { "Asr", "DerivedAsr", "Tts", "Other" })
        {
            var symbol = compilation.GetTypeByMetadataName("Example." + name + "`1")!;
            var cases = ((System.Collections.IEnumerable)compute.Invoke(null, new object[] { symbol })!).Cast<object>()
                .ToDictionary(item => (string)item.GetType().GetProperty("Property")!.GetValue(item)!,
                    item => item.GetType().GetProperty("SkipReason")!.GetValue(item) as string);
            Assert.NotNull(cases["Global"]);
            Assert.Null(cases["Live"]);
            Assert.Equal(name != "Other", cases["Repeated"] is not null);
            bool expected = scope == "null" || (name != "Other" && (name != "Tts" ? skipsAsr : !skipsAsr));
            Assert.Equal(expected, cases["Scoped"] is not null);
        }
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(true, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, true, false)]
    [InlineData(false, false, true)]
    [InlineData(true, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, true, true)]
    public async Task OnlyReachedValueHelpersAndGettersConsumeOptions(bool helper, bool getter, bool documentation)
    {
        string source = @"
namespace AiDotNet.Models.Options { public abstract class ModelOptions {} }
namespace AiDotNet.Interfaces { public interface IFullModel<T, TIn, TOut> {} }
namespace Example {
    public interface ISettings<T> { int InterfaceOnly { get; } }
    public class Options<T> : AiDotNet.Models.Options.ModelOptions, ISettings<T> {
        public int Direct { get; set; }
        public int ThroughHelper { get; set; }
        public int ThroughGetter { get; set; }
        public int NeverUsed { get; set; }
        public int ValidationOnly { get; set; }
        public int CopyOnly { get; set; }
        public int CloneOnly { get; set; }
        public int InterfaceOnly { get; set; }
        public int Canonical { get; set; }
        public int Alias { get => Canonical; set => Canonical = value; }
        public int TransitiveAlias { get => Alias; set => Alias = value; }
        public int ConstantAlias { get => Canonical; set => Canonical = 1; }
        public int Effective => ThroughGetter * 2;
        public int Configuration() => Nested();
        private int Nested() => ThroughHelper.GetHashCode();
        public int Uncalled() => NeverUsed;
        public void Validate() { if (ValidationOnly < 0) throw new System.ArgumentException(); }
        public Options() {}
        public Options(Options<T> other) { CopyOnly = other.CopyOnly; }
        public Options<T> Clone() => new Options<T> { CloneOnly = CloneOnly };
    }
    /// <summary>Documents <see cref=""Options{T}.NeverUsed""/> without consuming it.</summary>
    public class Model : AiDotNet.Interfaces.IFullModel<int, int, int> {
        public int Value { get; }
        public Model(Options<double> options) {
            options.Validate();
            options.CopyOnly = 0;
            string label = nameof(options.NeverUsed);
            Value = options.Direct + options.Canonical + ((ISettings<double>)options).InterfaceOnly" + (helper ? " + options.Configuration()" : "")
                + (getter ? " + options.Effective" : "") + @";
        }
    }
}";
        var references = AppDomain.CurrentDomain.GetAssemblies()
            .Where(a => !a.IsDynamic && !string.IsNullOrEmpty(a.Location))
            .Select(a => a.Location).Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(CachedMetadataReference.FromFile);
        var compilation = CSharpCompilation.Create("AiDotNet",
            new[] { CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(
                documentationMode: documentation ? DocumentationMode.Diagnose : DocumentationMode.None)) }, references,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));
        Assert.Empty(compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error));
        var diagnostics = await compilation.WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(
            new AiDotNet.Generators.OptionsFidelityAnalyzer())).GetAnalyzerDiagnosticsAsync();
        var unread = diagnostics.Where(d => d.Id == "ADNOPT001")
            .Select(d => d.GetMessage().Split('\'')[1].Split('.').Last()).OrderBy(n => n).ToArray();
        var expected = new List<string> { "NeverUsed", "ValidationOnly", "CopyOnly", "CloneOnly", "ConstantAlias" };
        if (!helper) expected.Add("ThroughHelper");
        if (!getter) expected.Add("ThroughGetter");
        Assert.Equal(expected.OrderBy(n => n), unread);
    }
}
