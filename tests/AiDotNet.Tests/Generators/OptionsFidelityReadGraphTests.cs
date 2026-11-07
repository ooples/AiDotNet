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
