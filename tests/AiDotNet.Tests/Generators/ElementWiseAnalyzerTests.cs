using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Generators;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Text;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Locks the element-wise gate: ADNELEM001 (tensor indexer in a loop), ADNELEM002 (numeric operation applied per
/// element in a loop) and ADNELEM003 (the shrink-only baseline), and in particular which loops ADNELEM002 treats as
/// loops over elements.
/// </summary>
public sealed class ElementWiseAnalyzerTests
{
    private const string Header = @"
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Attributes
{
    [AttributeUsage(AttributeTargets.All)]
    public sealed class ScalarKernelAttribute : Attribute
    {
        public ScalarKernelAttribute(string reason) { }
    }
}
";

    private static string Subject(string body) => Header + @"
public sealed class Subject<T>
{
    private readonly INumericOperations<T> _ops;
    public Subject(INumericOperations<T> ops) { _ops = ops; }
" + body + @"
}";

    private static ImmutableArray<MetadataReference> References()
    {
        // The compiled snippets use the real tensor types, so their assembly must be loaded before the scan.
        GC.KeepAlive(typeof(Tensor<double>));
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

    private static async Task<ImmutableArray<Diagnostic>> RunAsync(
        string source, string? baseline = null, bool gate = true, bool newestFramework = true)
    {
        var parseOptions = new CSharpParseOptions(LanguageVersion.Latest,
            preprocessorSymbols: newestFramework ? new[] { "NET10_0_OR_GREATER" } : Array.Empty<string>());
        var compilation = CSharpCompilation.Create(
            "ElementWiseGateSubject",
            new[] { CSharpSyntaxTree.ParseText(source, parseOptions) },
            References(),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        var compileErrors = compilation.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
        Assert.True(compileErrors.Length == 0, "The snippet must compile: " + string.Join("; ", compileErrors.Take(3)));

        var additionalFiles = baseline is null
            ? ImmutableArray<AdditionalText>.Empty
            : ImmutableArray.Create<AdditionalText>(new InMemoryText("ElementWiseBaseline.txt", baseline));
        var options = new AnalyzerOptions(additionalFiles, new GateOptionsProvider(gate));
        return await compilation
            .WithAnalyzers(ImmutableArray.Create<DiagnosticAnalyzer>(new ElementWiseAnalyzer()), options)
            .GetAnalyzerDiagnosticsAsync();
    }

    private static int Count(ImmutableArray<Diagnostic> diagnostics, string id) => diagnostics.Count(d => d.Id == id);

    // Mutation: return false from IsTensorType, or drop the PropertyReference action.
    [Fact]
    public async Task TensorIndexerAndNumericCallInAnElementLoop_AreBothReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public Vector<T> Scale(Vector<T> values, T factor)
    {
        var result = new Vector<T>(values.Length);
        for (int i = 0; i < values.Length; i++)
            result[i] = _ops.Multiply(values[i], factor);
        return result;
    }"));

        Assert.Equal(2, Count(diagnostics, "ADNELEM001"));
        Assert.Equal(1, Count(diagnostics, "ADNELEM002"));
        Assert.All(diagnostics, d => Assert.Equal(DiagnosticSeverity.Error, d.Severity));
    }

    // Mutation: make RunsPerElement return true unconditionally (the gate before this rule): the series loop
    // below, which has no element in it, is reported again.
    [Fact]
    public async Task NumericCallInALoopOverSeriesTerms_IsNotReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public T Exponential(T x, int order)
    {
        T sum = _ops.One, term = _ops.One;
        double factorial = 1.0;
        for (int n = 1; n <= order; n++)
        {
            factorial *= n;
            term = _ops.Multiply(term, x);
            sum = _ops.Add(sum, _ops.Multiply(term, _ops.FromDouble(1.0 / factorial)));
        }
        return sum;
    }"));

        Assert.Empty(diagnostics);
    }

    // Mutation: drop the Span/ReadOnlySpan or the indexer case from IsElementAccess: a span fill is no longer seen.
    [Fact]
    public async Task NumericCallWritingSpanElements_IsReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public void Ramp(Span<T> destination)
    {
        for (int i = 0; i < destination.Length; i++)
            destination[i] = _ops.FromDouble(i);
    }"));

        Assert.Equal(1, Count(diagnostics, "ADNELEM002"));
        Assert.Equal(0, Count(diagnostics, "ADNELEM001"));
    }

    // Mutation: compare only against the numeric type in Same (drop the host floating-point types): converting a
    // double[] through LINQ, which builds tensor data one element at a time, is no longer seen.
    [Fact]
    public async Task NumericCallConvertingHostDoubles_IsReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public T[] Convert(double[] values) => values.Select(v => _ops.FromDouble(v)).ToArray();

    public T[] ConvertIndexed(double[] values)
    {
        var result = new List<T>();
        for (int i = 0; i < values.Length; i++) result.Add(_ops.FromDouble(values[i]));
        return result.ToArray();
    }"));

        Assert.Equal(2, Count(diagnostics, "ADNELEM002"));
    }

    // Mutation: let TouchesElement descend into nested loops: the per-epoch step after the element loop is
    // counted as a third occurrence.
    [Fact]
    public async Task ScalarStepAfterANestedElementLoop_IsNotCountedAgainstTheOuterLoop()
    {
        var diagnostics = await RunAsync(Subject(@"
    public T Train(T[] weights, T rate, T decay, int epochs)
    {
        T total = _ops.Zero;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < weights.Length; i++)
                total = _ops.Add(total, _ops.Multiply(weights[i], rate));
            rate = _ops.Multiply(rate, decay);
        }
        return total;
    }"));

        Assert.Equal(2, Count(diagnostics, "ADNELEM002"));
    }

    // Mutation: test only the innermost loop in RunsPerElement: the power loop below runs once per element of
    // the outer loop but has no element access of its own, and stops being reported.
    [Fact]
    public async Task NumericCallNestedInsideAnElementLoop_IsReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public T SumOfPowers(T[] values, int power)
    {
        T total = _ops.Zero;
        for (int i = 0; i < values.Length; i++)
        {
            T value = values[i];
            T raised = _ops.One;
            for (int k = 0; k < power; k++) raised = _ops.Multiply(raised, value);
            total = _ops.Add(total, raised);
        }
        return total;
    }"));

        Assert.Equal(2, Count(diagnostics, "ADNELEM002"));
    }

    // Mutation: drop the IForEachLoopOperation case, or the collection-member case, from the element tests.
    [Fact]
    public async Task ForEachOverElementsAndCollectionMembers_AreReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public T Sum(T[] values)
    {
        T total = _ops.Zero;
        foreach (var value in values) total = _ops.Add(total, value);
        return total;
    }

    public List<T> Ones(int count)
    {
        var list = new List<T>();
        for (int i = 0; i < count; i++) list.Add(_ops.FromDouble(1.0));
        return list;
    }

    public T Drain(Tensor<T> tensor)
    {
        T total = _ops.Zero;
        for (int i = 0; i < tensor.Length; i++) total = _ops.Add(total, tensor.GetFlat(i));
        return total;
    }"));

        Assert.Equal(3, Count(diagnostics, "ADNELEM002"));
    }

    // Mutation: drop IsItemMember: values read from a field of the indexed or iterated item are no longer seen.
    // The third member is the boundary: a value computed by a call on the item is not an element of any buffer.
    [Fact]
    public async Task NumericCallOnAMemberOfTheLoopItem_IsReported()
    {
        var diagnostics = await RunAsync(Subject(@"
    public struct Item { public T Score; }

    public T Total(List<Item> items)
    {
        T total = _ops.Zero;
        for (int i = 0; i < items.Count; i++) total = _ops.Add(total, items[i].Score);
        return total;
    }

    public T TotalEach(List<Item> items)
    {
        T total = _ops.Zero;
        foreach (var item in items) total = _ops.Add(total, item.Score);
        return total;
    }

    public T TotalOfCalls(List<Item> items, Func<Item, T> score)
    {
        T total = _ops.Zero;
        foreach (var item in items) total = _ops.Add(total, score(item));
        return total;
    }"));

        Assert.Equal(2, Count(diagnostics, "ADNELEM002"));
    }

    // Mutation: return false from IsExempt.
    [Fact]
    public async Task ScalarKernel_ExemptsTheMember()
    {
        var diagnostics = await RunAsync(Subject(@"
    [AiDotNet.Attributes.ScalarKernel(""Reference implementation."")]
    public T Sum(Vector<T> values)
    {
        T total = _ops.Zero;
        for (int i = 0; i < values.Length; i++) total = _ops.Add(total, values[i]);
        return total;
    }"));

        Assert.Empty(diagnostics);
    }

    private const string SummingMember = @"
    public T Sum(Vector<T> values)
    {
        T total = _ops.Zero;
        for (int i = 0; i < values.Length; i++) total = _ops.Add(total, values[i]);
        return total;
    }";

    private const string SumKey = "Subject<T>.Sum(AiDotNet.Tensors.LinearAlgebra.Vector<T>)";

    // Mutation: report every occurrence regardless of the baseline (drop the 'over' filter).
    [Fact]
    public async Task ExactBaseline_SuppressesTheMember()
    {
        var diagnostics = await RunAsync(Subject(SummingMember),
            baseline: "# inventory\n1\tADNELEM001|" + SumKey + "\n1\tADNELEM002|" + SumKey + "\n");

        Assert.Empty(diagnostics);
    }

    // Mutation: skip the shrink check in the compilation-end action.
    [Fact]
    public async Task StaleBaseline_IsReportedAsShrinkRequired()
    {
        var diagnostics = await RunAsync(Subject(SummingMember),
            baseline: "3\tADNELEM001|" + SumKey + "\n1\tADNELEM002|" + SumKey + "\n1\tADNELEM002|Subject<T>.Gone()\n");

        Assert.Equal(2, Count(diagnostics, "ADNELEM003"));
        Assert.Contains(diagnostics, d => d.GetMessage().IndexOf("lower it to 1", StringComparison.Ordinal) >= 0);
        Assert.Contains(diagnostics, d => d.GetMessage().IndexOf("remove its line", StringComparison.Ordinal) >= 0);
        Assert.Equal(0, Count(diagnostics, "ADNELEM001"));
    }

    // Mutation: drop the ElementWiseGate property test or the NET10_0_OR_GREATER test in Initialize.
    [Theory]
    [InlineData(false, true)]
    [InlineData(true, false)]
    public async Task GateOffOrOlderFramework_ReportsNothing(bool gate, bool newestFramework)
    {
        var diagnostics = await RunAsync(Subject(SummingMember), gate: gate, newestFramework: newestFramework);

        Assert.Empty(diagnostics);
    }

    private sealed class InMemoryText : AdditionalText
    {
        private readonly SourceText _text;

        public InMemoryText(string path, string content)
        {
            Path = path;
            _text = SourceText.From(content);
        }

        public override string Path { get; }

        public override SourceText GetText(CancellationToken cancellationToken = default) => _text;
    }

    private sealed class GateOptionsProvider : AnalyzerConfigOptionsProvider
    {
        private readonly GateOptions _global;

        public GateOptionsProvider(bool gate) => _global = new GateOptions(gate);

        public override AnalyzerConfigOptions GlobalOptions => _global;

        public override AnalyzerConfigOptions GetOptions(SyntaxTree tree) => _global;

        public override AnalyzerConfigOptions GetOptions(AdditionalText textFile) => _global;
    }

    // The base signature carries [NotNullWhen(true)], which the .NET Framework targets do not define; the override
    // is declared nullable-oblivious so it matches on every target.
#nullable disable
    private sealed class GateOptions : AnalyzerConfigOptions
    {
        private readonly bool _gate;

        public GateOptions(bool gate) => _gate = gate;

        public override bool TryGetValue(string key, out string value)
        {
            bool found = _gate && key == "build_property.ElementWiseGate";
            value = found ? "true" : string.Empty;
            return found;
        }
    }
#nullable restore
}
