using System.Text.Json;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

// #2147: count parameter references inside syntax-bounded method bodies. Comments,
// strings and the next method's documentation cannot make an unused parameter look used.
static IEnumerable<(MethodDeclarationSyntax Method, ParameterSyntax Parameter, int References)> Audit(string source)
{
    var tree = CSharpSyntaxTree.ParseText(source);
    var errors = tree.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error).ToArray();
    if (errors.Length != 0)
        throw new InvalidDataException(string.Join(Environment.NewLine, errors.Select(d => d.ToString())));
    foreach (var method in tree.GetRoot().DescendantNodes().OfType<MethodDeclarationSyntax>())
    {
        if (!method.Modifiers.Any(SyntaxKind.PublicKeyword) || !method.Modifiers.Any(SyntaxKind.StaticKeyword))
            continue;
        foreach (var parameter in method.ParameterList.Parameters)
        {
            if (parameter.Type?.DescendantNodesAndSelf().OfType<GenericNameSyntax>()
                .Any(t => t.Identifier.ValueText == "NeuralNetworkArchitecture") != true)
                continue;
            SyntaxNode? body = method.Body ?? (SyntaxNode?)method.ExpressionBody;
            if (body is null) continue;
            int references = body.DescendantNodes().OfType<IdentifierNameSyntax>()
                .Count(identifier => identifier.Identifier.ValueText == parameter.Identifier.ValueText);
            yield return (method, parameter, references);
        }
    }
}

if (args.Contains("--self-test", StringComparer.Ordinal))
{
    const string fixture = """
        class Factory<T> {
            /// <param name="architecture">architecture is mentioned only in documentation</param>
            public static object Unused(NeuralNetworkArchitecture<T> architecture) {
                // architecture.OutputSize and a following method's docs are not reads.
                return "architecture";
            }
            /// <param name="architecture">the previous body ends before this comment</param>
            public static object Forward(NeuralNetworkArchitecture<T> architecture) {
                return Build(architecture);
            }
            public static int Expression(NeuralNetworkArchitecture<T> architecture) => architecture.OutputSize;
            public static int Qualified(global::Example.NeuralNetworkArchitecture<T> spec) => spec.OutputSize;
            private static int Private(NeuralNetworkArchitecture<T> architecture) => 0;
        }
        """;
    var actual = Audit(fixture).Select(x => (x.Method.Identifier.ValueText, x.References)).ToArray();
    (string, int)[] expected = [("Unused", 0), ("Forward", 1), ("Expression", 1), ("Qualified", 1)];
    if (!actual.SequenceEqual(expected)) throw new InvalidDataException("Architecture audit self-test failed.");
    Console.WriteLine("Architecture audit: comments, strings, method boundaries, forwarding, expression bodies and qualified types verified.");
    return;
}

string root = Path.GetFullPath(args.FirstOrDefault(argument => !argument.StartsWith("--", StringComparison.Ordinal))
    ?? Environment.CurrentDirectory);
string helperDirectory = Path.Combine(root, "src", "Helpers");
var files = Directory.GetFiles(helperDirectory, "LayerHelper*.cs").Order(StringComparer.Ordinal).ToArray();
if (files.Length == 0) throw new FileNotFoundException("No LayerHelper source files found.", helperDirectory);
var inventory = files.SelectMany(file => Audit(File.ReadAllText(file)).Select(item => new
{
    file = Path.GetRelativePath(root, file).Replace('\\', '/'),
    line = item.Method.GetLocation().GetLineSpan().StartLinePosition.Line + 1,
    factory = item.Method.Identifier.ValueText,
    parameter = item.Parameter.Identifier.ValueText,
    references = item.References,
    obsolete = item.Method.AttributeLists.SelectMany(list => list.Attributes)
        .Any(attribute => attribute.Name.ToString().Split('.').Last() is "Obsolete" or "ObsoleteAttribute"),
    expressionForwarder = item.Method.ExpressionBody?.Expression is InvocationExpressionSyntax,
})).ToArray();
Console.WriteLine(JsonSerializer.Serialize(new
{
    instrument = "C# syntax tree; public static methods with bodies; identifier references only",
    caveat = "A reference proves consumption, not correct sizing; each unused factory still needs per-model reconciliation.",
    total = inventory.Length,
    unused = inventory.Count(item => item.references == 0),
    unusedObsoleteFactories = inventory.Count(item => item.references == 0 && item.obsolete),
    unusedObsoleteNonForwarders = inventory.Count(item => item.references == 0 && item.obsolete && !item.expressionForwarder),
    unusedActiveFactories = inventory.Count(item => item.references == 0 && !item.obsolete),
    factories = inventory,
}, new JsonSerializerOptions { WriteIndented = true }));
if (args.Contains("--fail-on-active-unused", StringComparer.Ordinal)
    && inventory.Any(item => item.references == 0 && !item.obsolete))
    Environment.ExitCode = 1;
