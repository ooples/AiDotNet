// DocSnippetVerify — compiles example code against the real AiDotNet assemblies (via
// Roslyn) and reports which fail. Two input modes:
//   • a directory  -> every ```csharp fenced block in its *.md / *.mdx files
//   • an .xml file -> every <example><code> block in that XML doc-comment file
// This is the compile gate for both the documentation snippets and the source-code XML
// examples, catching the API drift that broke the samples.
//
// Usage: dotnet run --project tools/DocSnippetVerify [root1 root2 ...]
//        roots default to "docs" and "website"; pass a path to AiDotNet.xml to check the
//        source doc-comment examples.

using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

// --include-prose also compile-checks <code> found in <remarks>/<summary>, not just <example>.
bool includeProse = args.Any(a => string.Equals(a, "--include-prose", StringComparison.OrdinalIgnoreCase));
string[] roots = args.Where(a => !a.StartsWith("--", StringComparison.Ordinal)).ToArray();
if (roots.Length == 0) roots = new[] { "docs", "website" };

// ── Reference set: AiDotNet + all transitive deps (this tool's own output) + BCL ──
var refs = new List<MetadataReference>();
var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
void AddRef(string path)
{
    var name = Path.GetFileName(path);
    if (!seen.Add(name)) return;
    try { System.Reflection.AssemblyName.GetAssemblyName(path); }
    catch { return; }
    try { refs.Add(MetadataReference.CreateFromFile(path)); } catch { }
}
foreach (var dll in Directory.GetFiles(AppContext.BaseDirectory, "*.dll")) AddRef(dll);
if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string tpa)
    foreach (var p in tpa.Split(Path.PathSeparator)) AddRef(p);

var options = new CSharpCompilationOptions(
    OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true,
    nullableContextOptions: NullableContextOptions.Disable);
var parse = new CSharpParseOptions(LanguageVersion.Latest);

// Fully-qualified type names for generated code, so a stub never collides with an imported name.
var FullType = SymbolDisplayFormat.FullyQualifiedFormat
    .WithMiscellaneousOptions(SymbolDisplayMiscellaneousOptions.UseSpecialTypes);

// ── Usings the harness prepends to every snippet ──
// A documentation example names a type; it does not carry the library's import list, because a reader pastes it
// into a file that already has those usings (or lets the IDE add them). Importing only three AiDotNet namespaces
// therefore failed 1370 of 1403 snippets on CS0246 — "type not found" — which says nothing about the example and
// hides the handful that reference types the library does not actually have.
//
// The namespace list is DISCOVERED from the referenced assemblies rather than hardcoded, so it cannot drift as
// the library grows: a new namespace is covered the next time this runs. Only namespaces that actually contain
// a public type are imported, and `using` of a namespace is free when unused.
var discoveredUsings = new SortedSet<string>(StringComparer.Ordinal);

// Public types that can be built with no arguments, or with just an architecture. These let the
// declaration solver answer a name like `myTokenizer` or `myEncoder` with a real type from the library —
// something no fixed candidate list could contain, because the answer depends on what the library ships.
var constructibleTypes = new List<(string Simple, int Arity, bool Parameterless, bool TakesArchitecture)>();
// Returned-type name -> a call that builds one with no arguments.
var zeroArgFactories = new List<(string Returns, string Call)>();
{
    var probe = CSharpCompilation.Create("__ns_probe", Array.Empty<SyntaxTree>(), refs, options);
    foreach (var reference in refs)
    {
        if (probe.GetAssemblyOrModuleSymbol(reference) is not IAssemblySymbol assembly) continue;
        if (!assembly.Name.StartsWith("AiDotNet", StringComparison.Ordinal)) continue;

        var queue = new Queue<INamespaceSymbol>();
        queue.Enqueue(assembly.GlobalNamespace);
        while (queue.Count > 0)
        {
            var ns = queue.Dequeue();
            if (!ns.IsGlobalNamespace &&
                ns.GetTypeMembers().Any(t => t.DeclaredAccessibility == Accessibility.Public))
            {
                discoveredUsings.Add(ns.ToDisplayString());
            }

            foreach (var type in ns.GetTypeMembers())
            {
                if (type.DeclaredAccessibility != Accessibility.Public || type.IsAbstract ||
                    type.TypeKind != TypeKind.Class || type.Arity > 1)
                {
                    continue;
                }

                var ctors = type.InstanceConstructors
                    .Where(c => c.DeclaredAccessibility == Accessibility.Public)
                    .ToList();
                bool none = ctors.Any(c => c.Parameters.All(p => p.HasExplicitDefaultValue));
                bool arch = ctors.Any(c =>
                    c.Parameters.Length > 0 &&
                    c.Parameters[0].Type.Name == "NeuralNetworkArchitecture" &&
                    c.Parameters.Skip(1).All(p => p.HasExplicitDefaultValue));

                if (none || arch)
                {
                    constructibleTypes.Add((type.Name, type.Arity, none, arch));
                }

                // Static factories that need no arguments. Several types are only reachable this way —
                // CharacterTokenizer wants a vocabulary and special tokens, but CreateAscii() builds both,
                // so an example can name a tokenizer in one line instead of four. Preferring the
                // library's own factory is also what a reader should be shown.
                foreach (var method in type.GetMembers().OfType<IMethodSymbol>())
                {
                    if (!method.IsStatic || method.DeclaredAccessibility != Accessibility.Public) continue;
                    if (method.MethodKind != MethodKind.Ordinary) continue;
                    if (!method.Parameters.All(p => p.HasExplicitDefaultValue)) continue;
                    if (method.ReturnsVoid || method.Arity > 0) continue;

                    string owner = type.Arity == 1 ? $"{type.Name}<double>" : type.Name;
                    zeroArgFactories.Add((method.ReturnType.Name, $"{owner}.{method.Name}()"));
                }
            }

            foreach (var child in ns.GetNamespaceMembers()) queue.Enqueue(child);
        }
    }
}

string commonUsings =
    "using System;using System.Collections.Generic;using System.Linq;" +
    "using System.Threading;using System.Threading.Tasks;" +
    string.Concat(discoveredUsings.Select(n => $"using {n};")) + "\n";

Console.WriteLine($"Imported {discoveredUsings.Count} AiDotNet namespaces into the snippet harness.");

var blockRe = new Regex("```csharp\\s*?\\n(.*?)```", RegexOptions.Singleline);
// The kind label can contain spaces ("Base Classes"), so it is matched as anything between backticks.
var pageNamespaceRe = new Regex(@"^`[^`\n]+`\s*·\s*`([A-Za-z0-9_.]+)`", RegexOptions.Multiline);
var usingRe = new Regex(@"^\s*using\s+[A-Za-z_][\w.]*\s*;\s*$");
var typeStartRe = new Regex(@"^\s*(\[|public |internal |private |protected |static |abstract |sealed |partial |class |record |struct |enum |interface |namespace )");

int total = 0, pass = 0;
var failures = new List<(string file, int idx, string err)>();
var perFile = new Dictionary<string, (int total, int fail)>();

// --suggest-decls machinery. The candidate list is deliberately short and ordered from most to least
// specific: an input that binds as a Matrix should be declared a Matrix rather than falling through to a
// bare int that happens to satisfy some other overload.
bool suggestDecls = args.Any(a => string.Equals(a, "--suggest-decls", StringComparison.OrdinalIgnoreCase));
var suggestions = new List<(string file, int idx, string decl)>();
var unsolvable = new List<string>();
// --facade-check reports examples that train or predict on a model instead of an AiModelResult.
bool facadeCheck = args.Any(a => string.Equals(a, "--facade-check", StringComparison.OrdinalIgnoreCase));
var facadeViolations = new List<string>();
// --facade-plan <path> writes the facts a rewrite needs, one line per convertible call.
bool facadePlan = args.Any(a => string.Equals(a, "--facade-plan", StringComparison.OrdinalIgnoreCase));
var facadePlanLines = new List<string>();
int probeId = 0;
string[] DeclarationCandidates =
{
    "new Matrix<double>(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 }, { 7.0, 8.0 } })",
    "new Vector<double>(new double[] { 0.0, 1.0, 0.0, 1.0 })",
    "new Matrix<float>(new float[,] { { 1.0f, 2.0f }, { 3.0f, 4.0f } })",
    "new Vector<float>(new float[] { 0.0f, 1.0f })",
    "Tensor<float>.CreateRandom(1, 3, 32, 32)",
    "Tensor<double>.CreateRandom(2, 4)",
    "Tensor<float>.CreateRandom(2, 4)",
    "new double[] { 1.0, 2.0, 3.0 }",
    "new string[] { \"first\", \"second\" }",
    "new int[] { 0, 1, 2 }",
    "new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 4)",
    "new NeuralNetworkArchitecture<float>(inputFeatures: 8, outputSize: 4)",
    "new AiModelBuilder<double, Matrix<double>, Vector<double>>()",
    "new Random(42)",
    "new List<string> { \"first\", \"second\" }",
    "new List<double> { 1.0, 2.0, 3.0 }",
    "new Dictionary<string, double> { [\"alpha\"] = 1.0 }",
    "\"example\"",
    "32",
    "0.5",
    "true",
};

// Shapes that suit a name, tried before the generic list. Purely cosmetic — every one of these still has
// to satisfy the compiler before it is accepted — but it keeps an audio buffer from being declared with
// the dimensions of an RGB image.
// The type an example documents, derived from its member id, so the solver can offer "an instance of the
// thing this page is about". That is what an undefined `model` or `network` almost always means in a doc
// example, and no generic candidate list would ever supply it.
IEnumerable<string> SelfCandidates(string fileKey)
{
    var name = fileKey.Contains(':') ? fileKey[(fileKey.IndexOf(':') + 1)..] : fileKey;
    name = Regex.Replace(name, @"\(.*", "");
    var arity = Regex.Match(name, @"`(\d+)");
    name = Regex.Replace(name, @"`\d+", "");
    var simple = name.Split('.').LastOrDefault(s => s.Length > 0 && char.IsUpper(s[0]));
    if (simple is null) yield break;

    if (!arity.Success || arity.Groups[1].Value == "0")
    {
        yield return $"new {simple}()";
        yield break;
    }
    if (arity.Groups[1].Value == "1")
    {
        yield return $"new {simple}<double>()";
        yield return $"new {simple}<float>()";
        yield return $"new {simple}<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 4))";
        yield return $"new {simple}<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 8, outputSize: 4))";
    }
}

// Constructions of library types whose name matches the variable's. `myTokenizer` looks for a type whose
// name ends in "Tokenizer", `pool` for one ending in "Pool". The compiler still has to accept the result,
// so a coincidental name match that does not fit the surrounding code is rejected like any other
// candidate — this only widens what gets tried.
IEnumerable<string> TypeCandidatesFor(string name)
{
    var word = Regex.Replace(name, @"^(my|the|a|an)(?=[A-Z])", "", RegexOptions.IgnoreCase);
    word = Regex.Replace(word, @"\d+$", "");
    if (word.Length < 4) yield break;

    // Factories first: where the library offers one, it is both the shortest way to build the object and
    // the call a reader should be shown. A tokenizer takes a vocabulary and special tokens through its
    // constructor, but CharacterTokenizer.CreateAscii() supplies both.
    foreach (var f in zeroArgFactories
                 .Where(f => f.Returns.EndsWith(word, StringComparison.OrdinalIgnoreCase) ||
                             string.Equals(f.Returns, word, StringComparison.OrdinalIgnoreCase))
                 .OrderBy(f => f.Call.Length)
                 .Take(3))
    {
        yield return f.Call;
    }

    var matches = constructibleTypes
        .Where(t => t.Simple.EndsWith(word, StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(t.Simple, word, StringComparison.OrdinalIgnoreCase))
        .OrderBy(t => t.Simple.Length)          // the least-decorated name is the likeliest intent
        .Take(4)
        .ToList();

    foreach (var t in matches)
    {
        string generic = t.Arity == 1 ? "<double>" : "";
        if (t.Parameterless) yield return $"new {t.Simple}{generic}()";
        if (t.TakesArchitecture)
        {
            string archArg = t.Arity == 1
                ? "new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 4)"
                : "new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 4)";
            yield return $"new {t.Simple}{generic}({archArg})";
        }
    }
}

IEnumerable<string> OrderCandidatesFor(string name, string fileKey)
{
    var lower = name.ToLowerInvariant();
    var preferred = new List<string>();

    // An undefined model/network/instance in a type's own documentation means that type.
    if (lower is "model" or "network" or "instance" or "algorithm" || lower.EndsWith("model", StringComparison.Ordinal))
    {
        preferred.AddRange(SelfCandidates(fileKey));
    }
    // Names like myEncoder read as "some model of mine". These parameters are typed IFullModel, so a
    // plain NeuralNetwork is both valid and recognisable; searching the library for a type ending in
    // "Encoder" would instead offer whichever one sorts shortest — CIFEncoder, a speech
    // continuous-integrate-and-fire encoder, in a Matching Networks example.
    else if (lower.EndsWith("encoder", StringComparison.Ordinal) || lower.EndsWith("embedder", StringComparison.Ordinal) ||
             lower.EndsWith("backbone", StringComparison.Ordinal) || lower.EndsWith("basemodel", StringComparison.Ordinal))
    {
        preferred.Add("new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(inputFeatures: 8, outputSize: 4))");
        preferred.Add("new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(inputFeatures: 8, outputSize: 4))");
    }
    // A count or a size is a number. Without this the solver reaches the Matrix candidate first and
    // accepts it, because `var featureCount = new Matrix<double>(...)` compiles perfectly well and is
    // nonsense — the compiler cannot object to a name, only to a type.
    else if (lower.EndsWith("count", StringComparison.Ordinal) || lower.EndsWith("size", StringComparison.Ordinal) ||
             lower.EndsWith("dim", StringComparison.Ordinal) || lower.EndsWith("length", StringComparison.Ordinal) ||
             lower.StartsWith("num", StringComparison.Ordinal) || lower is "epochs" or "steps" or "index" or "seed")
    {
        preferred.Add("32");
        preferred.Add("0.5");
    }
    else if (lower.EndsWith("rate", StringComparison.Ordinal) || lower.EndsWith("threshold", StringComparison.Ordinal) ||
             lower.EndsWith("ratio", StringComparison.Ordinal) || lower is "alpha" or "beta" or "gamma" or "temperature")
    {
        preferred.Add("0.5");
        preferred.Add("32");
    }
    else if (lower.EndsWith("name", StringComparison.Ordinal) || lower.EndsWith("path", StringComparison.Ordinal) ||
             lower is "prompt" or "text" or "query" or "sentence")
    {
        preferred.Add("\"example\"");
    }

    else if (lower.Contains("audio") || lower.Contains("signal") || lower.Contains("wave") ||
        lower.Contains("melody") || lower.Contains("speech"))
    {
        preferred.Add("Tensor<float>.CreateRandom(1, 16000)");
        preferred.Add("Tensor<double>.CreateRandom(1, 16000)");
    }
    else if (lower.Contains("image") || lower.Contains("frame") || lower.Contains("photo") ||
             lower.Contains("mask") || lower.Contains("pixel"))
    {
        preferred.Add("Tensor<float>.CreateRandom(1, 3, 32, 32)");
        preferred.Add("Tensor<double>.CreateRandom(1, 3, 32, 32)");
    }
    else if (lower.Contains("token") || lower.Contains("sequence") || lower.Contains("series") ||
             lower.Contains("window") || lower.Contains("prices"))
    {
        preferred.Add("Tensor<float>.CreateRandom(1, 128)");
        preferred.Add("new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0 })");
    }

    // Name-matched library types come after the shape hints but before the generic fallbacks, so a
    // recognisable input still gets a Matrix or a Tensor while a domain object gets its real type.
    return preferred.Concat(TypeCandidatesFor(name)).Concat(DeclarationCandidates);
}


// Returns the namespace a documented member lives in, or null. Rather than parse "M:Ns.Type.Method(args)"
// by counting dots — which misreads nested types, explicit interface implementations and generic arity —
// this takes the longest known namespace that prefixes the name, using the set discovered above.
string? HomeNamespace(string memberName)
{
    int colon = memberName.IndexOf(':');
    string name = colon >= 0 ? memberName[(colon + 1)..] : memberName;
    int paren = name.IndexOf('(');
    if (paren >= 0) name = name[..paren];

    string? best = null;
    foreach (var ns in discoveredUsings)
    {
        if (name.StartsWith(ns + ".", StringComparison.Ordinal) &&
            (best is null || ns.Length > best.Length))
        {
            best = ns;
        }
    }
    return best;
}

// Compiles one snippet, recording pass/fail against the given key.
//
// homeNamespace, when given, wraps the snippet in that namespace. Importing every AiDotNet namespace makes
// the 33 type names that are declared in two namespaces ambiguous — an artefact of the harness, not of the
// example, since a reader imports the handful of namespaces they need rather than all 867. Compiling an
// example inside the namespace of the member that documents it reproduces the reader's situation exactly:
// C# resolves a type in the containing namespace ahead of anything a using directive brought in, so
// `Donut` in the docs for AiDotNet.VisionLanguage.Document.Donut means that one, as the reader intends.
// Builds the compilable unit for a snippet, optionally with extra declarations prepended to its body.
string Compose(string code, string? homeNamespace, IEnumerable<string>? extraDeclarations)
{
    var sb = new StringBuilder();
    var body = new StringBuilder();
    bool bodyStarted = false;
    foreach (var line in code.Replace("\r\n", "\n").Split('\n'))
    {
        if (!bodyStarted && usingRe.IsMatch(line)) sb.Append(line.Trim()).Append('\n');
        else { bodyStarted = bodyStarted || line.Trim().Length > 0; body.Append(line).Append('\n'); }
    }

    string bodyText = body.ToString();

    // Decide "declares types" from the first line that is actually code. A snippet routinely opens with a
    // line or two of // comment explaining itself, and testing the raw first line meant those examples
    // were compiled as statements — where a class declaration is a syntax error, reported as
    // "Invalid expression term 'partial'" rather than anything a reader could act on.
    string firstCode = bodyText.Replace("\r\n", "\n").Split('\n')
        .Select(l => l.Trim())
        .FirstOrDefault(l => l.Length > 0 && !l.StartsWith("//", StringComparison.Ordinal)) ?? "";
    bool isTypes = typeStartRe.IsMatch(firstCode);

    // Declarations only make sense inside a method body. A snippet that declares its own types is
    // compiled as top-level declarations, where a `var` line would be a class member and not compile.
    if (extraDeclarations is not null && !isTypes)
    {
        bodyText = string.Concat(extraDeclarations.Select(d => d + "\n")) + bodyText;
    }

    string unit = isTypes
        ? bodyText
        : "static class __Snippet { static async System.Threading.Tasks.Task __Run() {\n" + bodyText + "\n} }";

    if (homeNamespace is not null)
    {
        unit = $"namespace {homeNamespace} {{\n{unit}\n}}";
    }

    return commonUsings + sb + unit;
}

List<Diagnostic> Compile(string source, int id)
{
    var t = CSharpSyntaxTree.ParseText(source, parse);
    var c = CSharpCompilation.Create("probe" + id, new[] { t }, refs, options);
    return c.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001").ToList();
}

// ── Facade conformance ──
//
// AiModelBuilder and AiModelResult are the whole supported surface: users configure a model through the
// builder and predict through the result. An example that trains or predicts on the model object directly
// teaches people to reach past that, and every such example is a place where the facade's own guarantees
// — validation, preprocessing, the training pipeline, the callbacks — are silently skipped.
//
// Constructing a model is NOT the violation. ConfigureModel takes an IFullModel, so
// `.ConfigureModel(new SimpleRegression<double>())` is exactly right. What this looks for is a
// Train/Predict/Fit call whose RECEIVER is a model rather than an AiModelResult, resolved through the
// semantic model rather than matched by name — a receiver's type is not something a regex can know.
// Facts a rewrite needs, gathered where they can be known exactly. Which builder type arguments a model
// requires is a property of its IFullModel implementation, not of its name — a regression is
// <double, Matrix<double>, Vector<double>> while a network is <T, Tensor<T>, Tensor<T>> — so the plan is
// produced here, from the semantic model, and applied as text elsewhere.
List<string> FacadePlan(string source, string fileKey, int idx)
{
    var plan = new List<string>();
    var tree = CSharpSyntaxTree.ParseText(source, parse);
    var comp = CSharpCompilation.Create("plan" + (++probeId), new[] { tree }, refs, options);
    var model = comp.GetSemanticModel(tree);
    var root = tree.GetRoot();

    // A block that predicts without ever training is converted too. Those examples show a model being
    // constructed and immediately predicted from, which is not a workflow that works — a model has to be
    // fitted first — so routing them through the facade also corrects what they teach. The training
    // arguments are left empty and the applier synthesises data from TInput and TOutput.
    bool trains = root.DescendantNodes().OfType<InvocationExpressionSyntax>().Any(c =>
        c.Expression is MemberAccessExpressionSyntax a &&
        a.Name.Identifier.ValueText is "Train" or "Fit");

    foreach (var call in root.DescendantNodes().OfType<InvocationExpressionSyntax>())
    {
        if (call.Expression is not MemberAccessExpressionSyntax access) continue;
        var called = access.Name.Identifier.ValueText;
        if (called is not ("Train" or "Fit") && !(called == "Predict" && !trains)) continue;
        if (access.Expression is not IdentifierNameSyntax id) continue;

        var receiver = model.GetTypeInfo(access.Expression).Type;
        if (receiver is null || receiver.Name == "AiModelResult") continue;

        var full = receiver.AllInterfaces.FirstOrDefault(i => i.Name == "IFullModel");
        if (full is null || full.TypeArguments.Length != 3) continue;

        // the declaration this variable came from, so its constructor can move into ConfigureModel
        var declarator = root.DescendantNodes().OfType<VariableDeclaratorSyntax>()
            .FirstOrDefault(v => v.Identifier.ValueText == id.Identifier.ValueText);
        string ctor = declarator?.Initializer?.Value.ToString().Replace("\r", " ").Replace("\n", " ") ?? "";
        if (ctor.Length == 0) continue;

        string args = called == "Predict"
            ? "<synthesise>"
            : string.Join(" | ", call.ArgumentList.Arguments.Select(a => a.ToString()));
        string targs = string.Join(", ", full.TypeArguments.Select(a => a.ToDisplayString(FullType)));

        plan.Add(string.Join("\t", fileKey, idx.ToString(), id.Identifier.ValueText,
                             Regex.Replace(ctor, @"\s+", " "), targs, args));
    }
    return plan;
}

List<string> FacadeViolations(string source, string fileKey)
{
    var found = new List<string>();
    var tree = CSharpSyntaxTree.ParseText(source, parse);
    var comp = CSharpCompilation.Create("facade" + (++probeId), new[] { tree }, refs, options);
    var model = comp.GetSemanticModel(tree);

    foreach (var call in tree.GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>())
    {
        if (call.Expression is not MemberAccessExpressionSyntax access) continue;
        var name = access.Name.Identifier.ValueText;
        if (name is not ("Train" or "Predict" or "Fit" or "TrainAsync" or "PredictAsync")) continue;

        var receiver = model.GetTypeInfo(access.Expression).Type;
        if (receiver is null) continue;
        if (receiver.Name is "AiModelResult") continue;               // the supported path

        bool isModel = receiver.AllInterfaces.Any(i => i.Name is "IFullModel" or "IPredictiveModel")
                       || receiver.Name is "IFullModel" or "IPredictiveModel";
        if (isModel)
        {
            found.Add($"{fileKey}: {receiver.Name}.{name}(...) — train and predict through AiModelResult");
        }
    }
    return found;
}

// ── Completing a documented partial class ──
//
// Some examples exist to show what goes ON a type, not how to write a whole one: three attributes on
// three fields of a `partial class MyNorm<T> : LayerBase<T>`. Six lines that say exactly one thing. They
// cannot compile alone, because the base has abstract members the fragment does not implement — and
// writing those in would take the example to about twenty-five lines and bury the three that matter.
//
// So the harness supplies the other half of the partial class: it asks Roslyn which inherited members are
// unimplemented and emits a sibling part with stub overrides plus a constructor chaining to the base.
// Nothing is skipped and nothing is hidden from the reader — the fragment is still compiled against the
// real base type, so a renamed attribute, a changed base, or a member that no longer exists still fails
// the build. That is the difference between this and an opt-out marker.

string? CompletionFor(string source)
{
    var tree = CSharpSyntaxTree.ParseText(source, parse);
    var comp = CSharpCompilation.Create("complete", new[] { tree }, refs, options);
    var model = comp.GetSemanticModel(tree);

    var parts = tree.GetRoot().DescendantNodes().OfType<TypeDeclarationSyntax>()
        .Where(d => d.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
        .ToList();
    if (parts.Count == 0) return null;

    var sb = new StringBuilder();
    foreach (var part in parts)
    {
        if (model.GetDeclaredSymbol(part) is not INamedTypeSymbol type) continue;

        var missing = new List<string>();
        for (var b = type.BaseType; b is not null; b = b.BaseType)
        {
            foreach (var m in b.GetMembers())
            {
                if (!m.IsAbstract) continue;
                if (type.FindImplementationForInterfaceMember(m) is not null) continue;
                if (type.GetMembers(m.Name).Any()) continue;
                var stub = StubFor(m);
                if (stub is not null) missing.Add(stub);
            }
        }
        foreach (var iface in type.AllInterfaces)
        {
            foreach (var m in iface.GetMembers())
            {
                if (type.FindImplementationForInterfaceMember(m) is not null) continue;
                if (type.GetMembers(m.Name).Any()) continue;
                var stub = StubFor(m, asOverride: false);
                if (stub is not null) missing.Add(stub);
            }
        }

        // A base with no parameterless constructor needs one chained explicitly.
        string ctor = "";
        var baseCtor = type.BaseType?.InstanceConstructors
            .Where(c => c.DeclaredAccessibility != Accessibility.Private)
            .OrderBy(c => c.Parameters.Length)
            .FirstOrDefault();
        if (baseCtor is not null && baseCtor.Parameters.Length > 0 &&
            !type.InstanceConstructors.Any(c => c.Parameters.Length == 0 && !c.IsImplicitlyDeclared))
        {
            var args = string.Join(", ", baseCtor.Parameters.Select(
                p => $"default({p.Type.ToDisplayString(FullType)})"));
            ctor = $"    public {part.Identifier.ValueText}() : base({args}) {{ }}\n";
        }

        if (missing.Count == 0 && ctor.Length == 0) continue;

        var typeParams = part.TypeParameterList?.ToString() ?? "";
        sb.Append($"public partial class {part.Identifier.ValueText}{typeParams}\n{{\n")
          .Append(ctor)
          .Append(string.Concat(missing.Select(m => "    " + m + "\n")))
          .Append("}\n");
    }

    return sb.Length == 0 ? null : sb.ToString();
}

string? StubFor(ISymbol member, bool asOverride = true)
{
    // An override may not widen access. LayerBase.InitializeLayers is protected, so emitting the stub as
    // public is itself a compile error, and one that looks like a defect in the example rather than in
    // the harness.
    string access = member.DeclaredAccessibility switch
    {
        Accessibility.Protected => "protected",
        Accessibility.ProtectedOrInternal => "protected internal",
        Accessibility.Internal => "internal",
        _ => "public",
    };
    string mod = asOverride ? $"{access} override " : $"{access} ";
    switch (member)
    {
        case IMethodSymbol { MethodKind: MethodKind.Ordinary } method:
        {
            var ps = string.Join(", ", method.Parameters.Select(
                p => $"{p.Type.ToDisplayString(FullType)} {p.Name}"));
            var ret = method.ReturnType.ToDisplayString(FullType);
            return method.ReturnsVoid
                ? $"{mod}void {method.Name}({ps}) {{ }}"
                : $"{mod}{ret} {method.Name}({ps}) => default({ret});";
        }
        case IPropertySymbol property:
        {
            var t = property.Type.ToDisplayString(FullType);
            return property.SetMethod is null
                ? $"{mod}{t} {property.Name} => default({t});"
                : $"{mod}{t} {property.Name} {{ get; set; }}";
        }
        default:
            return null;
    }
}

void Check(string code, string fileKey, int idx, string? homeNamespace = null)
{
    total++;
    string source = Compose(code, homeNamespace, null);

    var tree = CSharpSyntaxTree.ParseText(source, parse);
    var comp = CSharpCompilation.Create("snip" + total, new[] { tree }, refs, options);
    var errors = comp.GetDiagnostics()
        .Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001")
        .ToList();

    // A documented partial class is a fragment by design; supply its other half and re-check.
    //
    // The generated part must sit in the SAME namespace as the fragment. Appended after the namespace
    // block it lands in the global one instead, which declares an unrelated type of the same name and
    // completes nothing — the errors come back identical and the whole mechanism looks inert.
    if (errors.Any(e => e.Id is "CS0534" or "CS0535" or "CS1729" or "CS0501") &&
        CompletionFor(source) is { } completion)
    {
        string placed = homeNamespace is null
            ? completion
            : $"namespace {homeNamespace} {{\n{completion}\n}}";
        var completed = Compile(source + "\n" + placed, ++probeId);
        if (completed.Count < errors.Count)
        {
            errors = completed;
        }
    }

    if (errors.Count == 0)
    {
        pass++;
    }
    else
    {
        // Record every error, not just the first. A snippet routinely carries several independent
        // defects, and reporting one at a time means a repair pass fixes one, a full rebuild follows,
        // and the next is revealed — turning what could be a single sweep into a dozen slow rounds.
        foreach (var e in errors)
        {
            failures.Add((fileKey, idx, $"{e.Id}: {e.GetMessage()}"));
        }
    }

    // --suggest-decls: for a snippet that only lacks variable declarations, work out declarations that
    // make it compile and report them, instead of guessing a type from the variable's name. Naming
    // conventions are not reliable enough — `q` and `r` are a QR factorisation as often as ARIMA orders —
    // and a guess that merely compiles can put a false statement in the documentation. Here the compiler
    // decides: a candidate is accepted only when it removes the error and introduces none.
    if (suggestDecls && errors.Count > 0 && errors.All(e => e.Id == "CS0103"))
    {
        var accepted = new List<string>();
        var undefined = errors
            .Select(e => Regex.Match(e.GetMessage(), @"The name '([^']+)'"))
            .Where(m => m.Success)
            .Select(m => m.Groups[1].Value)
            .Distinct(StringComparer.Ordinal)
            .ToList();

        foreach (var name in undefined)
        {
            // Score EVERY candidate and keep the best, rather than the first that helps.
            //
            // Taking the first improvement is what made this fail on a Train(matrix, vector) example: the
            // Matrix candidate resolved `targetVector` and turned its CS0103 into a CS1503, which counts
            // as fewer errors, so the Vector candidate that would have compiled cleanly was never tried.
            // Scoring all of them lets a zero-error answer win whenever one exists.
            //
            // The compiler decides whether a candidate is CORRECT; the variable's name only orders
            // candidates the compiler judges equally. A tensor for `noisyAudioTensor` binds just as well
            // at (1, 3, 32, 32) as at (1, 16000) — both compile, but one reads like an image — so among
            // equal scores the earlier (name-preferred) candidate wins.
            string? best = null;
            int bestScore = int.MaxValue;

            foreach (var candidate in OrderCandidatesFor(name, fileKey))
            {
                string decl = $"var {name} = {candidate};";
                var trial = Compile(Compose(code, homeNamespace, accepted.Append(decl)), ++probeId);
                bool nameResolved = !trial.Any(d =>
                    d.Id == "CS0103" && d.GetMessage().Contains($"'{name}'", StringComparison.Ordinal));
                if (!nameResolved) continue;

                if (trial.Count < bestScore)
                {
                    bestScore = trial.Count;
                    best = decl;
                    if (bestScore == 0) break;          // cannot do better than a clean compile
                }
            }

            if (best is not null && bestScore < errors.Count)
            {
                accepted.Add(best);
            }
            else
            {
                // No candidate resolved this name. Recording it is how the candidate list grows: one
                // unbuildable name blocks its whole snippet, so these are exactly the gaps worth filling.
                unsolvable.Add(name);
            }
        }

        // Only emit when the snippet compiles CLEANLY with the accepted set.
        //
        // Emitting partial solutions was tried and withdrawn. It looked safe — each candidate is accepted
        // only if it reduces the error count — but reducing a count is not the same as being right: two
        // CS0103s can collapse into one CS1503, so a wrongly-typed declaration still counts as progress.
        // Applied across 43 snippets it moved 51 errors from one class to another, converted none to
        // passing, and left declarations in the documentation whose types the compiler had never
        // actually endorsed. A clean compile is the only signal worth acting on here.
        if (accepted.Count > 0 &&
            Compile(Compose(code, homeNamespace, accepted), ++probeId).Count == 0)
        {
            foreach (var d in accepted)
            {
                suggestions.Add((fileKey, idx, d));
            }
        }
    }

    if (facadeCheck)
    {
        facadeViolations.AddRange(FacadeViolations(source, fileKey));
    }

    if (facadePlan)
    {
        facadePlanLines.AddRange(FacadePlan(source, fileKey, idx));
    }

    var cur = perFile.GetValueOrDefault(fileKey);
    perFile[fileKey] = (cur.total + 1, cur.fail + (errors.Count == 0 ? 0 : 1));
}

foreach (var root in roots)
{
    // XML doc-comment examples mode.
    if (File.Exists(root) && root.EndsWith(".xml", StringComparison.OrdinalIgnoreCase))
    {
        var xdoc = XDocument.Load(root);
        foreach (var member in xdoc.Descendants("member"))
        {
            var memberName = (string?)member.Attribute("name") ?? "?";
            var home = HomeNamespace(memberName);
            int idx = 0;
            foreach (var codeEl in member.Descendants("example").Elements("code"))
            {
                idx++;
                Check(codeEl.Value, memberName, idx, home);   // XDocument already unescaped &lt; etc.
            }

            // Code shown in <remarks>/<summary> prose, which is where most of the library's copy-and-paste
            // examples actually live. These were exempt from every gate: XML mode looked only inside <example>,
            // and WikiGenerator.NormalizeBlock rewrites ```csharp to ```cs in prose so the markdown pass skips
            // them too. That combination let 67 examples calling a non-existent Build(X, y) overload survive.
            // Opt-in, because prose snippets are often deliberate fragments; run with --include-prose to measure.
            if (includeProse)
            {
                foreach (var codeEl in member.Descendants("code"))
                {
                    if (codeEl.Parent is not null && codeEl.Parent.Name == "example") continue;   // already checked
                    idx++;
                    Check(codeEl.Value, memberName + " (prose)", idx);
                }
            }
        }
        continue;
    }

    if (!Directory.Exists(root)) continue;
    var files = Directory.EnumerateFiles(root, "*.md", SearchOption.AllDirectories)
        .Concat(Directory.EnumerateFiles(root, "*.mdx", SearchOption.AllDirectories))
        .OrderBy(f => f);

    foreach (var file in files)
    {
        var text = File.ReadAllText(file);
        int idx = 0;
        var key = file.Replace('\\', '/');

        // Generated API pages state the type's namespace on their first content line, as `kind` · `Ns`.
        // Using it scopes the page's snippets the same way the XML pass scopes a member's, so a type name
        // that exists in two namespaces resolves to the one the page is actually about.
        var nsMatch = pageNamespaceRe.Match(text);
        string? pageNamespace = nsMatch.Success && discoveredUsings.Contains(nsMatch.Groups[1].Value)
            ? nsMatch.Groups[1].Value
            : null;

        foreach (Match m in blockRe.Matches(text))
        {
            idx++;
            Check(m.Groups[1].Value, key, idx, pageNamespace);
        }
    }
}

Console.WriteLine($"\n=== Example compile results ===");
Console.WriteLine($"Total: {total}   PASS: {pass}   FAIL: {total - pass}\n");

var byCode = failures.GroupBy(f => f.err.Split(':')[0]).OrderByDescending(g => g.Count());
Console.WriteLine("Failures by error code:");
foreach (var g in byCode) Console.WriteLine($"  {g.Count(),4}  {g.Key}");

Console.WriteLine("\nUnits with the most failures:");
foreach (var kv in perFile.Where(p => p.Value.fail > 0).OrderByDescending(p => p.Value.fail).Take(25))
    Console.WriteLine($"  {kv.Value.fail,3}/{kv.Value.total,-3} {kv.Key}");

// --dump <path> writes every failure, not a sample. Needed to work out which namespaces the snippet
// harness must import: a sample cannot tell you which types are unresolved across 1400 snippets.
var dumpIndex = Array.FindIndex(args, a => string.Equals(a, "--dump", StringComparison.OrdinalIgnoreCase));
if (dumpIndex >= 0 && dumpIndex + 1 < args.Length)
{
    File.WriteAllLines(args[dumpIndex + 1], failures.Select(f => $"{f.file}#{f.idx}\t{f.err}"));
    Console.WriteLine($"\nwrote {failures.Count} failures -> {args[dumpIndex + 1]}");
}

// --suggest-decls output: one accepted declaration per line, keyed by snippet. Written next to the dump
// so a repair script can apply them; each was verified to make its snippet compile.
if (suggestDecls)
{
    int i = Array.FindIndex(args, a => string.Equals(a, "--suggest-decls", StringComparison.OrdinalIgnoreCase));
    string path = i >= 0 && i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal)
        ? args[i + 1]
        : "suggested-declarations.tsv";
    File.WriteAllLines(path, suggestions.Select(s => $"{s.file}\t{s.idx}\t{s.decl}"));
    Console.WriteLine(
        $"\nSolved declarations for {suggestions.Select(s => (s.file, s.idx)).Distinct().Count()} snippet(s) " +
        $"({suggestions.Count} declarations) -> {path}");
}

// Names the solver could not build. One of these blocks its entire snippet, so the list is the work
// queue: either the candidate set needs to grow, or the example needs writing by hand.
if (suggestDecls && unsolvable.Count > 0)
{
    Console.WriteLine("\nNames no candidate could build (each blocks its whole snippet):");
    foreach (var g in unsolvable.GroupBy(n => n, StringComparer.Ordinal)
                                .OrderByDescending(g => g.Count())
                                .Take(40))
    {
        Console.WriteLine($"  {g.Count(),3}  {g.Key}");
    }
}

// Ratchet state, declared before the gates that set it.
int failed = total - pass;
bool ratcheted = false;
int exit = 0;

// The conversion plan: one row per call that can be routed through the facade, carrying the model
// variable, its constructor text, the builder's three type arguments and the training arguments.
if (facadePlan)
{
    int p = Array.FindIndex(args, a => string.Equals(a, "--facade-plan", StringComparison.OrdinalIgnoreCase));
    string planPath = p >= 0 && p + 1 < args.Length && !args[p + 1].StartsWith("--", StringComparison.Ordinal)
        ? args[p + 1]
        : "facade-plan.tsv";
    File.WriteAllLines(planPath, facadePlanLines);
    Console.WriteLine($"\nwrote {facadePlanLines.Count} convertible call(s) -> {planPath}");
}

// Facade conformance, and the ratchet that stops it regressing. AiModelBuilder and AiModelResult are the
// whole supported surface; an example that trains or predicts on the model object teaches readers to reach
// past it, skipping the validation, preprocessing and training pipeline the facade exists to run.
if (facadeCheck)
{
    Console.WriteLine($"\nFacade conformance: {facadeViolations.Count} example call(s) train or predict " +
                      "on a model rather than an AiModelResult.");
    foreach (var g in facadeViolations.GroupBy(v => v.Contains(": ") ? v[..v.IndexOf(": ")] : v, StringComparer.Ordinal)
                                      .OrderByDescending(g => g.Count())
                                      .Take(10))
    {
        Console.WriteLine($"  {g.Count(),3}  {g.Key}");
    }

    int i = Array.FindIndex(args, a => string.Equals(a, "--max-facade-violations", StringComparison.OrdinalIgnoreCase));
    if (i >= 0 && i + 1 < args.Length && int.TryParse(args[i + 1], out int facadeCeiling))
    {
        // This is a ratchet in its own right, so it decides the exit code rather than falling through to
        // the all-or-nothing compile check at the bottom. Without this the facade step always failed:
        // examples still fail to compile, so the fallback returned 1 no matter what the facade count was.
        ratcheted = true;

        if (facadeViolations.Count > facadeCeiling)
        {
            Console.WriteLine(
                $"\nFAIL: {facadeViolations.Count} non-facade calls, above the ceiling of {facadeCeiling}. Route the " +
                "example through AiModelBuilder: configure the model, Build(features, labels), then predict " +
                "on the returned AiModelResult.");
            return 1;
        }
        Console.WriteLine(facadeViolations.Count < facadeCeiling
            ? $"OK: {facadeViolations.Count} non-facade calls, below the ceiling of {facadeCeiling}. Lower it."
            : $"OK: {facadeViolations.Count} non-facade calls, matching the ceiling.");
    }
}

Console.WriteLine("\nSample failures:");
foreach (var f in failures.Take(30))
    Console.WriteLine($"  {f.file} #{f.idx}: {f.err}");

// Two ratchets, because they catch different mistakes and neither alone is enough:
//
//   --max-fail <n>  the number of BROKEN examples may not rise. This is the one that catches a newly
//                   added broken example, which a pass-count floor cannot see: add a bad example and the
//                   pass count is unchanged, so a --min-pass gate stays green while the backlog grows.
//   --min-pass <n>  the number of COMPILING examples may not drop. Catches an existing example being
//                   broken even in the same change that deletes another, where the failure count could
//                   stay level.
//
// A plain all-or-nothing gate cannot be switched on part-way through repairing a large backlog — it would
// block every unrelated change until the last example is fixed — and a warn-only gate lets the count
// silently rot back down, which is how these examples decayed in the first place.

int Arg(string name)
{
    int i = Array.FindIndex(args, a => string.Equals(a, name, StringComparison.OrdinalIgnoreCase));
    return i >= 0 && i + 1 < args.Length && int.TryParse(args[i + 1], out int v) ? v : -1;
}

if (Arg("--max-fail") is var ceiling and >= 0)
{
    ratcheted = true;
    if (failed > ceiling)
    {
        Console.WriteLine(
            $"\nFAIL: {failed} examples do not compile, above the recorded ceiling of {ceiling}. This change " +
            "either broke a working example or added one that never compiled. Fix it, then lower the ceiling " +
            $"in the workflow to {failed}.");
        exit = 1;
    }
    else
    {
        Console.WriteLine(failed < ceiling
            ? $"\nOK: {failed} examples fail, below the ceiling of {ceiling}. Lower the ceiling to {failed}."
            : $"\nOK: {failed} examples fail, matching the ceiling.");
    }
}

if (Arg("--min-pass") is var floor and >= 0)
{
    ratcheted = true;
    if (pass < floor)
    {
        Console.WriteLine(
            $"\nFAIL: {pass} examples compile, below the recorded floor of {floor}. A change here broke an " +
            "example that used to compile. Fix it, or — if an example was deliberately removed — lower the " +
            "floor in the workflow with the reason in the commit message.");
        exit = 1;
    }
    else
    {
        Console.WriteLine(pass > floor
            ? $"\nOK: {pass} examples compile, above the floor of {floor}. Raise the floor to {pass}."
            : $"\nOK: {pass} examples compile, matching the floor.");
    }
}

if (ratcheted) return exit;

return failed == 0 ? 0 : 1;
