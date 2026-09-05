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
var usingRe = new Regex(@"^\s*using\s+[A-Za-z_][\w.]*\s*;\s*$");
var typeStartRe = new Regex(@"^\s*(\[|public |internal |private |protected |static |abstract |sealed |partial |class |record |struct |enum |interface |namespace )");

int total = 0, pass = 0;
var failures = new List<(string file, int idx, string err)>();
var perFile = new Dictionary<string, (int total, int fail)>();

// Compiles one snippet, recording pass/fail against the given key.
void Check(string code, string fileKey, int idx)
{
    total++;
    var sb = new StringBuilder();
    var body = new StringBuilder();
    bool bodyStarted = false;
    foreach (var line in code.Replace("\r\n", "\n").Split('\n'))
    {
        if (!bodyStarted && usingRe.IsMatch(line)) sb.Append(line.Trim()).Append('\n');
        else { bodyStarted = bodyStarted || line.Trim().Length > 0; body.Append(line).Append('\n'); }
    }

    string bodyText = body.ToString();
    bool isTypes = typeStartRe.IsMatch(bodyText.TrimStart());
    string source = commonUsings + sb +
        (isTypes
            ? bodyText
            : "static class __Snippet { static async System.Threading.Tasks.Task __Run() {\n" + bodyText + "\n} }");

    var tree = CSharpSyntaxTree.ParseText(source, parse);
    var comp = CSharpCompilation.Create("snip" + total, new[] { tree }, refs, options);
    var errors = comp.GetDiagnostics()
        .Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001")
        .ToList();

    if (errors.Count == 0) pass++;
    else failures.Add((fileKey, idx, $"{errors[0].Id}: {errors[0].GetMessage()}"));

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
            int idx = 0;
            foreach (var codeEl in member.Descendants("example").Elements("code"))
            {
                idx++;
                Check(codeEl.Value, memberName, idx);   // XDocument already unescaped &lt; etc.
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
        foreach (Match m in blockRe.Matches(text))
        {
            idx++;
            Check(m.Groups[1].Value, key, idx);
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

Console.WriteLine("\nSample failures:");
foreach (var f in failures.Take(30))
    Console.WriteLine($"  {f.file} #{f.idx}: {f.err}");

return total - pass == 0 ? 0 : 1;
