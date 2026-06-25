// DocSnippetVerify — compiles every ```csharp fenced block in the docs/website
// markdown against the real AiDotNet assemblies (via Roslyn) and reports which
// fail. This is the compile gate for documentation snippets: it catches the same
// API drift that broke the samples, without needing to run code that references
// external files/data.
//
// Usage: dotnet run --project tools/DocSnippetVerify [root1 root2 ...]
//        (defaults to "docs" and "website")

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

string[] roots = args.Length > 0 ? args : new[] { "docs", "website" };

// ── Reference set: AiDotNet + all transitive deps (this tool's own output) + BCL ──
var refs = new List<MetadataReference>();
var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
void AddRef(string path)
{
    var name = Path.GetFileName(path);
    if (!seen.Add(name)) return;
    // Skip native DLLs (libopenblas, runtime shims, …) — only managed assemblies
    // can be Roslyn metadata references.
    try { System.Reflection.AssemblyName.GetAssemblyName(path); }
    catch { return; }
    try { refs.Add(MetadataReference.CreateFromFile(path)); } catch { }
}
foreach (var dll in Directory.GetFiles(AppContext.BaseDirectory, "*.dll")) AddRef(dll);
if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string tpa)
    foreach (var p in tpa.Split(Path.PathSeparator)) AddRef(p);

// Common usings so fragments that assume obvious context still resolve; the
// snippet's own usings are kept too.
const string commonUsings =
    "using System;using System.Collections.Generic;using System.Linq;" +
    "using System.Threading;using System.Threading.Tasks;" +
    "using AiDotNet;using AiDotNet.Tensors;using AiDotNet.Tensors.LinearAlgebra;\n";

var options = new CSharpCompilationOptions(
    OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true,
    nullableContextOptions: NullableContextOptions.Disable);
var parse = new CSharpParseOptions(LanguageVersion.Latest);

var blockRe = new Regex("```csharp\\s*?\\n(.*?)```", RegexOptions.Singleline);
var usingRe = new Regex(@"^\s*using\s+[A-Za-z_][\w.]*\s*;\s*$");
var typeStartRe = new Regex(@"^\s*(\[|public |internal |private |protected |static |abstract |sealed |partial |class |record |struct |enum |interface |namespace )");

int total = 0, pass = 0;
var failures = new List<(string file, int idx, string err)>();
var perFile = new Dictionary<string, (int total, int fail)>();

foreach (var root in roots)
{
    if (!Directory.Exists(root)) continue;
    var files = Directory.EnumerateFiles(root, "*.md", SearchOption.AllDirectories)
        .Concat(Directory.EnumerateFiles(root, "*.mdx", SearchOption.AllDirectories))
        .OrderBy(f => f);

    foreach (var file in files)
    {
        var text = File.ReadAllText(file);
        int idx = 0;
        foreach (Match m in blockRe.Matches(text))
        {
            total++; idx++;
            var code = m.Groups[1].Value;

            // Separate the snippet's own using directives from its body.
            var sb = new StringBuilder();
            var body = new StringBuilder();
            bool bodyStarted = false;
            foreach (var line in code.Replace("\r\n", "\n").Split('\n'))
            {
                if (!bodyStarted && usingRe.IsMatch(line)) sb.Append(line.Trim()).Append('\n');
                else { bodyStarted = bodyStarted || line.Trim().Length > 0; body.Append(line).Append('\n'); }
            }

            // Type declarations compile as-is; loose statements wrap in an async method.
            string bodyText = body.ToString();
            bool isTypes = typeStartRe.IsMatch(bodyText.TrimStart());
            string source = commonUsings + sb +
                (isTypes
                    ? bodyText
                    : "static class __Snippet { static async System.Threading.Tasks.Task __Run() {\n" + bodyText + "\n} }");

            var tree = CSharpSyntaxTree.ParseText(source, parse);
            var comp = CSharpCompilation.Create("snip" + total, new[] { tree }, refs, options);
            var errors = comp.GetDiagnostics()
                .Where(d => d.Severity == DiagnosticSeverity.Error)
                // Ignore "unreachable/unused" style and entry-point noise that don't reflect API drift.
                .Where(d => d.Id != "CS5001")
                .ToList();

            if (errors.Count == 0) pass++;
            else failures.Add((file, idx, $"{errors[0].Id}: {errors[0].GetMessage()}"));

            var key = file.Replace('\\', '/');
            var cur = perFile.GetValueOrDefault(key);
            perFile[key] = (cur.total + 1, cur.fail + (errors.Count == 0 ? 0 : 1));
        }
    }
}

Console.WriteLine($"\n=== Doc snippet compile results ===");
Console.WriteLine($"Total snippets: {total}   PASS: {pass}   FAIL: {total - pass}\n");

// Categorize the first error of each failure. CS0103 (undefined name) and a
// non-AiDotNet CS0246 are usually fragment context (a snippet referencing a
// variable defined in surrounding prose); the API-drift signal lives in
// removed members/ctors/types.
var byCode = failures.GroupBy(f => f.err.Split(':')[0]).OrderByDescending(g => g.Count());
Console.WriteLine("Failures by error code:");
foreach (var g in byCode) Console.WriteLine($"  {g.Count(),4}  {g.Key}");
int drift = failures.Count(f => f.err.StartsWith("CS1061") || f.err.StartsWith("CS1501") ||
    f.err.StartsWith("CS7036") || f.err.StartsWith("CS1739") || f.err.StartsWith("CS1503") ||
    (f.err.StartsWith("CS0246") && f.err.Contains("AiDotNet")));
Console.WriteLine($"\n  ~{drift} look like genuine AiDotNet API drift (removed member/ctor/type);");
Console.WriteLine($"  the remainder are mostly fragment context (undefined names) or missing usings.\n");

Console.WriteLine("Files with the most failures:");
foreach (var kv in perFile.Where(p => p.Value.fail > 0).OrderByDescending(p => p.Value.fail).Take(20))
    Console.WriteLine($"  {kv.Value.fail,3}/{kv.Value.total,-3} {kv.Key}");

Console.WriteLine("\nSample failures:");
foreach (var f in failures.Take(30))
    Console.WriteLine($"  {f.file} #{f.idx}: {f.err}");

return total - pass == 0 ? 0 : 1;
