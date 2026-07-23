// ExampleFixer — the auto-correction half of the example-verification pipeline.
//
// For every <example> in the AiDotNet XML doc comments it: compiles the example,
// injects the `using` directives for any unresolved types (looked up by reflecting over
// the real assembly), applies known mechanical drift fixes, and re-compiles. It reports
// how many examples each layer fixes and what genuine drift remains. (A later pass writes
// the corrected examples back into the source .cs XML comments.)
//
// Usage: dotnet run --project tools/ExampleFixer [pathToAiDotNet.xml]

using System.Reflection;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

string xmlPath = args.Length > 0 ? args[0] : "src/bin/Debug/net8.0/AiDotNet.xml";
if (!File.Exists(xmlPath)) { Console.Error.WriteLine($"not found: {xmlPath}"); return 1; }

// ── Roslyn reference set (this tool's own output + BCL) ──
var refs = new List<MetadataReference>();
var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
void AddRef(string p)
{
    if (!seen.Add(Path.GetFileName(p))) return;
    try { AssemblyName.GetAssemblyName(p); } catch { return; }
    try { refs.Add(MetadataReference.CreateFromFile(p)); } catch { }
}
foreach (var dll in Directory.GetFiles(AppContext.BaseDirectory, "*.dll")) AddRef(dll);
if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string tpa)
    foreach (var p in tpa.Split(Path.PathSeparator)) AddRef(p);

// ── Reflect: simple type name -> namespace(s) ──
var nsMap = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
foreach (var name in new[] { "AiDotNet.dll", "AiDotNet.Tensors.dll" })
{
    var path = Path.Combine(AppContext.BaseDirectory, name);
    if (!File.Exists(path)) continue;
    try
    {
        foreach (var t in Assembly.LoadFrom(path).GetExportedTypes())
        {
            if (t.Namespace is null) continue;
            var simple = t.Name;
            int tick = simple.IndexOf('`');
            if (tick >= 0) simple = simple.Substring(0, tick);
            if (!nsMap.TryGetValue(simple, out var set)) nsMap[simple] = set = new();
            set.Add(t.Namespace);
        }
    }
    catch (Exception ex) { Console.Error.WriteLine($"reflect {name}: {ex.Message}"); }
}
Console.WriteLine($"Indexed {nsMap.Count} type names from the assembly.");

const string commonUsings =
    "using System;using System.Collections.Generic;using System.Linq;using System.Threading.Tasks;" +
    "using AiDotNet;using AiDotNet.Tensors;using AiDotNet.Tensors.LinearAlgebra;\n";
var compOptions = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
    allowUnsafe: true, nullableContextOptions: NullableContextOptions.Disable);
var parse = new CSharpParseOptions(LanguageVersion.Latest);
var typeNameRe = new Regex(@"'([A-Za-z_]\w*)");
var typeStartRe = new Regex(@"^\s*(\[|public |internal |static |class |record |struct |enum |namespace )");

List<Diagnostic> Compile(string usings, string body)
{
    bool isTypes = typeStartRe.IsMatch(body.TrimStart());
    string src = commonUsings + usings +
        (isTypes ? body : "static class __S { static async Task __R() {\n" + body + "\n} }");
    var comp = CSharpCompilation.Create("ex", new[] { CSharpSyntaxTree.ParseText(src, parse) }, refs, compOptions);
    return comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001").ToList();
}

// Known mechanical drift fixes applied to the example body.
string ApplyDriftFixes(string body) => body
    .Replace("LearningRate =", "InitialLearningRate =")
    .Replace("InitialInitialLearningRate", "InitialLearningRate");

int total = 0, alreadyOk = 0, fixedByUsings = 0, fixedByDrift = 0, residual = 0;
var residualHist = new Dictionary<string, int>();

var doc = XDocument.Load(xmlPath);
foreach (var code in doc.Descendants("example").Elements("code"))
{
    total++;
    string body = code.Value;

    if (Compile("", body).Count == 0) { alreadyOk++; continue; }

    // Layer 1: inject usings for unresolved (single-namespace) types.
    var add = new SortedSet<string>();
    foreach (var d in Compile("", body).Where(d => d.Id == "CS0246"))
    {
        var m = typeNameRe.Match(d.GetMessage());
        if (m.Success && nsMap.TryGetValue(m.Groups[1].Value, out var nss) && nss.Count == 1)
            add.Add(nss.First());
    }
    string usings = string.Concat(add.Select(n => $"using {n};\n"));
    if (add.Count > 0 && Compile(usings, body).Count == 0) { fixedByUsings++; continue; }

    // Layer 2: + mechanical drift fixes.
    string fixedBody = ApplyDriftFixes(body);
    if (Compile(usings, fixedBody).Count == 0) { fixedByDrift++; continue; }

    residual++;
    var errs = Compile(usings, fixedBody);
    var first = errs.Count > 0 ? errs[0].Id : "none";
    residualHist[first] = residualHist.GetValueOrDefault(first) + 1;
}

Console.WriteLine($"\nExamples: {total}");
Console.WriteLine($"  already compiled        : {alreadyOk}");
Console.WriteLine($"  fixed by using-injection: {fixedByUsings}");
Console.WriteLine($"  fixed by + drift rename : {fixedByDrift}");
Console.WriteLine($"  residual (real drift)   : {residual}");
double auto = total == 0 ? 0 : 100.0 * (alreadyOk + fixedByUsings + fixedByDrift) / total;
Console.WriteLine($"  => {auto:F1}% auto-greenable");
Console.WriteLine("\nResidual first-error histogram:");
foreach (var kv in residualHist.OrderByDescending(k => k.Value).Take(15))
    Console.WriteLine($"  {kv.Value,5}  {kv.Key}");
return 0;
