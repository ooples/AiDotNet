// WikiGenerator — generates a rich, per-item documentation wiki from the AiDotNet XML doc
// comments. The prose (summary, "For Beginners", "How It Works") is pulled from the source
// XML; the example is GENERATED from a per-category template and compiled in-process, so
// only examples that actually compile against the real assembly are emitted (the source
// examples are drifted fragments, so they are not used).
//
// Usage: dotnet run --project tools/WikiGenerator [pathToAiDotNet.xml] [outputDir]

using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

string xmlPath = args.Length > 0 ? args[0] : "src/bin/Debug/net8.0/AiDotNet.xml";
string outDir = args.Length > 1 ? args[1] : "wiki";
// URL prefix used for intra-wiki links (root-absolute, matching the Astro site's link style).
string urlBase = args.Length > 2 ? args[2].TrimEnd('/') : "/docs/reference/wiki";
// Link style: "relative" emits Jekyll-native ./Type.md links; otherwise root-absolute Astro links.
bool relLinks = args.Length > 3 && string.Equals(args[3], "relative", StringComparison.OrdinalIgnoreCase);

if (!File.Exists(xmlPath))
{
    Console.Error.WriteLine($"XML doc file not found: {xmlPath}. Build src/AiDotNet.csproj first.");
    return 1;
}

// ── Roslyn reference set for compile-checking examples ──
var refs = new List<MetadataReference>();
var seenRef = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
void AddRef(string p)
{
    if (!seenRef.Add(Path.GetFileName(p))) return;
    try { AssemblyName.GetAssemblyName(p); } catch { return; }
    try { refs.Add(MetadataReference.CreateFromFile(p)); } catch { }
}
foreach (var dll in Directory.GetFiles(AppContext.BaseDirectory, "*.dll")) AddRef(dll);
if (AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") is string tpa)
    foreach (var p in tpa.Split(Path.PathSeparator)) AddRef(p);

var compOptions = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
    allowUnsafe: true, nullableContextOptions: NullableContextOptions.Disable);
var parse = new CSharpParseOptions(LanguageVersion.Latest);
var usingLineRe = new Regex(@"^\s*using\s+[A-Za-z_][\w.]*\s*;\s*$", RegexOptions.None, TimeSpan.FromSeconds(1));
// Same implicit using set DocSnippetVerify prepends, so "compiles here" == "passes the gate".
const string commonUsings =
    "using System;using System.Collections.Generic;using System.Linq;" +
    "using System.Threading;using System.Threading.Tasks;" +
    "using AiDotNet;using AiDotNet.Tensors;using AiDotNet.Tensors.LinearAlgebra;\n";
int _dbg = 0;

bool Compiles(string code)
{
    // Hoist leading `using` directives above the synthetic wrapper class — directives are
    // illegal inside a method body. Mirrors DocSnippetVerify's gate so the in-process check
    // accepts exactly what the external markdown gate accepts.
    var usings = new StringBuilder();
    var body = new StringBuilder();
    bool bodyStarted = false;
    foreach (var line in code.Replace("\r\n", "\n").Split('\n'))
    {
        if (!bodyStarted && usingLineRe.IsMatch(line)) usings.Append(line.Trim()).Append('\n');
        else { bodyStarted = bodyStarted || line.Trim().Length > 0; body.Append(line).Append('\n'); }
    }
    var src = commonUsings + usings + "static class __S { static async System.Threading.Tasks.Task __R() {\n" + body + "\n} }";
    var comp = CSharpCompilation.Create("ex", new[] { CSharpSyntaxTree.ParseText(src, parse) }, refs, compOptions);
    var errs = comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001").ToList();
    if (errs.Count > 0 && Environment.GetEnvironmentVariable("WIKI_DEBUG") == "1" && _dbg++ < 5)
        foreach (var e in errs.Take(3)) Console.Error.WriteLine($"  [dbg] {e.Id}: {e.GetMessage()}");
    return errs.Count == 0;
}

var categories = new (string Ns, string Slug, string Title)[]
{
    ("AiDotNet.Optimizers.", "optimizers", "Optimizers"),
    ("AiDotNet.LossFunctions.", "loss-functions", "Loss Functions"),
    ("AiDotNet.Regression.", "regression", "Regression Models"),
    ("AiDotNet.Clustering.Partitioning.", "clustering", "Clustering"),
    ("AiDotNet.Clustering.Density.", "clustering", "Clustering"),
    ("AiDotNet.Clustering.Hierarchical.", "clustering", "Clustering"),
    ("AiDotNet.TimeSeries.", "time-series", "Time-Series Models"),
    ("AiDotNet.LoRA.Adapters.", "lora-adapters", "LoRA / PEFT Adapters"),
};

var doc = XDocument.Load(xmlPath);
var members = doc.Root?.Element("members")?.Elements("member").ToList() ?? new();

int totalPages = 0, withExample = 0;
Directory.CreateDirectory(outDir);
var catSummary = new List<(string Slug, string Title, int Count)>();

foreach (var grp in categories.GroupBy(c => c.Slug))
{
    var dir = Path.Combine(outDir, grp.Key);
    Directory.CreateDirectory(dir);
    var index = new List<(string Type, string Summary)>();
    string title = grp.First().Title;

    foreach (var cat in grp)
    {
        foreach (var m in members)
        {
            var name = (string?)m.Attribute("name");
            if (name is null || !name.StartsWith("T:" + cat.Ns)) continue;
            var rel = name.Substring(("T:" + cat.Ns).Length);
            if (rel.Contains('.')) continue;

            string typeName = StripArity(rel);
            string summary = NormalizeBlock(ToMarkdown(m.Element("summary")));
            if (summary.Length == 0) continue;

            string forBeginners = ExtractForBeginners(m.Element("remarks"));
            string remarks = ExtractRemarksExcludingForBeginners(m.Element("remarks"));

            // Generate + compile-check the example; only keep it if it compiles.
            string example = BuildExample(cat.Slug, typeName, cat.Ns.TrimEnd('.'));
            bool exampleOk = example.Length > 0 && Compiles(example);

            var sb = new StringBuilder();
            sb.AppendLine("---");
            sb.AppendLine($"title: {Yaml(typeName)}");
            sb.AppendLine($"description: {Yaml(FirstSentence(summary))}");
            sb.AppendLine("section: \"Reference\"");
            sb.AppendLine("---").AppendLine();
            sb.AppendLine($"_{title}_").AppendLine();
            sb.AppendLine(summary).AppendLine();
            if (forBeginners.Length > 0)
                sb.AppendLine("## For Beginners").AppendLine().AppendLine(forBeginners).AppendLine();
            if (remarks.Length > 0)
                sb.AppendLine("## How It Works").AppendLine().AppendLine(remarks).AppendLine();
            if (exampleOk)
            {
                sb.AppendLine("## Example").AppendLine();
                sb.AppendLine("```csharp").AppendLine(example).AppendLine("```").AppendLine();
                withExample++;
            }

            WriteFile(Path.Combine(dir, typeName + ".md"), sb.ToString());
            index.Add((typeName, FirstSentence(summary)));
            totalPages++;
        }
    }

    if (index.Count > 0)
    {
        var rows = index.OrderBy(r => r.Type).DistinctBy(r => r.Type).ToList();
        var idx = new StringBuilder();
        idx.AppendLine("---");
        idx.AppendLine($"title: {Yaml(title)}");
        idx.AppendLine($"description: {Yaml("Every " + title + " type in AiDotNet, auto-generated with compile-checked examples.")}");
        idx.AppendLine("section: \"Reference\"");
        idx.AppendLine("---").AppendLine();
        idx.AppendLine($"Every {title} type in AiDotNet — each with a beginner-friendly explanation and, where the snippet compiles against the live library, a runnable example.").AppendLine();
        idx.AppendLine("| Type | Summary |").AppendLine("|:-----|:--------|");
        foreach (var (type, sum) in rows)
        {
            // Astro lowercases route slugs; relative Jekyll links resolve to the real .md filename.
            string href = relLinks ? $"./{type}.md" : $"{urlBase}/{grp.Key}/{type.ToLowerInvariant()}/";
            idx.AppendLine($"| [`{type}`]({href}) | {EscapeCell(sum)} |");
        }
        WriteFile(Path.Combine(dir, "index.md"), idx.ToString());
        catSummary.Add((grp.Key, title, rows.Count));
        Console.WriteLine($"{grp.Key,-16} {index.Count,4} pages");
    }
}

// Top-level wiki landing page (links to each category index).
if (catSummary.Count > 0)
{
    var top = new StringBuilder();
    top.AppendLine("---");
    top.AppendLine("title: \"API Wiki\"");
    top.AppendLine($"description: {Yaml("One compile-checked reference page per optimizer, loss function, model, and adapter in AiDotNet.")}");
    top.AppendLine("section: \"Reference\"");
    top.AppendLine("---").AppendLine();
    top.AppendLine("A page for every type in the library's core families, generated from the source XML documentation. Each page carries the summary, a beginner-friendly explanation, how it works, and — where the snippet compiles against the live library — a runnable example built through the `AiModelBuilder` facade.").AppendLine();
    top.AppendLine("| Category | Types |").AppendLine("|:---------|------:|");
    foreach (var (slug, t, count) in catSummary.OrderBy(c => c.Title).DistinctBy(c => c.Slug))
    {
        string href = relLinks ? $"./{slug}/index.md" : $"{urlBase}/{slug}/";
        top.AppendLine($"| [{t}]({href}) | {count} |");
    }
    WriteFile(Path.Combine(outDir, "index.md"), top.ToString());
}

Console.WriteLine($"----\nTotal: {totalPages} pages ({withExample} with a compiling example) under {outDir}/");
return 0;

// ── frontmatter / table helpers ──────────────────────────────────────────────
// Always write LF so output is byte-identical on Windows and Linux — a CI drift check
// (regenerate + `git diff --exit-code`) would otherwise fail on CRLF/LF differences alone.
static void WriteFile(string path, string content) =>
    File.WriteAllText(path, content.Replace("\r\n", "\n"));

static string Yaml(string s) =>
    "\"" + (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
        .Replace("\r", " ").Replace("\n", " ").Trim() + "\"";

static string EscapeCell(string s) =>
    (s ?? "").Replace("|", "\\|").Replace("\r", " ").Replace("\n", " ").Trim();

// ── example templates (one per category) ────────────────────────────────────

static string BuildExample(string slug, string type, string ns) => slug switch
{
    "optimizers" => $$"""
        using AiDotNet;
        using AiDotNet.Data.Loaders;
        using AiDotNet.Enums;
        using AiDotNet.NeuralNetworks;
        using AiDotNet.Optimizers;
        using AiDotNet.Tensors.LinearAlgebra;

        var rng = new Random(0);
        var trainX = new Tensor<double>(new[] { 32, 8 });
        var trainY = new Tensor<double>(new[] { 32, 2 });
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 8; j++) trainX[new[] { i, j }] = rng.NextDouble();
            trainY[new[] { i, i % 2 }] = 1.0;
        }

        var model = new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            inputFeatures: 8, numClasses: 2, complexity: NetworkComplexity.Simple));

        var result = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(new {{type}}<double, Tensor<double>, Tensor<double>>(model))
            .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
            .BuildAsync();

        Console.WriteLine("Trained with {{type}}.");
        """,

    "loss-functions" => $$"""
        using {{ns}};
        using AiDotNet.Tensors.LinearAlgebra;

        var loss = new {{type}}<float>();
        var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
        var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

        float value = loss.CalculateLoss(predicted, actual);
        Console.WriteLine($"{{type}} = {value:F4}");
        """,

    "regression" => $$"""
        using AiDotNet;
        using AiDotNet.Data.Loaders;
        using {{ns}};
        using AiDotNet.Tensors.LinearAlgebra;

        double[][] features =
        {
            new[] { 1.0, 2.0 }, new[] { 2.0, 3.0 }, new[] { 3.0, 4.0 },
            new[] { 4.0, 5.0 }, new[] { 5.0, 6.0 }, new[] { 6.0, 7.0 }
        };
        double[] targets = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new {{type}}<double>())
            .ConfigureDataLoader(DataLoaders.FromArrays(features, targets))
            .BuildAsync();

        Console.WriteLine("Trained {{type}}.");
        """,

    "clustering" => $$"""
        using AiDotNet;
        using AiDotNet.Data.Loaders;
        using {{ns}};
        using AiDotNet.Tensors.LinearAlgebra;

        var data = new Matrix<double>(6, 2);
        double[][] rows = { new[] { 1.0, 1.0 }, new[] { 1.2, 0.9 }, new[] { 1.1, 1.1 },
                            new[] { 8.0, 8.0 }, new[] { 8.2, 7.9 }, new[] { 7.9, 8.1 } };
        for (int i = 0; i < 6; i++) { data[i, 0] = rows[i][0]; data[i, 1] = rows[i][1]; }

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new {{type}}<double>())
            .ConfigureDataLoader(DataLoaders.FromMatrix(data))
            .BuildAsync();

        var labels = result.Predict(data);
        Console.WriteLine($"{{type}}: clustered {labels.Length} points.");
        """,

    "time-series" => $$"""
        using AiDotNet;
        using AiDotNet.Data.Loaders;
        using {{ns}};
        using AiDotNet.Tensors.LinearAlgebra;

        double[] series =
        {
            120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
            140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
        };
        var x = new Matrix<double>(series.Length, 1);
        for (int i = 0; i < series.Length; i++) x[i, 0] = i;

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new {{type}}<double>())
            .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
            .BuildAsync();

        var forecast = result.Predict(new Matrix<double>(6, 1));
        Console.WriteLine($"{{type}}: forecast {forecast.Length} steps.");
        """,

    "lora-adapters" => $$"""
        using AiDotNet.LoRA;
        using {{ns}};

        var adapter = new {{type}}<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
        var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
        Console.WriteLine($"Configured {{type}} (rank {config.Rank}).");
        """,

    _ => ""
};

// ── XML → Markdown helpers ──────────────────────────────────────────────────

static string StripArity(string s)
{
    int tick = s.IndexOf('`');
    return tick >= 0 ? s.Substring(0, tick) : s;
}

static string ToMarkdown(XElement? el)
{
    if (el is null) return "";
    var sb = new StringBuilder();
    foreach (var node in el.Nodes())
    {
        switch (node)
        {
            case XText t: sb.Append(Regex.Replace(t.Value, @"\s+", " ")); break;
            case XElement e:
                switch (e.Name.LocalName)
                {
                    case "para": sb.Append("\n\n").Append(ToMarkdown(e)).Append("\n\n"); break;
                    case "b": case "strong": sb.Append("**").Append(ToMarkdown(e).Trim()).Append("**"); break;
                    case "i": case "em": sb.Append('*').Append(ToMarkdown(e).Trim()).Append('*'); break;
                    case "c": sb.Append('`').Append(Regex.Replace(e.Value, @"\s+", " ").Trim()).Append('`'); break;
                    case "see": case "seealso":
                        sb.Append('`').Append((string?)e.Attribute("langword") ?? StripCref((string?)e.Attribute("cref")) ?? "").Append('`');
                        break;
                    case "paramref": case "typeparamref":
                        sb.Append('`').Append((string?)e.Attribute("name") ?? "").Append('`'); break;
                    case "code": break;
                    case "list":
                        foreach (var item in e.Elements("item")) sb.Append("\n- ").Append(ToMarkdown(item).Trim());
                        sb.Append('\n'); break;
                    default: sb.Append(ToMarkdown(e)); break;
                }
                break;
        }
    }
    return sb.ToString();
}

static string? StripCref(string? cref)
{
    if (string.IsNullOrEmpty(cref)) return null;
    int colon = cref.IndexOf(':');
    var s = colon >= 0 ? cref.Substring(colon + 1) : cref;
    int dot = s.LastIndexOf('.');
    if (dot >= 0) s = s.Substring(dot + 1);
    return StripArity(s);
}

static string NormalizeBlock(string s)
{
    s = Regex.Replace(s, @"[ \t]+", " ");
    s = Regex.Replace(s, @"\n[ \t]+", "\n");
    s = Regex.Replace(s, @"\n{3,}", "\n\n");
    return s.Trim();
}

static string ExtractForBeginners(XElement? remarks)
{
    if (remarks is null) return "";
    foreach (var para in remarks.Elements("para"))
    {
        var bold = para.Element("b");
        if (bold != null && bold.Value.Trim().StartsWith("For Beginners", StringComparison.OrdinalIgnoreCase))
        {
            var md = ToMarkdown(para);
            md = Regex.Replace(md, @"^\s*\*\*For Beginners:?\*\*\s*", "", RegexOptions.IgnoreCase);
            return NormalizeBlock(md);
        }
    }
    return "";
}

static string ExtractRemarksExcludingForBeginners(XElement? remarks)
{
    if (remarks is null) return "";
    var sb = new StringBuilder();
    foreach (var node in remarks.Nodes())
    {
        if (node is XElement e && e.Name.LocalName == "para")
        {
            var bold = e.Element("b");
            if (bold != null && bold.Value.Trim().StartsWith("For Beginners", StringComparison.OrdinalIgnoreCase)) continue;
            var text = ToMarkdown(e);
            if (Regex.IsMatch(text.Trim(), @"^Based on the paper", RegexOptions.IgnoreCase)) continue;
            sb.Append("\n\n").Append(text).Append("\n\n");
        }
        else if (node is XElement other) sb.Append(ToMarkdown(other));
        else if (node is XText t) sb.Append(Regex.Replace(t.Value, @"\s+", " "));
    }
    return NormalizeBlock(sb.ToString());
}

static string FirstSentence(string s)
{
    s = s.Replace("\n", " ").Trim();
    int dot = s.IndexOf(". ");
    return dot > 0 ? s.Substring(0, dot + 1) : s;
}
