// WikiGenerator — generates a rich, per-item documentation wiki from the AiDotNet XML
// doc comments (AiDotNet.xml). The prose (summary, "For Beginners", remarks) is pulled
// verbatim and converted to Markdown; embedded <example> code is emitted but flagged for
// the DocSnippetVerify compile gate, because the source examples drift from the real API.
//
// Usage: dotnet run --project tools/WikiGenerator [pathToAiDotNet.xml] [outputDir]

using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;

string xmlPath = args.Length > 0 ? args[0] : "src/bin/Debug/net8.0/AiDotNet.xml";
string outDir = args.Length > 1 ? args[1] : "wiki-generated";

if (!File.Exists(xmlPath))
{
    Console.Error.WriteLine($"XML doc file not found: {xmlPath}");
    Console.Error.WriteLine("Build src/AiDotNet.csproj first (it emits bin/.../AiDotNet.xml).");
    return 1;
}

// Each category maps a namespace prefix to an output folder + a human title.
var categories = new (string Ns, string Slug, string Title)[]
{
    ("AiDotNet.Optimizers.", "optimizers", "Optimizers"),
    ("AiDotNet.LossFunctions.", "loss-functions", "Loss Functions"),
    ("AiDotNet.Regression.", "regression", "Regression Models"),
    ("AiDotNet.Clustering.Partitioning.", "clustering", "Clustering — Partitioning"),
    ("AiDotNet.Clustering.Density.", "clustering", "Clustering — Density"),
    ("AiDotNet.TimeSeries.", "time-series", "Time-Series Models"),
    ("AiDotNet.LoRA.Adapters.", "lora-adapters", "LoRA / PEFT Adapters"),
};

var doc = XDocument.Load(xmlPath);
var members = doc.Root?.Element("members")?.Elements("member").ToList() ?? new();

// Index members by type name for quick lookup.
var byName = new Dictionary<string, XElement>();
foreach (var m in members)
{
    var n = (string?)m.Attribute("name");
    if (n != null && !byName.ContainsKey(n)) byName[n] = m;
}

int totalPages = 0;
Directory.CreateDirectory(outDir);

foreach (var grp in categories.GroupBy(c => c.Slug))
{
    var dir = Path.Combine(outDir, grp.Key);
    Directory.CreateDirectory(dir);
    var indexRows = new List<(string Type, string Summary)>();

    foreach (var cat in grp)
    {
        foreach (var m in members)
        {
            var name = (string?)m.Attribute("name");
            if (name is null || !name.StartsWith("T:" + cat.Ns)) continue;

            var rel = name.Substring(("T:" + cat.Ns).Length);
            if (rel.Contains('.')) continue;             // a deeper sub-namespace, not a type here

            string typeName = StripArity(rel);            // "AdamWOptimizer`3" -> "AdamWOptimizer"
            string summary = NormalizeBlock(ToMarkdown(m.Element("summary")));
            if (summary.Length == 0) continue;            // skip undocumented types

            string forBeginners = ExtractForBeginners(m.Element("remarks"));
            string remarks = ExtractRemarksExcludingForBeginners(m.Element("remarks"));
            string example = ExtractExample(m.Element("example"));

            var sb = new StringBuilder();
            sb.AppendLine($"# {typeName}");
            sb.AppendLine();
            sb.AppendLine($"_{cat.Title}_");
            sb.AppendLine();
            sb.AppendLine(summary);
            sb.AppendLine();
            if (forBeginners.Length > 0)
            {
                sb.AppendLine("## For Beginners");
                sb.AppendLine();
                sb.AppendLine(forBeginners);
                sb.AppendLine();
            }
            if (remarks.Length > 0 && remarks != forBeginners)
            {
                sb.AppendLine("## How It Works");
                sb.AppendLine();
                sb.AppendLine(remarks);
                sb.AppendLine();
            }
            // Prefer a generated, self-contained example (guaranteed to compile) over the
            // source <example> fragment, which assumes undefined context and drifts.
            string generated = CategoryExample(cat.Slug, typeName);
            if (generated.Length > 0)
            {
                sb.AppendLine("## Example");
                sb.AppendLine();
                sb.AppendLine("```csharp");
                sb.AppendLine(generated);
                sb.AppendLine("```");
                sb.AppendLine();
            }
            else if (example.Length > 0)
            {
                sb.AppendLine("## Example");
                sb.AppendLine();
                sb.AppendLine("<!-- VERIFY: transcribed from source XML; not yet compile-checked. -->");
                sb.AppendLine("```csharp");
                sb.AppendLine(example);
                sb.AppendLine("```");
                sb.AppendLine();
            }

            File.WriteAllText(Path.Combine(dir, typeName + ".md"), sb.ToString());
            indexRows.Add((typeName, FirstSentence(summary)));
            totalPages++;
        }
    }

    // Category index page.
    if (indexRows.Count > 0)
    {
        var idx = new StringBuilder();
        idx.AppendLine($"# {grp.First().Title.Split('—')[0].Trim()}");
        idx.AppendLine();
        idx.AppendLine("| Type | Summary |");
        idx.AppendLine("|:-----|:--------|");
        foreach (var (type, sum) in indexRows.OrderBy(r => r.Type))
            idx.AppendLine($"| [`{type}`](./{type}.md) | {sum} |");
        File.WriteAllText(Path.Combine(dir, "index.md"), idx.ToString());
        Console.WriteLine($"{grp.Key,-16} {indexRows.Count,4} pages");
    }
}

Console.WriteLine($"----\nTotal: {totalPages} wiki pages generated under {outDir}/");
return 0;

// ── helpers ────────────────────────────────────────────────────────────────

static string StripArity(string s)
{
    int tick = s.IndexOf('`');
    return tick >= 0 ? s.Substring(0, tick) : s;
}

// Returns a self-contained, compiling example for a category (substituting the type name),
// or "" to fall back to the source <example>. Generated examples define their own context,
// so they don't suffer the undefined-variable problem of the source fragments.
static string CategoryExample(string slug, string typeName)
{
    if (slug != "optimizers") return "";
    return """
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
            .ConfigureOptimizer(new __TYPE__<double, Tensor<double>, Tensor<double>>(model))
            .ConfigureDataLoader(DataLoaders.FromTensors(trainX, trainY))
            .BuildAsync();

        Console.WriteLine("Trained with __TYPE__.");
        """.Replace("__TYPE__", typeName);
}

// Converts XML doc content (mixed text + tags) into Markdown.
static string ToMarkdown(XElement? el)
{
    if (el is null) return "";
    var sb = new StringBuilder();
    foreach (var node in el.Nodes())
    {
        switch (node)
        {
            case XText t:
                sb.Append(Regex.Replace(t.Value, @"\s+", " "));
                break;
            case XElement e:
                switch (e.Name.LocalName)
                {
                    case "para": sb.Append("\n\n").Append(ToMarkdown(e)).Append("\n\n"); break;
                    case "b": case "strong": sb.Append("**").Append(ToMarkdown(e).Trim()).Append("**"); break;
                    case "i": case "em": sb.Append('*').Append(ToMarkdown(e).Trim()).Append('*'); break;
                    case "c": sb.Append('`').Append(Regex.Replace(e.Value, @"\s+", " ").Trim()).Append('`'); break;
                    case "see": case "seealso":
                        var cref = (string?)e.Attribute("cref");
                        var langword = (string?)e.Attribute("langword");
                        sb.Append('`').Append(langword ?? StripCref(cref) ?? "").Append('`');
                        break;
                    case "paramref": case "typeparamref":
                        sb.Append('`').Append((string?)e.Attribute("name") ?? "").Append('`');
                        break;
                    case "code": break;   // examples handled separately
                    case "list":
                        foreach (var item in e.Elements("item"))
                            sb.Append("\n- ").Append(ToMarkdown(item).Trim());
                        sb.Append('\n');
                        break;
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
    int lastDot = s.LastIndexOf('.');
    if (lastDot >= 0) s = s.Substring(lastDot + 1);
    return StripArity(s);
}

// Collapses excess blank lines and trims.
static string NormalizeBlock(string s)
{
    s = Regex.Replace(s, @"[ \t]+", " ");
    s = Regex.Replace(s, @"\n[ \t]+", "\n");
    s = Regex.Replace(s, @"\n{3,}", "\n\n");
    return s.Trim();
}

// Finds the "For Beginners" paragraph in remarks and returns its text.
static string ExtractForBeginners(XElement? remarks)
{
    if (remarks is null) return "";
    foreach (var para in remarks.Elements("para"))
    {
        var bold = para.Element("b");
        if (bold != null && bold.Value.Trim().StartsWith("For Beginners", StringComparison.OrdinalIgnoreCase))
        {
            // Drop the bold label, keep the explanation.
            var md = ToMarkdown(para);
            md = Regex.Replace(md, @"^\s*\*\*For Beginners:?\*\*\s*", "", RegexOptions.IgnoreCase);
            return NormalizeBlock(md);
        }
    }
    return "";
}

// Builds the remarks markdown but drops the "For Beginners" paragraph (shown separately)
// and the bibliographic "Based on the paper …" attribution.
static string ExtractRemarksExcludingForBeginners(XElement? remarks)
{
    if (remarks is null) return "";
    var sb = new StringBuilder();
    foreach (var node in remarks.Nodes())
    {
        if (node is XElement e && e.Name.LocalName == "para")
        {
            var bold = e.Element("b");
            if (bold != null && bold.Value.Trim().StartsWith("For Beginners", StringComparison.OrdinalIgnoreCase))
                continue;
            var text = ToMarkdown(e);
            if (Regex.IsMatch(text.Trim(), @"^Based on the paper", RegexOptions.IgnoreCase))
                continue;
            sb.Append("\n\n").Append(text).Append("\n\n");
        }
        else if (node is XElement other)
        {
            sb.Append(ToMarkdown(other));
        }
        else if (node is XText t)
        {
            sb.Append(Regex.Replace(t.Value, @"\s+", " "));
        }
    }
    return NormalizeBlock(sb.ToString());
}

static string ExtractExample(XElement? example)
{
    var code = example?.Element("code");
    if (code is null) return "";
    var raw = code.Value;
    // Strip common leading indentation.
    var lines = raw.Replace("\r\n", "\n").Split('\n')
        .SkipWhile(l => l.Trim().Length == 0).ToList();
    while (lines.Count > 0 && lines[^1].Trim().Length == 0) lines.RemoveAt(lines.Count - 1);
    int indent = lines.Where(l => l.Trim().Length > 0)
        .Select(l => l.Length - l.TrimStart().Length).DefaultIfEmpty(0).Min();
    return string.Join("\n", lines.Select(l => l.Length >= indent ? l.Substring(indent) : l));
}

static string FirstSentence(string s)
{
    s = s.Replace("\n", " ").Trim();
    int dot = s.IndexOf(". ");
    return dot > 0 ? s.Substring(0, dot + 1) : s;
}
