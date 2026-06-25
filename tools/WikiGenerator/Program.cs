// WikiGenerator — generates a complete, namespace-organized API reference wiki from the
// AiDotNet assemblies and their XML doc comments. EVERY public type gets a page, grouped by
// its top-level namespace (one category per namespace) and, within a category, by kind
// (Models, Layers, Interfaces, Enums, Options, Helpers, ...). Each page carries the summary,
// beginner notes, how-it-works, its documented members, and — for concrete model types in
// supported domains — a compile-checked example.
//
// Usage: dotnet run --project tools/WikiGenerator [xmlPath] [outputDir]

using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

string xmlPath = args.Length > 0 ? args[0] : "src/bin/Debug/net8.0/AiDotNet.xml";
string outDir = args.Length > 1 ? args[1] : "website/src/content/docs/reference/wiki";
const string urlBase = "/docs/reference/wiki";

if (!File.Exists(xmlPath))
{
    Console.Error.WriteLine($"XML doc file not found: {xmlPath}. Build src/AiDotNet.csproj first.");
    return 1;
}

// ── Roslyn reference set (this tool's own output bin: AiDotNet + transitive deps) ──
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

// ── Reflect the full public type universe from the AiDotNet assemblies ──
var allTypes = new List<Type>();
foreach (var asmName in new[] { "AiDotNet.dll", "AiDotNet.Tensors.dll" })
{
    var path = Path.Combine(AppContext.BaseDirectory, asmName);
    if (!File.Exists(path)) continue;
    try { allTypes.AddRange(Assembly.LoadFrom(path).GetExportedTypes()); }
    catch (ReflectionTypeLoadException ex) { allTypes.AddRange(ex.Types.Where(t => t is not null).Select(t => t!)); }
    catch (Exception ex) { Console.Error.WriteLine($"reflect {asmName}: {ex.Message}"); }
}
Console.WriteLine($"Reflected {allTypes.Count} public types.");

// ── XML docs: type summaries/remarks + members grouped by declaring type ──
var xml = XDocument.Load(xmlPath);
var typeDoc = new Dictionary<string, XElement>(StringComparer.Ordinal);
// Member summaries keyed by (declaring type, member name, param count) for relaxed matching
// against reflected members (overloads with same name+arity share a summary — rare).
var memberSummary = new Dictionary<(string Decl, string Name, int Arity), XElement>();
foreach (var m in xml.Root?.Element("members")?.Elements("member") ?? Enumerable.Empty<XElement>())
{
    var name = (string?)m.Attribute("name");
    if (name is null || name.Length < 2 || name[1] != ':') continue;
    char tag = name[0];
    if (tag == 'T') { typeDoc[name.Substring(2)] = m; continue; }
    if (tag is 'M' or 'P' or 'F' or 'E')
    {
        var (decl, nm, arity) = DeclNameArity(name);
        if (decl.Length == 0) continue;
        memberSummary[(decl, nm, arity)] = m;
    }
}

// ── In-process compile gate (mirrors DocSnippetVerify) ──
const string commonUsings =
    "using System;using System.Collections.Generic;using System.Linq;" +
    "using System.Threading;using System.Threading.Tasks;" +
    "using AiDotNet;using AiDotNet.Tensors;using AiDotNet.Tensors.LinearAlgebra;\n";
var usingLineRe = new Regex(@"^\s*using\s+[A-Za-z_][\w.]*\s*;\s*$", RegexOptions.None, TimeSpan.FromSeconds(1));
var compOptions = new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary,
    allowUnsafe: true, nullableContextOptions: NullableContextOptions.Disable);
var parse = new CSharpParseOptions(LanguageVersion.Latest);
List<Diagnostic> Diagnose(string code)
{
    var usings = new StringBuilder();
    var body = new StringBuilder();
    bool started = false;
    foreach (var line in code.Replace("\r\n", "\n").Split('\n'))
    {
        if (!started && usingLineRe.IsMatch(line)) usings.Append(line.Trim()).Append('\n');
        else { started = started || line.Trim().Length > 0; body.Append(line).Append('\n'); }
    }
    var src = commonUsings + usings + "static class __S { static async System.Threading.Tasks.Task __R() {\n" + body + "\n} }";
    var comp = CSharpCompilation.Create("ex", new[] { CSharpSyntaxTree.ParseText(src, parse) }, refs, compOptions);
    return comp.GetDiagnostics().Where(d => d.Severity == DiagnosticSeverity.Error && d.Id != "CS5001").ToList();
}
bool Compiles(string code) => Diagnose(code).Count == 0;

// Simple type name -> namespaces, for injecting `using`s into drifted author examples.
var nsIndex = new Dictionary<string, HashSet<string>>(StringComparer.Ordinal);
foreach (var t in allTypes)
{
    if (t.Namespace is null) continue;
    string simple = StripArity(t.Name);
    if (!nsIndex.TryGetValue(simple, out var set)) nsIndex[simple] = set = new();
    set.Add(t.Namespace);
}
var crefRe = new Regex(@"'([A-Za-z_]\w*)'", RegexOptions.None, TimeSpan.FromSeconds(1));

// Try to make an author's <example> compile by injecting `using`s for unresolved (single-
// namespace) types. Returns the compiling code, or null if it still won't build.
string? FixCompile(string code)
{
    if (Compiles(code)) return code;
    var add = new SortedSet<string>(StringComparer.Ordinal);
    foreach (var d in Diagnose(code).Where(d => d.Id == "CS0246"))
    {
        var m = crefRe.Match(d.GetMessage());
        if (m.Success && nsIndex.TryGetValue(m.Groups[1].Value, out var nss) && nss.Count == 1)
            add.Add(nss.First());
    }
    if (add.Count == 0) return null;
    string fixedCode = string.Concat(add.Select(n => $"using {n};\n")) + code;
    return Compiles(fixedCode) ? fixedCode : null;
}

// Categories whose concrete model types get a runnable, compile-checked example.
var exampleSlugs = new HashSet<string>(StringComparer.Ordinal)
{
    "optimizers", "lossfunctions", "regression", "clustering", "timeseries", "lora",
    "activationfunctions", "windowfunctions", "waveletfunctions", "kernels", "radialbasisfunctions"
};

// ── Build a page for every public type ──
Directory.CreateDirectory(outDir);
int totalPages = 0, withExample = 0, illustrativeCount = 0;
var catSummary = new List<(string Slug, string Title, int Count)>();

var grouped = allTypes
    .Where(t => t.Namespace is not null && t.Namespace.StartsWith("AiDotNet", StringComparison.Ordinal)
        && !t.Name.Contains('<')   // skip compiler-generated nested types (fixed buffers, closures)
        && !t.IsDefined(typeof(System.Runtime.CompilerServices.CompilerGeneratedAttribute), false))
    .GroupBy(t => CategoryKey(t.Namespace!))
    .OrderBy(g => g.Key, StringComparer.Ordinal)
    .ToList();

// Pass 1: assign every type a category slug + collision-free page slug (global, so cross-links resolve).
var typeCat = new Dictionary<Type, string>();
var typeSlug = new Dictionary<Type, string>();
foreach (var grp in grouped)
{
    string slug = grp.Key.ToLowerInvariant();
    var used = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
    foreach (var t in grp.OrderBy(t => t.FullName ?? t.Name, StringComparer.Ordinal))
    {
        string b = Sanitize(StripArity(t.Name));
        if (b.Length == 0) b = "type";
        string cand = b;
        int n = 2;
        while (!used.Add(cand.ToLowerInvariant())) cand = $"{b}-{n++}";
        typeCat[t] = slug;
        typeSlug[t] = cand;
    }
}

// Pass 1b: reverse maps — base class -> derived types, interface -> implementors (documented types only).
var derivedOf = new Dictionary<Type, List<Type>>();
var implementorsOf = new Dictionary<Type, List<Type>>();
foreach (var t in typeCat.Keys)
{
    if (t.BaseType is { } bt && typeCat.ContainsKey(Norm(bt))) AddTo(derivedOf, Norm(bt), t);
    foreach (var itf in DirectInterfaces(t))
        if (typeCat.ContainsKey(Norm(itf))) AddTo(implementorsOf, Norm(itf), t);
}

// Pass 2: render a page for every type.
foreach (var grp in grouped)
{
    string title = Spaced(grp.Key);
    string slug = grp.Key.ToLowerInvariant();
    var dir = Path.Combine(outDir, slug);
    Directory.CreateDirectory(dir);
    var typesInCat = grp.OrderBy(t => t.FullName ?? t.Name, StringComparer.Ordinal).ToList();
    var byKind = new Dictionary<string, List<(Type T, string Slug, string Summary)>>();

    foreach (var t in typesInCat)
    {
        string xmlName = XmlName(t);
        typeDoc.TryGetValue(xmlName, out var md);
        string kind = KindOf(t);
        string summary = md is not null ? NormalizeBlock(ToMarkdown(md.Element("summary"))) : "";
        string forBeginners = md is not null ? ExtractForBeginners(md.Element("remarks")) : "";
        string remarks = md is not null ? ExtractRemarksExcludingForBeginners(md.Element("remarks")) : "";

        // Example selection (any domain):
        //   1. the author's <example> from the XML docs, auto-fixed to compile (verified);
        //   2. a per-domain template that compiles (verified);
        //   3. the author's <example> shown as illustrative if it can't be made to compile.
        bool exampleOk = false, illustrative = false;
        string example = "";
        var exNode = md?.Element("example");
        string srcExample = exNode is null ? "" : Dedent(exNode.Element("code")?.Value ?? exNode.Value);

        if (srcExample.Length > 0 && FixCompile(srcExample) is { } fixedSrc) { example = fixedSrc; exampleOk = true; }
        if (!exampleOk && kind == "Models & Types" && exampleSlugs.Contains(slug))
            foreach (var candidate in BuildExamples(slug, StripArity(t.Name), t.Namespace!))
                if (candidate.Length > 0 && Compiles(candidate)) { example = candidate; exampleOk = true; break; }
        if (!exampleOk && srcExample.Length > 0) { example = srcExample; illustrative = true; }

        string display = FriendlyName(t);
        var sb = new StringBuilder();
        sb.AppendLine("---");
        sb.AppendLine($"title: {Yaml(display)}");
        sb.AppendLine($"description: {Yaml(summary.Length > 0 ? FirstSentence(summary) : $"{display} — {kind} in {t.Namespace}.")}");
        sb.AppendLine("section: \"API Reference\"");
        sb.AppendLine("---").AppendLine();
        sb.AppendLine($"`{kind}` · `{t.Namespace}`").AppendLine();
        sb.AppendLine(summary.Length > 0 ? summary : "_No summary documentation available yet._").AppendLine();
        sb.Append(RenderHierarchy(t, typeCat, typeSlug, derivedOf, implementorsOf, urlBase));
        if (forBeginners.Length > 0) sb.AppendLine("## For Beginners").AppendLine().AppendLine(forBeginners).AppendLine();
        if (remarks.Length > 0) sb.AppendLine("## How It Works").AppendLine().AppendLine(remarks).AppendLine();
        if (exampleOk)
        {
            sb.AppendLine("## Example").AppendLine().AppendLine("```csharp").AppendLine(example).AppendLine("```").AppendLine();
            withExample++;
        }
        else if (illustrative)
        {
            sb.AppendLine("## Example").AppendLine();
            sb.AppendLine("_From the type's documentation (illustrative — not compile-verified):_").AppendLine();
            sb.AppendLine("```cs").AppendLine(example).AppendLine("```").AppendLine();
            illustrativeCount++;
        }
        sb.Append(RenderMembers(t, xmlName, memberSummary));

        WriteFile(Path.Combine(dir, typeSlug[t] + ".md"), sb.ToString());
        totalPages++;

        if (!byKind.TryGetValue(kind, out var bucket)) byKind[kind] = bucket = new();
        bucket.Add((t, typeSlug[t].ToLowerInvariant(), summary.Length > 0 ? FirstSentence(summary) : ""));
    }

    WriteCategoryIndex(dir, title, slug, byKind, urlBase);
    catSummary.Add((slug, title, typesInCat.Count));
    Console.WriteLine($"{slug,-26} {typesInCat.Count,5} types");
}

WriteTopIndex(outDir, catSummary, urlBase);
Console.WriteLine($"----\nTotal: {totalPages} type pages across {catSummary.Count} namespaces; {withExample} compile-verified + {illustrativeCount} illustrative examples under {outDir}/");
return 0;

// ── reflection: hierarchy, cross-links, members ──────────────────────────────

static Type Norm(Type t) => t.IsGenericType && !t.IsGenericTypeDefinition ? t.GetGenericTypeDefinition() : t;

static void AddTo(Dictionary<Type, List<Type>> d, Type k, Type v)
{
    if (!d.TryGetValue(k, out var l)) d[k] = l = new();
    l.Add(v);
}

static IEnumerable<Type> DirectInterfaces(Type t)
{
    var inherited = new HashSet<Type>();
    if (t.BaseType is { } bt) foreach (var i in bt.GetInterfaces()) inherited.Add(i);
    foreach (var i in t.GetInterfaces()) foreach (var sub in i.GetInterfaces()) inherited.Add(sub);
    return t.GetInterfaces().Where(i => !inherited.Contains(i));
}

static string Pretty(Type t)
{
    if (t.IsByRef) return Pretty(t.GetElementType()!);
    if (t.IsArray) return Pretty(t.GetElementType()!) + "[]";
    if (t.IsGenericParameter) return t.Name;
    if (t.IsGenericType)
    {
        if (t.GetGenericTypeDefinition() == typeof(Nullable<>)) return Pretty(t.GetGenericArguments()[0]) + "?";
        return $"{StripArity(t.Name)}<{string.Join(", ", t.GetGenericArguments().Select(Pretty))}>";
    }
    return t.FullName switch   // C# keyword aliases for the BCL primitives
    {
        "System.Boolean" => "bool", "System.Byte" => "byte", "System.SByte" => "sbyte",
        "System.Char" => "char", "System.Int16" => "short", "System.UInt16" => "ushort",
        "System.Int32" => "int", "System.UInt32" => "uint", "System.Int64" => "long",
        "System.UInt64" => "ulong", "System.Single" => "float", "System.Double" => "double",
        "System.Decimal" => "decimal", "System.String" => "string", "System.Object" => "object",
        "System.Void" => "void",
        _ => t.Name
    };
}

static string Link(Type t, Dictionary<Type, string> cat, Dictionary<Type, string> slug, string urlBase)
{
    var key = Norm(t.IsByRef || t.IsArray ? (t.GetElementType() ?? t) : t);
    if (cat.TryGetValue(key, out var c) && slug.TryGetValue(key, out var s))
        return $"[`{Pretty(t)}`]({urlBase}/{c}/{s.ToLowerInvariant()}/)";
    return $"`{Pretty(t)}`";
}

static string RenderHierarchy(Type t, Dictionary<Type, string> cat, Dictionary<Type, string> slug,
    Dictionary<Type, List<Type>> derivedOf, Dictionary<Type, List<Type>> implementorsOf, string urlBase)
{
    var sb = new StringBuilder();

    if (!t.IsInterface && !t.IsEnum && t.BaseType is not null && t.BaseType != typeof(object))
    {
        var chain = new List<Type>();
        for (var b = t.BaseType; b is not null && b != typeof(object); b = b.BaseType) chain.Add(b);
        chain.Reverse();
        sb.Append("**Inheritance:** ");
        foreach (var b in chain) sb.Append(Link(b, cat, slug, urlBase)).Append(" → ");
        sb.Append($"`{Pretty(t)}`").AppendLine().AppendLine();
    }

    var ifaces = DirectInterfaces(t).Where(i => i.IsPublic || i.IsNestedPublic).OrderBy(i => i.Name, StringComparer.Ordinal).ToList();
    if (ifaces.Count > 0)
        sb.Append("**Implements:** ").Append(string.Join(", ", ifaces.Select(i => Link(i, cat, slug, urlBase)))).AppendLine().AppendLine();

    if (derivedOf.TryGetValue(t, out var derived) && derived.Count > 0)
    {
        sb.Append(t.IsInterface ? "**Known implementations:** " : "**Known derived types:** ");
        sb.Append(string.Join(", ", derived.OrderBy(d => d.Name, StringComparer.Ordinal).Take(60).Select(d => Link(d, cat, slug, urlBase))));
        if (derived.Count > 60) sb.Append($" _(+{derived.Count - 60} more)_");
        sb.AppendLine().AppendLine();
    }
    if (implementorsOf.TryGetValue(t, out var impl) && impl.Count > 0)
    {
        sb.Append("**Known implementations:** ");
        sb.Append(string.Join(", ", impl.OrderBy(d => d.Name, StringComparer.Ordinal).Take(60).Select(d => Link(d, cat, slug, urlBase))));
        if (impl.Count > 60) sb.Append($" _(+{impl.Count - 60} more)_");
        sb.AppendLine().AppendLine();
    }
    return sb.ToString();
}

static string RenderMembers(Type t, string xmlName, Dictionary<(string, string, int), XElement> summaries)
{
    const System.Reflection.BindingFlags F = System.Reflection.BindingFlags.Public
        | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static
        | System.Reflection.BindingFlags.DeclaredOnly;

    string Sum(string name, int arity) =>
        summaries.TryGetValue((xmlName, name, arity), out var el) ? EscapeCell(FirstSentence(Inline(el.Element("summary")))) : "";

    var sb = new StringBuilder();
    void Section(string head, string col, List<(string Sig, string Sum)> rows)
    {
        if (rows.Count == 0) return;
        sb.AppendLine($"## {head}").AppendLine().AppendLine($"| {col} | Summary |").AppendLine("|:-----|:--------|");
        foreach (var (sig, sum) in rows.OrderBy(r => r.Sig, StringComparer.Ordinal)) sb.AppendLine($"| `{sig}` | {sum} |");
        sb.AppendLine();
    }

    try
    {
        var ctors = t.GetConstructors(F).Where(c => !c.IsPrivate)
            .Select(c => ($"{StripArity(t.Name)}({string.Join(", ", c.GetParameters().Select(p => Pretty(p.ParameterType) + " " + p.Name))})", Sum("#ctor", c.GetParameters().Length))).ToList();
        var props = t.GetProperties(F)
            .Select(p => ($"{Pretty(p.PropertyType)} {p.Name} {{ {(p.CanRead ? "get; " : "")}{(p.CanWrite ? "set; " : "")}}}", Sum(p.Name, 0))).ToList();
        var methods = t.GetMethods(F).Where(m => !(m.IsSpecialName && (m.Name.StartsWith("get_") || m.Name.StartsWith("set_") || m.Name.StartsWith("add_") || m.Name.StartsWith("remove_"))))
            .Select(m => ($"{Pretty(m.ReturnType)} {m.Name}({string.Join(", ", m.GetParameters().Select(p => Pretty(p.ParameterType) + " " + p.Name))})", Sum(m.Name, m.GetParameters().Length))).ToList();
        var fields = t.GetFields(F).Where(f => !f.IsPrivate)
            .Select(f => ($"{Pretty(f.FieldType)} {f.Name}", Sum(f.Name, 0))).ToList();

        Section("Constructors", "Constructor", ctors);
        Section("Properties", "Property", props);
        Section("Methods", "Method", methods);
        Section("Fields", "Field", fields);
    }
    catch { /* a few types fail to reflect members (load issues); skip their member tables */ }
    return sb.ToString();
}

// ── classification + naming ──────────────────────────────────────────────────

static string CategoryKey(string ns)
{
    var parts = ns.Split('.');
    return parts.Length >= 2 ? parts[1] : parts[0];   // AiDotNet.X.* -> X
}

static string Spaced(string s) => s switch
{
    "LoRA" => "LoRA",
    "NER" => "NER",
    "Onnx" => "ONNX",
    "AutoML" => "AutoML",
    _ => Regex.Replace(s, @"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", RegexOptions.None, TimeSpan.FromSeconds(1))
};

static string KindOf(Type t)
{
    if (t.IsEnum) return "Enums";
    if (t.IsInterface) return "Interfaces";
    if (typeof(Delegate).IsAssignableFrom(t)) return "Delegates";
    if (t.IsValueType) return "Structs";
    string n = StripArity(t.Name);
    string ns = t.Namespace ?? "";
    if (n.EndsWith("Exception", StringComparison.Ordinal)) return "Exceptions";
    if (n.EndsWith("Attribute", StringComparison.Ordinal)) return "Attributes";
    if (n.EndsWith("Options", StringComparison.Ordinal) || n.EndsWith("Configuration", StringComparison.Ordinal)
        || n.EndsWith("Config", StringComparison.Ordinal) || n.EndsWith("Settings", StringComparison.Ordinal))
        return "Options & Configuration";
    if (ns.Contains(".Layers", StringComparison.Ordinal) || n.EndsWith("Layer", StringComparison.Ordinal)) return "Layers";
    if (t.IsAbstract && t.IsSealed) return "Helpers & Utilities";   // static class
    if (n.EndsWith("Extensions", StringComparison.Ordinal) || n.EndsWith("Helper", StringComparison.Ordinal)
        || n.EndsWith("Helpers", StringComparison.Ordinal) || n.EndsWith("Factory", StringComparison.Ordinal)
        || n.EndsWith("Builder", StringComparison.Ordinal) || n.EndsWith("Node", StringComparison.Ordinal)
        || n.EndsWith("Util", StringComparison.Ordinal) || n.EndsWith("Utils", StringComparison.Ordinal))
        return "Helpers & Utilities";
    if (t.IsAbstract) return "Base Classes";
    return "Models & Types";
}

static string StripArity(string s)
{
    int tick = s.IndexOf('`');
    return tick >= 0 ? s.Substring(0, tick) : s;
}

static string Sanitize(string s) =>
    Regex.Replace(s, @"[^A-Za-z0-9_.-]", "", RegexOptions.None, TimeSpan.FromSeconds(1));

// Strip leading/trailing blank lines and the common left indentation from doc-comment code.
static string Dedent(string s)
{
    var lines = s.Replace("\r\n", "\n").Split('\n').ToList();
    while (lines.Count > 0 && lines[0].Trim().Length == 0) lines.RemoveAt(0);
    while (lines.Count > 0 && lines[^1].Trim().Length == 0) lines.RemoveAt(lines.Count - 1);
    if (lines.Count == 0) return "";
    int min = lines.Where(l => l.Trim().Length > 0)
        .Select(l => l.Length - l.TrimStart().Length).DefaultIfEmpty(0).Min();
    return string.Join("\n", lines.Select(l => l.Length >= min ? l.Substring(min) : l));
}

static string XmlName(Type t) => (t.FullName ?? (t.Namespace + "." + t.Name)).Replace('+', '.');

static string FriendlyName(Type t)
{
    string n = StripArity(t.Name);
    if (t.IsGenericTypeDefinition)
        return $"{n}<{string.Join(", ", t.GetGenericArguments().Select(a => a.Name))}>";
    return n;
}

// Parse an XML member doc-comment name into (declaring type, member name, param count).
static (string Decl, string Name, int Arity) DeclNameArity(string raw)
{
    string s = raw.Substring(2);                         // drop "M:" / "P:" / ...
    int paren = s.IndexOf('(');
    string head = paren >= 0 ? s.Substring(0, paren) : s;
    string args = "";
    if (paren >= 0)
    {
        int close = s.IndexOf(')', paren);
        if (close > paren) args = s.Substring(paren + 1, close - paren - 1);
    }
    int dot = head.LastIndexOf('.');
    if (dot < 0) return ("", "", 0);
    string decl = head.Substring(0, dot);
    string name = Regex.Replace(head.Substring(dot + 1), @"``\d+", "", RegexOptions.None, TimeSpan.FromSeconds(1));
    return (decl, name, args.Trim().Length == 0 ? 0 : SplitTopLevel(args));
}

// Count top-level comma-separated params (ignoring commas inside {}/[] generic args).
static int SplitTopLevel(string s)
{
    int depth = 0, count = 1;
    foreach (char ch in s)
    {
        if (ch is '{' or '[') depth++;
        else if (ch is '}' or ']') depth--;
        else if (ch == ',' && depth == 0) count++;
    }
    return count;
}

static void WriteCategoryIndex(string dir, string title, string slug,
    Dictionary<string, List<(Type T, string Slug, string Summary)>> byKind, string urlBase)
{
    string[] order =
    {
        "Models & Types", "Layers", "Base Classes", "Interfaces", "Enums", "Structs",
        "Delegates", "Options & Configuration", "Helpers & Utilities", "Attributes", "Exceptions"
    };
    int total = byKind.Sum(k => k.Value.Count);
    var sb = new StringBuilder();
    sb.AppendLine("---");
    sb.AppendLine($"title: {Yaml(title)}");
    sb.AppendLine($"description: {Yaml($"All {total} public types in the AiDotNet.{slug} namespace, organized by kind.")}");
    sb.AppendLine("section: \"API Reference\"");
    sb.AppendLine("---").AppendLine();
    sb.AppendLine($"**{total}** public types in this namespace, organized by kind.").AppendLine();
    foreach (var kind in order.Concat(byKind.Keys.Where(k => !order.Contains(k))))
    {
        if (!byKind.TryGetValue(kind, out var items) || items.Count == 0) continue;
        sb.AppendLine($"## {kind} ({items.Count})").AppendLine();
        sb.AppendLine("| Type | Summary |").AppendLine("|:-----|:--------|");
        foreach (var (t, s, sum) in items.OrderBy(i => i.T.Name, StringComparer.Ordinal))
            sb.AppendLine($"| [`{FriendlyName(t)}`]({urlBase}/{slug}/{s}/) | {EscapeCell(sum)} |");
        sb.AppendLine();
    }
    WriteFile(Path.Combine(dir, "index.md"), sb.ToString());
}

static void WriteTopIndex(string outDir, List<(string Slug, string Title, int Count)> cats, string urlBase)
{
    int total = cats.Sum(c => c.Count);
    var sb = new StringBuilder();
    sb.AppendLine("---");
    sb.AppendLine("title: \"API Reference\"");
    sb.AppendLine($"description: {Yaml($"Complete namespace-organized API reference for AiDotNet — {total} public types across {cats.Count} namespaces.")}");
    sb.AppendLine("section: \"API Reference\"");
    sb.AppendLine("---").AppendLine();
    sb.AppendLine($"Every public type in AiDotNet, organized by namespace. **{total} types** across **{cats.Count} namespaces** — each page carries the summary, a beginner-friendly explanation, how it works, its documented members, and a compiling example where one applies.").AppendLine();
    sb.AppendLine("| Namespace | Types |").AppendLine("|:----------|------:|");
    foreach (var (slug, t, count) in cats.OrderBy(c => c.Title, StringComparer.OrdinalIgnoreCase))
        sb.AppendLine($"| [{t}]({urlBase}/{slug}/) | {count} |");
    WriteFile(Path.Combine(outDir, "index.md"), sb.ToString());
}

// ── frontmatter / table helpers ──────────────────────────────────────────────

// Always write LF so output is byte-identical on Windows and Linux (stable CI drift check).
static void WriteFile(string path, string content) => File.WriteAllText(path, content.Replace("\r\n", "\n"));

static string Yaml(string s) =>
    "\"" + (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", " ").Replace("\n", " ").Trim() + "\"";

static string EscapeCell(string s) =>
    (s ?? "").Replace("|", "\\|").Replace("\r", " ").Replace("\n", " ").Trim();

// ── XML → Markdown ───────────────────────────────────────────────────────────

static string ToMarkdown(XElement? el)
{
    if (el is null) return "";
    var sb = new StringBuilder();
    foreach (var node in el.Nodes())
    {
        switch (node)
        {
            // Collapse horizontal whitespace but KEEP newlines so author-written lists survive.
            case XText t: sb.Append(Regex.Replace(t.Value, @"[^\S\n]+", " ")); break;
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

// Single-line conversion for table cells.
static string Inline(XElement? el) => Regex.Replace(ToMarkdown(el), @"\s+", " ").Trim();

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
    // Code fences embedded in doc-comment prose are illustrative, not our verified example —
    // downgrade to ```cs so the compile gate (keyed on ```csharp) skips them.
    s = Regex.Replace(s, @"```\s*(csharp|c#)", "```cs", RegexOptions.IgnoreCase, TimeSpan.FromSeconds(1));
    s = TidyLists(s);
    s = Regex.Replace(s, @"\n{3,}", "\n\n");
    return s.Trim();
}

// Ensure a blank line sits before and after each run of list items so markdown renders them.
static string TidyLists(string s)
{
    var outL = new List<string>();
    bool prevItem = false, prevBlank = true;
    foreach (var line in s.Replace("\r\n", "\n").Split('\n'))
    {
        bool blank = line.Trim().Length == 0;
        bool item = Regex.IsMatch(line, @"^\s*([-*]|\d+\.)\s+", RegexOptions.None, TimeSpan.FromSeconds(1));
        if (item && !prevItem && !prevBlank) outL.Add("");
        if (!item && !blank && prevItem) outL.Add("");
        outL.Add(line);
        if (blank) { prevBlank = true; prevItem = false; }
        else { prevBlank = false; prevItem = item; }
    }
    return string.Join("\n", outL);
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
            md = Regex.Replace(md, @"^\s*\*\*For Beginners:?\*\*\s*", "", RegexOptions.IgnoreCase, TimeSpan.FromSeconds(1));
            return NormalizeBlock(md);
        }
    }
    return "";
}

static string ExtractRemarksExcludingForBeginners(XElement? remarks)
{
    if (remarks is null) return "";
    var paras = remarks.Elements("para").ToList();
    if (paras.Count == 0) return NormalizeBlock(ToMarkdown(remarks));
    var sb = new StringBuilder();
    foreach (var para in paras)
    {
        var bold = para.Element("b");
        if (bold != null && bold.Value.Trim().StartsWith("For Beginners", StringComparison.OrdinalIgnoreCase)) continue;
        sb.Append(ToMarkdown(para)).Append("\n\n");
    }
    return NormalizeBlock(sb.ToString());
}

static string FirstSentence(string s)
{
    s = s.Replace("\n", " ").Trim();
    int dot = s.IndexOf(". ", StringComparison.Ordinal);
    if (dot > 0 && dot < 240) return s.Substring(0, dot + 1);
    return s.Length > 240 ? s.Substring(0, 240).TrimEnd() + "…" : s;
}

// ── example templates (concrete model types in supported domains) ─────────────

static string[] BuildExamples(string slug, string type, string ns) => slug switch
{
    "optimizers" => new[] { $$"""
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
        """ },

    "lossfunctions" => new[] { $$"""
        using {{ns}};
        using AiDotNet.Tensors.LinearAlgebra;

        var loss = new {{type}}<float>();
        var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
        var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

        float value = loss.CalculateLoss(predicted, actual);
        Console.WriteLine($"{{type}} = {value:F4}");
        """ },

    "regression" => new[] { $$"""
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
        """ },

    "clustering" => new[] { $$"""
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
        """ },

    "timeseries" => new[] { $$"""
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
        """ },

    "lora" => new[] { $$"""
        using AiDotNet.LoRA;
        using {{ns}};

        var adapter = new {{type}}<double>(null, rank: 8, alpha: 8, freezeBaseLayer: true);
        var config = new DefaultLoRAConfiguration<double>(rank: 8, alpha: 8, loraAdapter: adapter);
        Console.WriteLine($"Configured {{type}} (rank {config.Rank}).");
        """ },

    "activationfunctions" => new[] { $$"""
        using AiDotNet.ActivationFunctions;

        var activation = new {{type}}<double>();
        double y = activation.Activate(0.5);
        Console.WriteLine($"{{type}}: f(0.5) = {y:F4}");
        """ },

    "windowfunctions" => new[] { $$"""
        using AiDotNet.WindowFunctions;
        using AiDotNet.Tensors.LinearAlgebra;

        var window = new {{type}}<double>();
        Vector<double> w = window.Create(16);
        Console.WriteLine($"{{type}}: {w.Length}-point window, center = {w[w.Length / 2]:F4}");
        """ },

    "waveletfunctions" => new[] { $$"""
        using AiDotNet.WaveletFunctions;

        var wavelet = new {{type}}<double>();
        double value = wavelet.Calculate(0.5);
        Console.WriteLine($"{{type}}: psi(0.5) = {value:F4}");
        """ },

    "kernels" => new[] { $$"""
        using AiDotNet.Kernels;
        using AiDotNet.Tensors.LinearAlgebra;

        var kernel = new {{type}}<double>();
        var a = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var b = new Vector<double>(new[] { 1.5, 1.0, 2.5 });
        double similarity = kernel.Calculate(a, b);
        Console.WriteLine($"{{type}}: K(a, b) = {similarity:F4}");
        """ },

    // RBFs take a width parameter; fall back to a parameterless ctor where one exists.
    "radialbasisfunctions" => new[]
    {
        $$"""
        using AiDotNet.RadialBasisFunctions;

        var rbf = new {{type}}<double>(1.0);
        double value = rbf.Compute(0.5);
        Console.WriteLine($"{{type}}: f(0.5) = {value:F4}");
        """,
        $$"""
        using AiDotNet.RadialBasisFunctions;

        var rbf = new {{type}}<double>();
        double value = rbf.Compute(0.5);
        Console.WriteLine($"{{type}}: f(0.5) = {value:F4}");
        """
    },

    _ => System.Array.Empty<string>()
};
