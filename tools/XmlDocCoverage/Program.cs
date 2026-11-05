// Repo-wide XML documentation ratchet.
//
// Counts the externally visible members of AiDotNet that have no XML documentation (what CS1591 would report if
// it were on everywhere) and holds that count to a recorded ceiling that may only fall:
//
//   dotnet run --project tools/XmlDocCoverage -- --max-undocumented <N> [--list <path>]
//
// Exit 1 when the count rises above N (a change added undocumented API), when it falls below N (documentation
// was added, so the ceiling must be lowered to the new count or the gain can silently regress), or when any
// member of a public interface is undocumented. Interfaces are held at zero everywhere because they are the
// contracts a user implements and the facade exposes; src/Interfaces is also compiler-gated (CS1591 as an error),
// and this check covers the interfaces declared in every other folder.
//
// The documentation file is read from beside the loaded AiDotNet.dll, so the assembly and its XML always come
// from the same build.

using System.Reflection;
using System.Xml.Linq;
using AiDotNet.Tools.XmlDocCoverage;

int? ceiling = null;
string? listPath = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--max-undocumented" when i + 1 < args.Length && int.TryParse(args[i + 1], out int parsed) && parsed >= 0:
            ceiling = parsed;
            i++;
            break;
        case "--list" when i + 1 < args.Length:
            listPath = args[++i];
            break;
        default:
            Console.Error.WriteLine($"Unknown or incomplete argument '{args[i]}'.");
            Console.Error.WriteLine("Usage: XmlDocCoverage --max-undocumented <N> [--list <path>]");
            return 2;
    }
}

if (ceiling is null)
{
    Console.Error.WriteLine("--max-undocumented <N> is required.");
    return 2;
}

Assembly assembly = typeof(AiDotNet.Interfaces.IFullModel<,,>).Assembly;
string xmlPath = Path.ChangeExtension(assembly.Location, ".xml");
if (!File.Exists(xmlPath))
{
    Console.Error.WriteLine($"No XML documentation file beside {assembly.Location}.");
    return 2;
}

// AiDotNet's dependencies sit beside it in this tool's output; resolve any that the default context misses.
string probeDirectory = Path.GetDirectoryName(assembly.Location) ?? AppContext.BaseDirectory;
AppDomain.CurrentDomain.AssemblyResolve += (_, request) =>
{
    string? simpleName = new AssemblyName(request.Name).Name;
    if (string.IsNullOrEmpty(simpleName) || simpleName != Path.GetFileName(simpleName)) return null;
    // Join, unlike Combine, never discards the probe directory when the second segment looks rooted.
    string candidate = Path.Join(probeDirectory, Path.GetFileName(simpleName + ".dll"));
    return File.Exists(candidate) ? Assembly.LoadFrom(candidate) : null;
};

var undocumented = XmlDocCoverage.FindUndocumented(assembly, XDocument.Load(xmlPath));
int total = undocumented.Count;

Console.WriteLine($"Assembly: {assembly.Location}");
Console.WriteLine($"Undocumented visible members: {total} (ceiling {ceiling})");
Console.WriteLine();
Console.WriteLine("By namespace (top 25):");
foreach (var group in undocumented
    .GroupBy(entry => TopNamespace(entry.DeclaringType))
    .OrderByDescending(group => group.Count())
    .ThenBy(group => group.Key, StringComparer.Ordinal)
    .Take(25))
{
    Console.WriteLine($"  {group.Count(),6}  {group.Key}");
}

if (listPath is not null)
{
    File.WriteAllLines(listPath, undocumented.Select(entry => entry.Id));
    Console.WriteLine();
    Console.WriteLine($"Wrote the {total} IDs to {listPath}.");
}

bool failed = false;

var onInterfaces = undocumented.Where(entry => entry.DeclaringType.IsInterface).ToList();
if (onInterfaces.Count > 0)
{
    Console.WriteLine();
    Console.WriteLine($"FAIL: {onInterfaces.Count} member(s) of public interfaces are undocumented; interfaces are held at zero:");
    foreach (var entry in onInterfaces)
    {
        Console.WriteLine($"  {entry.Id}");
    }

    failed = true;
}

Console.WriteLine();
if (total > ceiling)
{
    Console.WriteLine($"FAIL: {total} undocumented members, above the recorded ceiling of {ceiling}. Document the API this change added.");
    failed = true;
}
else if (total < ceiling)
{
    Console.WriteLine($"FAIL: {total} undocumented members, below the recorded ceiling of {ceiling}. Lower the ceiling in the workflow to {total} so the gain cannot regress.");
    failed = true;
}
else
{
    Console.WriteLine($"OK: {total} undocumented members, matching the ceiling.");
}

return failed ? 1 : 0;

// The namespace two levels below AiDotNet (AiDotNet.Models.Options), or the whole namespace when it is shorter.
static string TopNamespace(Type type)
{
    string ns = type.Namespace ?? "(global)";
    string[] parts = ns.Split('.');
    return parts.Length <= 3 ? ns : string.Join('.', parts.Take(3));
}
