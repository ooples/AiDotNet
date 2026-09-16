using System.Security.Cryptography;
using System.Text.Json;
using AttributionRuntime;
using Mono.Cecil;
using Mono.Cecil.Cil;

if (args.Length != 2) throw new ArgumentException("Usage: instrumenter input.dll output.dll (distinct private copy)");
string input = Path.GetFullPath(args[0]);
string output = Path.GetFullPath(args[1]);
if (string.Equals(input, output, StringComparison.OrdinalIgnoreCase) || File.Exists(output))
    throw new InvalidOperationException("Instrumentation requires a new, separate output file.");
string inputHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(input)));
string pdbHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.ChangeExtension(input, ".pdb"))));
using var resolver = new DefaultAssemblyResolver();
resolver.AddSearchDirectory(Path.GetDirectoryName(input) ?? throw new InvalidOperationException("Input has no parent."));
resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location) ?? throw new InvalidOperationException("Runtime has no parent."));
using var assembly = AssemblyDefinition.ReadAssembly(input, new ReaderParameters
{
    ReadSymbols = true, InMemory = true, AssemblyResolver = resolver
});
if (assembly.Name.HasPublicKey || assembly.MainModule.AssemblyReferences.Any(reference => reference.Name == "AttributionRuntime"))
    throw new InvalidOperationException("Signed or already instrumented inputs are unsupported by this prototype.");
var hit = assembly.MainModule.ImportReference(typeof(Tracker).GetMethod(nameof(Tracker.Hit))
    ?? throw new InvalidOperationException("Missing tracking method."));
var methods = new List<object>();
foreach (TypeDefinition type in AllTypes(assembly.MainModule.Types))
foreach (MethodDefinition method in type.Methods)
{
    if (!method.HasBody || !method.DebugInformation.HasSequencePoints) continue;
    SequencePoint[] points = method.DebugInformation.SequencePoints.Where(point => !point.IsHidden).ToArray();
    if (points.Length == 0) continue;
    string key = $"{inputHash}:{method.MetadataToken.ToInt32():X8}";
    // Method entry is deliberately conservative: all its source spans are dependencies,
    // not claims that each line/branch executed. Async MoveNext is instrumented as well.
    ILProcessor il = method.Body.GetILProcessor();
    Instruction entry = method.Body.Instructions[0];
    il.InsertBefore(entry, il.Create(OpCodes.Ldstr, key));
    il.InsertBefore(entry, il.Create(OpCodes.Call, hit));
    method.Body.MaxStackSize = Math.Max(method.Body.MaxStackSize, 1);
    methods.Add(new
    {
        Key = key, Name = method.FullName,
        Spans = points.Select(point => new { Document = point.Document.Url, point.StartLine, point.EndLine }).ToArray()
    });
}
if (methods.Count == 0) throw new InvalidOperationException("No source-backed methods found.");
assembly.Write(output, new WriterParameters { WriteSymbols = true });
string outputHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(output)));
File.WriteAllText(output + ".map.json", JsonSerializer.Serialize(new
{
    Schema = 1, InputHash = inputHash, PdbHash = pdbHash, OutputHash = outputHash, Methods = methods
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Instrumented {methods.Count} source-backed methods into private copy {output}");

static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> roots)
{
    foreach (TypeDefinition type in roots)
    {
        yield return type;
        foreach (TypeDefinition nested in AllTypes(type.NestedTypes)) yield return nested;
    }
}
