using System.Security.Cryptography;
using System.Text.Json;
using AttributionRuntime;
using Mono.Cecil;
using Mono.Cecil.Cil;
using Mono.Cecil.Rocks;

if (args.Length is < 2 or > 3) throw new ArgumentException("Usage: instrumenter input.dll output.dll [Methods|TaskBoundaries]");
InstrumentationMode mode = args.Length == 2 ? InstrumentationMode.Methods :
    Enum.TryParse(args[2], out InstrumentationMode parsed) && Enum.IsDefined(parsed)
        ? parsed : throw new ArgumentException("Unsupported instrumentation mode.");
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
if (mode == InstrumentationMode.RemoveTaskObserverForMutationTest)
{
    if (assembly.Name.Name != "AttributionRuntime") throw new InvalidOperationException("Mutation control only accepts the fixture runtime.");
    MethodDefinition observer = assembly.MainModule.Types.Single(type => type.FullName == typeof(Tracker).FullName)
        .Methods.Single(method => method.Name == nameof(Tracker.ObserveTask));
    observer.Body = new MethodBody(observer);
    observer.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
    observer.DebugInformation.SequencePoints.Clear();
    assembly.Write(output, new WriterParameters { WriteSymbols = true });
    Console.WriteLine("Created unsafe private mutation control; never use as collection evidence.");
    return;
}
if (assembly.Name.HasPublicKey || AllTypes(assembly.MainModule.Types).SelectMany(type => type.Methods)
    .Where(method => method.HasBody).SelectMany(method => method.Body.Instructions)
    .Any(instruction => instruction.Operand is MethodReference reference &&
        reference.DeclaringType.FullName == typeof(Tracker).FullName && reference.Name is
            nameof(Tracker.Hit) or nameof(Tracker.ObserveTask) or nameof(Tracker.CheckProcessStart) or
            nameof(Tracker.UntrackedProcess) or nameof(Tracker.UntrackedConcurrency)))
    throw new InvalidOperationException("Signed or already instrumented inputs are unsupported by this prototype.");
var hit = assembly.MainModule.ImportReference(typeof(Tracker).GetMethod(nameof(Tracker.Hit))
    ?? throw new InvalidOperationException("Missing tracking method."));
var observe = assembly.MainModule.ImportReference(typeof(Tracker).GetMethod(nameof(Tracker.ObserveTask))
    ?? throw new InvalidOperationException("Missing task observer."));
MethodReference ImportTracker(string name) => assembly.MainModule.ImportReference(typeof(Tracker).GetMethod(name)
    ?? throw new InvalidOperationException($"Missing runtime hook {name}."));
var checkProcess = ImportTracker(nameof(Tracker.CheckProcessStart));
var untrackedProcess = ImportTracker(nameof(Tracker.UntrackedProcess));
var untrackedConcurrency = ImportTracker(nameof(Tracker.UntrackedConcurrency));
var methods = new List<object>();
var dependencyGraph = DependencyGraph.Read(assembly, inputHash);
int taskSites = 0;
int boundarySites = 0;
foreach (TypeDefinition type in AllTypes(assembly.MainModule.Types))
foreach (MethodDefinition method in type.Methods)
{
    if (!method.HasBody) continue;
    method.Body.SimplifyMacros();
    ILProcessor il = method.Body.GetILProcessor();
    foreach (Instruction instruction in method.Body.Instructions.ToArray())
    {
        if (instruction.OpCode.Code is not (Code.Call or Code.Callvirt or Code.Newobj) || instruction.Operand is not MethodReference call) continue;
        // Known escapes are rejected at execution time unless the process has an
        // explicit registered owner. We do not pretend to track arbitrary timers/threads.
        if (call.DeclaringType.FullName == "System.Diagnostics.Process" && call.Name == "Start")
        {
            if (!call.HasThis && call.Parameters.Count == 1 && call.Parameters[0].ParameterType.FullName == "System.Diagnostics.ProcessStartInfo")
            {
                InsertBeforeIncludingTargets(method, instruction, il.Create(OpCodes.Dup), il.Create(OpCodes.Call, checkProcess));
            }
            else InsertBeforeIncludingTargets(method, instruction, il.Create(OpCodes.Call, untrackedProcess));
            boundarySites++;
        }
        if ((call.DeclaringType.FullName is "System.Threading.Timer" or "System.Timers.Timer" && call.Name == ".ctor") ||
            (call.DeclaringType.FullName == "System.Threading.Thread" && call.Name == "Start") ||
            (call.DeclaringType.FullName == "System.Threading.ThreadPool" && call.Name.Contains("Queue", StringComparison.Ordinal)))
        {
            InsertBeforeIncludingTargets(method, instruction, il.Create(OpCodes.Call, untrackedConcurrency));
            boundarySites++;
        }
        string returnName = call.ReturnType is GenericInstanceType generic ? generic.ElementType.FullName : call.ReturnType.FullName;
        if (returnName is not ("System.Threading.Tasks.Task" or "System.Threading.Tasks.Task`1")) continue;
        if (instruction.Previous?.OpCode.Code == Code.Tail) throw new InvalidOperationException("Tail-call task instrumentation is unsupported.");
        // Preserve the original return value; the observer does not wrap or replace the task.
        Instruction duplicate = il.Create(OpCodes.Dup);
        il.InsertAfter(instruction, duplicate);
        il.InsertAfter(duplicate, il.Create(OpCodes.Call, observe));
        taskSites++;
    }
    method.Body.MaxStackSize += 1;
    if (mode == InstrumentationMode.TaskBoundaries || !method.DebugInformation.HasSequencePoints) continue;
    SequencePoint[] points = method.DebugInformation.SequencePoints.Where(point => !point.IsHidden).ToArray();
    if (points.Length == 0) continue;
    string key = $"{inputHash}:{method.MetadataToken.ToInt32():X8}";
    // Method entry is deliberately conservative: all its source spans are dependencies,
    // not claims that each line/branch executed. Async MoveNext is instrumented as well.
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
if (mode == InstrumentationMode.Methods && methods.Count == 0) throw new InvalidOperationException("No source-backed methods found.");
foreach (MethodDefinition method in AllTypes(assembly.MainModule.Types).SelectMany(type => type.Methods).Where(method => method.HasBody))
    method.Body.OptimizeMacros();
assembly.Write(output, new WriterParameters { WriteSymbols = true });
string outputHash = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(output)));
File.WriteAllText(output + ".map.json", JsonSerializer.Serialize(new
{
    Schema = 1, InputHash = inputHash, PdbHash = pdbHash, OutputHash = outputHash, Methods = methods, TaskSites = taskSites, BoundarySites = boundarySites, Mode = mode.ToString(), DependencyGraph = dependencyGraph
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Instrumented {methods.Count} source-backed methods and {taskSites} task sites into private copy {output}");

static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> roots)
{
    foreach (TypeDefinition type in roots)
    {
        yield return type;
        foreach (TypeDefinition nested in AllTypes(type.NestedTypes)) yield return nested;
    }
}

static void InsertBeforeIncludingTargets(MethodDefinition method, Instruction target, params Instruction[] inserted)
{
    if (target.Previous?.OpCode.OpCodeType == OpCodeType.Prefix)
        throw new InvalidOperationException("Prefixed escape calls are unsupported; refusing to emit invalid IL.");
    // A branch to the original call must not bypass its safety check.
    foreach (Instruction instruction in method.Body.Instructions)
    {
        if (ReferenceEquals(instruction.Operand, target)) instruction.Operand = inserted[0];
        else if (instruction.Operand is Instruction[] targets)
            for (int index = 0; index < targets.Length; index++)
                if (ReferenceEquals(targets[index], target)) targets[index] = inserted[0];
    }
    foreach (ExceptionHandler handler in method.Body.ExceptionHandlers)
    {
        if (handler.TryStart == target) handler.TryStart = inserted[0];
        if (handler.TryEnd == target) handler.TryEnd = inserted[0];
        if (handler.HandlerStart == target) handler.HandlerStart = inserted[0];
        if (handler.HandlerEnd == target) handler.HandlerEnd = inserted[0];
        if (handler.FilterStart == target) handler.FilterStart = inserted[0];
    }
    foreach (Instruction instruction in inserted) method.Body.GetILProcessor().InsertBefore(target, instruction);
}

enum InstrumentationMode { Methods, TaskBoundaries, RemoveTaskObserverForMutationTest }
