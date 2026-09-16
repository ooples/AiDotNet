using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;

// Read reachable managed IL from the actual bound output/runtime, not arbitrary
// assemblies in the machine's package cache. Native, virtual, reflection and
// missing/budget-limited targets remain explicit open graph boundaries.
internal sealed class ManagedDependencyReader : IDisposable
{
    private const int MaximumMethods = 50_000;
    private const int MaximumModules = 64;
    private readonly IReadOnlySet<string> sourcePaths;
    private readonly HashSet<string> sourceNames;
    private readonly string bundle;
    private readonly string runtime;
    private readonly IAssemblyResolver resolver;
    private readonly SourceDependencyBinary[] initialFiles;
    private readonly Dictionary<string, ModuleInput> modules = new(StringComparer.Ordinal);
    private readonly Dictionary<string, MethodDefinition> pending = new(StringComparer.Ordinal);
    private readonly Queue<MethodDefinition> queue = new();
    private readonly Dictionary<string, bool> types = new(StringComparer.Ordinal);
    private List<MethodDefinition>? candidates;
    private bool truncated;

    internal ManagedDependencyReader(string binary, IReadOnlySet<string> sourcePaths, IAssemblyResolver resolver)
    {
        this.sourcePaths = sourcePaths;
        sourceNames = sourcePaths.Select(Path.GetFileNameWithoutExtension).OfType<string>().ToHashSet(StringComparer.Ordinal);
        bundle = Path.GetDirectoryName(Path.GetFullPath(binary)) ?? throw new IOException("Missing bundle directory.");
        runtime = Path.GetDirectoryName(typeof(object).Assembly.Location) ?? throw new IOException("Missing runtime directory.");
        this.resolver = resolver;
        initialFiles = CaptureFiles();
    }

    internal bool Contains(string id) => pending.ContainsKey(id);

    internal bool IsBound(TypeReference type) => IsBound(type, 0);

    private bool IsBound(TypeReference type, int depth)
    {
        if (depth > 128 || type is PointerType or FunctionPointerType) return false;
        if (type is GenericParameter) return true; // Concrete call-site arguments are checked separately.
        if (type is GenericInstanceType generic && generic.GenericArguments.Any(argument => !IsBound(argument, depth + 1))) return false;
        if (type is RequiredModifierType required && !IsBound(required.ModifierType, depth + 1) ||
            type is OptionalModifierType optional && !IsBound(optional.ModifierType, depth + 1)) return false;
        if (type is TypeSpecification specification) return IsBound(specification.ElementType, depth + 1);
        if (type.IsPrimitive || type.MetadataType is MetadataType.Void or MetadataType.String or MetadataType.Object) return true;
        string key = type.FullName + "@" + type.Scope;
        if (types.TryGetValue(key, out bool known)) return known;
        types[key] = false;
        try
        {
            TypeDefinition? definition = type.Resolve();
            bool bound = definition is not null && (sourcePaths.Contains(Path.GetFullPath(definition.Module.FileName)) || Input(definition.Module) is not null);
            if (bound && definition is not null && definition.IsValueType)
                bound = definition.Fields.Where(field => !field.IsStatic).All(field => IsBound(field.FieldType, depth + 1));
            types[key] = bound;
            return bound;
        }
        catch (AssemblyResolutionException) { return false; }
        catch (ResolutionException) { return false; }
    }

    internal MethodDefinition? Resolve(MethodReference reference)
    {
        try
        {
            MethodDefinition? method = reference.Resolve();
            return method is not null && Input(method.Module) is not null ? method : null;
        }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }

    internal FieldDefinition? Resolve(FieldReference reference)
    {
        try
        {
            FieldDefinition? field = reference.Resolve();
            return field is not null && Input(field.Module) is not null ? field : null;
        }
        catch (AssemblyResolutionException) { return null; }
        catch (ResolutionException) { return null; }
    }

    internal void Observe(MethodDefinition method)
    {
        if (sourcePaths.Contains(Path.GetFullPath(method.Module.FileName))) return;
        if (candidates is not null) { candidates.Add(method); return; }
        string id = DependencyGraph.Stable(method);
        if (pending.ContainsKey(id)) return;
        if (pending.Count >= MaximumMethods || Input(method.Module) is null) { truncated = true; return; }
        pending.Add(id, method);
        queue.Enqueue(method);
    }

    internal void BeginMethod()
    {
        if (candidates is not null) throw new InvalidOperationException("Nested managed graph extraction.");
        candidates = [];
    }

    internal void EndMethod(bool closed)
    {
        List<MethodDefinition> found = candidates ?? throw new InvalidOperationException("Missing managed graph extraction.");
        candidates = null;
        // An already-open node cannot authorize skipping. Expanding all the
        // runtime implementation beneath it adds cost but no proof; preserve
        // its open boundary and only expand candidates of potentially closed IL.
        if (closed) foreach (MethodDefinition method in found) Observe(method);
    }

    internal void ObserveRoots(IEnumerable<string> roots)
    {
        foreach (string id in roots.Distinct(StringComparer.Ordinal))
        {
            int separator = id.IndexOf(':');
            if (separator <= 0) continue;
            string assembly = id[..separator];
            if (assembly is "unresolved" or "unmapped" or "xunit-fixture" || sourceNames.Contains(assembly)) continue;
            try
            {
                ModuleDefinition module = resolver.Resolve(new AssemblyNameReference(assembly, new Version(0, 0))).MainModule;
                ModuleInput? input = Input(module);
                if (input is not null && input.ById.TryGetValue(id, out MethodDefinition? method)) Observe(method);
            }
            catch (AssemblyResolutionException) { /* Missing root remains open in the graph. */ }
        }
    }

    internal SourceManagedDependencies Read()
    {
        var result = new List<SourceMethod>();
        while (queue.TryDequeue(out MethodDefinition? method))
        {
            ModuleInput input = Input(method.Module) ?? throw new InvalidDataException("Managed input disappeared.");
            MethodDependencyNode node = DependencyGraph.Read(method.Module.Assembly, input.Hash, sourcePaths, this, [method]).Methods.Single();
            string Normalize(string key) => input.ByKey.TryGetValue(key, out MethodDefinition? target) ? DependencyGraph.Stable(target) : key;
            result.Add(new(new(DependencyGraph.Stable(method), node.LocalCalls.Select(Normalize).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
                    node.StaticFields, node.OpenDependencies.Length == 0 ? DependencyBoundary.Closed : DependencyBoundary.Unresolved),
                method.Module.Assembly.Name.Name + ":" + method.DeclaringType.FullName.Replace('/', '+') + "." + method.Name,
                SourceSnapshotReader.BodyHash(input.Pe, method), [], false, SourceMethodScope.DependencyAssembly));
        }
        foreach (ModuleInput module in modules.Values)
            if (Hash(module.Module.FileName) != module.Hash) throw new InvalidDataException("Managed dependency changed during mapping.");
        if (!initialFiles.SequenceEqual(CaptureFiles())) throw new InvalidDataException("Dependency files changed during mapping.");
        return new(1, initialFiles,
            result.OrderBy(method => method.Dependency.Id, StringComparer.Ordinal).ToArray(), truncated);
    }

    private SourceDependencyBinary[] CaptureFiles()
    {
        var result = new List<SourceDependencyBinary>();
        var excluded = sourcePaths.Concat(sourcePaths.Select(path => Path.ChangeExtension(path, ".pdb"))).ToHashSet(
            OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        foreach ((string root, ManagedBinaryOrigin origin) in new[] { (bundle, ManagedBinaryOrigin.Bundle), (runtime, ManagedBinaryOrigin.Runtime) })
            foreach (string file in Files(root).Order(StringComparer.Ordinal))
                if (!excluded.Contains(file)) result.Add(new(Path.GetRelativePath(root, file).Replace('\\', '/'), Hash(file), origin));
        return result.OrderBy(file => file.Origin).ThenBy(file => file.File, StringComparer.Ordinal).ToArray();
    }

    private static IEnumerable<string> Files(string root)
    {
        if ((File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0) throw new InvalidDataException("Linked dependency paths are unsupported.");
        foreach (string path in Directory.EnumerateFileSystemEntries(root))
        {
            FileAttributes attributes = File.GetAttributes(path);
            if ((attributes & FileAttributes.ReparsePoint) != 0) throw new InvalidDataException("Linked dependency paths are unsupported.");
            if ((attributes & FileAttributes.Directory) != 0)
                foreach (string file in Files(path)) yield return file;
            else yield return path;
        }
    }

    private ModuleInput? Input(ModuleDefinition module)
    {
        string file = Path.GetFullPath(module.FileName);
        if (sourcePaths.Contains(file)) return null;
        if (modules.TryGetValue(file, out ModuleInput? existing)) return existing;
        if (modules.Count >= MaximumModules) { truncated = true; return null; }
        string? parent = Path.GetDirectoryName(file);
        StringComparison comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        ManagedBinaryOrigin origin;
        if (string.Equals(parent, bundle, comparison)) origin = ManagedBinaryOrigin.Bundle;
        else if (string.Equals(parent, runtime, comparison)) origin = ManagedBinaryOrigin.Runtime;
        else return null;
        // Do not guess CLR binding when the output and runtime contain different
        // binaries with the same simple filename. Their callers remain open.
        if (initialFiles.Where(input => Path.GetFileName(input.File).Equals(Path.GetFileName(file), StringComparison.OrdinalIgnoreCase))
                .Select(input => input.Hash).Distinct(StringComparer.Ordinal).Skip(1).Any()) return null;
        if ((File.GetAttributes(file) & FileAttributes.ReparsePoint) != 0 || module.Assembly.Modules.Count != 1) return null;
        var input = new ModuleInput(module, origin);
        modules.Add(file, input);
        return input;
    }

    internal static string Hash(string file)
    {
        using var stream = File.OpenRead(file);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    public void Dispose()
    {
        foreach (ModuleInput module in modules.Values) module.Pe.Dispose();
    }

    private sealed class ModuleInput
    {
        internal ModuleInput(ModuleDefinition module, ManagedBinaryOrigin origin)
        {
            Module = module;
            Origin = origin;
            Hash = ManagedDependencyReader.Hash(module.FileName);
            Pe = new PEReader(File.OpenRead(module.FileName));
            MethodDefinition[] methods = AllTypes(module.Types).SelectMany(type => type.Methods).ToArray();
            ById = methods.ToDictionary(DependencyGraph.Stable, StringComparer.Ordinal);
            ByKey = methods.ToDictionary(method => $"{Hash}:{method.MetadataToken.ToInt32():X8}", StringComparer.Ordinal);
        }
        internal ModuleDefinition Module { get; }
        internal ManagedBinaryOrigin Origin { get; }
        internal string Hash { get; }
        internal PEReader Pe { get; }
        internal Dictionary<string, MethodDefinition> ById { get; }
        internal Dictionary<string, MethodDefinition> ByKey { get; }
        private static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> types)
        {
            foreach (TypeDefinition type in types)
            {
                yield return type;
                foreach (TypeDefinition nested in AllTypes(type.NestedTypes)) yield return nested;
            }
        }
    }
}
