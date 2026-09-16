using System.Diagnostics;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using AiDotNet.TestImpact;
using Mono.Cecil;
using Mono.Cecil.Cil;
using MethodDefinition = Mono.Cecil.MethodDefinition;
using TypeDefinition = Mono.Cecil.TypeDefinition;
using Document = Mono.Cecil.Cil.Document;
using SequencePoint = Mono.Cecil.Cil.SequencePoint;
using TypeReference = Mono.Cecil.TypeReference;
using MemberReference = Mono.Cecil.MemberReference;
using AssemblyDefinition = Mono.Cecil.AssemblyDefinition;

internal static class SourceSnapshotReader
{
    public static SourceBundleSnapshot ReadBundle(string testBinary, string repository, string[] dependencies)
    {
        string[] binaries = new[] { testBinary }.Concat(dependencies).Select(Path.GetFullPath).ToArray();
        var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        var paths = binaries.ToHashSet(comparer);
        if (paths.Count != binaries.Length || binaries.Select(Path.GetFileName).Distinct(StringComparer.OrdinalIgnoreCase).Count() != binaries.Length)
            throw new InvalidDataException("Duplicate bundle assembly.");
        var hashes = binaries.ToDictionary(path => path, FileHash, comparer);
        var pdbHashes = binaries.ToDictionary(path => path, path => FileHash(Path.ChangeExtension(path, ".pdb")), comparer);
        var identities = new HashSet<string>(StringComparer.Ordinal);
        var methodIds = new HashSet<string>(StringComparer.Ordinal);
        foreach (string path in binaries)
        {
            using var definition = AssemblyDefinition.ReadAssembly(path);
            if (!identities.Add(definition.Name.Name)) throw new InvalidDataException("Ambiguous assembly name in source bundle.");
            foreach (MethodDefinition method in AllTypes(definition.MainModule.Types).SelectMany(type => type.Methods))
                if (method.HasBody || method.IsPInvokeImpl) methodIds.Add(DependencyGraph.Stable(method));
        }
        SourceSnapshot[] snapshots = binaries.Select(path => Read(path, repository, paths, methodIds)).ToArray();
        if (binaries.Any(path => FileHash(path) != hashes[path] || FileHash(Path.ChangeExtension(path, ".pdb")) != pdbHashes[path]) ||
            snapshots.Any(snapshot => snapshot.SourceTree != snapshots[0].SourceTree))
            throw new InvalidDataException("Source bundle changed during graph construction.");
        return new(1, snapshots[0].SourceTree, Path.GetFileName(testBinary), snapshots);
    }

    public static SourceSnapshot Read(string binary, string repository, IReadOnlySet<string>? linkedPaths = null,
        IReadOnlySet<string>? linkedMethodIds = null)
    {
        string root = Path.GetFullPath(repository);
        string source = Git(root, "rev-parse", "HEAD").Trim();
        if (Git(root, "status", "--porcelain", "--untracked-files=all").Length != 0)
            throw new InvalidDataException("Source snapshots require a clean checkout, not a claimed revision over modified files.");
        string hash = FileHash(binary);
        string pdb = Path.ChangeExtension(binary, ".pdb");
        string pdbHash = FileHash(pdb);
        using var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(binary)));
        resolver.AddSearchDirectory(Path.GetDirectoryName(typeof(object).Assembly.Location));
        if (linkedPaths is not null)
            foreach (string path in linkedPaths) resolver.AddSearchDirectory(Path.GetDirectoryName(path));
        using var assembly = AssemblyDefinition.ReadAssembly(binary, new ReaderParameters { ReadSymbols = true, InMemory = true, AssemblyResolver = resolver });
        if (assembly.Name.HasPublicKey || assembly.Modules.Count != 1)
            throw new InvalidDataException("Signed or multi-module snapshots are not supported.");
        IReadOnlySet<string> sourcePaths = linkedPaths ?? new HashSet<string>([Path.GetFullPath(binary)],
            OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        using var managed = new ManagedDependencyReader(binary, sourcePaths, resolver);
        MethodDependencyGraph raw = DependencyGraph.Read(assembly, hash, linkedPaths, managed);
        var definitions = AllTypes(assembly.MainModule.Types).SelectMany(type => type.Methods)
            .ToDictionary(method => $"{hash}:{method.MetadataToken.ToInt32():X8}", StringComparer.Ordinal);
        string Stable(MethodDependencyNode node) => DependencyGraph.Stable(definitions[node.Key]);
        var names = raw.Methods.ToDictionary(node => node.Key, Stable, StringComparer.Ordinal);
        bool verified = true;
        var documents = new Dictionary<string, string?>(StringComparer.Ordinal);
        var methods = new List<SourceMethod>();
        XunitLifecycleResult lifecycle = XunitLifecycleReader.Read(assembly);
        managed.ObserveRoots(lifecycle.Map.GroupRoots.Concat(lifecycle.Map.Tests.SelectMany(test => test.Roots))
            .Concat(lifecycle.SyntheticMethods.SelectMany(method => method.Dependency.Calls)));
        SourceManagedDependencies dependencies = managed.Read();
        using var stream = File.OpenRead(binary);
        using var pe = new PEReader(stream);
        foreach (MethodDependencyNode node in raw.Methods)
        {
            MethodDefinition method = definitions[node.Key];
            var spans = new List<SourceSpan>();
            foreach (SequencePoint point in method.DebugInformation.SequencePoints.Where(point => !point.IsHidden))
            {
                string? path = VerifyDocument(point.Document, root, documents);
                if (path is null)
                {
                    bool inertEntryPoint = method == assembly.EntryPoint && method.HasBody &&
                        method.Body.Instructions.All(instruction => instruction.OpCode.Code is Code.Nop or Code.Ret) &&
                        File.Exists(point.Document.Url) && MatchesChecksum(point.Document, File.ReadAllBytes(point.Document.Url));
                    if (!inertEntryPoint) verified = false;
                    continue;
                }
                spans.Add(new(path, point.StartLine, point.EndLine));
            }
            string bodyHash = method.HasBody ? BodyHash(pe, method) : hash;
            DependencyBoundary boundary = node.OpenDependencies.Length == 0 ? DependencyBoundary.Closed : DependencyBoundary.Unresolved;
            // The CLR's empty Object constructor has no callbacks or shared state.
            // Other external calls remain open; do not guess purity from a name.
            if (node.OpenDependencies.Length > 0 && node.OpenDependencies.All(open => open.Kind == OpenDependencyKind.ExternalCall &&
                open.Target == "System.Void System.Object::.ctor()") && method.HasBody && method.Body.Instructions
                .Where(instruction => instruction.Operand is MethodReference reference && reference.FullName == "System.Void System.Object::.ctor()")
                .All(instruction => IsRuntimeObjectConstructor((MethodReference)instruction.Operand)))
                boundary = DependencyBoundary.Closed;
            // Generic definitions still have concrete IL dependencies. Indirect,
            // constrained/virtual and unresolved targets remain open in the
            // extracted graph; generic syntax alone is not an unknown call.
            methods.Add(new(new(Stable(node), node.LocalCalls.Select(key => names.TryGetValue(key, out string? id) ? id :
                    linkedMethodIds?.Contains(key) == true || managed.Contains(key) ? key : "unresolved:" + key).ToArray(),
                node.StaticFields, boundary),
                assembly.Name.Name + ":" + method.DeclaringType.FullName.Replace('/', '+') + "." + method.Name,
                bodyHash, spans.ToArray(), method.IsConstructor || method.IsVirtual));
        }
        // Canonicalize every body to the same never-executed placeholder, leaving
        // declarations, resources, field initializers, attributes and references.
        // Any change to this envelope forces full execution. MVID/timestamp alone
        // must not make otherwise identical metadata look changed.
        assembly.MainModule.Mvid = Guid.Empty;
        foreach (MethodDefinition method in definitions.Values.Where(method => method.HasBody))
        {
            method.Body = new MethodBody(method);
            method.Body.Instructions.Add(Instruction.Create(OpCodes.Ret));
            method.DebugInformation.SequencePoints.Clear();
            method.DebugInformation.Scope = null;
        }
        using var normalized = new MemoryStream();
        assembly.Write(normalized, new WriterParameters { WriteSymbols = false, Timestamp = 0 });
        string configuration = Convert.ToHexStringLower(SHA256.HashData(System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(new
        {
            Metadata = Convert.ToHexStringLower(SHA256.HashData(normalized.ToArray())),
            dependencies.Files
        })));
        verified &= documents.Count > 0 && methods.Any(method => method.Spans.Length > 0);
        if (FileHash(binary) != hash || FileHash(pdb) != pdbHash || Git(root, "rev-parse", "HEAD").Trim() != source ||
            Git(root, "status", "--porcelain", "--untracked-files=all").Length != 0)
            throw new InvalidDataException("Inputs changed while constructing the source snapshot.");
        methods.AddRange(lifecycle.SyntheticMethods);
        return new(1, source, Path.GetFileName(binary), hash, pdbHash, configuration, verified ? SourceMapStatus.Verified : SourceMapStatus.Unverifiable,
            methods.ToArray(), lifecycle.Map, dependencies);
    }

    private static string FileHash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    internal static string BodyHash(PEReader pe, MethodDefinition method)
    {
        // Raw tokens alone miss changes to the metadata they reference, especially
        // local-variable signatures removed from the normalized envelope.
        string TypeName(TypeReference type) => type.FullName + "@" + type.Scope;
        var semantic = new
        {
            Raw = pe.GetSectionData(method.RVA).GetContent(0, pe.GetMethodBody(method.RVA).Size).ToArray(),
            Locals = method.Body.Variables.Select(variable => TypeName(variable.VariableType)).ToArray(),
            Catches = method.Body.ExceptionHandlers.Select(handler => handler.CatchType is null ? null : TypeName(handler.CatchType)).ToArray(),
            References = method.Body.Instructions.Select(instruction => instruction.Operand switch
            {
                TypeReference type => TypeName(type),
                MemberReference member => member.FullName + "@" + member.DeclaringType.Scope,
                string literal => literal,
                _ => null
            }).ToArray()
        };
        return Convert.ToHexStringLower(SHA256.HashData(System.Text.Json.JsonSerializer.SerializeToUtf8Bytes(semantic)));
    }

    private static bool IsRuntimeObjectConstructor(MethodReference reference)
    {
        try
        {
            MethodDefinition? definition = reference.Resolve();
            return definition is not null && Path.GetFullPath(definition.Module.FileName).Equals(
                Path.GetFullPath(typeof(object).Assembly.Location), OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
        }
        catch (AssemblyResolutionException) { return false; }
    }

    private static string? VerifyDocument(Document document, string root, Dictionary<string, string?> cache)
    {
        string key = document.Url + ":" + document.HashAlgorithm + ":" + Convert.ToHexString(document.Hash);
        if (cache.TryGetValue(key, out string? cached)) return cached;
        string full = Path.GetFullPath(document.Url);
        string relative = Path.GetRelativePath(root, full);
        bool contained = !Path.IsPathRooted(relative) && relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal);
        string? path = null;
        if (contained && File.Exists(full))
        {
            bool linked = false;
            for (string? current = full; current is not null; current = Path.GetDirectoryName(current))
            {
                linked |= (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0;
                if (current == root) break;
            }
            byte[] bytes = File.ReadAllBytes(full);
            if (!linked && MatchesChecksum(document, bytes)) path = relative.Replace('\\', '/');
        }
        cache.Add(key, path);
        return path;
    }

    private static bool MatchesChecksum(Document document, byte[] bytes)
    {
        byte[]? checksum = document.HashAlgorithm switch
        {
            DocumentHashAlgorithm.SHA256 => SHA256.HashData(bytes),
            DocumentHashAlgorithm.SHA1 => SHA1.HashData(bytes),
            _ => null
        };
        return checksum is not null && checksum.SequenceEqual(document.Hash);
    }

    private static string Git(string root, params string[] arguments)
    {
        var start = new ProcessStartInfo("git") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false, CreateNoWindow = true };
        start.ArgumentList.Add("-C"); start.ArgumentList.Add(root);
        foreach (string argument in arguments) start.ArgumentList.Add(argument);
        using Process process = Process.Start(start) ?? throw new IOException("Cannot start git.");
        Task<string> error = process.StandardError.ReadToEndAsync();
        string result = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0) throw new IOException(error.GetAwaiter().GetResult());
        return result;
    }

    private static IEnumerable<TypeDefinition> AllTypes(IEnumerable<TypeDefinition> types)
    {
        foreach (TypeDefinition type in types)
        {
            yield return type;
            foreach (TypeDefinition nested in AllTypes(type.NestedTypes)) yield return nested;
        }
    }
}
