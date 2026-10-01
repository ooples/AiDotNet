using System.Collections.Immutable;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;

namespace AiDotNet.TestImpact.TypeImpact;

/// <summary>An assembly and its portable PDB, kept open for the life of the process.</summary>
internal sealed class LoadedAssembly
{
    // Roslyn records the source documents of a type with no sequence points (an interface, an
    // enum, a field-only class) under this custom-debug-information kind.
    private static readonly Guid TypeDefinitionDocuments = new("932E74BC-DBA9-4478-8D46-0F32A7BAB3D3");

    private readonly MetadataReaderProvider _pdbProvider;
    private readonly string _repoRoot;

    private LoadedAssembly(string name, PEReader pe, MetadataReaderProvider pdbProvider, string repoRoot)
    {
        Name = name;
        PE = pe;
        Metadata = pe.GetMetadataReader();
        _pdbProvider = pdbProvider;
        Pdb = pdbProvider.GetMetadataReader();
        _repoRoot = repoRoot;
        foreach (var handle in Metadata.TypeDefinitions)
        {
            var type = Metadata.GetTypeDefinition(handle);
            if (type.GetDeclaringType().IsNil)
            {
                DefinitionsByFullName[AssemblyIndex.FullName(Metadata, handle)] = handle;
            }
        }

        ReferencesXunit = Metadata.AssemblyReferences
            .Select(r => Metadata.GetString(Metadata.GetAssemblyReference(r).Name))
            .Any(n => n.StartsWith("xunit", StringComparison.OrdinalIgnoreCase));
    }

    public string Name { get; }
    public PEReader PE { get; }
    public MetadataReader Metadata { get; }
    public MetadataReader Pdb { get; }
    public bool ReferencesXunit { get; }
    public Dictionary<TypeDefinitionHandle, TypeNode> NodeOf { get; } = [];
    public Dictionary<string, TypeDefinitionHandle> DefinitionsByFullName { get; } = new(StringComparer.Ordinal);

    /// <summary>Opens the pair, or returns null when none of its sources is under the repository.</summary>
    public static LoadedAssembly? TryOpen(string dll, string pdb, string repoRoot)
    {
        PEReader? pe = null;
        MetadataReaderProvider? provider = null;
        LoadedAssembly? loaded = null;
        try
        {
            pe = new PEReader(File.OpenRead(dll));
            if (!pe.HasMetadata)
            {
                return null;
            }

            provider = MetadataReaderProvider.FromPortablePdbStream(File.OpenRead(pdb));
            var reader = provider.GetMetadataReader();
            var root = repoRoot;
            bool ours = reader.Documents.Any(d => RepoRelative(reader.GetString(reader.GetDocument(d).Name), root) is { } relative
                && !relative.StartsWith(GeneratedDocumentPrefix, StringComparison.Ordinal));
            if (!ours)
            {
                return null;
            }

            var name = pe.GetMetadataReader().GetString(pe.GetMetadataReader().GetAssemblyDefinition().Name);
            loaded = new LoadedAssembly(name, pe, provider, repoRoot);
            return loaded;
        }
        catch (BadImageFormatException)
        {
            // Native or Windows-PDB binaries in the output folder are not ours to map.
            return null;
        }
        finally
        {
            // Every exit that did not hand the readers to a LoadedAssembly, exceptions included.
            if (loaded is null)
            {
                provider?.Dispose();
                pe?.Dispose();
            }
        }
    }

    /// <summary>Every repository source document that contributes to each type definition.</summary>
    public IEnumerable<(TypeDefinitionHandle Type, IReadOnlyCollection<string> Documents)> DocumentsByType()
    {
        var result = new Dictionary<TypeDefinitionHandle, HashSet<string>>();
        void Add(TypeDefinitionHandle type, DocumentHandle document)
        {
            var path = RepoRelative(Pdb.GetString(Pdb.GetDocument(document).Name), _repoRoot);
            if (path is null)
            {
                return;
            }

            if (!result.TryGetValue(type, out var set))
            {
                result[type] = set = new HashSet<string>(StringComparer.Ordinal);
            }

            set.Add(path);
        }

        foreach (var methodDebugHandle in Pdb.MethodDebugInformation)
        {
            var info = Pdb.GetMethodDebugInformation(methodDebugHandle);
            if (info.SequencePointsBlob.IsNil)
            {
                continue;
            }

            var method = Metadata.GetMethodDefinition(methodDebugHandle.ToDefinitionHandle());
            var type = method.GetDeclaringType();
            if (!info.Document.IsNil)
            {
                Add(type, info.Document);
            }
            else
            {
                // Multi-document method (partial or #line-mapped): walk its points.
                foreach (var point in info.GetSequencePoints())
                {
                    Add(type, point.Document);
                }
            }
        }

        foreach (var typeHandle in Metadata.TypeDefinitions)
        {
            foreach (var cdi in Pdb.GetCustomDebugInformation(typeHandle).Select(Pdb.GetCustomDebugInformation))
            {
                if (Pdb.GetGuid(cdi.Kind) != TypeDefinitionDocuments)
                {
                    continue;
                }

                var blob = Pdb.GetBlobReader(cdi.Value);
                while (blob.RemainingBytes > 0)
                {
                    Add(typeHandle, MetadataTokens.DocumentHandle(blob.ReadCompressedInteger()));
                }
            }
        }

        return result.Select(kv => (kv.Key, (IReadOnlyCollection<string>)kv.Value));
    }

    /// <summary>
    /// A PDB document path as a repository-relative, forward-slash path; null when it lies outside
    /// the repository (generated sources in obj/, SDK sources, other checkouts).
    /// </summary>
    /// <summary>Prefix of the document keys for source-generator output; see <see cref="RepoRelative"/>.</summary>
    public const string GeneratedDocumentPrefix = "<generated>/";

    // "<assembly>/<namespace-qualified generator type>/<hint>": relative, at least three segments, the second dotted.
    private static bool IsGeneratedDocument(string path)
    {
        if (path.Length == 0 || path[0] == '/' || (path.Length > 1 && path[1] == ':')) return false;
        var segments = path.Split('/');
        return segments.Length >= 3 && segments[1].Contains('.', StringComparison.Ordinal);
    }

    public static string? RepoRelative(string documentPath, string repoRoot)
    {
        var path = documentPath.Replace('\\', '/');
        string? relative = null;
        // Deterministic CI builds map the source root to /_/ .
        if (path.StartsWith("/_/", StringComparison.Ordinal))
        {
            relative = path[3..];
        }
        else
        {
            var root = repoRoot.Replace('\\', '/').TrimEnd('/') + "/";
            if (path.StartsWith(root, StringComparison.OrdinalIgnoreCase))
            {
                relative = path[root.Length..];
            }
        }

        // A source generator's output is recorded as ".../obj/<configuration>/<framework>/<generator assembly>/<generator
        // type>/<hint>", or as just "<generator assembly>/<generator type>/<hint>" when nothing roots it. It is kept as
        // "<generated>/<generator assembly>/<generator type>/<hint>", a key no repository path can have, so a change to a
        // generator maps to exactly the types that generator emitted. Everything else under obj/ is dropped.
        if (relative is null && IsGeneratedDocument(path))
        {
            return GeneratedDocumentPrefix + path;
        }

        if (relative is not null && (relative.StartsWith("obj/", StringComparison.Ordinal) || relative.Contains("/obj/", StringComparison.Ordinal)))
        {
            var parts = relative.Split('/');
            int obj = Array.LastIndexOf(parts, "obj");
            return obj >= 0 && parts.Length - obj >= 6 && parts[^2].Contains('.', StringComparison.Ordinal)
                ? GeneratedDocumentPrefix + string.Join('/', parts[^3..])
                : null;
        }

        if (relative is null || relative.Contains("/obj/", StringComparison.Ordinal) || relative.StartsWith("obj/", StringComparison.Ordinal))
        {
            return null;
        }

        return relative;
    }
}
