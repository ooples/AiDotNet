using System.Security.Cryptography;
using System.Text;
using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using Mono.Cecil;

internal enum FactDiscoveryContract { SkippableFact185MetadataOnly }

// Catalog for discovery ONLY. SkippableFact execution remains a custom runner
// and cannot satisfy ReviewedOwnerCompletion. The reviewed discovery iterator
// reads exception type names and constructs a case with null method arguments;
// it never invokes the test method. Metadata changes remain full-selection inputs.
internal sealed class ReviewedFactDiscovery
{
    internal const string PackageHash = "f8fb7e54fb771f40c0a6b773e20954545277fb0ba71286c87bfd07410c3c1160";
    private readonly Dictionary<string, string> hashes = new(StringComparer.Ordinal);

    internal bool IsStandardTrait(CustomAttribute attribute)
    {
        try
        {
            TypeDefinition? type = attribute.AttributeType.Resolve();
            return type is not null && type.FullName == "Xunit.TraitAttribute" &&
                Hash(type.Module.FileName) == ReviewedOwnerCompletion.CoreHash && attribute.Fields.Count == 0 &&
                attribute.Properties.Count == 0 && attribute.ConstructorArguments.Count == 2 &&
                attribute.ConstructorArguments.All(argument => argument.Value is string);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException) { return false; }
    }

    internal SourceMethod? Read(CustomAttribute attribute)
    {
        if (attribute.AttributeType.FullName != "Xunit.SkippableFactAttribute") return null;
        try
        {
            TypeDefinition? type = attribute.AttributeType.Resolve();
            if (type is null || type.Module.Assembly.Name.Name != "Xunit.SkippableFact" || Hash(type.Module.FileName) != PackageHash)
                return null;
            string bundle = Path.GetDirectoryName(type.Module.FileName) ?? throw new IOException("Missing package directory.");
            if (Hash(Path.Combine(bundle, "xunit.core.dll")) != ReviewedOwnerCompletion.CoreHash ||
                Hash(Path.Combine(bundle, "xunit.execution.dotnet.dll")) != ReviewedOwnerCompletion.ExecutionHash ||
                Hash(typeof(object).Assembly.Location) != ReviewedOwnerCompletion.RuntimeHash ||
                Hash(Path.Combine(bundle, "Attribution.Xunit.dll")) != Hash(typeof(AttributionTestFramework).Assembly.Location)) return null;
            // Resolve the actual base used by this attribute, not a same-named
            // DLL elsewhere in the resolver's search path.
            TypeDefinition? parent = type.BaseType?.Resolve();
            if (parent is null || parent.FullName != "Xunit.FactAttribute" ||
                Hash(parent.Module.FileName) != ReviewedOwnerCompletion.CoreHash) return null;
            if (attribute.Constructor.FullName != "System.Void Xunit.SkippableFactAttribute::.ctor(System.Type[])" ||
                attribute.ConstructorArguments.Count != 1 || attribute.Fields.Count != 0 ||
                attribute.Properties.Any(property => property.Name is not ("DisplayName" or "Skip" or "Timeout"))) return null;
            // The exception list contains metadata type references, never
            // factories/instances or custom conversion callbacks.
            object? arguments = attribute.ConstructorArguments[0].Value;
            if (arguments is not null && (arguments is not CustomAttributeArgument[] values ||
                values.Any(value => value.Value is not TypeReference))) return null;
            string id = "reviewed-discovery:" + FactDiscoveryContract.SkippableFact185MetadataOnly;
            string hash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes(string.Join("|",
                PackageHash, ReviewedOwnerCompletion.CoreHash, ReviewedOwnerCompletion.ExecutionHash,
                ReviewedOwnerCompletion.RuntimeHash, Hash(typeof(AttributionTestFramework).Assembly.Location)))));
            return new(new(id, [], [], DependencyBoundary.Closed), id, hash, [], true);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return null;
        }
    }

    private string Hash(string path)
    {
        path = Path.GetFullPath(path);
        if (hashes.TryGetValue(path, out string? found)) return found;
        for (FileSystemInfo? entry = new FileInfo(path); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if ((entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Linked discovery contract input.");
        using var stream = File.OpenRead(path);
        string hash = Convert.ToHexStringLower(SHA256.HashData(stream));
        hashes.Add(path, hash);
        return hash;
    }
}
