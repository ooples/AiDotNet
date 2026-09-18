using System.Security.Cryptography;
using AiDotNet.TestImpact;
using AiDotNet.TestImpact.Xunit;
using AttributionRuntime;
using Mono.Cecil;

internal sealed record OwnerCompletionObservation(string Owner, OwnerCompletionProof Proof);

// This closes ONLY the returned-task observation requirement. Initialization,
// file/AsyncLocal effects and source-change completeness remain separate gates.
// The pinned xUnit path awaits the reflected Task; its timeout path requires
// the underlying invocation to win WhenAny and reads its Result. Neither a
// timeout nor a fault can be reported as a passing standard case on that path.
internal static class ReviewedOwnerCompletion
{
    internal const string ExecutionHash = "e70febe547c15f8155db3e2d95c3ba4445a59cbc21a3e7541ab22ed49d3dcf7b";
    internal const string CoreHash = "2f7feeec8c6dec7df684c25fb80aadb292b963d09723460f03d53c5ad2088acc";

    // Bounded prototype catalog entry. Other runtimes require a review rather
    // than inheriting task semantics from a filename or framework version label.
    internal const string RuntimeHash = "1125acc8106c43fc8bad2d203c4c4485df6182d292846c2fff415c1040c54678";

    internal static OwnerCompletionProof Read(string bundle, string assemblyFile, string owner, VerifiedObservedExecution observed)
        => ReadAll(bundle, assemblyFile, [owner], observed)[0].Proof;

    internal static OwnerCompletionObservation[] ReadAll(string bundle, string assemblyFile, string[] owners, VerifiedObservedExecution observed)
    {
        OwnerCompletionObservation[] Unknown() => owners.Select(owner => new OwnerCompletionObservation(owner, OwnerCompletionProof.Unresolved)).ToArray();
        try
        {
            if (owners.Length == 0) return [];
            if (Path.GetFileName(assemblyFile) != assemblyFile ||
                Path.GetExtension(assemblyFile) != ".dll" || RunnerBinding.HashBundle(bundle) != observed.Execution.Context.BuildFingerprint)
                return Unknown();
            string runtime = typeof(object).Assembly.Location;
            if (Hash(runtime) != RuntimeHash || Hash(Path.Combine(bundle, "xunit.execution.dotnet.dll")) != ExecutionHash ||
                Hash(Path.Combine(bundle, "xunit.core.dll")) != CoreHash ||
                Hash(Path.Combine(bundle, "Attribution.Xunit.dll")) != Hash(typeof(AttributionTestFramework).Assembly.Location) ||
                Hash(Path.Combine(bundle, "AttributionRuntime.dll")) != Hash(typeof(PlannedEvidence).Assembly.Location))
                return Unknown();
            using var resolver = new DefaultAssemblyResolver();
            resolver.AddSearchDirectory(Path.GetFullPath(bundle));
            resolver.AddSearchDirectory(Path.GetDirectoryName(runtime));
            using var assembly = AssemblyDefinition.ReadAssembly(Path.Combine(bundle, assemblyFile),
                new ReaderParameters { AssemblyResolver = resolver, InMemory = true });
            if (!StandardFramework(assembly, bundle)) return Unknown();
            var methods = Types(assembly.MainModule.Types).SelectMany(type => type.Methods).ToLookup(method =>
                assembly.Name.Name + ":" + method.DeclaringType.FullName.Replace('/', '+') + "." + method.Name, StringComparer.Ordinal);
            var result = owners.Select(owner => new OwnerCompletionObservation(owner, observed.HasStandardOwnerCompletion(owner)
                ? Entry(methods[owner].ToArray(), bundle) : OwnerCompletionProof.Unresolved)).ToArray();
            return RunnerBinding.HashBundle(bundle) == observed.Execution.Context.BuildFingerprint && Hash(runtime) == RuntimeHash
                ? result : Unknown();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    private static OwnerCompletionProof Entry(MethodDefinition[] entries, string bundle)
    {
        if (entries.Length != 1) return OwnerCompletionProof.Unresolved;
        MethodDefinition entry = entries[0];
        CustomAttribute[] facts = entry.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.FactAttribute").ToArray();
        if (facts.Length != 1 || !FromBundle(facts[0].AttributeType.Resolve(), bundle, "xunit.core.dll"))
            return OwnerCompletionProof.Unresolved;
        CustomAttribute[] markers = entry.CustomAttributes.Where(attribute =>
            attribute.AttributeType.FullName == "System.Runtime.CompilerServices.AsyncStateMachineAttribute").ToArray();
        if (markers.Length != 1 || markers[0].ConstructorArguments.Count != 1 ||
            markers[0].ConstructorArguments[0].Value is not TypeReference state || state.Resolve() is not TypeDefinition stateType)
            return OwnerCompletionProof.Unresolved;
        MethodDefinition[] bodies = stateType.Methods.Where(method => method.Name == "MoveNext").ToArray();
        if (bodies.Length != 1 || AsyncOwnerReader.Read(entry, bodies[0]) != AsyncOwnerBinding.ReturnsStateMachineTask)
            return OwnerCompletionProof.Unresolved;
        return OwnerCompletionProof.ReviewedStandardTaskObserved;
    }

    private static bool StandardFramework(AssemblyDefinition assembly, string bundle)
    {
        CustomAttribute[] attributes = assembly.CustomAttributes.Where(attribute => attribute.AttributeType.FullName == "Xunit.TestFrameworkAttribute").ToArray();
        if (attributes.Length != 1 || !FromBundle(attributes[0].AttributeType.Resolve(), bundle, "xunit.core.dll") ||
            attributes[0].ConstructorArguments.Count != 2 || attributes[0].ConstructorArguments[0].Value is not string typeName ||
            attributes[0].ConstructorArguments[1].Value is not string assemblyName) return false;
        AssemblyDefinition frameworkAssembly = assemblyName == assembly.Name.Name ? assembly :
            assembly.MainModule.AssemblyResolver.Resolve(new AssemblyNameReference(assemblyName, new Version(0, 0)));
        TypeDefinition? type = frameworkAssembly.MainModule.GetType(typeName);
        var visited = new HashSet<string>(StringComparer.Ordinal);
        while (type is not null && visited.Add(type.FullName + "@" + type.Module.FileName))
        {
            if (type.FullName == typeof(AttributionTestFramework).FullName)
                return FromBundle(type, bundle, "Attribution.Xunit.dll");
            // CPU-only wrappers may initialize in constructors, but must not
            // replace executor/discoverer dispatch. Their startup effects still
            // need the distinct lifecycle contract before any test can be reused.
            if (type.Methods.Any(method => !method.IsConstructor) || type.BaseType is null) return false;
            type = type.BaseType.Resolve();
        }
        return false;
    }

    private static bool FromBundle(TypeDefinition? type, string bundle, string file) => type is not null &&
        string.Equals(Path.GetFullPath(type.Module.FileName), Path.GetFullPath(Path.Combine(bundle, file)),
            OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if ((entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Linked runtime contract input.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    private static IEnumerable<TypeDefinition> Types(IEnumerable<TypeDefinition> roots)
    {
        foreach (TypeDefinition type in roots)
        {
            yield return type;
            foreach (TypeDefinition nested in Types(type.NestedTypes)) yield return nested;
        }
    }
}
