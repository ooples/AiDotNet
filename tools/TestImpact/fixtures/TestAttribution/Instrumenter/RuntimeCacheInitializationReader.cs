using System.Collections.Concurrent;
using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum RuntimeCacheInitializationContract { Unresolved, PrivateTypeAccelerationCache }
internal sealed record RuntimeCacheInitializationAssessment(RuntimeCacheInitializationContract Contract,
    string Field, string RuntimeHash, string CollectionsHash);

// One reviewed normal-return allocation, not a general dictionary-purity rule.
// The pinned parameterless constructor allocates private buckets/locks and uses
// the runtime's default Type comparer; it neither enumerates values nor invokes
// a user comparer. Resource failures/observers and later cache access remain
// separate obligations. In particular this does not prove a package module init.
internal static class RuntimeCacheInitializationReader
{
    internal const string CollectionsHash = "082f4bc0da1141eb65111b6da633d75624adfb7bb067d7ee90c5622c8b30885b";

    internal static RuntimeCacheInitializationAssessment Read(TypeDefinition type)
    {
        RuntimeCacheInitializationAssessment Unknown() => new(RuntimeCacheInitializationContract.Unresolved, "", "", "");
        try
        {
            string runtime = typeof(object).Assembly.Location;
            string collections = typeof(ConcurrentDictionary<,>).Assembly.Location;
            if (Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash || Hash(collections) != CollectionsHash ||
                type.HasGenericParameters || type.HasSecurityDeclarations || !type.IsAbstract || !type.IsSealed ||
                type.BaseType is null || !RuntimeType(type.BaseType, "System.Object", runtime)) return Unknown();
            MethodDefinition[] initializers = type.Methods.Where(method => method.IsConstructor && method.IsStatic).ToArray();
            if (initializers.Length != 1) return Unknown();
            MethodDefinition initializer = initializers[0];
            if (!initializer.HasBody || initializer.ImplAttributes != MethodImplAttributes.IL || initializer.HasThis ||
                initializer.IsPInvokeImpl || initializer.HasSecurityDeclarations || initializer.HasOverrides ||
                initializer.HasGenericParameters || initializer.Parameters.Count != 0 ||
                !RuntimeType(initializer.ReturnType, "System.Void", runtime) ||
                initializer.Body.HasVariables || initializer.Body.HasExceptionHandlers ||
                !initializer.Body.Instructions.Select(instruction => instruction.OpCode.Code)
                    .SequenceEqual(new[] { Code.Newobj, Code.Stsfld, Code.Ret })) return Unknown();
            var instructions = initializer.Body.Instructions;
            if (instructions[0].Operand is not MethodReference constructor || constructor is GenericInstanceMethod || !constructor.HasThis || constructor.ExplicitThis ||
                constructor.HasGenericParameters || constructor.CallingConvention != MethodCallingConvention.Default ||
                constructor.Name != ".ctor" || constructor.Parameters.Count != 0 || !RuntimeType(constructor.ReturnType, "System.Void", runtime) ||
                constructor.DeclaringType is not GenericInstanceType cache || !CacheType(cache, runtime, collections) ||
                instructions[1].Operand is not FieldReference target || target.Resolve() is not FieldDefinition field ||
                field.DeclaringType != type || !field.IsPrivate || !field.IsStatic || !field.IsInitOnly ||
                field.CustomAttributes.Any(attribute => attribute.AttributeType.FullName is "System.ThreadStaticAttribute" or "System.ContextStaticAttribute") ||
                field.FieldType is not GenericInstanceType fieldType || !CacheType(fieldType, runtime, collections) ||
                constructor.Resolve() is not MethodDefinition resolved || !SameFile(resolved.Module.FileName, collections)) return Unknown();
            // Do not trust a mutable Cecil definition's claimed file path.
            using var pinned = AssemblyDefinition.ReadAssembly(collections);
            if (pinned.MainModule.LookupToken(resolved.MetadataToken) is not MethodDefinition original ||
                original.FullName != resolved.FullName || original.DeclaringType.FullName != "System.Collections.Concurrent.ConcurrentDictionary`2" ||
                !original.IsConstructor || original.IsStatic || original.HasGenericParameters || original.Parameters.Count != 0 ||
                original.DeclaringType.Methods.Any(method => method.IsConstructor && method.IsStatic) ||
                pinned.MainModule.Types.Where(candidate => candidate.Name == "<Module>")
                    .SelectMany(candidate => candidate.Methods).Any(method => method.IsConstructor)) return Unknown();
            if (Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash || Hash(collections) != CollectionsHash) return Unknown();
            return new(RuntimeCacheInitializationContract.PrivateTypeAccelerationCache,
                field.Module.Assembly.Name.Name + ":" + field.FullName, ReviewedOwnerCompletion.RuntimeHash, CollectionsHash);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    private static bool CacheType(GenericInstanceType cache, string runtime, string collections) =>
        cache.ElementType.FullName == "System.Collections.Concurrent.ConcurrentDictionary`2" &&
        cache.ElementType.Resolve() is TypeDefinition definition && SameFile(definition.Module.FileName, collections) &&
        cache.GenericArguments.Count == 2 && RuntimeType(cache.GenericArguments[0], "System.Type", runtime) &&
        cache.GenericArguments[1] is GenericInstanceType tuple && tuple.GenericArguments.Count == 2 &&
        RuntimeType(tuple.ElementType, "System.ValueTuple`2", runtime) &&
        tuple.GenericArguments.All(argument => RuntimeType(argument, "System.Boolean", runtime));

    private static bool RuntimeType(TypeReference type, string name, string runtime) => type is not TypeSpecification &&
        type.FullName == name && type.Resolve() is TypeDefinition definition && definition.FullName == name && SameFile(definition.Module.FileName, runtime);

    private static bool SameFile(string first, string second) => string.Equals(Path.GetFullPath(first), Path.GetFullPath(second),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists || (entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Missing or linked cache-contract input.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
