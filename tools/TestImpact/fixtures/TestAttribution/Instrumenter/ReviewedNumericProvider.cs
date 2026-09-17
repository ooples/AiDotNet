using System.Security.Cryptography;
using Mono.Cecil;

internal enum NumericProviderContract { Unresolved, TensorsDoubleCache }
internal enum NumericProviderRequirement { SharedInitialization, SuccessfulOwner, NoExternalCacheMutation }
internal sealed record NumericProviderAssessment(NumericProviderContract Contract, string Method,
    string PackageHash, string RuntimeHash, NumericProviderRequirement[] Requirements);

// Review of one concrete successful cache path, not of MathHelper as a whole.
// In these bytes CreateNumericOperations<double> takes its first type branch,
// allocates DoubleOperations (object constructor only), and caches the result.
// DoubleOperations.FromDouble returns its scalar argument without conversion.
// MathHelper's own dictionary initializer is deliberately NOT discharged here.
internal static class ReviewedNumericProvider
{
    internal const string PackageHash = "eb681ae60f23b03cf08e0bf3ab70a372673927acd87a428c74536d424846d5e7";
    private const string Helper = "AiDotNet.Tensors.Helpers.MathHelper";

    internal static NumericProviderAssessment Read(MethodReference call, MethodReference? context = null)
    {
        NumericProviderAssessment Unknown() => new(NumericProviderContract.Unresolved, call.FullName, "", "", []);
        if (call is not GenericInstanceMethod instance || instance.GenericArguments.Count != 1 ||
            !HasProviderSignature(instance.ElementMethod)) return Unknown();
        try
        {
            TypeReference? argument = ConcreteGenericBinding.Read(instance.GenericArguments[0], context);
            TypeDefinition? scalar = argument?.Resolve();
            MethodDefinition? definition = instance.ElementMethod.Resolve();
            string runtime = typeof(double).Assembly.Location;
            if (scalar is null || scalar.FullName != "System.Double" ||
                definition is null || definition.DeclaringType.FullName != Helper ||
                definition.Name != call.Name || !definition.IsStatic || definition.GenericParameters.Count != 1 ||
                definition.Parameters.Count != 0 || definition.Module.Assembly.Modules.Count != 1 ||
                Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash ||
                !SamePath(scalar.Module.FileName, runtime) ||
                Hash(definition.Module.FileName) != PackageHash) return Unknown();
            // Re-read the pinned file. A mutable/synthetic Cecil definition must
            // not obtain a contract merely by pointing at a trusted filename.
            using var pinned = AssemblyDefinition.ReadAssembly(definition.Module.FileName);
            MethodDefinition? bound = pinned.MainModule.LookupToken(definition.MetadataToken) as MethodDefinition;
            if (bound is null || bound.FullName != definition.FullName ||
                !HasProviderSignature(bound) || instance.ElementMethod.ReturnType.Resolve() is not TypeDefinition operations ||
                operations.Module.FileName != definition.Module.FileName ||
                operations.FullName != "AiDotNet.Tensors.Interfaces.INumericOperations`1")
                return Unknown();
            if (Hash(definition.Module.FileName) != PackageHash || Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash)
                return Unknown();
            return new(NumericProviderContract.TensorsDoubleCache, call.FullName, PackageHash,
                ReviewedOwnerCompletion.RuntimeHash,
                [NumericProviderRequirement.SharedInitialization, NumericProviderRequirement.SuccessfulOwner,
                 NumericProviderRequirement.NoExternalCacheMutation]);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or
            BadImageFormatException or AssemblyResolutionException or ResolutionException or InvalidOperationException)
        {
            return Unknown();
        }
    }

    // Cecil prints an imported method parameter as !!0 and a definition's as T.
    // Compare its owner/kind/position, not display names, without accepting a
    // same-named type parameter or a different closed result type.
    internal static bool HasProviderSignature(MethodReference method) =>
        method.Name == "GetNumericOperations" && method.DeclaringType.FullName == Helper &&
        method.CallingConvention is MethodCallingConvention.Default or MethodCallingConvention.Generic &&
        !method.HasThis && !method.ExplicitThis && method.GenericParameters.Count == 1 && method.Parameters.Count == 0 &&
        method.ReturnType is GenericInstanceType result && result.ElementType.FullName == "AiDotNet.Tensors.Interfaces.INumericOperations`1" &&
        result.GenericArguments.Count == 1 && result.GenericArguments[0] is GenericParameter parameter &&
        parameter.Type == GenericParameterType.Method && parameter.Position == 0 && ReferenceEquals(parameter.Owner, method);

    private static bool SamePath(string left, string right) => string.Equals(Path.GetFullPath(left), Path.GetFullPath(right),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
             entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists) throw new FileNotFoundException("Missing numeric contract input.", entry.FullName);
            else if ((entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Linked numeric contract input.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }
}
