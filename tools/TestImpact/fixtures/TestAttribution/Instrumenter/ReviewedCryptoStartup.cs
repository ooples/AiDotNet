using System.Security.Cryptography;
using Mono.Cecil;
using Mono.Cecil.Cil;

internal enum CryptoStartupContract { Unresolved, OwnedEd25519KeyPair }
internal enum CryptoStartupRequirement { DeclaringTypeInitialization, SuccessfulInitialization, NoExceptionObservers, NoExternalProviderMutation }
internal sealed record CryptoStartupAssessment(CryptoStartupContract Contract, string PackageHash, string CryptoRuntimeHash,
    CryptoStartupRequirement[] Requirements);

// A concrete allocation/call protocol, not a general contract for SecureRandom
// subclasses, caller-supplied generators, or the BouncyCastle package as a whole.
// OS entropy consumption and private provider/precomputation caches are effects;
// this contract must never be used to claim that key generation is pure.
internal static class ReviewedCryptoStartup
{
    internal const string PackageHash = "ef92fc661e8d7ba8ec4d39a7cdfcdba41f14cb327f0d24fe79d0b5bb428899b5";
    internal const string CryptoRuntimeHash = "e0a839a540e47343b53a936b32d91599a3470005228d50fd0b896f8bbdcc65b0";

    internal static CryptoStartupAssessment Read(MethodDefinition factory)
    {
        CryptoStartupAssessment Unknown() => new(CryptoStartupContract.Unresolved, "", "", []);
        try
        {
            if (!ReadShape(factory) || factory.Body.Instructions[0].Operand is not MethodReference allocation ||
                allocation.Resolve() is not MethodDefinition constructor) return Unknown();
            string package = constructor.Module.FileName;
            string crypto = typeof(RandomNumberGenerator).Assembly.Location;
            string runtime = typeof(object).Assembly.Location;
            if (Hash(package) != PackageHash || Hash(crypto) != CryptoRuntimeHash || Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash)
                return Unknown();
            using var pinned = AssemblyDefinition.ReadAssembly(package,
                new ReaderParameters { AssemblyResolver = constructor.Module.AssemblyResolver });
            if (pinned.Modules.Count != 1 || pinned.MainModule.Mvid != constructor.Module.Mvid ||
                pinned.MainModule.LookupToken(constructor.MetadataToken) is not MethodDefinition original || original.FullName != constructor.FullName)
                return Unknown();
            // Bind the runtime reached by the package's resolver, not merely
            // an unrelated trusted copy already loaded into this verifier.
            MethodReference entropy = pinned.MainModule.GetType("Org.BouncyCastle.Security.SecureRandom").Methods
                .Single(method => method.Name == "AutoSeed").Body.Instructions.Select(instruction => instruction.Operand)
                .OfType<MethodReference>().Single(call => call.DeclaringType.FullName == "System.Security.Cryptography.RandomNumberGenerator");
            if (entropy.Resolve() is not MethodDefinition fill || !SamePath(fill.Module.FileName, crypto) ||
                pinned.MainModule.TypeSystem.Object.Resolve() is not TypeDefinition root || !SamePath(root.Module.FileName, runtime)) return Unknown();
            // The fresh exact SecureRandom uses SHA256 DigestRandomGenerator.
            // Its static VMPC generator and its instance generator seed via
            // synchronous RandomNumberGenerator.Fill, not ThreadedSeedGenerator.
            // KeyGenerationParameters retains the supplied non-null random.
            // GeneratePublicKey's one Func is the pinned static CreatePublicKey,
            // invoked synchronously by EnsureSingletonInitialized; it cannot be
            // replaced by a caller callback. Ed25519 precomputation is lock-bound
            // arithmetic over private tables. No timer/thread/task registration
            // occurs on these concrete normal paths. Exceptional observers and
            // outside mutations of package caches remain separate obligations.
            if (Hash(package) != PackageHash || Hash(crypto) != CryptoRuntimeHash || Hash(runtime) != ReviewedOwnerCompletion.RuntimeHash)
                return Unknown();
            return new(CryptoStartupContract.OwnedEd25519KeyPair, PackageHash, CryptoRuntimeHash,
                Enum.GetValues<CryptoStartupRequirement>());
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return Unknown(); }
    }

    internal static bool ReadShape(MethodDefinition factory)
    {
        try
        {
            if (!factory.IsStatic || factory.HasThis || factory.ExplicitThis || factory.HasGenericParameters || factory.Parameters.Count != 0 ||
                factory.CallingConvention != MethodCallingConvention.Default || !factory.HasBody || factory.Body.HasExceptionHandlers ||
                factory.Body.Variables.Count != 1 || factory.ImplAttributes != MethodImplAttributes.IL || factory.IsPInvokeImpl ||
                factory.HasSecurityDeclarations || factory.DeclaringType.HasGenericParameters) return false;
            var il = factory.Body.Instructions;
            if (!il.Select(instruction => instruction.OpCode.Code).SequenceEqual(new[] { Code.Newobj, Code.Stloc_0, Code.Ldloc_0,
                Code.Newobj, Code.Newobj, Code.Callvirt, Code.Ldloc_0, Code.Callvirt, Code.Ret })) return false;
            string[] signatures =
            [
                "System.Void Org.BouncyCastle.Crypto.Generators.Ed25519KeyPairGenerator::.ctor()",
                "System.Void Org.BouncyCastle.Security.SecureRandom::.ctor()",
                "System.Void Org.BouncyCastle.Crypto.Parameters.Ed25519KeyGenerationParameters::.ctor(Org.BouncyCastle.Security.SecureRandom)",
                "System.Void Org.BouncyCastle.Crypto.Generators.Ed25519KeyPairGenerator::Init(Org.BouncyCastle.Crypto.KeyGenerationParameters)",
                "Org.BouncyCastle.Crypto.AsymmetricCipherKeyPair Org.BouncyCastle.Crypto.Generators.Ed25519KeyPairGenerator::GenerateKeyPair()"
            ];
            int[] indices = [0, 3, 4, 5, 7];
            ModuleDefinition? package = null;
            for (int index = 0; index < indices.Length; index++)
            {
                if (il[indices[index]].Operand is not MethodReference call || call.FullName != signatures[index] ||
                    !call.HasThis || call.ExplicitThis || call.HasGenericParameters || call.CallingConvention != MethodCallingConvention.Default ||
                    call.DeclaringType is TypeSpecification || call.Resolve() is not MethodDefinition target || target.IsStatic ||
                    target.FullName != call.FullName || target.HasGenericParameters) return false;
                package ??= target.Module;
                if (target.Module != package || call.DeclaringType.Resolve() != target.DeclaringType ||
                    !OwnedFieldBinding.SameType(call.ReturnType, target.ReturnType) || call.Parameters.Count != target.Parameters.Count ||
                    call.Parameters.Where((parameter, position) => !OwnedFieldBinding.SameType(parameter.ParameterType, target.Parameters[position].ParameterType)).Any()) return false;
            }
            return il[0].Operand is MethodReference generator &&
                OwnedFieldBinding.SameType(factory.Body.Variables[0].VariableType, generator.DeclaringType) &&
                il[7].Operand is MethodReference generate && OwnedFieldBinding.SameType(factory.ReturnType, generate.ReturnType);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return false; }
    }

    private static string Hash(string path)
    {
        for (FileSystemInfo? entry = new FileInfo(Path.GetFullPath(path)); entry is not null;
            entry = entry is FileInfo file ? file.Directory : ((DirectoryInfo)entry).Parent)
            if (!entry.Exists || (entry.Attributes & FileAttributes.ReparsePoint) != 0 || entry.LinkTarget is not null)
                throw new InvalidDataException("Missing or linked crypto-contract input.");
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    private static bool SamePath(string first, string second) => string.Equals(Path.GetFullPath(first), Path.GetFullPath(second),
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
