using AiDotNet.TestImpact;
using Mono.Cecil;

internal enum CpuStartupContract { Unresolved, PlainCpuResetWithoutCallbacks }

// This contract includes the two declaring-type initializers and the pinned
// GPU module opt-out. It requires exclusive startup: the entry sample alone
// cannot exclude a concurrent engine replacement or environment mutation.
internal static class ReviewedCpuStartup
{
    internal static CpuStartupContract Read(MethodReference reset, MethodReference parallelSetter, RuntimeContractProfile? profile)
    {
        try
        {
            if (!RuntimeProfileEvidence.HasObservedSuccessfulCpuReset(profile) ||
                !Signature(reset, "System.Void AiDotNet.Tensors.Engines.AiDotNetEngine::ResetToCpu()", 0) ||
                !Signature(parallelSetter, "System.Void AiDotNet.Tensors.Helpers.CpuParallelSettings::set_MaxDegreeOfParallelism(System.Int32)", 1) ||
                reset.Resolve() is not MethodDefinition resetMethod || parallelSetter.Resolve() is not MethodDefinition setter ||
                resetMethod.Module != setter.Module || !SignedLicenseReader.Type(reset.ReturnType, typeof(void)) ||
                !SignedLicenseReader.Type(parallelSetter.ReturnType, typeof(void)) ||
                !SignedLicenseReader.Type(parallelSetter.Parameters[0].ParameterType, typeof(int)) ||
                ReviewedGpuStartup.Read(resetMethod.Module.Assembly, profile) != GpuStartupContract.DisabledWithoutDiagnosticCallbacks)
                return CpuStartupContract.Unresolved;
            using var pinned = AssemblyDefinition.ReadAssembly(resetMethod.Module.FileName,
                new ReaderParameters { AssemblyResolver = resetMethod.Module.AssemblyResolver });
            if (pinned.MainModule.LookupToken(resetMethod.MetadataToken) is not MethodDefinition original || original.FullName != reset.FullName ||
                pinned.MainModule.LookupToken(setter.MetadataToken) is not MethodDefinition originalSetter || originalSetter.FullName != parallelSetter.FullName ||
                !SignedLicenseReader.Type(pinned.MainModule.TypeSystem.Object, typeof(object))) return CpuStartupContract.Unresolved;
            // Reviewed exact package path: engine cctor allocates only CpuEngine;
            // that constructor calls Object. Reset replaces Current via Volatile
            // and cannot dispose a DirectGpuTensorEngine when entry is exact CPU.
            // Nonempty QUIET returns before Logger.Invoke and Console.WriteLine.
            // CpuParallelSettings cctor reads ProcessorCount/COOP_POOL and stores
            // scalar fields; it does not invoke physical-core probing, parallel
            // execution, native BLAS initialization, or the persistent pool.
            return ReviewedGpuStartup.Read(resetMethod.Module.Assembly, profile) == GpuStartupContract.DisabledWithoutDiagnosticCallbacks
                ? CpuStartupContract.PlainCpuResetWithoutCallbacks : CpuStartupContract.Unresolved;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or ArgumentException or BadImageFormatException or
            AssemblyResolutionException or ResolutionException or InvalidOperationException)
        { return CpuStartupContract.Unresolved; }
    }

    private static bool Signature(MethodReference method, string signature, int parameters) => method is not MethodSpecification &&
        method.FullName == signature && !method.HasThis && !method.ExplicitThis && !method.HasGenericParameters &&
        method.Parameters.Count == parameters && method.CallingConvention == MethodCallingConvention.Default &&
        method.DeclaringType is not TypeSpecification;
}
