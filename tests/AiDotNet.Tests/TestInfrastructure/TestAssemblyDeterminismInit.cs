using System.Runtime.CompilerServices;

#if NETFRAMEWORK
// net471 has no ModuleInitializerAttribute; provide the well-known polyfill so the
// initializer below compiles on both target frameworks. The block-scoped polyfill
// namespace forces the enclosing namespace below to also be block-scoped (CS8956:
// file-scoped namespace must precede all other members).
namespace System.Runtime.CompilerServices
{
    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    internal sealed class ModuleInitializerAttribute : Attribute
    {
    }
}
#endif

namespace AiDotNet.Tests.TestInfrastructure
{
    /// <summary>
    /// Establishes deterministic CPU-BLAS reduction order for the entire test assembly at
    /// load time.
    /// </summary>
    /// <remarks>
    /// The BLAS determinism flag (OpenBLAS thread count + DeterministicReductions) is a
    /// process-global in AiDotNet.Tensors. Production code re-asserts it on every
    /// Build/Predict, but there is a startup window — before any model has run — where the
    /// process default (multi-threaded, non-deterministic reduction order) is active. Tests
    /// that exercise BLAS during that window, or that run concurrently with a path that has
    /// not yet pinned the flag, can observe non-reproducible floating-point results under
    /// xUnit's parallel execution (e.g. a facade-vs-direct prediction parity check diverging
    /// only in the full parallel run). Pinning deterministic mode once at module load removes
    /// that window so the whole suite shares a stable, reproducible BLAS configuration.
    /// </remarks>
    internal static class TestAssemblyDeterminismInit
    {
        [ModuleInitializer]
        internal static void Init()
        {
            AiDotNet.Tensors.Engines.AiDotNetEngine.SetDeterministicMode(true);
        }
    }
}
