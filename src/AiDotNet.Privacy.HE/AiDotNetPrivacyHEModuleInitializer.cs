using System;
using System.Runtime.CompilerServices;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Cryptography
{
    /// <summary>
    /// Registers <see cref="SealHomomorphicEncryptionProvider{T}"/> as the default homomorphic-encryption
    /// provider for federated training the moment this assembly loads.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Applications that add a package reference to <c>AiDotNet.Privacy.HE</c> get SEAL/CKKS/BFV
    /// available with no startup code — the runtime invokes <see cref="Initialize"/> exactly once,
    /// before any method of any type in this assembly runs, and we plug into
    /// <see cref="HomomorphicEncryptionOptions.DefaultProviderFactory"/>.
    /// </para>
    /// <para>
    /// Applications that do NOT reference this package and try to enable HE on the federated trainer
    /// will hit a clear <see cref="InvalidOperationException"/> from
    /// <c>InMemoryFederatedTrainer.ResolveDefaultHomomorphicEncryptionProvider</c> telling them to
    /// install <c>AiDotNet.Privacy.HE</c>. This replaces the audit-2026-05-pre hard-coded SEAL fallback
    /// that pinned <c>Microsoft.Research.SEALNet</c> as a transitive dep of every core consumer.
    /// </para>
    /// </remarks>
    internal static class AiDotNetPrivacyHEModuleInitializer
    {
        [ModuleInitializer]
        internal static void Initialize()
        {
            HomomorphicEncryptionOptions.DefaultProviderFactory ??= type =>
            {
                var providerType = typeof(SealHomomorphicEncryptionProvider<>).MakeGenericType(type);
                return Activator.CreateInstance(providerType)
                       ?? throw new InvalidOperationException(
                           $"Activator.CreateInstance returned null for {providerType.FullName}.");
            };
        }
    }
}

#if !NET5_0_OR_GREATER
namespace System.Runtime.CompilerServices
{
    // Polyfill so Roslyn (C# 9+) emits a module initializer even when targeting net471.
    // The attribute presence is what the compiler looks for; the runtime treats marked
    // methods as module-load callbacks regardless of TFM. Marked public so the AiDotNet
    // core assembly's internal polyfill (if present) does not shadow this one.
    [AttributeUsage(AttributeTargets.Method, Inherited = false)]
    public sealed class ModuleInitializerAttribute : Attribute { }
}
#endif
