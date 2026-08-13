using System.Collections.Concurrent;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Supplies the complete executable behavior for a user-defined tensor value domain. Requiring all
/// four operations prevents a custom validator from becoming a test-fixture blind spot.
/// </summary>
public interface IInputDomainProvider
{
    /// <summary>Stable key referenced by <c>CustomProviderKey</c> declarations.</summary>
    string Key { get; }

    /// <summary>Proves whether this consumer accepts a producer domain.</summary>
    LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer);

    /// <summary>Validates a tensor, throwing <see cref="InputContractViolationException"/> on failure.</summary>
    void Validate<T>(Tensor<T> input, string ownerName, string portName);

    /// <summary>Creates a valid tensor for generated fixtures and tooling.</summary>
    Tensor<T> CreateValid<T>(int[] shape, Random random);

    /// <summary>Creates a nearby valid tensor without leaving the custom domain.</summary>
    Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon);

    /// <summary>Creates an invalid tensor used to prove the validator's negative path.</summary>
    Tensor<T> CreateInvalid<T>(int[] shape);
}

/// <summary>
/// Process-wide registry for explicit custom input-domain providers. Registration rejects duplicate
/// keys and returns a scope so parallel tools can cleanly restore global state.
/// </summary>
public static class InputDomainProviderRegistry
{
    private static readonly ConcurrentDictionary<string, IInputDomainProvider> Providers =
        new(StringComparer.Ordinal);

    public static IDisposable Register(IInputDomainProvider provider)
    {
        if (provider is null) throw new ArgumentNullException(nameof(provider));
        if (string.IsNullOrWhiteSpace(provider.Key))
            throw new ArgumentException("A custom input-domain provider must have a stable key.", nameof(provider));
        if (!Providers.TryAdd(provider.Key, provider))
            throw new InvalidOperationException(
                $"A custom input-domain provider named '{provider.Key}' is already registered.");
        return new Registration(provider);
    }

    public static bool TryResolve(string? key, out IInputDomainProvider provider)
    {
        if (!string.IsNullOrWhiteSpace(key) && Providers.TryGetValue(key!, out var resolved))
        {
            provider = resolved;
            return true;
        }

        provider = null!;
        return false;
    }

    internal static IInputDomainProvider Require(string? key)
    {
        if (TryResolve(key, out var provider)) return provider;
        throw new InputContractBindingException(
            $"Custom input domain '{key ?? "<missing>"}' has no registered provider. Register an "
            + $"{nameof(IInputDomainProvider)} before binding or executing the component.");
    }

    private sealed class Registration : IDisposable
    {
        private readonly IInputDomainProvider _provider;
        private bool _disposed;

        public Registration(IInputDomainProvider provider) => _provider = provider;

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (Providers.TryGetValue(_provider.Key, out var current)
                && ReferenceEquals(current, _provider))
                Providers.TryRemove(_provider.Key, out _);
        }
    }
}
