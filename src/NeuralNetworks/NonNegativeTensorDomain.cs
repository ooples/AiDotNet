using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// The domain of finite values in <c>[0, +inf)</c>, for a public output that is non-negative by
/// construction rather than by convention.
/// </summary>
/// <remarks>
/// <para>
/// A variance and a standard deviation are the motivating cases: Bollerslev 1986 defines the GARCH
/// conditional variance as a strictly positive quantity, and every volatility head in this library
/// ends in an activation whose range is non-negative. Reporting such an output as
/// <see cref="LayerInputDomain.Continuous"/> is not a harmless over-approximation. A generic caller
/// that believes the output is continuous builds a target containing negative values, the model
/// cannot reach it at any parameter setting, and training converges on the boundary - every input
/// mapping to the same saturated zero. The failure then reads as a collapsed network rather than as
/// an unreachable objective.
/// </para>
/// <para>
/// Mirrors <c>UnitIntervalTensorDomain</c>: the provider is registered on first use so a model can
/// name the domain without a consumer having to register anything.
/// </para>
/// </remarks>
internal static class NonNegativeTensorDomain
{
    private const string ProviderKey = "AiDotNet.NeuralNetworks.NonNegative";

    private static readonly Lazy<IDisposable> Registration =
        new(() => InputDomainProviderRegistry.Register(new Provider()));

    public static LayerInputDomain Value
    {
        get
        {
            _ = Registration.Value;
            return LayerInputDomain.Custom(ProviderKey);
        }
    }

    private sealed class Provider : IInputDomainProvider
    {
        public string Key => ProviderKey;

        public LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer) =>
            producer.Kind == LayerInputDomainKind.Custom && producer.Detail == ProviderKey
                ? LayerInputDomainCompatibility.Compatible
                : LayerInputDomainCompatibility.Incompatible;

        public void Validate<T>(Tensor<T> input, string ownerName, string portName)
        {
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = operations.ToDouble(input[i]);
                if (!double.IsNaN(value) && !double.IsInfinity(value) && value >= 0.0)
                    continue;

                throw new InputContractViolationException(
                    $"{ownerName}.{portName} requires finite non-negative values, "
                    + $"but element {i} is {value}.",
                    portName);
            }
        }

        public Tensor<T> CreateValid<T>(int[] shape, Random random)
        {
            var tensor = new Tensor<T>(shape);
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = operations.FromDouble(random.NextDouble());
            return tensor;
        }

        public Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon)
        {
            var nearby = new Tensor<T>(input.Shape.ToArray());
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = Math.Max(operations.ToDouble(input[i]) + epsilon, 0.0);
                nearby[i] = operations.FromDouble(value);
            }
            return nearby;
        }

        public Tensor<T> CreateInvalid<T>(int[] shape)
        {
            var tensor = new Tensor<T>(shape);
            var invalid = MathHelper.GetNumericOperations<T>().FromDouble(-1.0);
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = invalid;
            return tensor;
        }
    }
}