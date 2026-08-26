using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Shared value-domain contract for tensors whose elements are probabilities or
/// other normalized continuous values in the closed unit interval.
/// </summary>
internal static class UnitIntervalTensorDomain
{
    private const string ProviderKey = "AiDotNet.NeuralNetworks.UnitInterval";

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
                if (!double.IsNaN(value) && !double.IsInfinity(value)
                    && value >= 0.0 && value <= 1.0)
                    continue;

                throw new InputContractViolationException(
                    $"{ownerName}.{portName} requires continuous values in [0, 1], "
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
                double value = Math.Min(Math.Max(operations.ToDouble(input[i]) + epsilon, 0.0), 1.0);
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
