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
            var engine = AiDotNetEngine.Current;

            // Non-zero where the element is NaN or lies outside [0, 1], 0 where it is valid
            var outOfRange = engine.TensorWhere(
                engine.TensorIsNan(input),
                engine.TensorOnesLike(input),
                engine.TensorAdd(
                    engine.TensorLessThan(input, operations.Zero), engine.TensorGreaterThan(input, operations.One)));
            if (engine.TensorCountNonzero(outOfRange) == 0)
                return;

            int i = TieBreakArgMax.FirstMaxIndex(outOfRange);
            double value = operations.ToDouble(input.GetFlat(i));
            throw new InputContractViolationException(
                $"{ownerName}.{portName} requires continuous values in [0, 1], "
                + $"but element {i} is {value}.",
                portName);
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
            var operations = MathHelper.GetNumericOperations<T>();
            var engine = AiDotNetEngine.Current;

            // min(max(x + epsilon, 0), 1), with NaN passed through
            var shifted = engine.TensorAddScalar(input, operations.FromDouble(epsilon));
            var clamped = engine.TensorClamp(shifted, operations.Zero, operations.One);
            return engine.TensorWhere(engine.TensorIsNan(shifted), shifted, clamped);
        }

        public Tensor<T> CreateInvalid<T>(int[] shape)
        {
            var invalid = MathHelper.GetNumericOperations<T>().FromDouble(-1.0);
            return Tensor<T>.CreateDefault(shape, invalid);
        }
    }
}
