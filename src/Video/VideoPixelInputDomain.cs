using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Video;

/// <summary>
/// Shared value-domain contract for public video-frame tensors represented on the
/// conventional unsigned 8-bit intensity scale.
/// </summary>
internal static class VideoPixelInputDomain
{
    private const string RawProviderKey = "AiDotNet.Video.RawPixel0To255";
    private const string NormalizedProviderKey = "AiDotNet.Video.NormalizedPixel0To1";

    // The provider registry is process-wide while generic video bases have one static
    // constructor per closed T. Keep registration in this non-generic holder so float,
    // double, and other numeric models cannot race to register the same domain twice.
    private static readonly Lazy<IDisposable> RawRegistration =
        new(() => InputDomainProviderRegistry.Register(new RawPixelProvider()));

    private static readonly Lazy<IDisposable> NormalizedRegistration =
        new(() => InputDomainProviderRegistry.Register(new NormalizedPixelProvider()));

    public static LayerInputDomain Value
    {
        get
        {
            _ = RawRegistration.Value;
            return LayerInputDomain.Custom(RawProviderKey);
        }
    }

    public static LayerInputDomain NormalizedValue
    {
        get
        {
            _ = NormalizedRegistration.Value;
            return LayerInputDomain.Custom(NormalizedProviderKey);
        }
    }

    private sealed class RawPixelProvider : IInputDomainProvider
    {
        public string Key => RawProviderKey;

        public LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer) =>
            producer.Kind == LayerInputDomainKind.Custom && producer.Detail == RawProviderKey
                ? LayerInputDomainCompatibility.Compatible
                : LayerInputDomainCompatibility.Incompatible;

        public void Validate<T>(Tensor<T> input, string ownerName, string portName)
        {
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = operations.ToDouble(input[i]);
                if (double.IsFinite(value) && value >= 0.0 && value <= 255.0)
                    continue;

                throw new InputContractViolationException(
                    $"{ownerName}.{portName} requires raw pixel values in [0, 255], "
                    + $"but element {i} is {value}.",
                    portName);
            }
        }

        public Tensor<T> CreateValid<T>(int[] shape, Random random)
        {
            var tensor = new Tensor<T>(shape);
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = operations.FromDouble(random.NextDouble() * 255.0);
            return tensor;
        }

        public Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon)
        {
            var nearby = new Tensor<T>(input.Shape.ToArray());
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = Math.Clamp(operations.ToDouble(input[i]) + epsilon, 0.0, 255.0);
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

    private sealed class NormalizedPixelProvider : IInputDomainProvider
    {
        public string Key => NormalizedProviderKey;

        public LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer) =>
            producer.Kind == LayerInputDomainKind.Custom && producer.Detail == NormalizedProviderKey
                ? LayerInputDomainCompatibility.Compatible
                : LayerInputDomainCompatibility.Incompatible;

        public void Validate<T>(Tensor<T> input, string ownerName, string portName)
        {
            var operations = MathHelper.GetNumericOperations<T>();
            for (int i = 0; i < input.Length; i++)
            {
                double value = operations.ToDouble(input[i]);
                if (double.IsFinite(value) && value >= 0.0 && value <= 1.0)
                    continue;

                throw new InputContractViolationException(
                    $"{ownerName}.{portName} requires normalized pixel values in [0, 1], "
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
                double value = Math.Clamp(operations.ToDouble(input[i]) + epsilon, 0.0, 1.0);
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
