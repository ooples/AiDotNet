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
            int bad = PixelRange.FirstOutOfRange(input, 255.0);
            if (bad < 0)
                return;

            double value = MathHelper.GetNumericOperations<T>().ToDouble(input[bad]);
            throw new InputContractViolationException(
                $"{ownerName}.{portName} requires raw pixel values in [0, 255], "
                + $"but element {bad} is {value}.",
                portName);
        }

        public Tensor<T> CreateValid<T>(int[] shape, Random random)
            => PixelRange.Uniform<T>(shape, random, 255.0);

        public Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon)
            => PixelRange.Shifted(input, epsilon, 255.0);

        public Tensor<T> CreateInvalid<T>(int[] shape)
            => Tensor<T>.CreateDefault(shape, MathHelper.GetNumericOperations<T>().FromDouble(-1.0));
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
            int bad = PixelRange.FirstOutOfRange(input, 1.0);
            if (bad < 0)
                return;

            double value = MathHelper.GetNumericOperations<T>().ToDouble(input[bad]);
            throw new InputContractViolationException(
                $"{ownerName}.{portName} requires normalized pixel values in [0, 1], "
                + $"but element {bad} is {value}.",
                portName);
        }

        public Tensor<T> CreateValid<T>(int[] shape, Random random)
            => PixelRange.Uniform<T>(shape, random, 1.0);

        public Tensor<T> CreateNearby<T>(Tensor<T> input, double epsilon)
            => PixelRange.Shifted(input, epsilon, 1.0);

        public Tensor<T> CreateInvalid<T>(int[] shape)
            => Tensor<T>.CreateDefault(shape, MathHelper.GetNumericOperations<T>().FromDouble(-1.0));
    }

    /// <summary>Element-wise pixel range checks and generators shared by the two providers.</summary>
    private static class PixelRange
    {
        /// <summary>Index of the first element that is not a finite value in [0, <paramref name="high"/>], or -1.</summary>
        public static int FirstOutOfRange<T>(Tensor<T> input, double high)
        {
            var engine = AiDotNetEngine.Current;
            var operations = MathHelper.GetNumericOperations<T>();
            var flat = engine.Reshape(input, new[] { input.Length });
            var outside = engine.TensorAdd(
                engine.TensorLessThan(flat, operations.Zero), engine.TensorGreaterThan(flat, operations.FromDouble(high)));
            var bad = engine.TensorWhere(engine.TensorIsFinite(flat), outside, engine.TensorOnesLike(flat));
            var coordinates = engine.TensorNonzero(bad);
            return coordinates.Length == 0 ? -1 : coordinates[0, 0];
        }

        /// <summary>One uniform draw per element, in element order, scaled to [0, <paramref name="high"/>).</summary>
        public static Tensor<T> Uniform<T>(int[] shape, Random random, double high)
        {
            int length = 1;
            foreach (int extent in shape) length *= extent;
            var draws = new double[length];
            for (int i = 0; i < length; i++)
                draws[i] = random.NextDouble() * high;
            return new Tensor<double>(draws, shape).Cast<T>();
        }

        /// <summary>The input shifted by <paramref name="epsilon"/> and clamped back into [0, <paramref name="high"/>].</summary>
        public static Tensor<T> Shifted<T>(Tensor<T> input, double epsilon, double high)
        {
            var engine = AiDotNetEngine.Current;
            var operations = MathHelper.GetNumericOperations<T>();
            return engine.TensorClamp(
                engine.TensorAddScalar(input, operations.FromDouble(epsilon)), operations.Zero, operations.FromDouble(high));
        }
    }
}
