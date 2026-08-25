using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Every layer that overrides <c>Deserialize</c> and rebuilds its weight tensors must re-register
/// them as trainable parameters.
/// </summary>
/// <remarks>
/// <para>
/// <c>Deserialize</c> replaces the tensor field references outright. The trainable-parameter registry
/// holds the OLD references, so unless the restored tensors are registered, one of two things
/// happens. On a lazily constructed layer the registry was never populated at all, so
/// <c>GetParameters()</c> reports zero and every restored weight is silently discarded. On a layer
/// that had already run a forward pass, the registry still points at the pre-deserialize tensors, so
/// optimizers and tape training update dead references while <c>Forward</c> reads the new ones — the
/// model appears to train and never changes.
/// </para>
/// <para>
/// Conv3DLayer had exactly this defect and its generated model-family test caught it
/// (<c>Serialize_Deserialize_ShouldPreserveBehavior</c>, expected 56 parameters and got 0).
/// ConvolutionalLayer does it correctly and carries a comment explaining why. The four layers below
/// have no generated coverage, which is why they went unnoticed, so they are pinned here.
/// </para>
/// </remarks>
public class LayerDeserializeReregistersParametersTests
{
    private static void AssertRoundTripPreservesParameters<TLayer>(
        Func<TLayer> create, Action<TLayer> configure, Tensor<double> input, string what)
        where TLayer : ILayer<double>
    {
        var layer = create();
        configure(layer);
        layer.SetTrainingMode(false);
        layer.Forward(input);

        var original = layer.GetParameters();
        Assert.True(original.Length > 0, $"{what}: the layer reported no parameters before serializing");

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            layer.Serialize(writer);
        }

        var restoredLayer = create();
        configure(restoredLayer);
        stream.Position = 0;
        using (var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            restoredLayer.Deserialize(reader);
        }

        var restored = restoredLayer.GetParameters();

        Assert.True(original.Length == restored.Length,
            $"{what}: {original.Length} parameters before the round trip, {restored.Length} after. " +
            "Deserialize rebuilt the tensors without re-registering them as trainable parameters.");

        for (int i = 0; i < original.Length; i++)
        {
            Assert.True(original[i] == restored[i],
                $"{what}: parameter {i} changed across the round trip: {original[i]} then {restored[i]}");
        }
    }

    /// <summary>Each element linked to its neighbours on a ring, which is a valid mesh topology.</summary>
    private static int[,] RingAdjacency(int elements, int neighbours)
    {
        var adjacency = new int[elements, neighbours];

        for (int i = 0; i < elements; i++)
        {
            for (int j = 0; j < neighbours; j++)
            {
                adjacency[i, j] = (i + j + 1) % elements;
            }
        }

        return adjacency;
    }

    private static Tensor<double> Ramp(params int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        var span = tensor.AsWritableSpan();

        for (int i = 0; i < span.Length; i++)
        {
            span[i] = 0.05 * ((i % 17) - 8);
        }

        return tensor;
    }

    [Fact]
    public void MeshEdgeConvLayer_RoundTripsItsParameters()
    {
        AssertRoundTripPreservesParameters(
            () => new MeshEdgeConvLayer<double>(inputChannels: 3, outputChannels: 4, numNeighbors: 4,
                activationFunction: (IActivationFunction<double>?)null),
            layer => layer.SetEdgeAdjacency(RingAdjacency(6, 4)),
            Ramp(6, 3),
            nameof(MeshEdgeConvLayer<double>));
    }

    [Fact]
    public void MeshPoolLayer_RoundTripsItsParameters()
    {
        AssertRoundTripPreservesParameters(
            () => new MeshPoolLayer<double>(inputChannels: 3, targetEdges: 4, numNeighbors: 4),
            layer => layer.SetEdgeAdjacency(RingAdjacency(8, 4)),
            Ramp(8, 3),
            nameof(MeshPoolLayer<double>));
    }

    [Fact]
    public void SpiralConvLayer_RoundTripsItsParameters()
    {
        AssertRoundTripPreservesParameters(
            () => new SpiralConvLayer<double>(inputChannels: 3, outputChannels: 4, spiralLength: 5,
                activationFunction: (IActivationFunction<double>?)null),
            layer => layer.SetSpiralIndices(RingAdjacency(6, 5)),
            Ramp(6, 3),
            nameof(SpiralConvLayer<double>));
    }

    [Fact]
    public void DiffusionConvLayer_RoundTripsItsParameters()
    {
        AssertRoundTripPreservesParameters(
            () => new DiffusionConvLayer<double>(outputChannels: 4, numTimeScales: 2, numEigenvectors: 4,
                activation: (IActivationFunction<double>?)null),
            layer => layer.SetEigenbasis(
                new[] { 0.0, 0.4, 0.9, 1.6 },
                Ramp(6, 4)),
            Ramp(6, 3),
            nameof(DiffusionConvLayer<double>));
    }

    [Fact]
    public void Conv3DLayer_RoundTripsItsParameters()
    {
        // The layer that surfaced the pattern, pinned here alongside the others so the sweep is
        // visible in one place rather than only in a generated test.
        AssertRoundTripPreservesParameters(
            () => new Conv3DLayer<double>(outputChannels: 2, kernelSize: 3, stride: 1, padding: 1,
                activationFunction: (IActivationFunction<double>?)null),
            _ => { },
            Ramp(1, 1, 4, 4, 4),
            nameof(Conv3DLayer<double>));
    }
}
