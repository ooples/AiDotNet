using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Format v8 stores a float network's parameters as 4-byte floats. Every earlier version wrote each value as an 8-byte
/// double whatever T was, which doubled the file and made Serialize, building one byte array, overflow for any float
/// model above about 268M parameters.
/// </summary>
public class NetworkSerializationParameterWidthTests
{
    private static NeuralNetworkArchitecture<TNum> Architecture<TNum>()
        => new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
               inputSize: 64, outputSize: 32);

    [Fact]
    public void AFloatNetwork_StoresFourBytesPerParameter_AndRestoresExactly()
    {
        using var source = new FeedForwardNeuralNetwork<float>(Architecture<float>());
        var parameters = source.GetParameters();
        byte[] bytes = source.Serialize();

        // Layer headers, shapes and metadata are a few kilobytes; at 8 bytes per value the payload alone would exceed
        // this bound.
        Assert.True(bytes.Length < parameters.Length * 6L,
            $"{bytes.Length} bytes for {parameters.Length} float parameters: they were not stored as 4-byte floats.");

        using var restored = new FeedForwardNeuralNetwork<float>(Architecture<float>());
        restored.Deserialize(bytes);
        var back = restored.GetParameters();
        Assert.Equal(parameters.Length, back.Length);
        for (int i = 0; i < parameters.Length; i++)
            Assert.Equal(parameters[i], back[i]);
    }

    [Fact]
    public void ADoubleNetwork_StillStoresEightBytesPerParameter_AndRestoresExactly()
    {
        using var source = new FeedForwardNeuralNetwork<double>(Architecture<double>());
        var parameters = source.GetParameters();
        byte[] bytes = source.Serialize();
        Assert.True(bytes.Length >= parameters.Length * 8L);

        using var restored = new FeedForwardNeuralNetwork<double>(Architecture<double>());
        restored.Deserialize(bytes);
        var back = restored.GetParameters();
        Assert.Equal(parameters.Length, back.Length);
        for (int i = 0; i < parameters.Length; i++)
            Assert.Equal(parameters[i], back[i]);
    }
}
