using System;
using AiDotNet.NeuralRadianceFields.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// The engine-op frequency encoding NeRF trains through must reproduce the engine kernel's layout exactly; a
/// misplaced sin or cos would silently feed the field different features.
/// </summary>
public class PositionalEncoderTests
{
    [Fact]
    public void Encode_MatchesTheEngineKernel()
    {
        var positions = new Tensor<double>([5, 3]);
        for (int i = 0; i < positions.Length; i++) positions[i] = Math.Sin(0.37 * i) * 1.7;

        var engine = AiDotNetEngine.Current;
        var expected = engine.PositionalEncoding(positions, 4);
        var actual = PositionalEncoder<double>.Encode(engine, positions, 4);

        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-12, $"element {i}: kernel {expected[i]}, encoder {actual[i]}");
    }
}
