using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.SyntheticData;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.SyntheticData;

/// <summary>
/// Reference behaviour for the synthetic-data members moved to engine ops: raw-T auxiliary
/// layer state, the VGM transformer's serialized state, and the per-row first-argmax decode
/// of one-hot blocks.
/// </summary>
public class SyntheticDataEngineOpsReferenceTests
{
    private const int Seed = 1234;

    [Fact]
    public void AuxElements_RoundTripBitForBit_InT()
    {
        var values = new Vector<float>(new[] { 0.1f, -3.25f, float.Epsilon, 1e30f, float.NaN });

        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            AuxLayerSerialization.WriteElements(writer, values);
        }

        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        var restored = AuxLayerSerialization.ReadElements<float>(reader);

        Assert.Equal(values.Length, restored.Length);
        for (int i = 0; i < values.Length; i++)
        {
            Assert.Equal(BitConverter.ToInt32(BitConverter.GetBytes(values[i]), 0),
                BitConverter.ToInt32(BitConverter.GetBytes(restored[i]), 0));
        }

        // count (4) + element size (4) + 5 four-byte floats
        Assert.Equal(4 + 4 + 5 * sizeof(float), (int)stream.Length);
    }

    [Fact]
    public void AuxElements_ReadAsDifferentWidth_Throws()
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            AuxLayerSerialization.WriteElements(writer, new Vector<float>(new[] { 1f, 2f }));
        }

        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        Assert.Throws<InvalidDataException>(() => AuxLayerSerialization.ReadElements<double>(reader));
    }

    [Fact]
    public void AuxElements_Empty_WritesOnlyTheCount()
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            AuxLayerSerialization.WriteElements(writer, new Vector<double>(0));
        }

        Assert.Equal(4, (int)stream.Length);
        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        Assert.Equal(0, AuxLayerSerialization.ReadElements<double>(reader).Length);
    }

    [Fact]
    public void CategoricalInverse_PicksFirstMaximum_AndZeroForAllNaN()
    {
        var columns = new List<ColumnMetadata>
        {
            new("C", ColumnDataType.Categorical, categories: new[] { "a", "b", "c" }, columnIndex: 0)
        };
        var fitData = new Matrix<double>(3, 1);
        fitData[0, 0] = 0;
        fitData[1, 0] = 1;
        fitData[2, 0] = 2;

        var transformer = new TabularDataTransformer<double>(random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(fitData, columns);

        var encoded = new Matrix<double>(4, 3);
        // tie between categories 0 and 1 -> first wins (0)
        encoded[0, 0] = 0.5; encoded[0, 1] = 0.5; encoded[0, 2] = 0.1;
        // every entry NaN -> nothing beats the starting value -> 0
        encoded[1, 0] = double.NaN; encoded[1, 1] = double.NaN; encoded[1, 2] = double.NaN;
        // NaN never wins; the finite maximum does
        encoded[2, 0] = double.NaN; encoded[2, 1] = 0.2; encoded[2, 2] = 0.9;
        // tie between categories 1 and 2 -> first wins (1)
        encoded[3, 0] = -1.0; encoded[3, 1] = 3.0; encoded[3, 2] = 3.0;

        var decoded = transformer.InverseTransform(encoded);

        Assert.Equal(0.0, decoded[0, 0]);
        Assert.Equal(0.0, decoded[1, 0]);
        Assert.Equal(2.0, decoded[2, 0]);
        Assert.Equal(1.0, decoded[3, 0]);
    }

    [Fact]
    public void CategoricalForward_EncodesIndexCellsOneHot()
    {
        var columns = new List<ColumnMetadata>
        {
            new("C", ColumnDataType.Categorical, categories: new[] { "a", "b", "c" }, columnIndex: 0)
        };
        var data = new Matrix<double>(3, 1);
        data[0, 0] = 2;
        data[1, 0] = 0;
        data[2, 0] = 7; // neither an index nor a category name -> all zeros

        var transformer = new TabularDataTransformer<double>(random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);
        var encoded = transformer.Transform(data);

        Assert.Equal(3, encoded.Columns);
        Assert.Equal(new[] { 0.0, 0.0, 1.0 }, new[] { encoded[0, 0], encoded[0, 1], encoded[0, 2] });
        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, new[] { encoded[1, 0], encoded[1, 1], encoded[1, 2] });
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, new[] { encoded[2, 0], encoded[2, 1], encoded[2, 2] });
    }

    [Fact]
    public void ContinuousTransform_RowsAreOneHotModes_WithClippedNormalizedValue()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        const int perMode = 200;
        var data = new Matrix<double>(perMode * 2, 1);
        for (int i = 0; i < perMode; i++) data[i, 0] = -4.0 + 0.3 * (rng.NextDouble() - 0.5);
        for (int i = 0; i < perMode; i++) data[perMode + i, 0] = 4.0 + 0.3 * (rng.NextDouble() - 0.5);
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var transformer = new TabularDataTransformer<double>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);
        var encoded = transformer.Transform(data);
        int width = transformer.GetTransformInfo(0).Width;

        Assert.Equal(width, encoded.Columns);
        for (int i = 0; i < encoded.Rows; i++)
        {
            Assert.InRange(encoded[i, 0], -0.99, 0.99);
            double modeSum = 0;
            int ones = 0;
            for (int m = 1; m < width; m++)
            {
                Assert.True(encoded[i, m] == 0.0 || encoded[i, m] == 1.0, $"row {i} mode entry {encoded[i, m]} is not one-hot");
                modeSum += encoded[i, m];
                if (encoded[i, m] == 1.0) ones++;
            }

            Assert.Equal(1.0, modeSum);
            Assert.Equal(1, ones);
        }
    }

    [Fact]
    public void TransformerState_RoundTrips_AndDecodesIdentically()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var data = new Matrix<float>(120, 2);
        for (int i = 0; i < data.Rows; i++)
        {
            data[i, 0] = (float)((i % 2 == 0 ? -2.0 : 3.0) + 0.2 * (rng.NextDouble() - 0.5));
            data[i, 1] = i % 3;
        }

        var columns = new List<ColumnMetadata>
        {
            new("X", ColumnDataType.Continuous, columnIndex: 0),
            new("C", ColumnDataType.Categorical, categories: new[] { "p", "q", "r" }, columnIndex: 1)
        };

        var original = new TabularDataTransformer<float>(random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        original.Fit(data, columns);
        var encoded = original.Transform(data);

        var restored = new TabularDataTransformer<float>();
        restored.Deserialize(original.Serialize());

        Assert.Equal(original.TransformedWidth, restored.TransformedWidth);
        var a = original.InverseTransform(encoded);
        var b = restored.InverseTransform(encoded);
        for (int i = 0; i < a.Rows; i++)
        {
            for (int j = 0; j < a.Columns; j++)
            {
                Assert.Equal(a[i, j], b[i, j]);
            }
        }

        // the categorical column decodes back to its category index exactly
        for (int i = 0; i < data.Rows; i++)
        {
            Assert.Equal(data[i, 1], a[i, 1]);
        }
    }
}
