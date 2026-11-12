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

    [Fact]
    public void AuxElements_ReadsLegacyCountPlusDoubles_AsT()
    {
        // master's layout: a count followed directly by that many doubles, no format marker
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(3);
            writer.Write(1.5);
            writer.Write(-0.25);
            writer.Write(1e10);
            writer.Write(0);
        }

        stream.Position = 0;
        using var reader = new BinaryReader(stream);
        var restored = AuxLayerSerialization.ReadElements<float>(reader);
        var empty = AuxLayerSerialization.ReadElements<float>(reader);

        Assert.Equal(3, restored.Length);
        Assert.Equal(new[] { 1.5f, -0.25f, 1e10f }, new[] { restored[0], restored[1], restored[2] });
        Assert.Equal(0, empty.Length);
        Assert.Equal(stream.Length, stream.Position);
    }

    [Fact]
    public void TransformerState_LoadsLegacyCheckpoint_AndNewFormatRoundTrips()
    {
        var meta = new ColumnMetadata("X", ColumnDataType.Continuous, columnIndex: 0);

        // Bytes exactly as master's Serialize wrote them: each VGM array is count + doubles.
        byte[] legacy;
        using (var stream = new MemoryStream())
        {
            using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true))
            {
                writer.Write(true);   // IsFitted
                writer.Write(3);      // transformed width: 1 + 2 modes
                writer.Write(1);      // column count
                meta.Serialize(writer);
                writer.Write(1);      // transform count
                writer.Write(0);      // key
                writer.Write(true);   // continuous
                writer.Write(0);      // index
                writer.Write(0);      // start offset
                writer.Write(3);      // width
                writer.Write(1);      // continuous column count
                writer.Write(2); writer.Write(-2.0); writer.Write(3.0);   // means
                writer.Write(2); writer.Write(0.5); writer.Write(0.25);   // stds
                writer.Write(2); writer.Write(0.4); writer.Write(0.6);    // weights
                writer.Write(0);      // categorical column count
            }

            legacy = stream.ToArray();
        }

        var loaded = new TabularDataTransformer<float>();
        loaded.Deserialize(legacy);

        Assert.True(loaded.IsFitted);
        Assert.Equal(3, loaded.TransformedWidth);
        var info = loaded.GetContinuousColumnInfo(0);
        Assert.Equal(new[] { -2f, 3f }, info.Means.ToArray());
        Assert.Equal(new[] { 0.5f, 0.25f }, info.Stds.ToArray());
        Assert.Equal(new[] { 0.4f, 0.6f }, info.Weights.ToArray());

        // normalized 0.5 in mode 1: 0.5 * 4 * 0.25 + 3 = 3.5
        var encoded = new Matrix<float>(1, 3);
        encoded[0, 0] = 0.5f; encoded[0, 1] = 0f; encoded[0, 2] = 1f;
        Assert.Equal(3.5f, loaded.InverseTransform(encoded)[0, 0]);

        // The new format (with its marker) round-trips the same values.
        var reloaded = new TabularDataTransformer<float>();
        reloaded.Deserialize(loaded.Serialize());
        var info2 = reloaded.GetContinuousColumnInfo(0);
        Assert.Equal(info.Means.ToArray(), info2.Means.ToArray());
        Assert.Equal(info.Stds.ToArray(), info2.Stds.ToArray());
        Assert.Equal(info.Weights.ToArray(), info2.Weights.ToArray());
        Assert.Equal(3.5f, reloaded.InverseTransform(encoded)[0, 0]);
    }

    [Fact]
    public void Vgm_FloatColumnAtPlusMinus1e20_FitsFiniteModes()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        const int perMode = 100;
        var data = new Matrix<float>(perMode * 2, 1);
        for (int i = 0; i < perMode; i++) data[i, 0] = (float)(-1e20 * (1.0 + 0.01 * (rng.NextDouble() - 0.5)));
        for (int i = 0; i < perMode; i++) data[perMode + i, 0] = (float)(1e20 * (1.0 + 0.01 * (rng.NextDouble() - 0.5)));
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var transformer = new TabularDataTransformer<float>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);

        var info = transformer.GetContinuousColumnInfo(0);
        foreach (float v in info.Means.ToArray()) Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"mean {v} is not finite");
        foreach (float v in info.Stds.ToArray()) Assert.True(!float.IsNaN(v) && !float.IsInfinity(v) && v > 0f, $"std {v} is not finite and positive");
        foreach (float v in info.Weights.ToArray()) Assert.False(float.IsNaN(v) || float.IsInfinity(v), $"weight {v} is not finite");

        // both real modes are found, near -1e20 and +1e20
        var means = info.Means.ToArray();
        Assert.Contains(means, m => Math.Abs(m + 1e20) < 0.05 * 1e20);
        Assert.Contains(means, m => Math.Abs(m - 1e20) < 0.05 * 1e20);

        var recon = transformer.InverseTransform(transformer.Transform(data));
        for (int i = 0; i < data.Rows; i++)
        {
            Assert.True(Math.Abs((double)recon[i, 0] - data[i, 0]) < 0.05 * 1e20,
                $"row {i}: reconstructed {recon[i, 0]} from {data[i, 0]}");
        }
    }

    [Fact]
    public void Vgm_DecimalConstantColumn_FitsWithoutDividingByZero()
    {
        var data = new Matrix<decimal>(50, 1);
        for (int i = 0; i < data.Rows; i++) data[i, 0] = 7m;
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var transformer = new TabularDataTransformer<decimal>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);

        var info = transformer.GetContinuousColumnInfo(0);
        Assert.True(info.NumActiveModes >= 1);
        decimal weightSum = 0m;
        foreach (decimal w in info.Weights.ToArray()) weightSum += w;
        Assert.Equal(1m, Math.Round(weightSum, 6));

        var recon = transformer.InverseTransform(transformer.Transform(data));
        for (int i = 0; i < data.Rows; i++)
        {
            Assert.True(Math.Abs(recon[i, 0] - 7m) < 0.01m, $"row {i}: reconstructed {recon[i, 0]}");
        }
    }

    [Fact]
    public void VgmParameters_SurviveALaterArenaScope()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var data = new Matrix<double>(200, 1);
        for (int i = 0; i < data.Rows; i++) data[i, 0] = (i % 2 == 0 ? -3.0 : 3.0) + 0.2 * (rng.NextDouble() - 0.5);
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };
        var transformer = new TabularDataTransformer<double>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));

        double[] means;
        double[] stds;
        double[] weights;
        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            transformer.Fit(data, columns);
            var fitted = transformer.GetContinuousColumnInfo(0);
            means = fitted.Means.ToArray();
            stds = fitted.Stds.ToArray();
            weights = fitted.Weights.ToArray();
        }

        // A later arena scope rents (and overwrites) whatever buffers the first one released.
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            for (int round = 0; round < 50; round++)
            {
                var noise = AiDotNet.Tensors.LinearAlgebra.Tensor<double>.CreateDefault(new[] { means.Length }, -12345.0 - round);
                engine.TensorMultiplyScalar(engine.TensorAddScalar(noise, 1.0), 3.0);
                engine.TensorDivideScalar(noise, 7.0);
            }
        }

        var info = transformer.GetContinuousColumnInfo(0);
        Assert.Equal(means, info.Means.ToArray());
        Assert.Equal(stds, info.Stds.ToArray());
        Assert.Equal(weights, info.Weights.ToArray());
    }

    [Fact]
    public void Vgm_ConstantColumn_KeepsFloorsInDataUnits()
    {
        // A constant 7 column collapses onto one mode at 7. With every row in that mode
        // (N_k = n), zero spread and x̄ = m0, the Gamma rate stays at its prior b0, which is the
        // 1e-6 variance floor in DATA units, and a_k = 1 + n/2; so std = sqrt(1e-6 / (1 + n/2)),
        // exactly as master's unscaled fit gives. A floor applied in scaled units (x / 7) would
        // instead give 7 times that.
        const int n = 50;
        var data = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++) data[i, 0] = 7.0;
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var transformer = new TabularDataTransformer<double>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);

        var info = transformer.GetContinuousColumnInfo(0);
        Assert.Equal(1, info.NumActiveModes);
        Assert.Equal(7.0, info.Means.ToArray()[0], 12);
        double expectedStd = Math.Sqrt(1e-6 / (1.0 + n / 2.0));
        Assert.Equal(expectedStd, info.Stds.ToArray()[0], 9);
    }

    [Fact]
    public void Vgm_NormalRangeColumn_MatchesTheUnscaledFit()
    {
        // Values in [-2.3, 2.3]. Reference means and stds come from the unscaled VB fit (the
        // algorithm before the max|x| rescale, mirrored in double precision).
        const int n = 200;
        var data = new Matrix<double>(n, 1);
        for (int i = 0; i < n; i++) data[i, 0] = (i % 2 == 0 ? -2.0 : 2.0) + 0.3 * Math.Sin(0.7 * i);
        var columns = new List<ColumnMetadata> { new("X", ColumnDataType.Continuous, columnIndex: 0) };

        var transformer = new TabularDataTransformer<double>(vgmModes: 10, random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed));
        transformer.Fit(data, columns);

        var info = transformer.GetContinuousColumnInfo(0);
        var means = info.Means.ToArray();
        var stds = info.Stds.ToArray();
        Assert.Equal(2, means.Length);
        Assert.Equal(-1.9795247119079302, means[0], 5);
        Assert.Equal(1.982976489694219, means[1], 5);
        Assert.Equal(0.4024556911096269, stds[0], 5);
        Assert.Equal(0.4030624712189378, stds[1], 5);
    }
}
