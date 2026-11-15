using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.ThreeD;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.ThreeD;

/// <summary>
/// Zero-1-to-3's camera conditioning against the official code: <c>cc_projection(cat(clip_emb, T))</c> with
/// <c>T = [deg2rad(polar), sin(deg2rad(azimuth)), cos(deg2rad(azimuth)), radius]</c> and the identity initialisation
/// of <c>ldm/models/diffusion/ddpm.py</c>.
/// </summary>
public class Zero123CameraProjectionTests
{
    [Fact]
    public void Initialisation_IsIdentityOnTheEmbeddingAndDefaultOnTheCamera()
    {
        const int d = 5;
        var projection = new ClipCameraProjection<double>(embedDim: d, seed: 3);
        Assert.Equal((long)(d * (d + 4)) + d, projection.ParameterCount);
        var p = projection.GetParameters().ToArray();
        double bound = 1.0 / Math.Sqrt(d + 4);
        for (int row = 0; row < d; row++)
        {
            for (int col = 0; col < d; col++)
                Assert.Equal(row == col ? 1.0 : 0.0, p[(row * (d + 4)) + col]);
            for (int k = 0; k < 4; k++)
                Assert.InRange(p[(row * (d + 4)) + d + k], -bound, bound);
            Assert.Equal(0.0, p[(d * (d + 4)) + row]);
        }
    }

    [Fact]
    public void CameraFeatures_AreTheOfficialEncoding()
    {
        var t = ClipCameraProjection<double>.CameraFeatures(new GuidanceViewOffset(-30.0, 90.0, 0.25));
        Assert.Equal(-Math.PI / 6.0, t[0], 12);
        Assert.Equal(1.0, t[1], 12);
        Assert.Equal(0.0, t[2], 12);
        Assert.Equal(0.25, t[3], 12);
    }

    [Fact]
    public void Project_MatchesAHandComputedLinearLayer()
    {
        const int d = 3;
        var projection = new ClipCameraProjection<double>(embedDim: d, seed: 1);
        // weight [3, 7] and bias [3], PyTorch nn.Linear layout.
        double[] weight =
        {
            1, 2, 0, 0.5, -1, 0, 3,
            0, 1, 1, 0, 2, 1, 0,
            -1, 0, 2, 1, 0, -2, 1,
        };
        double[] bias = { 0.1, -0.2, 0.3 };
        projection.LoadWeights(weight, bias);

        var embedding = new Tensor<double>(new[] { 1, 1, d }, new Vector<double>(new[] { 0.5, -1.0, 2.0 }));
        var views = new[] { new GuidanceViewOffset(45.0, 30.0, 0.0), new GuidanceViewOffset(-10.0, -120.0, 0.5) };
        var output = projection.Project(embedding, views);
        Assert.Equal(new[] { 2, 1, d }, output.Shape.ToArray());
        var values = output.ToArray();

        for (int b = 0; b < views.Length; b++)
        {
            double polar = views[b].Elevation * Math.PI / 180.0, azimuth = views[b].Azimuth * Math.PI / 180.0;
            double[] input = { 0.5, -1.0, 2.0, polar, Math.Sin(azimuth), Math.Cos(azimuth), views[b].Radius };
            for (int row = 0; row < d; row++)
            {
                double expected = bias[row];
                for (int col = 0; col < 7; col++)
                    expected += weight[(row * 7) + col] * input[col];
                Assert.Equal(expected, values[(b * d) + row], 12);
            }
        }
    }

    [Fact]
    public void Zero123Model_LoadsTheOfficialCcProjection()
    {
        var unet = new UNetNoisePredictor<double>(
            inputChannels: 8, outputChannels: 4, baseChannels: 64, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocks: 1, attentionResolutions: new[] { 1, 2 }, contextDim: 0, numHeads: 4, inputHeight: 16, seed: 42);
        var vae = new StandardVAE<double>(
            inputChannels: 3, latentChannels: 4, baseChannels: 32, channelMultipliers: new[] { 1, 2, 4 },
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 42);
        var model = new Zero123Model<double>(unet: unet, vae: vae, seed: 2);

        var requested = new List<(string Name, int[] Shape)>();
        model.LoadCameraProjectionWeights((name, shape) =>
        {
            requested.Add((name, shape));
            int length = shape.Aggregate(1, (a, b) => a * b);
            // A recognisable state dict: weight[r, c] = 0.001 r - 0.002 c, bias[r] = 0.5.
            return Enumerable.Range(0, length).Select(i => shape.Length == 2
                ? (0.001 * (i / shape[1])) - (0.002 * (i % shape[1]))
                : 0.5).ToArray();
        });

        Assert.Equal(2, requested.Count);
        Assert.Equal("cc_projection.weight", requested[0].Name);
        Assert.Equal(new[] { 768, 772 }, requested[0].Shape);
        Assert.Equal("cc_projection.bias", requested[1].Name);
        Assert.Equal(new[] { 768 }, requested[1].Shape);

        var p = model.CameraProjection.GetParameters().ToArray();
        Assert.Equal(0.001 * 7 - 0.002 * 771, p[(7 * 772) + 771], 12);
        Assert.Equal(0.5, p[(768 * 772) + 100], 12);
    }
}
