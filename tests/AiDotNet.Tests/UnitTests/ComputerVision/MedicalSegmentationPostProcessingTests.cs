using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.Common;
using AiDotNet.ComputerVision.Segmentation.Medical;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Post-processing of the medical segmenters: per-structure statistics, slice-by-slice volume assembly and the
/// class-axis softmax of a volumetric model, each checked against a hand-computed reference.
/// </summary>
public class MedicalSegmentationPostProcessingTests
{
    private const double Tolerance = 1e-9;

    private static NeuralNetworkArchitecture<double> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression, NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<double> Rand(params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    [Fact]
    public void Summarize_CountsAreaAndMeanConfidencePerPresentClass()
    {
        // 2x3 label map with classes 2, 0, 2 / 2, 0, 0 (class 1 absent) and per-class probabilities.
        var labels = new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new double[] { 2, 0, 2, 2, 0, 0 }));
        var probs = new Tensor<double>(new[] { 3, 2, 3 }, new Vector<double>(new double[]
        {
            0.1, 0.7, 0.2, 0.3, 0.6, 0.8, // class 0
            0.1, 0.2, 0.1, 0.1, 0.3, 0.1, // class 1
            0.8, 0.1, 0.7, 0.6, 0.1, 0.1, // class 2
        }));

        var structures = SegmentationStructureStatistics.Summarize(labels, probs);

        Assert.Equal(new[] { 0, 2 }, structures.Select(s => s.ClassId).ToArray());
        Assert.Equal(3, structures[0].VolumeOrArea);
        Assert.Equal((0.7 + 0.6 + 0.8) / 3, structures[0].MeanConfidence, Tolerance);
        Assert.Equal(3, structures[1].VolumeOrArea);
        Assert.Equal((0.8 + 0.7 + 0.6) / 3, structures[1].MeanConfidence, Tolerance);
        Assert.Equal("Class_2", structures[1].Name);
    }

    [Fact]
    public void Summarize_FirstOccurrenceOrderAndSkippedClass()
    {
        // Raster order of first appearance: 3, 0, 1 - background 0 is skipped.
        var labels = new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 3, 0, 1, 3 }));
        var probs = new Tensor<double>(new[] { 4, 4 }, new Vector<double>(new double[]
        {
            0.1, 0.9, 0.1, 0.1,
            0.1, 0.0, 0.5, 0.2,
            0.1, 0.0, 0.2, 0.1,
            0.7, 0.1, 0.2, 0.6,
        }));

        var structures = SegmentationStructureStatistics.Summarize(labels, probs, skipClassId: 0, firstOccurrenceOrder: true);

        Assert.Equal(new[] { 3, 1 }, structures.Select(s => s.ClassId).ToArray());
        Assert.Equal(2, structures[0].VolumeOrArea);
        Assert.Equal(0.65, structures[0].MeanConfidence, Tolerance);
        Assert.Equal(1, structures[1].VolumeOrArea);
        Assert.Equal(0.5, structures[1].MeanConfidence, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task NnUNet_SegmentVolume_ProbabilityVolumeHasOnePlanePerClass()
    {
        // 3 input channels, 5 classes: the stacked probabilities are [5, D, H, W] (one plane per class, summing to
        // one per voxel), not one plane per input channel.
        var model = new NnUNet<double>(Arch(), options: new NnUNetOptions { NumClasses = 5, ModelSize = NnUNetModelSize.UNet2D });
        var result = model.SegmentVolume(Rand(3, 2, 32, 32));
        await Task.CompletedTask;

        var probs = result.Probabilities;
        Assert.Equal(4, probs.Rank);
        Assert.Equal(5, probs.Shape[0]);
        Assert.Equal(2, probs.Shape[1]);
        Assert.Equal(new[] { 2, probs.Shape[2], probs.Shape[3] }, result.Labels.Shape.ToArray());
        AssertClassAxisSumsToOne(probs);
    }

    [Fact(Timeout = 120000)]
    public async Task SegMamba_SegmentVolume_SoftmaxIsOverTheClassAxis()
    {
        var model = new SegMamba<double>(Arch(), options: new SegMambaOptions { NumClasses = 3 });
        var result = model.SegmentVolume(Rand(1, 3, 32, 32));
        await Task.CompletedTask;

        var probs = result.Probabilities;
        Assert.Equal(4, probs.Rank);
        Assert.Equal(3, probs.Shape[0]);
        AssertClassAxisSumsToOne(probs);

        // Labels are the per-voxel argmax of the class probabilities.
        int spatial = probs.Length / probs.Shape[0];
        var p = probs.ToArray();
        var labels = result.Labels.ToArray();
        for (int i = 0; i < spatial; i++)
        {
            int best = 0;
            for (int c = 1; c < probs.Shape[0]; c++)
            {
                if (p[c * spatial + i] > p[best * spatial + i]) best = c;
            }

            Assert.Equal(best, (int)labels[i]);
        }
    }

    [Fact]
    public void SegMamba_ResolvedFullResolutionLayersTakeTheVolumeDepth()
    {
        // A 3-deep, 32 x 32 volume. The last decoder convolution and the class projection run at the input volume's
        // full resolution, so they are resolved for depth 3; the height (32) was read as the depth before.
        var model = new SegMamba<double>(Arch(), options: new SegMambaOptions { NumClasses = 3 });
        model.GetParameterLayout();

        var finalDecoderConv = model.Layers[model.Layers.Count - 3].GetInputShape();
        var classProjection = model.Layers[model.Layers.Count - 1].GetInputShape();

        Assert.Equal(new[] { 3, 32, 32 }, finalDecoderConv.Skip(1).ToArray());
        Assert.Equal(new[] { 3, 32, 32 }, classProjection.Skip(1).ToArray());
    }

    [Fact]
    public void Summarize_Half_AreaAndMeanStayExactBeyondTheHalfRange()
    {
        // 300 x 300 = 90000 pixels of class 1 (more than Half's largest finite value, 65504) plus 4 of class 0.
        // Summing the 0/1 mask or the probabilities in Half rounded the area and overflowed to infinity.
        int spatial = 90004;
        var labelValues = Enumerable.Repeat((Half)1f, spatial).ToArray();
        for (int i = 0; i < 4; i++) labelValues[i] = (Half)0f;
        var probValues = new Half[2 * spatial];
        for (int i = 0; i < spatial; i++)
        {
            probValues[i] = (Half)0.25f;
            probValues[spatial + i] = (Half)0.75f;
        }

        var structures = SegmentationStructureStatistics.Summarize(
            new Tensor<Half>(new[] { spatial }, new Vector<Half>(labelValues)),
            new Tensor<Half>(new[] { 2, spatial }, new Vector<Half>(probValues)));

        Assert.Equal(new[] { 0, 1 }, structures.Select(s => s.ClassId).ToArray());
        Assert.Equal(4, structures[0].VolumeOrArea);
        Assert.Equal(90000, structures[1].VolumeOrArea);
        Assert.Equal(0.25, structures[0].MeanConfidence, 1e-3);
        Assert.Equal(0.75, structures[1].MeanConfidence, 1e-3);
    }

    [Fact]
    public void MaskedMean_Half_CountsInIntAndIgnoresValuesOutsideTheMask()
    {
        // MedSAM2's binary result: 100000 foreground pixels at 0.9 and 3 background pixels, one of them NaN.
        int n = 100003;
        var values = Enumerable.Repeat((Half)0.9f, n).ToArray();
        var mask = Enumerable.Repeat((Half)1f, n).ToArray();
        for (int i = 0; i < 3; i++) mask[i] = (Half)0f;
        values[0] = Half.NaN;
        var maskTensor = new Tensor<Half>(new[] { n }, new Vector<Half>(mask));

        int count = SegmentationStructureStatistics.CountNonZero(maskTensor);
        Half mean = SegmentationStructureStatistics.MaskedMean(
            new Tensor<Half>(new[] { n }, new Vector<Half>(values)), maskTensor, count);

        Assert.Equal(100000, count);
        Assert.Equal(0.9, (double)(float)mean, 2e-3);
        Assert.Equal(0.0, (double)(float)SegmentationStructureStatistics.MaskedMean(maskTensor, maskTensor, 0), Tolerance);
    }

    private static void AssertClassAxisSumsToOne(Tensor<double> probs)
    {
        int classes = probs.Shape[0];
        int spatial = probs.Length / classes;
        var p = probs.ToArray();
        for (int i = 0; i < spatial; i++)
        {
            double sum = 0;
            for (int c = 0; c < classes; c++) sum += p[c * spatial + i];
            Assert.Equal(1.0, sum, 1e-6);
        }
    }
}
