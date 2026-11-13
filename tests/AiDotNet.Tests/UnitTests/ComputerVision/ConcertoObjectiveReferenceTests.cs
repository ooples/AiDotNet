using System;
using System.Collections.Generic;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Concerto's pretraining objectives (arXiv:2510.23607) against hand-computed references: the DINO-style
/// intra-modal cross-entropy with centering and sharpening, the EMA teacher and center updates, and the
/// 2D-3D cross-modal cosine objective with projection, depth-based visibility and patch mean-pooling.
/// </summary>
public class ConcertoObjectiveReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Tensor<double> T(int[] shape, params double[] values) => new(shape, new Vector<double>(values));

    [Fact]
    public void IntraModalLoss_MatchesHandComputedCrossEntropy_AndTheTapeLoss()
    {
        var student = T(new[] { 2, 3 }, 0.2, -0.1, 0.4, 1.0, 0.0, -0.5);
        var teacher = T(new[] { 2, 3 }, 0.3, 0.1, -0.2, -0.4, 0.6, 0.2);
        var center = T(new[] { 3 }, 0.05, -0.05, 0.1);

        double expected = 0;
        for (int p = 0; p < 2; p++)
        {
            var t = new double[3];
            var s = new double[3];
            for (int k = 0; k < 3; k++)
            {
                t[k] = (teacher[p, k] - center[k]) / 0.04;
                s[k] = student[p, k] / 0.1;
            }

            double tMax = Math.Max(t[0], Math.Max(t[1], t[2]));
            double tSum = Math.Exp(t[0] - tMax) + Math.Exp(t[1] - tMax) + Math.Exp(t[2] - tMax);
            double sMax = Math.Max(s[0], Math.Max(s[1], s[2]));
            double logZ = sMax + Math.Log(Math.Exp(s[0] - sMax) + Math.Exp(s[1] - sMax) + Math.Exp(s[2] - sMax));
            for (int k = 0; k < 3; k++)
            {
                expected -= Math.Exp(t[k] - tMax) / tSum * (s[k] - logZ);
            }
        }

        expected /= 2;

        double loss = ConcertoIntraModalObjective<double>.ComputeLoss(student, teacher, center);
        Assert.Equal(expected, loss, Tolerance);

        var tape = ConcertoIntraModalObjective<double>.ComputeTapeLoss(student, teacher, center);
        Assert.Equal(expected, tape.ToArray()[0], Tolerance);
    }

    [Fact]
    public void IntraModalLoss_NaNTeacherLogit_PropagatesLikeTheScalarReference()
    {
        // The scalar reference skips a prototype only when its teacher probability is <= 0. A NaN teacher logit
        // makes that point's probabilities NaN, which are not skipped, so the loss is NaN; selecting on "> 0"
        // instead dropped them and reported the other point's loss as if nothing were wrong.
        var student = T(new[] { 2, 2 }, 0.2, -0.1, 1.0, 0.0);
        var teacher = T(new[] { 2, 2 }, double.NaN, 0.1, -0.4, 0.6);

        Assert.True(double.IsNaN(ConcertoIntraModalObjective<double>.ComputeLoss(student, teacher)));
    }

    [Fact]
    public void UpdateTeacher_IsTheExponentialMovingAverageInPlace()
    {
        var teacher = new Vector<double>(new[] { 1.0, 2.0, -4.0 });
        var student = new Vector<double>(new[] { 3.0, 6.0, 0.0 });

        ConcertoIntraModalObjective<double>.UpdateTeacher(teacher, student, 0.75);

        Assert.Equal(1.5, teacher[0], Tolerance);
        Assert.Equal(3.0, teacher[1], Tolerance);
        Assert.Equal(-3.0, teacher[2], Tolerance);
    }

    [Fact]
    public void UpdateCenter_MovesTowardTheBatchMeanInPlace()
    {
        var center = T(new[] { 2 }, 1.0, -1.0);
        var logits = T(new[] { 2, 2 }, 1.0, 2.0, 3.0, 4.0);

        ConcertoIntraModalObjective<double>.UpdateCenter(center, logits, 0.9);

        Assert.Equal((0.9 * 1.0) + (0.1 * 2.0), center[0], Tolerance);
        Assert.Equal((0.9 * -1.0) + (0.1 * 3.0), center[1], Tolerance);
    }

    private static ConcertoPairedView<double> UnitCameraView() => new()
    {
        // Identity pose and a unit pinhole: pixel (u, v) = (x / z, y / z); a 2x2 depth map at depth 1 and a
        // 2x2 patch grid, so every pixel is its own patch.
        Extrinsics = T(new[] { 4, 4 }, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
        Intrinsics = T(new[] { 3, 3 }, 1, 0, 0, 0, 1, 0, 0, 0, 1),
        DepthMap = T(new[] { 2, 2 }, 1, 1, 1, 1),
        PatchGridSize = 2,
        ImagePatchFeatures = T(new[] { 4, 2 }, 1, 0, 0, 1, 1, 1, 1, 1),
    };

    private static Tensor<double> Coordinates() => T(new[] { 5, 3 },
        0.5, 0.5, 1.0,   // pixel (0, 0) -> patch 0
        1.5, 0.5, 1.0,   // pixel (1, 0) -> patch 1
        0.2, 0.2, 1.0,   // pixel (0, 0) -> patch 0
        0.5, 0.5, -1.0,  // behind the camera
        1.5, 4.5, 3.0);  // pixel (0, 1) but at depth 3 against an observed depth of 1: occluded

    [Fact]
    public void CrossModalLoss_ProjectsChecksVisibilityAndMeanPoolsPerPatch()
    {
        var features = T(new[] { 5, 2 }, 1, 0, 0, 1, 1, 2, 5, 5, 7, 7);
        var views = new List<ConcertoPairedView<double>> { UnitCameraView() };

        // Patch 0 pools points 0 and 2 to (1, 1) against (1, 0): cos = 1 / sqrt(2). Patch 1 holds point 1,
        // (0, 1) against (0, 1): cos = 1. The two invisible points are ignored.
        double expected = 1.0 - ((1.0 / Math.Sqrt(2.0)) + 1.0) / 2.0;

        Assert.Equal(expected, ConcertoCrossModalObjective<double>.ComputeLoss(features, Coordinates(), views, 0.5), Tolerance);
        var tape = ConcertoCrossModalObjective<double>.ComputeTapeLoss(features, Coordinates(), views, 0.5);
        Assert.Equal(expected, tape.ToArray()[0], Tolerance);
    }

    [Fact]
    public void CrossModalLoss_ZeroPooledFeature_CountsAsZeroSimilarityInBothEntryPoints()
    {
        // Patch 1's only point has a zero feature: like torch.nn.functional.cosine_similarity it scores 0 and is
        // still one of the matched patches, in the reported loss and the training loss alike.
        var features = T(new[] { 5, 2 }, 1, 0, 0, 0, 1, 2, 5, 5, 7, 7);
        var views = new List<ConcertoPairedView<double>> { UnitCameraView() };
        double expected = 1.0 - ((1.0 / Math.Sqrt(2.0)) + 0.0) / 2.0;

        Assert.Equal(expected, ConcertoCrossModalObjective<double>.ComputeLoss(features, Coordinates(), views, 0.5), Tolerance);
        var tape = ConcertoCrossModalObjective<double>.ComputeTapeLoss(features, Coordinates(), views, 0.5);
        Assert.Equal(expected, tape.ToArray()[0], Tolerance);
    }

    [Fact]
    public void CrossModalLoss_NoVisiblePoint_IsZero()
    {
        var features = T(new[] { 1, 2 }, 1, 0);
        var coordinates = T(new[] { 1, 3 }, 0.5, 0.5, -1.0);
        var views = new List<ConcertoPairedView<double>> { UnitCameraView() };

        Assert.Equal(0.0, ConcertoCrossModalObjective<double>.ComputeLoss(features, coordinates, views, 0.5), Tolerance);
    }

    [Fact]
    public void CrossModalLoss_NonFiniteFeatureOnAnInvisiblePoint_DoesNotReachAnyPatch()
    {
        // The point behind the camera carries NaN and the occluded one +inf: neither is visible, so neither may
        // touch a patch (a 0/1 mask multiply turned them into NaN inside patch 0 / patch 2).
        var features = T(new[] { 5, 2 }, 1, 0, 0, 1, 1, 2, double.NaN, double.NaN, double.PositiveInfinity, 7);
        var views = new List<ConcertoPairedView<double>> { UnitCameraView() };
        double expected = 1.0 - ((1.0 / Math.Sqrt(2.0)) + 1.0) / 2.0;

        Assert.Equal(expected, ConcertoCrossModalObjective<double>.ComputeLoss(features, Coordinates(), views, 0.5), Tolerance);
        var tape = ConcertoCrossModalObjective<double>.ComputeTapeLoss(features, Coordinates(), views, 0.5);
        Assert.Equal(expected, tape.ToArray()[0], Tolerance);
    }

    [Fact]
    public void CrossModalLoss_UnoccupiedPatchWithNonFiniteImageFeature_IsIgnored()
    {
        // Patches 2 and 3 receive no visible point; their image features never enter the loss.
        var unit = UnitCameraView();
        var view = new ConcertoPairedView<double>
        {
            Extrinsics = unit.Extrinsics,
            Intrinsics = unit.Intrinsics,
            DepthMap = unit.DepthMap,
            PatchGridSize = unit.PatchGridSize,
            ImagePatchFeatures = T(new[] { 4, 2 }, 1, 0, 0, 1, double.NaN, double.NaN, double.PositiveInfinity, 1),
        };
        var features = T(new[] { 5, 2 }, 1, 0, 0, 1, 1, 2, 5, 5, 7, 7);
        var views = new List<ConcertoPairedView<double>> { view };
        double expected = 1.0 - ((1.0 / Math.Sqrt(2.0)) + 1.0) / 2.0;

        Assert.Equal(expected, ConcertoCrossModalObjective<double>.ComputeLoss(features, Coordinates(), views, 0.5), Tolerance);
        var tape = ConcertoCrossModalObjective<double>.ComputeTapeLoss(features, Coordinates(), views, 0.5);
        Assert.Equal(expected, tape.ToArray()[0], Tolerance);
    }

    [Fact]
    public void CrossModalLoss_Half_ProjectsInDoubleSoAPointStaysInItsPixel()
    {
        // u = 1 * 300.25 + 300.5 = 600.75 lies in pixel 600, which a 1202-wide map with a 2x2 grid puts in patch 0.
        // Projected in Half, 600.75 rounds to 601 -- pixel 601, patch 1 -- and the point matched the wrong patch.
        static Tensor<Half> H(int[] shape, params double[] values)
            => new(shape, new Vector<Half>(Array.ConvertAll(values, v => (Half)(float)v)));

        var depth = new double[1202];
        for (int i = 0; i < depth.Length; i++) depth[i] = 1.0;
        var view = new ConcertoPairedView<Half>
        {
            Extrinsics = H(new[] { 4, 4 }, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1),
            Intrinsics = H(new[] { 3, 3 }, 1, 0, 300.5, 0, 1, 0, 0, 0, 1),
            DepthMap = H(new[] { 1, 1202 }, depth),
            PatchGridSize = 2,
            ImagePatchFeatures = H(new[] { 4, 2 }, 1, 0, 0, 1, 1, 1, 1, 1),
        };
        var coordinates = H(new[] { 1, 3 }, 300.25, 0.5, 1.0);
        var features = H(new[] { 1, 2 }, 1, 0);

        Half loss = ConcertoCrossModalObjective<Half>.ComputeLoss(
            features, coordinates, new List<ConcertoPairedView<Half>> { view }, 0.5);

        Assert.Equal(0.0, (double)(float)loss, 1e-3);
    }

    [Fact]
    public void IntraModalLoss_Half_ManyPointsDoesNotOverflow()
    {
        // 10000 points whose teacher picks prototype 0 while the student prefers prototype 1 by 10 nats:
        // H = 10 per point, so the raw total (100000) overflows Half; the mean is still 10.
        const int points = 10000;
        var studentValues = new Half[points * 2];
        var teacherValues = new Half[points * 2];
        for (int p = 0; p < points; p++)
        {
            studentValues[2 * p] = (Half)0f;
            studentValues[2 * p + 1] = (Half)1f;
            teacherValues[2 * p] = (Half)1f;
            teacherValues[2 * p + 1] = (Half)0f;
        }

        Half loss = ConcertoIntraModalObjective<Half>.ComputeLoss(
            new Tensor<Half>(new[] { points, 2 }, new Vector<Half>(studentValues)),
            new Tensor<Half>(new[] { points, 2 }, new Vector<Half>(teacherValues)));

        double expected = 10.0 + Math.Log(1.0 + Math.Exp(-10.0));
        Assert.Equal(expected, (double)(float)loss, 0.05);
    }

    private static Tensor<Half> HalfTensor(int[] shape, Func<int, double> valueAt)
    {
        int size = 1;
        foreach (int dim in shape) size *= dim;
        var values = new Half[size];
        for (int i = 0; i < size; i++) values[i] = (Half)(float)valueAt(i);
        return new Tensor<Half>(shape, new Vector<Half>(values));
    }

    [Fact]
    public void UpdateCenter_Half_ManyPointsAccumulatesTheMeanInDouble()
    {
        // 4096 points of (1, 30). Summed in Half the first column stalls at 2048 (1 is half an ulp there) and
        // the second overflows past 65504, so the old center was (0.25, inf); the mean is (1, 30).
        const int points = 4096;
        var center = HalfTensor(new[] { 2 }, _ => 0.0);
        var logits = HalfTensor(new[] { points, 2 }, i => i % 2 == 0 ? 1.0 : 30.0);

        ConcertoIntraModalObjective<Half>.UpdateCenter(center, logits, 0.5);

        Assert.Equal(0.5, (double)(float)center[0], 1e-3);
        Assert.Equal(15.0, (double)(float)center[1], 1e-2);
    }

    private static ConcertoPairedView<Half> HalfGridView(int grid, Func<int, double> imageFeatureAt) => new()
    {
        // Identity pose, unit pinhole, a grid x grid depth map at depth 1 and a grid x grid patch grid, so
        // every pixel is its own patch.
        Extrinsics = HalfTensor(new[] { 4, 4 }, i => i % 5 == 0 ? 1.0 : 0.0),
        Intrinsics = HalfTensor(new[] { 3, 3 }, i => i % 4 == 0 ? 1.0 : 0.0),
        DepthMap = HalfTensor(new[] { grid, grid }, _ => 1.0),
        PatchGridSize = grid,
        ImagePatchFeatures = HalfTensor(new[] { grid * grid, 2 }, imageFeatureAt),
    };

    [Fact]
    public void CrossModalLoss_Half_LargePatchMeanPoolsInDouble()
    {
        // 8192 points in patch 0: 6144 with feature (1, 0), then 2048 with (0, 1), against image feature (1, -1).
        // The pooled mean is (0.75, 0.25): cos = 0.5 / (sqrt(0.625) * sqrt(2)) = 1 / sqrt(5). Pooled in Half each
        // point adds 2^-13, which is half an ulp once the first column reaches 0.25, so the old pooled feature
        // was (0.25, 0.25), orthogonal to the target, and the loss was 1.
        const int points = 8192;
        const int firstGroup = 6144;
        var view = HalfGridView(2, i => i == 0 ? 1.0 : i == 1 ? -1.0 : 1.0);
        var coordinates = HalfTensor(new[] { points, 3 }, i => i % 3 == 2 ? 1.0 : 0.5);
        var features = HalfTensor(new[] { points, 2 }, i => (i / 2 < firstGroup) == (i % 2 == 0) ? 1.0 : 0.0);
        var views = new List<ConcertoPairedView<Half>> { view };

        double expected = 1.0 - (1.0 / Math.Sqrt(5.0));

        Half loss = ConcertoCrossModalObjective<Half>.ComputeLoss(features, coordinates, views, 0.5);
        Assert.Equal(expected, (double)(float)loss, 2e-3);
        var tape = ConcertoCrossModalObjective<Half>.ComputeTapeLoss(features, coordinates, views, 0.5);
        Assert.Equal(expected, (double)(float)tape.ToArray()[0], 2e-3);
    }

    [Fact]
    public void CrossModalLoss_Half_SmallPooledFeatureKeepsItsCosine()
    {
        // One visible point in patch 0 with feature (0.003, 0.004) against image feature (3, 4): cos = 1, loss 0.
        // Half cannot hold eps^2 = 1e-16, so its squared-norm floor is 2^-14 (a norm floor near 0.0078). The pooled
        // norm 0.005 sat below it and was divided by the floor, giving cos = 0.64 and a loss of 0.36.
        var view = HalfGridView(2, i => i == 0 ? 3.0 : i == 1 ? 4.0 : 1.0);
        var coordinates = HalfTensor(new[] { 1, 3 }, i => i == 2 ? 1.0 : 0.5);
        var features = HalfTensor(new[] { 1, 2 }, i => i == 0 ? 0.003 : 0.004);
        var views = new List<ConcertoPairedView<Half>> { view };

        Half loss = ConcertoCrossModalObjective<Half>.ComputeLoss(features, coordinates, views, 0.5);
        Assert.Equal(0.0, (double)(float)loss, 2e-3);
        var tape = ConcertoCrossModalObjective<Half>.ComputeTapeLoss(features, coordinates, views, 0.5);
        Assert.Equal(0.0, (double)(float)tape.ToArray()[0], 2e-3);
    }

    [Fact]
    public void PretrainEpochLoss_Half_ManySamplesDoesNotOverflow()
    {
        // 10000 samples at loss 10: the raw total (100000) overflows Half; each loss is scaled by 1 / count first.
        var losses = new Half[10000];
        for (int i = 0; i < losses.Length; i++) losses[i] = (Half)10f;

        Assert.Equal(10.0, (double)(float)Concerto<Half>.MeanOfLosses(losses), 0.02);
    }

    [Fact]
    public void PretrainEpochLoss_Half_SmallLossesDoNotUnderflowWhenScaled()
    {
        // 10000 samples at loss 0.001: scaled by the full 1 / count first, each loss became 1e-7, a Half subnormal
        // whose step is 6e-8, so every term rounded up to 1.19e-7 and the mean came out 0.00119 (19% high).
        var losses = new Half[10000];
        for (int i = 0; i < losses.Length; i++) losses[i] = (Half)0.001f;

        Assert.Equal(0.001, (double)(float)Concerto<Half>.MeanOfLosses(losses), 2e-5);
    }

    [Fact]
    public void CrossModalLoss_Half_ManyMatchedPatchesAveragesInDouble()
    {
        // A 64 x 64 grid with one point per pixel, every point and patch feature (1, 0): all 4096 similarities are
        // 1 and the loss is 0. Averaged in Half each similarity adds 2^-12, half an ulp once the sum reaches 0.5,
        // so the old mean stalled at 0.5 and the loss was 0.5.
        const int grid = 64;
        const int points = grid * grid;
        var view = HalfGridView(grid, i => i % 2 == 0 ? 1.0 : 0.0);
        var coordinates = HalfTensor(new[] { points, 3 }, i =>
        {
            int point = i / 3;
            return (i % 3) switch
            {
                0 => (point % grid) + 0.5,
                1 => (point / grid) + 0.5,
                _ => 1.0,
            };
        });
        var features = HalfTensor(new[] { points, 2 }, i => i % 2 == 0 ? 1.0 : 0.0);
        var views = new List<ConcertoPairedView<Half>> { view };

        Half loss = ConcertoCrossModalObjective<Half>.ComputeLoss(features, coordinates, views, 0.5);
        Assert.Equal(0.0, (double)(float)loss, 2e-3);
        var tape = ConcertoCrossModalObjective<Half>.ComputeTapeLoss(features, coordinates, views, 0.5);
        Assert.Equal(0.0, (double)(float)tape.ToArray()[0], 2e-3);
    }

    [Fact]
    public void CrossModalLoss_Half_TapeGradientMatchesDouble()
    {
        // The double-accumulated Half pooling is one recorded operation; its backward must give the same
        // feature gradient as the double objective.
        var doubleFeatures = T(new[] { 5, 2 }, 1, 0, 0, 1, 1, 2, 5, 5, 7, 7);
        Tensor<double> doubleGrad;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var loss = ConcertoCrossModalObjective<double>.ComputeTapeLoss(
                doubleFeatures, Coordinates(), new List<ConcertoPairedView<double>> { UnitCameraView() }, 0.5);
            doubleGrad = tape.ComputeGradients(loss, new[] { doubleFeatures })[doubleFeatures];
        }

        static Tensor<Half> ToHalf(Tensor<double> source)
            => new(source.Shape.ToArray(), new Vector<Half>(Array.ConvertAll(source.ToArray(), v => (Half)(float)v)));

        var reference = UnitCameraView();
        var halfView = new ConcertoPairedView<Half>
        {
            Extrinsics = ToHalf(reference.Extrinsics),
            Intrinsics = ToHalf(reference.Intrinsics),
            DepthMap = ToHalf(reference.DepthMap),
            PatchGridSize = reference.PatchGridSize,
            ImagePatchFeatures = ToHalf(reference.ImagePatchFeatures),
        };
        var halfFeatures = ToHalf(doubleFeatures);
        Tensor<Half> halfGrad;
        using (var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<Half>())
        {
            var loss = ConcertoCrossModalObjective<Half>.ComputeTapeLoss(
                halfFeatures, ToHalf(Coordinates()), new List<ConcertoPairedView<Half>> { halfView }, 0.5);
            halfGrad = tape.ComputeGradients(loss, new[] { halfFeatures })[halfFeatures];
        }

        double[] expected = doubleGrad.ToArray();
        Half[] actual = halfGrad.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], (double)(float)actual[i], 5e-3);
        }
    }
}
