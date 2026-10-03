using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;
using AiDotNet.Tensors.Helpers;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for Instance segmentation models:
/// YOLOv8Seg, YOLOv9Seg, YOLO11Seg, YOLO26Seg, YOLOv12Seg, MaskRCNN, SOLOv2, YOLOSeg.
/// </summary>
public class InstanceSegmentationIntegrationTests
{
    private static NeuralNetworkArchitecture<double> Arch(int h = 32, int w = 32, int d = 3)
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, h, w, d, 0);

    private static Tensor<double> Rand(params int[] shape)
    {
        int total = 1; foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #region YOLOv8Seg

    [Fact(Timeout = 120000)]
    public async Task YOLOv8Seg_Construction_Succeeds()
    {
        var model = new YOLOv8Seg<double>(Arch(), options: new YOLOv8SegOptions { ModelSize = YOLOv8SegModelSize.N });
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv8Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv8Seg<double>(Arch(), options: new YOLOv8SegOptions { ModelSize = YOLOv8SegModelSize.N });
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv8Seg_Train_DoesNotThrow()
    {
        var model = new YOLOv8Seg<double>(Arch(), options: new YOLOv8SegOptions { ModelSize = YOLOv8SegModelSize.N });
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape.ToArray());
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv8Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv8Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLOv9Seg

    [Fact(Timeout = 120000)]
    public async Task YOLOv9Seg_Construction_Succeeds()
    {
        var model = new YOLOv9Seg<double>(Arch(), options: new YOLOv9SegOptions { ModelSize = YOLOv9SegModelSize.C });
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv9Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv9Seg<double>(Arch(), options: new YOLOv9SegOptions { ModelSize = YOLOv9SegModelSize.C });
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv9Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv9Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLO11Seg

    [Fact(Timeout = 120000)]
    public async Task YOLO11Seg_Construction_Succeeds()
    {
        var model = new YOLO11Seg<double>(Arch(), options: new YOLO11SegOptions { ModelSize = YOLO11SegModelSize.N });
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO11Seg_Predict_ReturnsOutput()
    {
        var model = new YOLO11Seg<double>(Arch(), options: new YOLO11SegOptions { ModelSize = YOLO11SegModelSize.N });
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO11Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLO11Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLO26Seg

    [Fact(Timeout = 120000)]
    public async Task YOLO26Seg_Construction_Succeeds()
    {
        var model = new YOLO26Seg<double>(Arch(), options: new YOLO26SegOptions { ModelSize = YOLO26SegModelSize.N });
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO26Seg_Predict_ReturnsOutput()
    {
        var model = new YOLO26Seg<double>(Arch(), options: new YOLO26SegOptions { ModelSize = YOLO26SegModelSize.N });
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLO26Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLO26Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLOv12Seg

    [Fact(Timeout = 120000)]
    public async Task YOLOv12Seg_Construction_Succeeds()
    {
        var model = new YOLOv12Seg<double>(Arch(), options: new YOLOv12SegOptions { ModelSize = YOLOv12SegModelSize.N });
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv12Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv12Seg<double>(Arch(), options: new YOLOv12SegOptions { ModelSize = YOLOv12SegModelSize.N });
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOv12Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv12Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MaskRCNN

    [Fact(Timeout = 120000)]
    public async Task MaskRCNN_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.MaskRCNN,
            InputSize = new[] { 32, 32 }
        };
        var model = new MaskRCNN<double>(options);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskRCNN_RpnLaysAnchorsOnEveryPyramidLevelAtItsOwnStride()
    {
        // Regression for #2172: the RPN read only P2 (stride 4) but laid its anchors out at stride 16,
        // so every anchor centre sat at 4x its true position, and P3-P6 never proposed anything.
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.MaskRCNN,
            InputSize = new[] { 64, 64 }
        };
        var model = new MaskRCNN<double>(options);
        const int imageSize = 64;

        var (fpnFeatures, proposals, anchors, levelAnchorCounts) = model.ProposeRegions(Rand(1, 3, imageSize, imageSize), 1000);

        int[] strides = { 4, 8, 16, 32, 64 };
        int[] anchorSizes = { 32, 64, 128, 256, 512 };
        Assert.Equal(imageSize / 4, fpnFeatures[0].Shape[2]);
        Assert.Equal(strides.Length, levelAnchorCounts.Length);
        Assert.Equal(anchors.Count, levelAnchorCounts.Sum());

        int start = 0;
        for (int level = 0; level < strides.Length; level++)
        {
            int stride = strides[level];
            int cells = imageSize / stride;
            Assert.Equal(cells * cells * 3, levelAnchorCounts[level]);

            for (int i = start; i < start + levelAnchorCounts[level]; i++)
            {
                var a = anchors[i];
                double cx = (a.X1 + a.X2) / 2 / stride - 0.5;
                double cy = (a.Y1 + a.Y2) / 2 / stride - 0.5;
                Assert.InRange(cx, 0, cells - 1);
                Assert.InRange(cy, 0, cells - 1);
                Assert.Equal(Math.Round(cx), cx, 9);
                Assert.Equal(Math.Round(cy), cy, 9);
                Assert.Equal(anchorSizes[level], Math.Sqrt((a.X2 - a.X1) * (a.Y2 - a.Y1)), 6);
            }

            start += levelAnchorCounts[level];
        }

        Assert.Equal(4, proposals.Shape[1]);
        Assert.True(proposals.Shape[0] > 0, "The RPN produced no proposals, so the bounds checks below would be vacuous.");
        Assert.True(proposals.Shape[0] <= 1000, $"{proposals.Shape[0]} proposals exceed the requested 1000.");
        for (int p = 0; p < proposals.Shape[0]; p++)
        {
            Assert.InRange(proposals[p, 0], 0, proposals[p, 2]);
            Assert.InRange(proposals[p, 1], 0, proposals[p, 3]);
            Assert.InRange(proposals[p, 2], proposals[p, 0], imageSize);
            Assert.InRange(proposals[p, 3], proposals[p, 1], imageSize);
        }

        // The cap keeps the highest-scoring proposals: with a cap below the uncapped count, the result
        // is exactly the first rows of the uncapped ranking.
        int cap = Math.Max(1, proposals.Shape[0] / 2);
        Assert.True(cap < proposals.Shape[0], "Need more than one proposal to exercise the cap.");
        var (_, capped, _, _) = model.ProposeRegions(Rand(1, 3, imageSize, imageSize), cap);
        Assert.Equal(new[] { cap, 4 }, capped.Shape.ToArray());
        for (int p = 0; p < cap; p++)
            for (int k = 0; k < 4; k++)
                Assert.Equal(proposals[p, k], capped[p, k]);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskRCNN_Segment_ReturnsResult()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.MaskRCNN,
            InputSize = new[] { 32, 32 }
        };
        var model = new MaskRCNN<double>(options);
        var result = model.Segment(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task MaskRCNN_GetParameterCount_ReturnsPositive()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.MaskRCNN,
            InputSize = new[] { 32, 32 }
        };
        var model = new MaskRCNN<double>(options);
        Assert.True(model.GetParameterCount() >= 0);
    }

    #endregion

    #region SOLOv2

    [Fact(Timeout = 120000)]
    public async Task SOLOv2_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.SOLOv2,
            InputSize = new[] { 32, 32 }
        };
        var model = new SOLOv2<double>(options);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task SOLOv2_Segment_ReturnsResult()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.SOLOv2,
            InputSize = new[] { 32, 32 }
        };
        var model = new SOLOv2<double>(options);
        var result = model.Segment(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task SOLOv2_GetParameterCount_ReturnsPositive()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.SOLOv2,
            InputSize = new[] { 32, 32 }
        };
        var model = new SOLOv2<double>(options);
        Assert.True(model.GetParameterCount() >= 0);
    }

    #endregion

    #region YOLOSeg

    [Fact(Timeout = 120000)]
    public async Task YOLOSeg_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.YOLOSeg,
            InputSize = new[] { 32, 32 }
        };
        var model = new YOLOSeg<double>(options);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOSeg_Segment_ReturnsResult()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.YOLOSeg,
            InputSize = new[] { 32, 32 }
        };
        var model = new YOLOSeg<double>(options);
        var result = model.Segment(Rand(1, 3, 32, 32));
        Assert.NotNull(result);
    }

    [Fact(Timeout = 120000)]
    public async Task YOLOSeg_GetParameterCount_ReturnsPositive()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.YOLOSeg,
            InputSize = new[] { 32, 32 }
        };
        var model = new YOLOSeg<double>(options);
        Assert.True(model.GetParameterCount() >= 0);
    }

    #endregion
}
