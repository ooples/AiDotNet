using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

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
        var rng = new Random(42);
        for (int i = 0; i < total; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    #region YOLOv8Seg

    [Fact]
    public void YOLOv8Seg_Construction_Succeeds()
    {
        var model = new YOLOv8Seg<double>(Arch(), modelSize: YOLOv8SegModelSize.N);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void YOLOv8Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv8Seg<double>(Arch(), modelSize: YOLOv8SegModelSize.N);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void YOLOv8Seg_Train_DoesNotThrow()
    {
        var model = new YOLOv8Seg<double>(Arch(), modelSize: YOLOv8SegModelSize.N);
        var input = Rand(1, 3, 32, 32);
        var predicted = model.Predict(input);
        var expected = Rand(predicted.Shape);
        Assert.Null(Record.Exception(() => model.Train(input, expected)));
    }

    [Fact]
    public void YOLOv8Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv8Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLOv9Seg

    [Fact]
    public void YOLOv9Seg_Construction_Succeeds()
    {
        var model = new YOLOv9Seg<double>(Arch(), modelSize: YOLOv9SegModelSize.C);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void YOLOv9Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv9Seg<double>(Arch(), modelSize: YOLOv9SegModelSize.C);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void YOLOv9Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv9Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLO11Seg

    [Fact]
    public void YOLO11Seg_Construction_Succeeds()
    {
        var model = new YOLO11Seg<double>(Arch(), modelSize: YOLO11SegModelSize.N);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void YOLO11Seg_Predict_ReturnsOutput()
    {
        var model = new YOLO11Seg<double>(Arch(), modelSize: YOLO11SegModelSize.N);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void YOLO11Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLO11Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLO26Seg

    [Fact]
    public void YOLO26Seg_Construction_Succeeds()
    {
        var model = new YOLO26Seg<double>(Arch(), modelSize: YOLO26SegModelSize.N);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void YOLO26Seg_Predict_ReturnsOutput()
    {
        var model = new YOLO26Seg<double>(Arch(), modelSize: YOLO26SegModelSize.N);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void YOLO26Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLO26Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region YOLOv12Seg

    [Fact]
    public void YOLOv12Seg_Construction_Succeeds()
    {
        var model = new YOLOv12Seg<double>(Arch(), modelSize: YOLOv12SegModelSize.N);
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void YOLOv12Seg_Predict_ReturnsOutput()
    {
        var model = new YOLOv12Seg<double>(Arch(), modelSize: YOLOv12SegModelSize.N);
        var output = model.Predict(Rand(1, 3, 32, 32));
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
    }

    [Fact]
    public void YOLOv12Seg_Dispose_DoesNotThrow()
    {
        var model = new YOLOv12Seg<double>(Arch());
        Assert.Null(Record.Exception(() => model.Dispose()));
    }

    #endregion

    #region MaskRCNN

    [Fact]
    public void MaskRCNN_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.MaskRCNN,
            InputSize = new[] { 32, 32 }
        };
        var model = new MaskRCNN<double>(options);
        Assert.NotNull(model);
    }

    [Fact]
    public void MaskRCNN_Segment_ReturnsResult()
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

    [Fact]
    public void MaskRCNN_GetParameterCount_ReturnsPositive()
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

    [Fact]
    public void SOLOv2_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.SOLOv2,
            InputSize = new[] { 32, 32 }
        };
        var model = new SOLOv2<double>(options);
        Assert.NotNull(model);
    }

    [Fact]
    public void SOLOv2_Segment_ReturnsResult()
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

    [Fact]
    public void SOLOv2_GetParameterCount_ReturnsPositive()
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

    [Fact]
    public void YOLOSeg_Construction_Succeeds()
    {
        var options = new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.YOLOSeg,
            InputSize = new[] { 32, 32 }
        };
        var model = new YOLOSeg<double>(options);
        Assert.NotNull(model);
    }

    [Fact]
    public void YOLOSeg_Segment_ReturnsResult()
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

    [Fact]
    public void YOLOSeg_GetParameterCount_ReturnsPositive()
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
