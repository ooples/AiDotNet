using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.ComputerVision.Tracking;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Integration tests for the ComputerVision module.
/// Tests BoundingBox, Detection, NMS, Loss Functions, and Tracking components.
/// </summary>
public class ComputerVisionIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region BoundingBox Tests

    [Fact]
    public void BoundingBox_XYXYFormat_CreatesCorrectly()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        Assert.Equal(10, box.X1);
        Assert.Equal(20, box.Y1);
        Assert.Equal(50, box.X2);
        Assert.Equal(80, box.Y2);
        Assert.Equal(BoundingBoxFormat.XYXY, box.Format);
    }

    [Fact]
    public void BoundingBox_XYXYToXYXY_ReturnsCorrectCoordinates()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        var (xMin, yMin, xMax, yMax) = box.ToXYXY();

        Assert.Equal(10, xMin, Tolerance);
        Assert.Equal(20, yMin, Tolerance);
        Assert.Equal(50, xMax, Tolerance);
        Assert.Equal(80, yMax, Tolerance);
    }

    [Fact]
    public void BoundingBox_XYWHToXYXY_ConvertsCorrectly()
    {
        // XYWH: (10, 20, 40, 60) -> XYXY: (10, 20, 50, 80)
        var box = new BoundingBox<double>(10, 20, 40, 60, BoundingBoxFormat.XYWH);

        var (xMin, yMin, xMax, yMax) = box.ToXYXY();

        Assert.Equal(10, xMin, Tolerance);
        Assert.Equal(20, yMin, Tolerance);
        Assert.Equal(50, xMax, Tolerance); // 10 + 40
        Assert.Equal(80, yMax, Tolerance); // 20 + 60
    }

    [Fact]
    public void BoundingBox_CXCYWHToXYXY_ConvertsCorrectly()
    {
        // CXCYWH: (30, 50, 40, 60) -> center at (30, 50), w=40, h=60
        // XYXY: (10, 20, 50, 80)
        var box = new BoundingBox<double>(30, 50, 40, 60, BoundingBoxFormat.CXCYWH);

        var (xMin, yMin, xMax, yMax) = box.ToXYXY();

        Assert.Equal(10, xMin, Tolerance); // 30 - 40/2
        Assert.Equal(20, yMin, Tolerance); // 50 - 60/2
        Assert.Equal(50, xMax, Tolerance); // 30 + 40/2
        Assert.Equal(80, yMax, Tolerance); // 50 + 60/2
    }

    [Fact]
    public void BoundingBox_ToXYWH_ConvertsCorrectly()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        var (x, y, width, height) = box.ToXYWH();

        Assert.Equal(10, x, Tolerance);
        Assert.Equal(20, y, Tolerance);
        Assert.Equal(40, width, Tolerance);  // 50 - 10
        Assert.Equal(60, height, Tolerance); // 80 - 20
    }

    [Fact]
    public void BoundingBox_ToCXCYWH_ConvertsCorrectly()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        var (cx, cy, width, height) = box.ToCXCYWH();

        Assert.Equal(30, cx, Tolerance);     // 10 + 40/2
        Assert.Equal(50, cy, Tolerance);     // 20 + 60/2
        Assert.Equal(40, width, Tolerance);  // 50 - 10
        Assert.Equal(60, height, Tolerance); // 80 - 20
    }

    [Fact]
    public void BoundingBox_ToYOLO_WithImageDimensions_ConvertsCorrectly()
    {
        var box = new BoundingBox<double>(100, 100, 300, 400, BoundingBoxFormat.XYXY)
        {
            ImageWidth = 800,
            ImageHeight = 600
        };

        var (cx, cy, width, height) = box.ToYOLO();

        Assert.Equal(0.25, cx, Tolerance);           // 200 / 800
        Assert.Equal(0.41666666, cy, 1e-5);          // 250 / 600
        Assert.Equal(0.25, width, Tolerance);        // 200 / 800
        Assert.Equal(0.5, height, Tolerance);        // 300 / 600
    }

    [Fact]
    public void BoundingBox_ToYOLO_WithoutImageDimensions_ThrowsException()
    {
        var box = new BoundingBox<double>(100, 100, 300, 400, BoundingBoxFormat.XYXY);

        Assert.Throws<InvalidOperationException>(() => box.ToYOLO());
    }

    [Fact]
    public void BoundingBox_Area_CalculatesCorrectly()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        double area = box.Area();

        Assert.Equal(2400, area, Tolerance); // 40 * 60
    }

    [Fact]
    public void BoundingBox_Area_ForZeroSizeBox_ReturnsZero()
    {
        var box = new BoundingBox<double>(10, 20, 10, 20, BoundingBoxFormat.XYXY);

        double area = box.Area();

        Assert.Equal(0, area, Tolerance);
    }

    [Fact]
    public void BoundingBox_IoU_IdenticalBoxes_ReturnsOne()
    {
        var box1 = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        double iou = box1.IoU(box2);

        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void BoundingBox_IoU_NoOverlap_ReturnsZero()
    {
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY);

        double iou = box1.IoU(box2);

        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void BoundingBox_IoU_PartialOverlap_ReturnsCorrectValue()
    {
        // Box1: (0, 0, 20, 20), area = 400
        // Box2: (10, 10, 30, 30), area = 400
        // Intersection: (10, 10, 20, 20), area = 100
        // Union: 400 + 400 - 100 = 700
        // IoU: 100 / 700 = 0.142857...
        var box1 = new BoundingBox<double>(0, 0, 20, 20, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(10, 10, 30, 30, BoundingBoxFormat.XYXY);

        double iou = box1.IoU(box2);

        Assert.Equal(100.0 / 700.0, iou, Tolerance);
    }

    [Fact]
    public void BoundingBox_Clip_ClipsToImageBoundaries()
    {
        var box = new BoundingBox<double>(-10, -20, 150, 180, BoundingBoxFormat.XYXY);

        box.Clip(100, 100);

        var (xMin, yMin, xMax, yMax) = box.ToXYXY();
        Assert.Equal(0, xMin, Tolerance);
        Assert.Equal(0, yMin, Tolerance);
        Assert.Equal(100, xMax, Tolerance);
        Assert.Equal(100, yMax, Tolerance);
    }

    [Fact]
    public void BoundingBox_IsValid_ValidBox_ReturnsTrue()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        Assert.True(box.IsValid());
    }

    [Fact]
    public void BoundingBox_IsValid_ZeroSizeBox_ReturnsFalse()
    {
        var box = new BoundingBox<double>(10, 20, 10, 20, BoundingBoxFormat.XYXY);

        Assert.False(box.IsValid());
    }

    [Fact]
    public void BoundingBox_IsValid_InvertedBox_ReturnsFalse()
    {
        var box = new BoundingBox<double>(50, 80, 10, 20, BoundingBoxFormat.XYXY);

        Assert.False(box.IsValid());
    }

    [Fact]
    public void BoundingBox_Clone_CreatesDeepCopy()
    {
        var original = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY)
        {
            ClassIndex = 5,
            ClassName = "person",
            Confidence = 0.95,
            ImageWidth = 640,
            ImageHeight = 480
        };

        var clone = original.Clone();

        Assert.Equal(original.X1, clone.X1);
        Assert.Equal(original.X2, clone.X2);
        Assert.Equal(original.ClassIndex, clone.ClassIndex);
        Assert.Equal(original.ClassName, clone.ClassName);
        Assert.NotSame(original, clone);
    }

    #endregion

    #region Detection Tests

    [Fact]
    public void Detection_Constructor_CreatesCorrectly()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var detection = new Detection<double>(box, 1, 0.95, "person");

        Assert.Equal(box, detection.Box);
        Assert.Equal(1, detection.ClassId);
        Assert.Equal(0.95, detection.Confidence);
        Assert.Equal("person", detection.ClassName);
    }

    [Fact]
    public void Detection_Area_ReturnsBoxArea()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var detection = new Detection<double>(box, 1, 0.95, "person");

        Assert.Equal(2400, detection.Area, Tolerance);
    }

    [Fact]
    public void Detection_Center_ReturnsCorrectCoordinates()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var detection = new Detection<double>(box, 1, 0.95, "person");

        Assert.Equal(30, detection.CenterX, Tolerance);
        Assert.Equal(50, detection.CenterY, Tolerance);
    }

    [Fact]
    public void Detection_ToString_ReturnsFormattedString()
    {
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var detection = new Detection<double>(box, 1, 0.95, "person");

        string str = detection.ToString();

        Assert.Contains("person", str);
        Assert.Contains("95", str); // 95%
    }

    #endregion

    #region DetectionResult Tests

    [Fact]
    public void DetectionResult_FilterByConfidence_FiltersCorrectly()
    {
        var result = CreateSampleDetectionResult();

        var filtered = result.FilterByConfidence(0.6);

        Assert.Equal(2, filtered.Count); // Only 0.9 and 0.7 remain
        Assert.All(filtered.Detections, d => Assert.True(d.Confidence >= 0.6));
    }

    [Fact]
    public void DetectionResult_FilterByClass_FiltersCorrectly()
    {
        var result = CreateSampleDetectionResult();

        var filtered = result.FilterByClass(0, 2);

        Assert.All(filtered.Detections, d => Assert.True(d.ClassId == 0 || d.ClassId == 2));
    }

    [Fact]
    public void DetectionResult_SortByConfidence_SortsDescending()
    {
        var result = CreateSampleDetectionResult();

        var sorted = result.SortByConfidence();

        for (int i = 0; i < sorted.Count - 1; i++)
        {
            Assert.True(sorted.Detections[i].Confidence >= sorted.Detections[i + 1].Confidence);
        }
    }

    [Fact]
    public void DetectionResult_TopN_ReturnsTopDetections()
    {
        var result = CreateSampleDetectionResult();

        var top2 = result.TopN(2);

        Assert.Equal(2, top2.Count);
        Assert.Equal(0.9, top2.Detections[0].Confidence, Tolerance);
        Assert.Equal(0.7, top2.Detections[1].Confidence, Tolerance);
    }

    [Fact]
    public void DetectionResult_Count_ReturnsCorrectCount()
    {
        var result = CreateSampleDetectionResult();

        Assert.Equal(4, result.Count);
    }

    private static DetectionResult<double> CreateSampleDetectionResult()
    {
        return new DetectionResult<double>
        {
            ImageWidth = 640,
            ImageHeight = 480,
            Detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9, "cat"),
                new(new BoundingBox<double>(60, 60, 100, 100, BoundingBoxFormat.XYXY), 1, 0.7, "dog"),
                new(new BoundingBox<double>(120, 120, 180, 180, BoundingBoxFormat.XYXY), 0, 0.5, "cat"),
                new(new BoundingBox<double>(200, 200, 280, 280, BoundingBoxFormat.XYXY), 2, 0.3, "bird")
            }
        };
    }

    #endregion

    #region DetectionStatistics Tests

    [Fact]
    public void DetectionStatistics_FromResult_ComputesCorrectly()
    {
        var result = CreateSampleDetectionResult();

        var stats = DetectionStatistics<double>.FromResult(result);

        Assert.Equal(4, stats.TotalDetections);
        Assert.Equal(2, stats.CountByClass[0]); // Two cats
        Assert.Equal(1, stats.CountByClass[1]); // One dog
        Assert.Equal(1, stats.CountByClass[2]); // One bird
        Assert.Equal((0.9 + 0.7 + 0.5 + 0.3) / 4, stats.AverageConfidence, Tolerance);
    }

    [Fact]
    public void DetectionStatistics_FromResult_EmptyResult_ReturnsZeros()
    {
        var result = new DetectionResult<double>();

        var stats = DetectionStatistics<double>.FromResult(result);

        Assert.Equal(0, stats.TotalDetections);
        Assert.Empty(stats.CountByClass);
    }

    #endregion

    #region BatchDetectionResult Tests

    [Fact]
    public void BatchDetectionResult_Properties_ComputeCorrectly()
    {
        var batch = new BatchDetectionResult<double>
        {
            Results = new List<DetectionResult<double>>
            {
                CreateSampleDetectionResult(),
                CreateSampleDetectionResult()
            },
            TotalInferenceTime = TimeSpan.FromMilliseconds(100)
        };

        Assert.Equal(2, batch.BatchSize);
        Assert.Equal(8, batch.TotalDetections);
        Assert.Equal(TimeSpan.FromMilliseconds(50), batch.AverageInferenceTime);
    }

    [Fact]
    public void BatchDetectionResult_Indexer_ReturnsCorrectResult()
    {
        var result1 = new DetectionResult<double> { ImageWidth = 100 };
        var result2 = new DetectionResult<double> { ImageWidth = 200 };

        var batch = new BatchDetectionResult<double>
        {
            Results = new List<DetectionResult<double>> { result1, result2 }
        };

        Assert.Equal(100, batch[0].ImageWidth);
        Assert.Equal(200, batch[1].ImageWidth);
    }

    #endregion

    #region NMS Tests

    [Fact]
    public void NMS_Apply_RemovesDuplicates()
    {
        var nms = new NMS<double>();
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(5, 5, 105, 105, BoundingBoxFormat.XYXY), 0, 0.8), // Overlaps with first
            new(new BoundingBox<double>(200, 200, 300, 300, BoundingBoxFormat.XYXY), 0, 0.7) // No overlap
        };

        var result = nms.Apply(detections, 0.5);

        Assert.Equal(2, result.Count);
        Assert.Equal(0.9, result[0].Confidence);
        Assert.Equal(0.7, result[1].Confidence);
    }

    [Fact]
    public void NMS_Apply_EmptyList_ReturnsEmpty()
    {
        var nms = new NMS<double>();

        var result = nms.Apply(new List<Detection<double>>(), 0.5);

        Assert.Empty(result);
    }

    [Fact]
    public void NMS_Apply_SingleDetection_ReturnsSameDetection()
    {
        var nms = new NMS<double>();
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9)
        };

        var result = nms.Apply(detections, 0.5);

        Assert.Single(result);
        Assert.Equal(0.9, result[0].Confidence);
    }

    [Fact]
    public void NMS_ApplyClassAware_SeparatesClasses()
    {
        var nms = new NMS<double>();
        var detections = new List<Detection<double>>
        {
            // Class 0 - overlapping
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(5, 5, 105, 105, BoundingBoxFormat.XYXY), 0, 0.8),
            // Class 1 - overlapping with class 0 boxes but different class
            new(new BoundingBox<double>(10, 10, 110, 110, BoundingBoxFormat.XYXY), 1, 0.85)
        };

        var result = nms.ApplyClassAware(detections, 0.5);

        // Both class 0 (0.9) and class 1 (0.85) should remain
        Assert.Equal(2, result.Count);
        var class0 = result.Where(d => d.ClassId == 0).ToList();
        var class1 = result.Where(d => d.ClassId == 1).ToList();
        Assert.Single(class0);
        Assert.Single(class1);
    }

    [Fact]
    public void NMS_ApplyBatched_ProcessesMultipleImages()
    {
        var nms = new NMS<double>();
        var image1Detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(5, 5, 55, 55, BoundingBoxFormat.XYXY), 0, 0.8)
        };
        var image2Detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(100, 100, 150, 150, BoundingBoxFormat.XYXY), 1, 0.7)
        };

        var result = nms.ApplyBatched(
            new List<List<Detection<double>>> { image1Detections, image2Detections },
            0.5);

        Assert.Equal(2, result.Count);
        Assert.Single(result[0]); // Image 1: one remaining after NMS
        Assert.Single(result[1]); // Image 2: one detection
    }

    [Fact]
    public void NMS_ComputeIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeIoU_NoOverlap_ReturnsZero()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(100, 100, 150, 150, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeGIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);

        double giou = nms.ComputeGIoU(box1, box2);

        Assert.Equal(1.0, giou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeGIoU_NoOverlap_ReturnsNegativeOrZero()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(100, 100, 150, 150, BoundingBoxFormat.XYXY);

        double giou = nms.ComputeGIoU(box1, box2);

        // GIoU can be negative when boxes don't overlap
        Assert.True(giou <= 0);
    }

    [Fact]
    public void NMS_ComputeDIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);

        double diou = nms.ComputeDIoU(box1, box2);

        Assert.Equal(1.0, diou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeCIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);

        double ciou = nms.ComputeCIoU(box1, box2);

        Assert.Equal(1.0, ciou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeCIoU_DifferentAspectRatios_PenalizesCorrectly()
    {
        var nms = new NMS<double>();
        // Same position and IoU, but different aspect ratios
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY); // Square
        var box2 = new BoundingBox<double>(0, 0, 100, 50, BoundingBoxFormat.XYXY);  // Wide

        double ciou = nms.ComputeCIoU(box1, box2);

        // CIoU should be less than DIoU due to aspect ratio penalty
        double diou = nms.ComputeDIoU(box1, box2);
        Assert.True(ciou <= diou);
    }

    #endregion

    #region GIoU Loss Tests

    [Fact]
    public void GIoULoss_CalculateLoss_IdenticalBoxes_ReturnsZero()
    {
        var loss = new GIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });

        double lossValue = loss.CalculateLoss(predicted, actual);

        Assert.Equal(0.0, lossValue, 0.001);
    }

    [Fact]
    public void GIoULoss_CalculateLoss_DifferentBoxes_ReturnsPositive()
    {
        var loss = new GIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 50.0, 50.0, 150.0, 150.0 });

        double lossValue = loss.CalculateLoss(predicted, actual);

        Assert.True(lossValue > 0);
        Assert.True(lossValue <= 2); // GIoU loss ranges from 0 to 2
    }

    [Fact]
    public void GIoULoss_CalculateLoss_MultipleBoxes_ComputesMean()
    {
        var loss = new GIoULoss<double>();
        // Two boxes: first identical, second different
        var predicted = new Vector<double>(new[]
        {
            0.0, 0.0, 100.0, 100.0,
            200.0, 200.0, 300.0, 300.0
        });
        var actual = new Vector<double>(new[]
        {
            0.0, 0.0, 100.0, 100.0,  // Identical
            250.0, 250.0, 350.0, 350.0  // Different
        });

        double lossValue = loss.CalculateLoss(predicted, actual);

        // Loss should be mean of 0 and positive value
        Assert.True(lossValue > 0);
    }

    [Fact]
    public void GIoULoss_CalculateLoss_InvalidVectorLength_ThrowsException()
    {
        var loss = new GIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0 }); // Not multiple of 4
        var actual = new Vector<double>(new[] { 0.0, 0.0, 100.0 });

        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void GIoULoss_CalculateDerivative_ReturnsGradient()
    {
        var loss = new GIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 10.0, 10.0, 110.0, 110.0 });

        var gradient = loss.CalculateDerivative(predicted, actual);

        Assert.Equal(4, gradient.Length);
        // Gradient should not be all zeros when boxes differ
        Assert.True(gradient.Any(g => Math.Abs(g) > 1e-10));
    }

    [Fact]
    public void GIoULoss_CalculateLossForBox_ComputesCorrectly()
    {
        var loss = new GIoULoss<double>();
        var predicted = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var target = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);

        double lossValue = loss.CalculateLossForBox(predicted, target);

        Assert.Equal(0.0, lossValue, Tolerance);
    }

    [Fact]
    public void GIoULoss_TensorInput_ComputesCorrectly()
    {
        var loss = new GIoULoss<double>();
        // Shape: [batch=1, num_boxes=1, 4]
        var predicted = new Tensor<double>(new[] { 1, 1, 4 });
        predicted[0, 0, 0] = 0;
        predicted[0, 0, 1] = 0;
        predicted[0, 0, 2] = 100;
        predicted[0, 0, 3] = 100;

        var targets = new Tensor<double>(new[] { 1, 1, 4 });
        targets[0, 0, 0] = 0;
        targets[0, 0, 1] = 0;
        targets[0, 0, 2] = 100;
        targets[0, 0, 3] = 100;

        double lossValue = loss.CalculateLoss(predicted, targets);

        Assert.Equal(0.0, lossValue, 0.001);
    }

    #endregion

    #region Tracking Tests

    [Fact]
    public void SORT_Update_CreatesNewTracks()
    {
        var options = new TrackingOptions<double>();
        var tracker = new SORT<double>(options);

        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        };

        var result = tracker.Update(detections);

        Assert.Equal(1, result.FrameNumber);
        // Track might be tentative initially
        Assert.NotNull(result.Tracks);
    }

    [Fact]
    public void SORT_Update_TracksAcrossFrames()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 1,
            IouThreshold = 0.3
        };
        var tracker = new SORT<double>(options);

        // Frame 1
        var detections1 = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        };
        var result1 = tracker.Update(detections1);

        // Frame 2 - slightly moved
        var detections2 = new List<Detection<double>>
        {
            new(new BoundingBox<double>(5, 5, 55, 55, BoundingBoxFormat.XYXY), 0, 0.85)
        };
        var result2 = tracker.Update(detections2);

        Assert.Equal(2, result2.FrameNumber);
    }

    [Fact]
    public void SORT_Update_AssignsTrackIds()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 1
        };
        var tracker = new SORT<double>(options);

        // Multiple detections
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(200, 200, 250, 250, BoundingBoxFormat.XYXY), 1, 0.8)
        };

        var result = tracker.Update(detections);

        // Track IDs should be unique if tracks are confirmed
        var trackIds = result.Tracks.Select(t => t.TrackId).ToList();
        Assert.Equal(trackIds.Count, trackIds.Distinct().Count());
    }

    [Fact]
    public void SORT_Reset_ClearsState()
    {
        var options = new TrackingOptions<double> { MinHits = 1 };
        var tracker = new SORT<double>(options);

        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        };
        tracker.Update(detections);

        tracker.Reset();

        var result = tracker.Update(detections);
        Assert.Equal(1, result.FrameNumber);
    }

    [Fact]
    public void SORT_Name_ReturnsSORTName()
    {
        var options = new TrackingOptions<double>();
        var tracker = new SORT<double>(options);

        Assert.Equal("SORT", tracker.Name);
    }

    [Fact]
    public void SORT_GetConfirmedTracks_ReturnsOnlyConfirmed()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 3 // Requires 3 hits to confirm
        };
        var tracker = new SORT<double>(options);

        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        };

        // First update - track is tentative
        tracker.Update(detections);
        var confirmed = tracker.GetConfirmedTracks();

        // Should be empty as track needs 3 hits
        Assert.Empty(confirmed);
    }

    #endregion

    #region Track Tests

    [Fact]
    public void Track_Constructor_InitializesCorrectly()
    {
        var box = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var track = new Track<double>(1, box, 0, 0.9);

        Assert.Equal(1, track.TrackId);
        Assert.Equal(box, track.Box);
        Assert.Equal(0, track.ClassId);
        Assert.Equal(0.9, track.Confidence);
        Assert.Equal(TrackState.Tentative, track.State);
        Assert.Equal(1, track.Hits);
        Assert.Equal(1, track.Age);
    }

    [Fact]
    public void TrackState_Enum_HasExpectedValues()
    {
        Assert.Equal(0, (int)TrackState.Tentative);
        Assert.Equal(1, (int)TrackState.Confirmed);
        Assert.Equal(2, (int)TrackState.Lost);
        Assert.Equal(3, (int)TrackState.Deleted);
    }

    #endregion

    #region TrackingOptions Tests

    [Fact]
    public void TrackingOptions_DefaultValues_AreReasonable()
    {
        var options = new TrackingOptions<double>();

        Assert.Equal(30, options.MaxAge);
        Assert.Equal(3, options.MinHits);
        Assert.False(options.UseAppearance);
    }

    [Fact]
    public void TrackingOptions_CanBeCustomized()
    {
        var options = new TrackingOptions<double>
        {
            MaxAge = 50,
            MinHits = 5,
            UseAppearance = true,
            AppearanceWeight = 0.7,
            MaxCosineDistance = 0.5
        };

        Assert.Equal(50, options.MaxAge);
        Assert.Equal(5, options.MinHits);
        Assert.True(options.UseAppearance);
        Assert.Equal(0.7, options.AppearanceWeight, Tolerance);
        Assert.Equal(0.5, options.MaxCosineDistance, Tolerance);
    }

    #endregion

    #region TrackingResult Tests

    [Fact]
    public void TrackingResult_DefaultValues_AreEmpty()
    {
        var result = new TrackingResult<double>();

        Assert.Empty(result.Tracks);
        Assert.Equal(0, result.FrameNumber);
        Assert.Equal(TimeSpan.Zero, result.TrackingTime);
    }

    [Fact]
    public void TrackingResult_CanHoldTracks()
    {
        var box = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var result = new TrackingResult<double>
        {
            Tracks = new List<Track<double>>
            {
                new(1, box, 0, 0.9),
                new(2, box, 1, 0.8)
            },
            FrameNumber = 10,
            TrackingTime = TimeSpan.FromMilliseconds(5)
        };

        Assert.Equal(2, result.Tracks.Count);
        Assert.Equal(10, result.FrameNumber);
        Assert.Equal(TimeSpan.FromMilliseconds(5), result.TrackingTime);
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void Detection_Pipeline_FromBoxToFiltered_WorksEndToEnd()
    {
        // Create bounding boxes
        var boxes = new List<BoundingBox<double>>
        {
            new(10, 10, 60, 60, BoundingBoxFormat.XYXY),
            new(15, 15, 65, 65, BoundingBoxFormat.XYXY), // Overlaps with first
            new(200, 200, 250, 250, BoundingBoxFormat.XYXY),
            new(300, 300, 350, 350, BoundingBoxFormat.XYXY) { ClassIndex = 1 }
        };

        // Create detections
        var detections = boxes.Select((b, i) =>
            new Detection<double>(b, b.ClassIndex, 0.9 - i * 0.1, $"class_{b.ClassIndex}")).ToList();

        // Create detection result
        var result = new DetectionResult<double>
        {
            Detections = detections,
            ImageWidth = 640,
            ImageHeight = 480
        };

        // Apply NMS
        var nms = new NMS<double>();
        var nmsFiltered = nms.ApplyClassAware(result.Detections, 0.5);

        // Filter by confidence
        var confidenceFiltered = nmsFiltered.Where(d => d.Confidence >= 0.7).ToList();

        // Verify pipeline
        Assert.True(nmsFiltered.Count < detections.Count); // NMS removed some
        Assert.All(confidenceFiltered, d => Assert.True(d.Confidence >= 0.7));
    }

    [Fact]
    public void Tracking_Pipeline_MultiFrame_WorksEndToEnd()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 2,
            IouThreshold = 0.3
        };
        var tracker = new SORT<double>(options);

        // Simulate object moving across frames
        for (int frame = 0; frame < 5; frame++)
        {
            int x = frame * 10;
            var detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(x, x, x + 50, x + 50, BoundingBoxFormat.XYXY), 0, 0.9)
            };

            var result = tracker.Update(detections);

            Assert.Equal(frame + 1, result.FrameNumber);
        }

        // After 5 frames, track should be confirmed
        var finalTracks = tracker.GetConfirmedTracks();
        Assert.NotEmpty(finalTracks);
    }

    [Fact]
    public void BoundingBox_FormatConversion_RoundTrip_PreservesCoordinates()
    {
        var originalXYXY = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        // Convert to XYWH
        var (x, y, w, h) = originalXYXY.ToXYWH();
        var xywh = new BoundingBox<double>(x, y, w, h, BoundingBoxFormat.XYWH);

        // Convert back to XYXY
        var (x1, y1, x2, y2) = xywh.ToXYXY();

        Assert.Equal(10, x1, Tolerance);
        Assert.Equal(20, y1, Tolerance);
        Assert.Equal(50, x2, Tolerance);
        Assert.Equal(80, y2, Tolerance);
    }

    [Fact]
    public void BoundingBox_FormatConversion_CXCYWH_RoundTrip_PreservesCoordinates()
    {
        var originalXYXY = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        // Convert to CXCYWH
        var (cx, cy, w, h) = originalXYXY.ToCXCYWH();
        var cxcywh = new BoundingBox<double>(cx, cy, w, h, BoundingBoxFormat.CXCYWH);

        // Convert back to XYXY
        var (x1, y1, x2, y2) = cxcywh.ToXYXY();

        Assert.Equal(10, x1, Tolerance);
        Assert.Equal(20, y1, Tolerance);
        Assert.Equal(50, x2, Tolerance);
        Assert.Equal(80, y2, Tolerance);
    }

    #endregion
}
