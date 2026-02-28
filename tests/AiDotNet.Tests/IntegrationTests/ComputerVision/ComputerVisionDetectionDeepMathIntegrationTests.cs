using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Anchors;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Deep math integration tests for Computer Vision detection components:
/// BoundingBox format conversions, IoU/GIoU/DIoU/CIoU losses, anchor generation.
/// </summary>
public class ComputerVisionDetectionDeepMathIntegrationTests
{
    private const double Tolerance = 1e-6;

    // ============================
    // BoundingBox Format Conversion Tests
    // ============================

    [Fact]
    public void BoundingBox_XYXY_ToXYXY_IsIdentity()
    {
        // Box: (10, 20, 50, 80) in XYXY format
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var (x1, y1, x2, y2) = box.ToXYXY();

        Assert.Equal(10.0, x1, Tolerance);
        Assert.Equal(20.0, y1, Tolerance);
        Assert.Equal(50.0, x2, Tolerance);
        Assert.Equal(80.0, y2, Tolerance);
    }

    [Fact]
    public void BoundingBox_XYWH_ToXYXY_AddsWidthHeight()
    {
        // XYWH: (10, 20, 40, 60) => XYXY: (10, 20, 10+40, 20+60) = (10, 20, 50, 80)
        var box = new BoundingBox<double>(10, 20, 40, 60, BoundingBoxFormat.XYWH);
        var (x1, y1, x2, y2) = box.ToXYXY();

        Assert.Equal(10.0, x1, Tolerance);
        Assert.Equal(20.0, y1, Tolerance);
        Assert.Equal(50.0, x2, Tolerance);
        Assert.Equal(80.0, y2, Tolerance);
    }

    [Fact]
    public void BoundingBox_CXCYWH_ToXYXY_CentersCorrectly()
    {
        // CXCYWH: (30, 50, 40, 60)
        // XYXY: (30-20, 50-30, 30+20, 50+30) = (10, 20, 50, 80)
        var box = new BoundingBox<double>(30, 50, 40, 60, BoundingBoxFormat.CXCYWH);
        var (x1, y1, x2, y2) = box.ToXYXY();

        Assert.Equal(10.0, x1, Tolerance);
        Assert.Equal(20.0, y1, Tolerance);
        Assert.Equal(50.0, x2, Tolerance);
        Assert.Equal(80.0, y2, Tolerance);
    }

    [Fact]
    public void BoundingBox_ToCXCYWH_ComputesCenterAndDimensions()
    {
        // XYXY: (10, 20, 50, 80)
        // Center: ((10+50)/2, (20+80)/2) = (30, 50), W=40, H=60
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var (cx, cy, w, h) = box.ToCXCYWH();

        Assert.Equal(30.0, cx, Tolerance);
        Assert.Equal(50.0, cy, Tolerance);
        Assert.Equal(40.0, w, Tolerance);
        Assert.Equal(60.0, h, Tolerance);
    }

    [Fact]
    public void BoundingBox_ToXYWH_ComputesMinAndDimensions()
    {
        // XYXY: (10, 20, 50, 80)
        // XYWH: (10, 20, 40, 60)
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        var (x, y, w, h) = box.ToXYWH();

        Assert.Equal(10.0, x, Tolerance);
        Assert.Equal(20.0, y, Tolerance);
        Assert.Equal(40.0, w, Tolerance);
        Assert.Equal(60.0, h, Tolerance);
    }

    [Fact]
    public void BoundingBox_Area_IsWidthTimesHeight()
    {
        // XYXY: (10, 20, 50, 80), area = 40 * 60 = 2400
        var box = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);
        Assert.Equal(2400.0, box.Area(), Tolerance);
    }

    [Fact]
    public void BoundingBox_Area_DegenerateBox_IsZero()
    {
        // Zero-width box
        var box = new BoundingBox<double>(10, 20, 10, 80, BoundingBoxFormat.XYXY);
        Assert.Equal(0.0, box.Area(), Tolerance);
    }

    // ============================
    // IoU Tests
    // ============================

    [Fact]
    public void IoU_IdenticalBoxes_IsOne()
    {
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double iou = box.IoU(box);

        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_NonOverlapping_IsZero()
    {
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY);
        double iou = box1.IoU(box2);

        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_HandComputed_PartialOverlap()
    {
        // Box1: (0,0,10,10) area=100
        // Box2: (5,5,15,15) area=100
        // Intersection: (5,5,10,10) area=25
        // Union: 100+100-25=175
        // IoU = 25/175 = 1/7 ≈ 0.142857
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);
        double iou = box1.IoU(box2);

        Assert.Equal(1.0 / 7.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_ContainedBox_IsAreaRatio()
    {
        // Box1: (0,0,10,10) area=100
        // Box2: (2,2,8,8) area=36
        // Intersection = 36 (smaller box entirely inside)
        // Union = 100+36-36 = 100
        // IoU = 36/100 = 0.36
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(2, 2, 8, 8, BoundingBoxFormat.XYXY);
        double iou = box1.IoU(box2);

        Assert.Equal(0.36, iou, Tolerance);
    }

    [Fact]
    public void IoU_IsSymmetric()
    {
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        Assert.Equal(box1.IoU(box2), box2.IoU(box1), Tolerance);
    }

    [Fact]
    public void IoU_BoundedBetween0And1()
    {
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(3, 3, 12, 12, BoundingBoxFormat.XYXY);
        double iou = box1.IoU(box2);

        Assert.InRange(iou, 0.0, 1.0);
    }

    // ============================
    // NMS IoU Computation Tests (via loss classes)
    // ============================

    [Fact]
    public void NMS_ComputeIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double iou = nms.ComputeIoU(box, box);

        Assert.Equal(1.0, iou, Tolerance);
    }

    [Fact]
    public void NMS_ComputeIoU_HandComputed_MatchesBoundingBoxIoU()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        double nmsIoU = nms.ComputeIoU(box1, box2);
        double boxIoU = box1.IoU(box2);

        Assert.Equal(boxIoU, nmsIoU, Tolerance);
    }

    // ============================
    // GIoU Tests
    // ============================

    [Fact]
    public void GIoU_IdenticalBoxes_IsOne()
    {
        var nms = new NMS<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double giou = nms.ComputeGIoU(box, box);

        // Identical: IoU=1, enclosing=union=area, so (enclosing-union)/enclosing=0
        // GIoU = 1 - 0 = 1
        Assert.Equal(1.0, giou, Tolerance);
    }

    [Fact]
    public void GIoU_NonOverlapping_IsNegative()
    {
        var nms = new NMS<double>();
        // Far apart non-overlapping boxes
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY);
        double giou = nms.ComputeGIoU(box1, box2);

        // IoU = 0
        // Enclosing: (0,0,30,30), area=900
        // Union: 100+100=200
        // GIoU = 0 - (900-200)/900 = -700/900 ≈ -0.7778
        Assert.True(giou < 0, "GIoU should be negative for non-overlapping boxes");
        Assert.Equal(-700.0 / 900.0, giou, Tolerance);
    }

    [Fact]
    public void GIoU_HandComputed_PartialOverlap()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        // IoU = 25/175 (from earlier test)
        // Enclosing: (0,0,15,15), area=225
        // Union: 175
        // GIoU = 25/175 - (225-175)/225 = 25/175 - 50/225
        double expectedIoU = 25.0 / 175.0;
        double expectedGIoU = expectedIoU - (225.0 - 175.0) / 225.0;

        double giou = nms.ComputeGIoU(box1, box2);
        Assert.Equal(expectedGIoU, giou, Tolerance);
    }

    [Fact]
    public void GIoU_BoundedBetweenNeg1And1()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 50, 60, 60, BoundingBoxFormat.XYXY);
        double giou = nms.ComputeGIoU(box1, box2);

        Assert.InRange(giou, -1.0, 1.0);
    }

    [Fact]
    public void GIoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        Assert.Equal(nms.ComputeGIoU(box1, box2), nms.ComputeGIoU(box2, box1), Tolerance);
    }

    // ============================
    // DIoU Tests
    // ============================

    [Fact]
    public void DIoU_IdenticalBoxes_IsOne()
    {
        var nms = new NMS<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double diou = nms.ComputeDIoU(box, box);

        // center distance = 0, so penalty = 0, DIoU = IoU = 1
        Assert.Equal(1.0, diou, Tolerance);
    }

    [Fact]
    public void DIoU_HandComputed_CenterDistancePenalty()
    {
        var nms = new NMS<double>();
        // Box1: (0,0,10,10), center=(5,5)
        // Box2: (5,5,15,15), center=(10,10)
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        // Center distance squared: (10-5)^2 + (10-5)^2 = 50
        // Enclosing box: (0,0,15,15), diagonal^2 = 15^2+15^2 = 450
        // IoU = 25/175
        // DIoU = IoU - 50/450 = 25/175 - 1/9
        double expectedIoU = 25.0 / 175.0;
        double expectedDIoU = expectedIoU - 50.0 / 450.0;

        double diou = nms.ComputeDIoU(box1, box2);
        Assert.Equal(expectedDIoU, diou, Tolerance);
    }

    [Fact]
    public void DIoU_ConcentrieBoxes_NoCenterPenalty()
    {
        var nms = new NMS<double>();
        // Boxes with same center but different sizes
        // Box1: (0,0,10,10), center=(5,5)
        // Box2: (2,2,8,8), center=(5,5)
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(2, 2, 8, 8, BoundingBoxFormat.XYXY);

        // Same center => d^2 = 0 => DIoU = IoU
        double diou = nms.ComputeDIoU(box1, box2);
        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(iou, diou, Tolerance);
    }

    [Fact]
    public void DIoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        Assert.Equal(nms.ComputeDIoU(box1, box2), nms.ComputeDIoU(box2, box1), Tolerance);
    }

    // ============================
    // CIoU Tests
    // ============================

    [Fact]
    public void CIoU_IdenticalBoxes_IsOne()
    {
        var nms = new NMS<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double ciou = nms.ComputeCIoU(box, box);

        // Identical: DIoU=1, same aspect ratio => v=0, CIoU=1
        Assert.Equal(1.0, ciou, Tolerance);
    }

    [Fact]
    public void CIoU_SameAspectRatio_EqualsDIoU()
    {
        var nms = new NMS<double>();
        // Both 10x10 (same aspect ratio 1:1)
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        // Same aspect ratio => v=0 => alpha*v=0 => CIoU=DIoU
        double ciou = nms.ComputeCIoU(box1, box2);
        double diou = nms.ComputeDIoU(box1, box2);

        Assert.Equal(diou, ciou, Tolerance);
    }

    [Fact]
    public void CIoU_DifferentAspectRatio_LessThanDIoU()
    {
        var nms = new NMS<double>();
        // Box1: 10x10 (square, aspect ratio=1)
        // Box2: 10x20 (tall rectangle, aspect ratio=2)
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 10, 20, BoundingBoxFormat.XYXY);

        double ciou = nms.ComputeCIoU(box1, box2);
        double diou = nms.ComputeDIoU(box1, box2);

        // CIoU = DIoU - alpha*v, with v > 0, so CIoU < DIoU
        Assert.True(ciou < diou, $"CIoU ({ciou}) should be less than DIoU ({diou}) for different aspect ratios");
    }

    [Fact]
    public void CIoU_HandComputed_AspectRatioPenalty()
    {
        var nms = new NMS<double>();
        // Box1: (0,0,10,10), w1=10, h1=10, arctan(10/10)=pi/4
        // Box2: (0,0,10,20), w2=10, h2=20, arctan(10/20)=arctan(0.5)
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 10, 20, BoundingBoxFormat.XYXY);

        double w1 = 10, h1 = 10, w2 = 10, h2 = 20;
        double arctan1 = Math.Atan(w1 / h1);
        double arctan2 = Math.Atan(w2 / h2);
        double v = (4.0 / (Math.PI * Math.PI)) * Math.Pow(arctan1 - arctan2, 2);

        double iou = nms.ComputeIoU(box1, box2);
        double alpha = v / (1 - iou + v + 1e-7);

        double diou = nms.ComputeDIoU(box1, box2);
        double expectedCIoU = diou - alpha * v;

        double ciou = nms.ComputeCIoU(box1, box2);
        Assert.Equal(expectedCIoU, ciou, Tolerance);
    }

    [Fact]
    public void CIoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        // Use boxes with different aspect ratios
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(2, 2, 12, 22, BoundingBoxFormat.XYXY);

        // CIoU may NOT be symmetric in general due to aspect ratio term direction
        // But checking the formula: v depends on |arctan1 - arctan2|^2, which is symmetric
        // alpha depends on IoU and v (both symmetric)
        // DIoU is symmetric
        // So CIoU should be symmetric
        Assert.Equal(nms.ComputeCIoU(box1, box2), nms.ComputeCIoU(box2, box1), Tolerance);
    }

    // ============================
    // IoU Loss Tests (1 - IoU variants)
    // ============================

    [Fact]
    public void GIoULoss_IdenticalBoxes_IsZero()
    {
        var loss = new GIoULoss<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double lossVal = loss.CalculateLossForBox(box, box);

        // 1 - GIoU(identical) = 1 - 1 = 0
        Assert.Equal(0.0, lossVal, Tolerance);
    }

    [Fact]
    public void DIoULoss_IdenticalBoxes_IsZero()
    {
        var loss = new DIoULoss<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double lossVal = loss.CalculateLossForBox(box, box);

        Assert.Equal(0.0, lossVal, Tolerance);
    }

    [Fact]
    public void CIoULoss_IdenticalBoxes_IsZero()
    {
        var loss = new CIoULoss<double>();
        var box = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        double lossVal = loss.CalculateLossForBox(box, box);

        Assert.Equal(0.0, lossVal, Tolerance);
    }

    [Fact]
    public void GIoULoss_NonOverlapping_GreaterThanOne()
    {
        var loss = new GIoULoss<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY);
        double lossVal = loss.CalculateLossForBox(box1, box2);

        // GIoU < 0 for non-overlapping => Loss = 1 - GIoU > 1
        Assert.True(lossVal > 1.0, $"GIoU loss ({lossVal}) should be > 1 for non-overlapping boxes");
    }

    [Fact]
    public void GIoULoss_VectorForm_HandComputed()
    {
        var loss = new GIoULoss<double>();
        // Single box as flat vector [x1,y1,x2,y2]
        var pred = new Vector<double>(new double[] { 0, 0, 10, 10 });
        var actual = new Vector<double>(new double[] { 5, 5, 15, 15 });

        double lossVal = NumOps<double>.ToDouble(loss.CalculateLoss(pred, actual));

        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);
        double expectedLoss = 1.0 - nms.ComputeGIoU(box1, box2);

        Assert.Equal(expectedLoss, lossVal, Tolerance);
    }

    [Fact]
    public void CIoULoss_VectorForm_MultipleBoxes_IsMeanLoss()
    {
        var loss = new CIoULoss<double>();
        var nms = new NMS<double>();

        // Two boxes: [box1_pred, box2_pred] and [box1_target, box2_target]
        var pred = new Vector<double>(new double[] {
            0, 0, 10, 10,     // box1 pred
            20, 20, 40, 40    // box2 pred
        });
        var actual = new Vector<double>(new double[] {
            5, 5, 15, 15,     // box1 target
            20, 20, 40, 40    // box2 target (identical)
        });

        double totalLoss = NumOps<double>.ToDouble(loss.CalculateLoss(pred, actual));

        // Compute individual losses
        var predBox1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var targetBox1 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);
        var predBox2 = new BoundingBox<double>(20, 20, 40, 40, BoundingBoxFormat.XYXY);
        var targetBox2 = new BoundingBox<double>(20, 20, 40, 40, BoundingBoxFormat.XYXY);

        double loss1 = 1.0 - nms.ComputeCIoU(predBox1, targetBox1);
        double loss2 = 1.0 - nms.ComputeCIoU(predBox2, targetBox2); // 0 for identical
        double expectedMean = (loss1 + loss2) / 2.0;

        Assert.Equal(expectedMean, totalLoss, Tolerance);
    }

    [Fact]
    public void IoULoss_Ordering_GIoU_LE_IoU_For_Overlapping()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);
        double giou = nms.ComputeGIoU(box1, box2);

        // GIoU <= IoU always (subtracts non-negative penalty)
        Assert.True(giou <= iou + Tolerance, $"GIoU ({giou}) should be <= IoU ({iou})");
    }

    [Fact]
    public void IoULoss_Ordering_DIoU_LE_IoU()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(5, 5, 15, 15, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);
        double diou = nms.ComputeDIoU(box1, box2);

        // DIoU = IoU - d^2/c^2, penalty >= 0, so DIoU <= IoU
        Assert.True(diou <= iou + Tolerance, $"DIoU ({diou}) should be <= IoU ({iou})");
    }

    [Fact]
    public void IoULoss_Ordering_CIoU_LE_DIoU()
    {
        var nms = new NMS<double>();
        // Use different aspect ratios so alpha*v > 0
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 10, 20, BoundingBoxFormat.XYXY);

        double diou = nms.ComputeDIoU(box1, box2);
        double ciou = nms.ComputeCIoU(box1, box2);

        // CIoU = DIoU - alpha*v, penalty >= 0, so CIoU <= DIoU
        Assert.True(ciou <= diou + Tolerance, $"CIoU ({ciou}) should be <= DIoU ({diou})");
    }

    // ============================
    // Numerical Gradient Tests
    // ============================

    [Fact]
    public void GIoULoss_NumericalGradient_IsFinite()
    {
        var loss = new GIoULoss<double>();
        var pred = new Vector<double>(new double[] { 0, 0, 10, 10 });
        var actual = new Vector<double>(new double[] { 5, 5, 15, 15 });

        var gradient = loss.CalculateDerivative(pred, actual);

        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.True(!double.IsNaN(NumOps<double>.ToDouble(gradient[i])) && !double.IsInfinity(NumOps<double>.ToDouble(gradient[i])),
                $"Gradient[{i}] should be finite");
        }
    }

    [Fact]
    public void DIoULoss_NumericalGradient_IsFinite()
    {
        var loss = new DIoULoss<double>();
        var pred = new Vector<double>(new double[] { 0, 0, 10, 10 });
        var actual = new Vector<double>(new double[] { 5, 5, 15, 15 });

        var gradient = loss.CalculateDerivative(pred, actual);

        for (int i = 0; i < gradient.Length; i++)
        {
            Assert.True(!double.IsNaN(NumOps<double>.ToDouble(gradient[i])) && !double.IsInfinity(NumOps<double>.ToDouble(gradient[i])),
                $"Gradient[{i}] should be finite");
        }
    }

    // ============================
    // Anchor Generator Tests
    // ============================

    [Fact]
    public void AnchorGenerator_BaseAnchors_CountIsScalesTimesRatios()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0, 1.26 },
            strides: new int[] { 8 });

        Assert.Equal(6, gen.NumAnchorsPerLocation); // 3 ratios * 2 scales
    }

    [Fact]
    public void AnchorGenerator_AnchorCount_FeatureSize()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 8 });

        // Feature 4x4 with 3 anchors per location = 48 anchors
        var anchors = gen.GenerateAnchorsForLevel(4, 4, 8, 32);
        Assert.Equal(48, anchors.Count);
    }

    [Fact]
    public void AnchorGenerator_BaseAnchor_AreaPreservation()
    {
        // For baseSize=32, scale=1.0:
        // width = 32 / sqrt(aspectRatio), height = 32 * sqrt(aspectRatio)
        // area = width * height = 32^2 = 1024 regardless of aspect ratio
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 8 });

        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 32);

        // 3 anchors for 1x1 feature map with 3 aspect ratios
        Assert.Equal(3, anchors.Count);

        foreach (var anchor in anchors)
        {
            // All should have area = 32^2 = 1024
            Assert.Equal(1024.0, anchor.Area(), 1.0); // Allow small floating point error
        }
    }

    [Fact]
    public void AnchorGenerator_AspectRatio_Formula()
    {
        // baseSize=32, aspectRatio=2.0 (height/width), scale=1.0
        // width = 32 / sqrt(2), height = 32 * sqrt(2)
        // aspect ratio of generated anchor: height/width = 32*sqrt(2) / (32/sqrt(2)) = 2.0
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 2.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 8 });

        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 32);
        Assert.Single(anchors);

        var (cx, cy, w, h) = anchors[0].ToCXCYWH();
        double actualAR = h / w;
        Assert.Equal(2.0, actualAR, 1e-4);
    }

    [Fact]
    public void AnchorGenerator_CenterPosition_UsesStride()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 1.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 16 });

        // 2x2 feature map with stride 16
        var anchors = gen.GenerateAnchorsForLevel(2, 2, 16, 32);

        // Centers should be at (0.5*16, 0.5*16), (1.5*16, 0.5*16), etc.
        // = (8, 8), (24, 8), (8, 24), (24, 24)
        var centers = anchors.Select(a => a.ToCXCYWH()).ToList();

        Assert.Equal(8.0, centers[0].cx, Tolerance);
        Assert.Equal(8.0, centers[0].cy, Tolerance);
        Assert.Equal(24.0, centers[1].cx, Tolerance);
        Assert.Equal(8.0, centers[1].cy, Tolerance);
        Assert.Equal(8.0, centers[2].cx, Tolerance);
        Assert.Equal(24.0, centers[2].cy, Tolerance);
        Assert.Equal(24.0, centers[3].cx, Tolerance);
        Assert.Equal(24.0, centers[3].cy, Tolerance);
    }

    [Fact]
    public void AnchorGenerator_ScaleFactor_ScalesBaseSize()
    {
        // baseSize=32, scale=2.0 => scaledSize=64
        // aspectRatio=1.0 => width=height=64, area=4096
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 1.0 },
            scales: new double[] { 2.0 },
            strides: new int[] { 8 });

        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 32);
        Assert.Single(anchors);

        double area = anchors[0].Area();
        Assert.Equal(64.0 * 64.0, area, 1.0);
    }

    [Fact]
    public void AnchorGenerator_TotalAnchorCount_MultiLevel()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32, 64, 128 },
            aspectRatios: new double[] { 0.5, 1.0, 2.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 8, 16, 32 });

        // Image 256x256:
        // Level 0: ceil(256/8) = 32, 32x32 = 1024 cells, 1024*3 = 3072 anchors
        // Level 1: ceil(256/16) = 16, 16x16 = 256 cells, 256*3 = 768 anchors
        // Level 2: ceil(256/32) = 8, 8x8 = 64 cells, 64*3 = 192 anchors
        // Total = 3072 + 768 + 192 = 4032
        int total = gen.GetTotalAnchorCount(256, 256);
        Assert.Equal(4032, total);
    }

    [Fact]
    public void AnchorGenerator_FeatureMapSize_CeilDivision()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new double[] { 32 },
            aspectRatios: new double[] { 1.0 },
            scales: new double[] { 1.0 },
            strides: new int[] { 8 });

        // Image 100x100, stride 8:
        // Feature size: ceil(100/8) = 13
        // Total: 13*13*1 = 169
        int total = gen.GetTotalAnchorCount(100, 100);
        Assert.Equal(169, total);
    }

    [Fact]
    public void AnchorGenerator_FasterRCNN_HasCorrectConfig()
    {
        var gen = AnchorGenerator<double>.CreateFasterRCNNAnchors();

        Assert.Equal(5, gen.BaseSizes.Length);
        Assert.Equal(3, gen.AspectRatios.Length);
        Assert.Equal(new double[] { 0.5, 1.0, 2.0 }, gen.AspectRatios);
        Assert.Equal(new int[] { 4, 8, 16, 32, 64 }, gen.Strides);
        Assert.Equal(3, gen.NumAnchorsPerLocation); // 3 ratios * 1 scale
    }

    [Fact]
    public void AnchorGenerator_RetinaNet_HasCorrectConfig()
    {
        var gen = AnchorGenerator<double>.CreateRetinaNetAnchors();

        Assert.Equal(5, gen.BaseSizes.Length);
        Assert.Equal(3, gen.AspectRatios.Length);
        Assert.Equal(3, gen.Scales.Length);
        Assert.Equal(9, gen.NumAnchorsPerLocation); // 3 ratios * 3 scales
    }

    // ============================
    // BoundingBox Edge Cases
    // ============================

    [Fact]
    public void IoU_TouchingEdge_IsZero()
    {
        // Boxes share an edge but don't overlap in area
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(10, 0, 20, 10, BoundingBoxFormat.XYXY);

        double iou = box1.IoU(box2);
        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void GIoU_AdjacentBoxes_PenaltyReflectsGap()
    {
        var nms = new NMS<double>();
        // Adjacent boxes (no gap)
        var adj1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var adj2 = new BoundingBox<double>(10, 0, 20, 10, BoundingBoxFormat.XYXY);

        // Far apart boxes
        var far1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var far2 = new BoundingBox<double>(100, 0, 110, 10, BoundingBoxFormat.XYXY);

        double giouAdj = nms.ComputeGIoU(adj1, adj2);
        double giouFar = nms.ComputeGIoU(far1, far2);

        // Adjacent should have higher GIoU (less penalty) than far apart
        Assert.True(giouAdj > giouFar, $"Adjacent GIoU ({giouAdj}) should be > far GIoU ({giouFar})");
    }

    [Fact]
    public void IoULoss_Vector_InvalidLength_Throws()
    {
        var loss = new GIoULoss<double>();
        var pred = new Vector<double>(new double[] { 0, 0, 10 }); // Length 3, not multiple of 4
        var actual = new Vector<double>(new double[] { 0, 0, 10 });

        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(pred, actual));
    }
}

/// <summary>
/// Helper to access NumOps static methods.
/// </summary>
internal static class NumOps<T>
{
    private static readonly INumericOperations<T> Ops = Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    public static double ToDouble(T value) => Ops.ToDouble(value);
}
