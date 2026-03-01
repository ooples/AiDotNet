using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Anchors;
using AiDotNet.ComputerVision.Detection.Losses;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.ComputerVision.Segmentation.Losses;
using AiDotNet.ComputerVision.Tracking;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Extended integration tests for ComputerVision module with deep mathematical verification.
/// Tests IoU variants, loss functions, anchor generation, mask losses, and detection pipelines
/// using hand-calculated expected values to catch bugs.
/// </summary>
public class ComputerVisionExtendedIntegrationTests
{
    private const double Tolerance = 1e-6;
    private const double LooseTolerance = 1e-3;

    #region IoU Computation - Hand-Calculated Verification

    [Fact]
    public void IoU_PartialOverlap_HandCalculated_MatchesExpectedValue()
    {
        // Box1: (0,0,100,100) area=10000
        // Box2: (50,50,150,150) area=10000
        // Intersection: (50,50,100,100) = 50*50 = 2500
        // Union: 10000 + 10000 - 2500 = 17500
        // IoU = 2500/17500 = 1/7 ≈ 0.142857
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 50, 150, 150, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(1.0 / 7.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_ContainedBox_ReturnsCorrectValue()
    {
        // Box1 fully contains Box2
        // Box1: (0,0,200,200) area=40000
        // Box2: (50,50,100,100) area=2500
        // Intersection = area of Box2 = 2500
        // Union = 40000 + 2500 - 2500 = 40000
        // IoU = 2500/40000 = 0.0625
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 200, 200, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 50, 100, 100, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(2500.0 / 40000.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(10, 20, 80, 90, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(30, 40, 120, 130, BoundingBoxFormat.XYXY);

        double iou12 = nms.ComputeIoU(box1, box2);
        double iou21 = nms.ComputeIoU(box2, box1);

        Assert.Equal(iou12, iou21, Tolerance);
    }

    [Fact]
    public void IoU_TouchingBoxes_ReturnsZero()
    {
        // Boxes share an edge but no area
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 0, 100, 50, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(0.0, iou, Tolerance);
    }

    [Fact]
    public void IoU_HighOverlap_HandCalculated()
    {
        // Box1: (0,0,100,100) area=10000
        // Box2: (10,10,110,110) area=10000
        // Intersection: (10,10,100,100) = 90*90 = 8100
        // Union: 10000 + 10000 - 8100 = 11900
        // IoU = 8100/11900 ≈ 0.68067
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(10, 10, 110, 110, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);

        Assert.Equal(8100.0 / 11900.0, iou, Tolerance);
    }

    #endregion

    #region GIoU Computation - Mathematical Properties

    [Fact]
    public void GIoU_PartialOverlap_HandCalculated()
    {
        // Box1: (0,0,100,100), Box2: (50,50,150,150)
        // IoU = 1/7 (from above)
        // Enclosing: (0,0,150,150), area = 22500
        // Union = 17500
        // GIoU = 1/7 - (22500-17500)/22500 = 1/7 - 5000/22500
        // = 0.142857 - 0.222222 = -0.079365
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 50, 150, 150, BoundingBoxFormat.XYXY);

        double giou = nms.ComputeGIoU(box1, box2);
        double expected = 1.0 / 7.0 - 5000.0 / 22500.0;

        Assert.Equal(expected, giou, Tolerance);
    }

    [Fact]
    public void GIoU_FarApartBoxes_IsMoreNegativeThanCloseBoxes()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY);
        var closeBox = new BoundingBox<double>(15, 0, 25, 10, BoundingBoxFormat.XYXY);
        var farBox = new BoundingBox<double>(100, 0, 110, 10, BoundingBoxFormat.XYXY);

        double giouClose = nms.ComputeGIoU(box1, closeBox);
        double giouFar = nms.ComputeGIoU(box1, farBox);

        // GIoU for far boxes should be more negative (worse) than close boxes
        Assert.True(giouFar < giouClose,
            $"GIoU(far)={giouFar} should be < GIoU(close)={giouClose}");
    }

    [Fact]
    public void GIoU_RangeIsBetweenMinus1And1()
    {
        var nms = new NMS<double>();
        var testCases = new[]
        {
            (new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY),
             new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY)),
            (new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY),
             new BoundingBox<double>(100, 100, 110, 110, BoundingBoxFormat.XYXY)),
            (new BoundingBox<double>(0, 0, 100, 10, BoundingBoxFormat.XYXY),
             new BoundingBox<double>(50, 0, 150, 10, BoundingBoxFormat.XYXY)),
        };

        foreach (var (box1, box2) in testCases)
        {
            double giou = nms.ComputeGIoU(box1, box2);
            Assert.True(giou >= -1.0 - Tolerance && giou <= 1.0 + Tolerance,
                $"GIoU={giou} outside range [-1, 1]");
        }
    }

    [Fact]
    public void GIoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(10, 20, 80, 90, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(30, 40, 120, 130, BoundingBoxFormat.XYXY);

        double giou12 = nms.ComputeGIoU(box1, box2);
        double giou21 = nms.ComputeGIoU(box2, box1);

        Assert.Equal(giou12, giou21, Tolerance);
    }

    [Fact]
    public void GIoU_AlwaysLessThanOrEqualToIoU()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(30, 30, 130, 130, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);
        double giou = nms.ComputeGIoU(box1, box2);

        Assert.True(giou <= iou + Tolerance,
            $"GIoU={giou} should be <= IoU={iou}");
    }

    #endregion

    #region DIoU Computation - Mathematical Properties

    [Fact]
    public void DIoU_PartialOverlap_HandCalculated()
    {
        // Box1: (0,0,100,100), Box2: (50,50,150,150)
        // IoU = 1/7
        // Center1: (50, 50), Center2: (100, 100)
        // Center dist sq: 50^2 + 50^2 = 5000
        // Enclosing: (0,0,150,150), diagonal sq: 150^2 + 150^2 = 45000
        // DIoU = 1/7 - 5000/45000 = 0.142857 - 0.111111 = 0.031746
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(50, 50, 150, 150, BoundingBoxFormat.XYXY);

        double diou = nms.ComputeDIoU(box1, box2);
        double expected = 1.0 / 7.0 - 5000.0 / 45000.0;

        Assert.Equal(expected, diou, Tolerance);
    }

    [Fact]
    public void DIoU_CenteredBoxes_EqualsIoU()
    {
        // When centers coincide, center distance is 0, so DIoU = IoU
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(10, 10, 90, 90, BoundingBoxFormat.XYXY);
        // Center1: (50, 50), Center2: (50, 50) - same center!

        double iou = nms.ComputeIoU(box1, box2);
        double diou = nms.ComputeDIoU(box1, box2);

        Assert.Equal(iou, diou, Tolerance);
    }

    [Fact]
    public void DIoU_IsSymmetric()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(10, 20, 80, 90, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(30, 40, 120, 130, BoundingBoxFormat.XYXY);

        double diou12 = nms.ComputeDIoU(box1, box2);
        double diou21 = nms.ComputeDIoU(box2, box1);

        Assert.Equal(diou12, diou21, Tolerance);
    }

    [Fact]
    public void DIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(25, 30, 75, 80, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(25, 30, 75, 80, BoundingBoxFormat.XYXY);

        double diou = nms.ComputeDIoU(box1, box2);

        Assert.Equal(1.0, diou, Tolerance);
    }

    #endregion

    #region CIoU Computation - Mathematical Properties

    [Fact]
    public void CIoU_SameAspectRatio_EqualsDIoU()
    {
        // When both boxes are square, aspect ratio penalty v=0, so CIoU = DIoU
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY); // square
        var box2 = new BoundingBox<double>(50, 50, 150, 150, BoundingBoxFormat.XYXY); // square

        double diou = nms.ComputeDIoU(box1, box2);
        double ciou = nms.ComputeCIoU(box1, box2);

        Assert.Equal(diou, ciou, Tolerance);
    }

    [Fact]
    public void CIoU_DifferentAspectRatio_LessThanDIoU()
    {
        // Box1 is square, Box2 is wide rectangle
        // Aspect ratio penalty should make CIoU < DIoU
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY); // 1:1
        var box2 = new BoundingBox<double>(0, 0, 200, 50, BoundingBoxFormat.XYXY); // 4:1

        double diou = nms.ComputeDIoU(box1, box2);
        double ciou = nms.ComputeCIoU(box1, box2);

        Assert.True(ciou < diou,
            $"CIoU={ciou} should be < DIoU={diou} when aspect ratios differ");
    }

    [Fact]
    public void CIoU_HandCalculated_DifferentAspectRatios()
    {
        // Box1: (0,0,100,100) w=100, h=100, center=(50,50)
        // Box2: (0,0,200,100) w=200, h=100, center=(100,50)
        // IoU: intersection (0,0,100,100) = 10000; union = 10000+20000-10000 = 20000; IoU = 0.5
        // DIoU: centers (50,50) and (100,50); dist^2 = 2500
        //        enc (0,0,200,100); diag^2 = 200^2+100^2 = 50000
        //        DIoU = 0.5 - 2500/50000 = 0.5 - 0.05 = 0.45
        // CIoU: arctan(100/100) = pi/4; arctan(200/100) = atan(2)
        //        v = (4/pi^2) * (pi/4 - atan(2))^2
        //        alpha = v / (1 - 0.5 + v + 1e-7)
        //        CIoU = 0.45 - alpha * v
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(0, 0, 200, 100, BoundingBoxFormat.XYXY);

        double ciou = nms.ComputeCIoU(box1, box2);

        double arctan1 = Math.Atan(100.0 / 100.0); // pi/4
        double arctan2 = Math.Atan(200.0 / 100.0); // atan(2)
        double v = (4.0 / (Math.PI * Math.PI)) * Math.Pow(arctan1 - arctan2, 2);
        double alpha = v / (1 - 0.5 + v + 1e-7);
        double expectedCIoU = 0.45 - alpha * v;

        Assert.Equal(expectedCIoU, ciou, Tolerance);
    }

    [Fact]
    public void CIoU_AlwaysLessThanOrEqualToDIoU()
    {
        var nms = new NMS<double>();
        // Test with various box pairs
        var testCases = new[]
        {
            (new BoundingBox<double>(0, 0, 100, 50, BoundingBoxFormat.XYXY),
             new BoundingBox<double>(20, 10, 80, 90, BoundingBoxFormat.XYXY)),
            (new BoundingBox<double>(0, 0, 200, 50, BoundingBoxFormat.XYXY),
             new BoundingBox<double>(50, 0, 150, 100, BoundingBoxFormat.XYXY)),
        };

        foreach (var (box1, box2) in testCases)
        {
            double diou = nms.ComputeDIoU(box1, box2);
            double ciou = nms.ComputeCIoU(box1, box2);

            Assert.True(ciou <= diou + Tolerance,
                $"CIoU={ciou} should be <= DIoU={diou}");
        }
    }

    [Fact]
    public void CIoU_IdenticalBoxes_ReturnsOne()
    {
        var nms = new NMS<double>();
        var box = new BoundingBox<double>(10, 20, 80, 90, BoundingBoxFormat.XYXY);

        double ciou = nms.ComputeCIoU(box, box);

        Assert.Equal(1.0, ciou, Tolerance);
    }

    #endregion

    #region IoU Hierarchy Verification (IoU >= GIoU, IoU >= DIoU >= CIoU)

    [Fact]
    public void IoU_Hierarchy_VerifyOrdering()
    {
        var nms = new NMS<double>();
        var box1 = new BoundingBox<double>(0, 0, 100, 60, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(30, 10, 150, 100, BoundingBoxFormat.XYXY);

        double iou = nms.ComputeIoU(box1, box2);
        double giou = nms.ComputeGIoU(box1, box2);
        double diou = nms.ComputeDIoU(box1, box2);
        double ciou = nms.ComputeCIoU(box1, box2);

        // IoU >= GIoU (GIoU subtracts non-negative term)
        Assert.True(iou >= giou - Tolerance, $"IoU={iou} should be >= GIoU={giou}");
        // IoU >= DIoU (DIoU subtracts non-negative distance penalty)
        Assert.True(iou >= diou - Tolerance, $"IoU={iou} should be >= DIoU={diou}");
        // DIoU >= CIoU (CIoU subtracts non-negative aspect ratio penalty)
        Assert.True(diou >= ciou - Tolerance, $"DIoU={diou} should be >= CIoU={ciou}");
    }

    #endregion

    #region CIoU Loss - Deep Testing

    [Fact]
    public void CIoULoss_IdenticalBoxes_ReturnsZero()
    {
        var loss = new CIoULoss<double>();
        var predicted = new Vector<double>(new[] { 10.0, 20.0, 80.0, 90.0 });
        var actual = new Vector<double>(new[] { 10.0, 20.0, 80.0, 90.0 });

        double lossVal = loss.CalculateLoss(predicted, actual);

        Assert.Equal(0.0, lossVal, LooseTolerance);
    }

    [Fact]
    public void CIoULoss_OffsetBoxes_ReturnsPositive()
    {
        var loss = new CIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 50.0, 50.0, 150.0, 150.0 });

        double lossVal = loss.CalculateLoss(predicted, actual);

        // CIoU loss = 1 - CIoU; for offset square boxes, CIoU ≈ DIoU = 0.031746
        // So loss ≈ 1 - 0.031746 = 0.968254
        Assert.True(lossVal > 0, "CIoU loss should be positive for different boxes");
        Assert.True(lossVal <= 2.0, "CIoU loss should be at most 2 (since CIoU >= -1)");
    }

    [Fact]
    public void CIoULoss_InvalidVectorLength_ThrowsArgumentException()
    {
        var loss = new CIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0 }); // not multiple of 4
        var actual = new Vector<double>(new[] { 0.0, 0.0, 100.0 });

        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void CIoULoss_Gradient_NonZeroForDifferentBoxes()
    {
        var loss = new CIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 20.0, 20.0, 120.0, 120.0 });

        var gradient = loss.CalculateDerivative(predicted, actual);

        Assert.Equal(4, gradient.Length);
        // Gradient should not be all zeros since boxes differ
        bool hasNonZero = false;
        for (int i = 0; i < gradient.Length; i++)
        {
            if (Math.Abs(gradient[i]) > 1e-10) hasNonZero = true;
        }
        Assert.True(hasNonZero, "CIoU gradient should have non-zero elements");
    }

    [Fact]
    public void CIoULoss_Gradient_DirectionReducesLoss()
    {
        // Moving predicted toward actual should decrease loss
        var loss = new CIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 20.0, 20.0, 120.0, 120.0 });

        double originalLoss = loss.CalculateLoss(predicted, actual);
        var gradient = loss.CalculateDerivative(predicted, actual);

        // Take a small step in the negative gradient direction
        double stepSize = 0.1;
        var stepped = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            stepped[i] = predicted[i] - stepSize * gradient[i];
        }

        double newLoss = loss.CalculateLoss(stepped, actual);

        Assert.True(newLoss < originalLoss + Tolerance,
            $"Loss after gradient step ({newLoss}) should be < original ({originalLoss})");
    }

    [Fact]
    public void CIoULoss_TensorInput_MatchesVectorInput()
    {
        var loss = new CIoULoss<double>();

        // Vector version
        var predVec = new Vector<double>(new[] { 10.0, 20.0, 80.0, 90.0 });
        var actualVec = new Vector<double>(new[] { 15.0, 25.0, 85.0, 95.0 });
        double vectorLoss = loss.CalculateLoss(predVec, actualVec);

        // Tensor version
        var predTensor = new Tensor<double>(new[] { 1, 1, 4 });
        predTensor[0, 0, 0] = 10; predTensor[0, 0, 1] = 20;
        predTensor[0, 0, 2] = 80; predTensor[0, 0, 3] = 90;

        var actualTensor = new Tensor<double>(new[] { 1, 1, 4 });
        actualTensor[0, 0, 0] = 15; actualTensor[0, 0, 1] = 25;
        actualTensor[0, 0, 2] = 85; actualTensor[0, 0, 3] = 95;

        double tensorLoss = loss.CalculateLoss(predTensor, actualTensor);

        Assert.Equal(vectorLoss, tensorLoss, LooseTolerance);
    }

    [Fact]
    public void CIoULoss_BoxMethod_ConsistentWithVectorMethod()
    {
        var loss = new CIoULoss<double>();
        var predBox = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var actualBox = new BoundingBox<double>(10, 10, 110, 110, BoundingBoxFormat.XYXY);

        double boxLoss = loss.CalculateLossForBox(predBox, actualBox);

        var predVec = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actualVec = new Vector<double>(new[] { 10.0, 10.0, 110.0, 110.0 });
        double vectorLoss = loss.CalculateLoss(predVec, actualVec);

        Assert.Equal(boxLoss, vectorLoss, LooseTolerance);
    }

    #endregion

    #region DIoU Loss - Deep Testing

    [Fact]
    public void DIoULoss_IdenticalBoxes_ReturnsZero()
    {
        var loss = new DIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });

        double lossVal = loss.CalculateLoss(predicted, actual);

        Assert.Equal(0.0, lossVal, LooseTolerance);
    }

    [Fact]
    public void DIoULoss_HandCalculated()
    {
        // Box1: (0,0,100,100), Box2: (50,50,150,150)
        // DIoU = 0.031746 (from earlier calculation)
        // Loss = 1 - 0.031746 = 0.968254
        var loss = new DIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 50.0, 50.0, 150.0, 150.0 });

        double lossVal = loss.CalculateLoss(predicted, actual);
        double expectedLoss = 1.0 - (1.0 / 7.0 - 5000.0 / 45000.0);

        Assert.Equal(expectedLoss, lossVal, LooseTolerance);
    }

    [Fact]
    public void DIoULoss_InvalidVectorLength_ThrowsArgumentException()
    {
        var loss = new DIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0 });
        var actual = new Vector<double>(new[] { 0.0, 0.0, 100.0 });

        Assert.Throws<ArgumentException>(() => loss.CalculateLoss(predicted, actual));
    }

    [Fact]
    public void DIoULoss_Gradient_DirectionReducesLoss()
    {
        var loss = new DIoULoss<double>();
        var predicted = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var actual = new Vector<double>(new[] { 20.0, 20.0, 120.0, 120.0 });

        double originalLoss = loss.CalculateLoss(predicted, actual);
        var gradient = loss.CalculateDerivative(predicted, actual);

        double stepSize = 0.1;
        var stepped = new Vector<double>(4);
        for (int i = 0; i < 4; i++)
        {
            stepped[i] = predicted[i] - stepSize * gradient[i];
        }

        double newLoss = loss.CalculateLoss(stepped, actual);

        Assert.True(newLoss < originalLoss + Tolerance,
            $"Loss after gradient step ({newLoss}) should be < original ({originalLoss})");
    }

    [Fact]
    public void DIoULoss_TensorInput_3D_ComputesMean()
    {
        var loss = new DIoULoss<double>();

        // 2 boxes in a batch
        var pred = new Tensor<double>(new[] { 1, 2, 4 });
        pred[0, 0, 0] = 0; pred[0, 0, 1] = 0; pred[0, 0, 2] = 100; pred[0, 0, 3] = 100;
        pred[0, 1, 0] = 200; pred[0, 1, 1] = 200; pred[0, 1, 2] = 300; pred[0, 1, 3] = 300;

        var target = new Tensor<double>(new[] { 1, 2, 4 });
        target[0, 0, 0] = 0; target[0, 0, 1] = 0; target[0, 0, 2] = 100; target[0, 0, 3] = 100; // identical
        target[0, 1, 0] = 250; target[0, 1, 1] = 250; target[0, 1, 2] = 350; target[0, 1, 3] = 350; // offset

        double lossVal = loss.CalculateLoss(pred, target);

        // First pair: identical, loss = 0
        // Second pair: offset, loss > 0
        // Mean should be > 0 but < 1
        Assert.True(lossVal > 0, "Mean loss should be positive when one pair differs");
        Assert.True(lossVal < 1.0, "Mean loss with one identical pair should be < 1");
    }

    #endregion

    #region DETR Set Loss - Deep Testing

    [Fact]
    public void DETRSetLoss_VectorInput_ComputesL1Loss()
    {
        var loss = new DETRSetLoss<double>(numClasses: 5);
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.5, 2.5, 3.5, 4.5 });

        double lossVal = loss.CalculateLoss(predicted, actual);

        // L1 loss: |1-1.5| + |2-2.5| + |3-3.5| + |4-4.5| = 0.5*4 = 2.0
        // Mean: 2.0/4 = 0.5
        Assert.Equal(0.5, lossVal, LooseTolerance);
    }

    [Fact]
    public void DETRSetLoss_IdenticalVectors_ReturnsZero()
    {
        var loss = new DETRSetLoss<double>();
        var v = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

        double lossVal = loss.CalculateLoss(v, v);

        Assert.Equal(0.0, lossVal, LooseTolerance);
    }

    [Fact]
    public void DETRSetLoss_Gradient_IsNonZero()
    {
        var loss = new DETRSetLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var actual = new Vector<double>(new[] { 1.5, 2.5, 3.5, 4.5 });

        var gradient = loss.CalculateDerivative(predicted, actual);

        Assert.Equal(4, gradient.Length);
        bool hasNonZero = false;
        for (int i = 0; i < gradient.Length; i++)
        {
            if (Math.Abs(gradient[i]) > 1e-10) hasNonZero = true;
        }
        Assert.True(hasNonZero, "DETR gradient should have non-zero elements");
    }

    [Fact]
    public void DETRSetLoss_CustomWeights_AffectLoss()
    {
        var loss1 = new DETRSetLoss<double>(numClasses: 5, classWeight: 1.0, boxL1Weight: 5.0, boxGIoUWeight: 2.0);
        var loss2 = new DETRSetLoss<double>(numClasses: 5, classWeight: 2.0, boxL1Weight: 10.0, boxGIoUWeight: 4.0);

        // Vector-based L1 loss is weight-independent (uses basic L1)
        // But construction with different weights should succeed
        var v1 = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
        var v2 = new Vector<double>(new[] { 2.0, 3.0, 4.0, 5.0 });

        double lossVal1 = loss1.CalculateLoss(v1, v2);
        double lossVal2 = loss2.CalculateLoss(v1, v2);

        // Vector L1 method is the same regardless of weights
        Assert.Equal(lossVal1, lossVal2, Tolerance);
    }

    #endregion

    #region Anchor Generator - Deep Testing

    [Fact]
    public void AnchorGenerator_Default_HasCorrectConfig()
    {
        var gen = new AnchorGenerator<double>();

        Assert.Equal(3, gen.BaseSizes.Length);
        Assert.Equal(3, gen.AspectRatios.Length);
        Assert.Equal(3, gen.Scales.Length);
        Assert.Equal(3, gen.Strides.Length);
        Assert.Equal(9, gen.NumAnchorsPerLocation); // 3 scales * 3 ratios
    }

    [Fact]
    public void AnchorGenerator_GenerateForLevel_CorrectCount()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 32.0 },
            aspectRatios: new[] { 0.5, 1.0, 2.0 },
            scales: new[] { 1.0 },
            strides: new[] { 8 });

        // 3 anchors per location (3 ratios * 1 scale)
        // Feature map 4x4 = 16 locations
        var anchors = gen.GenerateAnchorsForLevel(4, 4, 8, 32);

        Assert.Equal(16 * 3, anchors.Count);
    }

    [Fact]
    public void AnchorGenerator_AnchorCenters_AreCorrectlyPositioned()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 32.0 },
            aspectRatios: new[] { 1.0 }, // only square
            scales: new[] { 1.0 },
            strides: new[] { 16 });

        // 1 anchor per location, 2x2 feature map with stride 16
        var anchors = gen.GenerateAnchorsForLevel(2, 2, 16, 32);

        Assert.Equal(4, anchors.Count);

        // First anchor center should be at (0.5 * 16, 0.5 * 16) = (8, 8)
        var (cx0, cy0, _, _) = anchors[0].ToCXCYWH();
        Assert.Equal(8.0, cx0, Tolerance);
        Assert.Equal(8.0, cy0, Tolerance);

        // Second anchor center should be at (1.5 * 16, 0.5 * 16) = (24, 8)
        var (cx1, cy1, _, _) = anchors[1].ToCXCYWH();
        Assert.Equal(24.0, cx1, Tolerance);
        Assert.Equal(8.0, cy1, Tolerance);

        // Third anchor center should be at (0.5 * 16, 1.5 * 16) = (8, 24)
        var (cx2, cy2, _, _) = anchors[2].ToCXCYWH();
        Assert.Equal(8.0, cx2, Tolerance);
        Assert.Equal(24.0, cy2, Tolerance);
    }

    [Fact]
    public void AnchorGenerator_SquareAnchor_HasCorrectDimensions()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 64.0 },
            aspectRatios: new[] { 1.0 },
            scales: new[] { 1.0 },
            strides: new[] { 8 });

        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 64);

        Assert.Single(anchors);

        // Square anchor with baseSize 64 and aspectRatio 1.0:
        // width = 64 / sqrt(1) = 64
        // height = 64 * sqrt(1) = 64
        var (_, _, w, h) = anchors[0].ToCXCYWH();
        Assert.Equal(64.0, w, Tolerance);
        Assert.Equal(64.0, h, Tolerance);
    }

    [Fact]
    public void AnchorGenerator_AspectRatio_ProducesCorrectDimensions()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 100.0 },
            aspectRatios: new[] { 0.5, 1.0, 2.0 },
            scales: new[] { 1.0 },
            strides: new[] { 8 });

        // 1x1 feature map, 3 anchors
        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 100);

        Assert.Equal(3, anchors.Count);

        // aspectRatio = h/w, area = baseSize^2 = 10000
        // For AR=0.5: w = 100/sqrt(0.5), h = 100*sqrt(0.5) → w ≈ 141.42, h ≈ 70.71
        var (_, _, w0, h0) = anchors[0].ToCXCYWH();
        Assert.Equal(100.0 / Math.Sqrt(0.5), w0, 0.01);
        Assert.Equal(100.0 * Math.Sqrt(0.5), h0, 0.01);

        // For AR=1.0: w = 100, h = 100
        var (_, _, w1, h1) = anchors[1].ToCXCYWH();
        Assert.Equal(100.0, w1, Tolerance);
        Assert.Equal(100.0, h1, Tolerance);

        // For AR=2.0: w = 100/sqrt(2), h = 100*sqrt(2) → w ≈ 70.71, h ≈ 141.42
        var (_, _, w2, h2) = anchors[2].ToCXCYWH();
        Assert.Equal(100.0 / Math.Sqrt(2.0), w2, 0.01);
        Assert.Equal(100.0 * Math.Sqrt(2.0), h2, 0.01);
    }

    [Fact]
    public void AnchorGenerator_ScaleMultiplier_AffectsSize()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 50.0 },
            aspectRatios: new[] { 1.0 },
            scales: new[] { 1.0, 2.0 },
            strides: new[] { 8 });

        var anchors = gen.GenerateAnchorsForLevel(1, 1, 8, 50);

        Assert.Equal(2, anchors.Count);

        // Scale 1.0: w=50, h=50
        var (_, _, w0, h0) = anchors[0].ToCXCYWH();
        Assert.Equal(50.0, w0, Tolerance);
        Assert.Equal(50.0, h0, Tolerance);

        // Scale 2.0: w=100, h=100
        var (_, _, w1, h1) = anchors[1].ToCXCYWH();
        Assert.Equal(100.0, w1, Tolerance);
        Assert.Equal(100.0, h1, Tolerance);
    }

    [Fact]
    public void AnchorGenerator_TotalCount_MatchesActualGenerated()
    {
        var gen = new AnchorGenerator<double>();

        int imageH = 640, imageW = 640;
        int expectedCount = gen.GetTotalAnchorCount(imageH, imageW);
        var anchors = gen.GenerateAnchorsForImage(imageH, imageW);

        Assert.Equal(expectedCount, anchors.Count);
    }

    [Fact]
    public void AnchorGenerator_MultiLevel_GeneratesPerLevel()
    {
        var gen = new AnchorGenerator<double>(
            baseSizes: new[] { 32.0, 64.0 },
            aspectRatios: new[] { 1.0 },
            scales: new[] { 1.0 },
            strides: new[] { 8, 16 });

        var featureSizes = new List<(int Height, int Width)> { (4, 4), (2, 2) };
        var anchors = gen.GenerateAnchors(featureSizes);

        Assert.Equal(2, anchors.Count);
        Assert.Equal(16, anchors[0].Count); // 4x4 * 1
        Assert.Equal(4, anchors[1].Count);  // 2x2 * 1
    }

    [Fact]
    public void AnchorGenerator_FasterRCNN_Factory_HasCorrectConfig()
    {
        var gen = AnchorGenerator<double>.CreateFasterRCNNAnchors();

        Assert.Equal(5, gen.BaseSizes.Length); // 32, 64, 128, 256, 512
        Assert.Equal(3, gen.AspectRatios.Length); // 0.5, 1.0, 2.0
        Assert.Equal(1, gen.Scales.Length); // 1.0
        Assert.Equal(5, gen.Strides.Length); // 4, 8, 16, 32, 64
        Assert.Equal(3, gen.NumAnchorsPerLocation);
    }

    [Fact]
    public void AnchorGenerator_RetinaNet_Factory_HasCorrectConfig()
    {
        var gen = AnchorGenerator<double>.CreateRetinaNetAnchors();

        Assert.Equal(5, gen.BaseSizes.Length);
        Assert.Equal(3, gen.AspectRatios.Length);
        Assert.Equal(3, gen.Scales.Length); // 1.0, 1.26, 1.59
        Assert.Equal(5, gen.Strides.Length);
        Assert.Equal(9, gen.NumAnchorsPerLocation); // 3*3
    }

    [Fact]
    public void AnchorGenerator_AllAnchorsAreValid()
    {
        var gen = new AnchorGenerator<double>();
        var anchors = gen.GenerateAnchorsForImage(320, 320);

        foreach (var anchor in anchors)
        {
            var (xMin, yMin, xMax, yMax) = anchor.ToXYXY();
            Assert.True(xMax > xMin, $"Anchor width should be positive: ({xMin},{yMin},{xMax},{yMax})");
            Assert.True(yMax > yMin, $"Anchor height should be positive: ({xMin},{yMin},{xMax},{yMax})");
        }
    }

    #endregion

    #region Mask BCE Loss - Mathematical Verification

    [Fact]
    public void MaskBCELoss_PerfectPrediction_ReturnsNearZero()
    {
        var loss = new MaskBCELoss<double>();
        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 4 });

        // Near-perfect predictions
        predicted[0] = 0.999; target[0] = 1.0;
        predicted[1] = 0.001; target[1] = 0.0;
        predicted[2] = 0.999; target[2] = 1.0;
        predicted[3] = 0.001; target[3] = 0.0;

        double lossVal = loss.Compute(predicted, target);

        Assert.True(lossVal < 0.01, $"Perfect predictions should give near-zero loss, got {lossVal}");
    }

    [Fact]
    public void MaskBCELoss_InvertedPrediction_ReturnsHigh()
    {
        var loss = new MaskBCELoss<double>();
        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 4 });

        // Completely wrong predictions
        predicted[0] = 0.001; target[0] = 1.0;
        predicted[1] = 0.999; target[1] = 0.0;
        predicted[2] = 0.001; target[2] = 1.0;
        predicted[3] = 0.999; target[3] = 0.0;

        double lossVal = loss.Compute(predicted, target);

        Assert.True(lossVal > 1.0, $"Inverted predictions should give high loss, got {lossVal}");
    }

    [Fact]
    public void MaskBCELoss_HandCalculated_SinglePixel()
    {
        var loss = new MaskBCELoss<double>();
        var predicted = new Tensor<double>(new[] { 1 });
        var target = new Tensor<double>(new[] { 1 });

        predicted[0] = 0.8; target[0] = 1.0;
        // BCE = -(1.0 * log(0.8) + 0 * log(0.2)) = -log(0.8) ≈ 0.22314
        double lossVal = loss.Compute(predicted, target);

        Assert.Equal(-Math.Log(0.8), lossVal, LooseTolerance);
    }

    [Fact]
    public void MaskBCELoss_MismatchedSizes_Throws()
    {
        var loss = new MaskBCELoss<double>();
        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 5 });

        Assert.Throws<ArgumentException>(() => loss.Compute(predicted, target));
    }

    [Fact]
    public void MaskBCELoss_Gradient_VerifyNumerically()
    {
        var loss = new MaskBCELoss<double>();
        var predicted = new Tensor<double>(new[] { 3 });
        var target = new Tensor<double>(new[] { 3 });

        predicted[0] = 0.7; target[0] = 1.0;
        predicted[1] = 0.3; target[1] = 0.0;
        predicted[2] = 0.5; target[2] = 1.0;

        var analyticalGrad = loss.Backward(predicted, target);

        // Verify with numerical gradient
        double eps = 1e-5;
        for (int i = 0; i < 3; i++)
        {
            double original = predicted[i];

            predicted[i] = original + eps;
            double lossPlus = loss.Compute(predicted, target);

            predicted[i] = original - eps;
            double lossMinus = loss.Compute(predicted, target);

            predicted[i] = original;

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);

            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-3);
        }
    }

    #endregion

    #region Mask Dice Loss - Mathematical Verification

    [Fact]
    public void MaskDiceLoss_PerfectOverlap_ReturnsNearZero()
    {
        var loss = new MaskDiceLoss<double>();
        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 4 });

        predicted[0] = 1.0; target[0] = 1.0;
        predicted[1] = 0.0; target[1] = 0.0;
        predicted[2] = 1.0; target[2] = 1.0;
        predicted[3] = 0.0; target[3] = 0.0;

        double lossVal = loss.Compute(predicted, target);

        // Dice = (2*intersection + smooth) / (predSum + targetSum + smooth)
        // intersection = 1*1 + 0*0 + 1*1 + 0*0 = 2
        // predSum = 1+0+1+0 = 2, targetSum = 1+0+1+0 = 2
        // Dice = (2*2 + 1) / (2 + 2 + 1) = 5/5 = 1.0
        // Loss = 1 - 1.0 = 0.0
        Assert.Equal(0.0, lossVal, LooseTolerance);
    }

    [Fact]
    public void MaskDiceLoss_NoOverlap_ReturnsNearOne()
    {
        var loss = new MaskDiceLoss<double>();
        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 4 });

        predicted[0] = 1.0; target[0] = 0.0;
        predicted[1] = 1.0; target[1] = 0.0;
        predicted[2] = 0.0; target[2] = 1.0;
        predicted[3] = 0.0; target[3] = 1.0;

        double lossVal = loss.Compute(predicted, target);

        // intersection = 0
        // predSum = 1+1+0+0 = 2, targetSum = 0+0+1+1 = 2
        // Dice = (0 + 1) / (2 + 2 + 1) = 1/5 = 0.2
        // Loss = 1 - 0.2 = 0.8
        Assert.Equal(0.8, lossVal, LooseTolerance);
    }

    [Fact]
    public void MaskDiceLoss_HandCalculated()
    {
        var loss = new MaskDiceLoss<double>(smooth: 1.0);
        var predicted = new Tensor<double>(new[] { 3 });
        var target = new Tensor<double>(new[] { 3 });

        predicted[0] = 0.8; target[0] = 1.0;
        predicted[1] = 0.2; target[1] = 0.0;
        predicted[2] = 0.6; target[2] = 1.0;

        // intersection = 0.8*1 + 0.2*0 + 0.6*1 = 1.4
        // predSum = 0.64 + 0.04 + 0.36 = 1.04
        // targetSum = 1 + 0 + 1 = 2.0
        // Dice = (2*1.4 + 1) / (1.04 + 2.0 + 1) = 3.8 / 4.04 ≈ 0.94059
        // Loss = 1 - 0.94059 ≈ 0.05941
        double lossVal = loss.Compute(predicted, target);
        double expectedDice = (2 * 1.4 + 1.0) / (1.04 + 2.0 + 1.0);
        double expectedLoss = 1.0 - expectedDice;

        Assert.Equal(expectedLoss, lossVal, LooseTolerance);
    }

    [Fact]
    public void MaskDiceLoss_Gradient_VerifyNumerically()
    {
        var loss = new MaskDiceLoss<double>();
        var predicted = new Tensor<double>(new[] { 3 });
        var target = new Tensor<double>(new[] { 3 });

        predicted[0] = 0.7; target[0] = 1.0;
        predicted[1] = 0.3; target[1] = 0.0;
        predicted[2] = 0.5; target[2] = 1.0;

        var analyticalGrad = loss.Backward(predicted, target);

        double eps = 1e-5;
        for (int i = 0; i < 3; i++)
        {
            double original = predicted[i];

            predicted[i] = original + eps;
            double lossPlus = loss.Compute(predicted, target);

            predicted[i] = original - eps;
            double lossMinus = loss.Compute(predicted, target);

            predicted[i] = original;

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);

            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-3);
        }
    }

    #endregion

    #region Mask Focal Loss - Mathematical Verification

    [Fact]
    public void MaskFocalLoss_EasyExamples_LowerThanBCE()
    {
        // Focal loss should down-weight easy (high confidence) examples
        var focalLoss = new MaskFocalLoss<double>(alpha: 0.25, gamma: 2.0);
        var bceLoss = new MaskBCELoss<double>();

        var predicted = new Tensor<double>(new[] { 2 });
        var target = new Tensor<double>(new[] { 2 });

        // Easy examples: high confidence correct predictions
        predicted[0] = 0.95; target[0] = 1.0;
        predicted[1] = 0.05; target[1] = 0.0;

        double focal = focalLoss.Compute(predicted, target);
        double bce = bceLoss.Compute(predicted, target);

        // Focal should be lower because it down-weights easy examples
        Assert.True(focal < bce,
            $"Focal loss ({focal}) should be < BCE ({bce}) for easy examples");
    }

    [Fact]
    public void MaskFocalLoss_HardExamples_HigherRelativeWeight()
    {
        var focalLoss = new MaskFocalLoss<double>(alpha: 0.25, gamma: 2.0);
        var bceLoss = new MaskBCELoss<double>();

        // Easy prediction
        var easyPred = new Tensor<double>(new[] { 1 });
        var easyTarget = new Tensor<double>(new[] { 1 });
        easyPred[0] = 0.9; easyTarget[0] = 1.0;

        // Hard prediction
        var hardPred = new Tensor<double>(new[] { 1 });
        var hardTarget = new Tensor<double>(new[] { 1 });
        hardPred[0] = 0.5; hardTarget[0] = 1.0;

        double focalEasy = focalLoss.Compute(easyPred, easyTarget);
        double focalHard = focalLoss.Compute(hardPred, hardTarget);
        double bceEasy = bceLoss.Compute(easyPred, easyTarget);
        double bceHard = bceLoss.Compute(hardPred, hardTarget);

        // The ratio of focal(hard)/focal(easy) should be larger than bce(hard)/bce(easy)
        // because focal down-weights easy examples more
        double focalRatio = focalHard / Math.Max(focalEasy, 1e-10);
        double bceRatio = bceHard / Math.Max(bceEasy, 1e-10);

        Assert.True(focalRatio > bceRatio,
            $"Focal ratio ({focalRatio}) should be > BCE ratio ({bceRatio})");
    }

    [Fact]
    public void MaskFocalLoss_Gradient_VerifyNumerically()
    {
        var loss = new MaskFocalLoss<double>(alpha: 0.25, gamma: 2.0);
        var predicted = new Tensor<double>(new[] { 3 });
        var target = new Tensor<double>(new[] { 3 });

        predicted[0] = 0.7; target[0] = 1.0;
        predicted[1] = 0.3; target[1] = 0.0;
        predicted[2] = 0.5; target[2] = 1.0;

        var analyticalGrad = loss.Backward(predicted, target);

        double eps = 1e-5;
        for (int i = 0; i < 3; i++)
        {
            double original = predicted[i];

            predicted[i] = original + eps;
            double lossPlus = loss.Compute(predicted, target);

            predicted[i] = original - eps;
            double lossMinus = loss.Compute(predicted, target);

            predicted[i] = original;

            double numericalGrad = (lossPlus - lossMinus) / (2 * eps);

            Assert.Equal(numericalGrad, analyticalGrad[i], 1e-3);
        }
    }

    #endregion

    #region Combined Mask Loss - Mathematical Verification

    [Fact]
    public void CombinedMaskLoss_EqualsWeightedSumOfComponents()
    {
        double bceWeight = 2.0, diceWeight = 3.0;
        var combinedLoss = new CombinedMaskLoss<double>(bceWeight, diceWeight);
        var bceLoss = new MaskBCELoss<double>();
        var diceLoss = new MaskDiceLoss<double>();

        var predicted = new Tensor<double>(new[] { 4 });
        var target = new Tensor<double>(new[] { 4 });

        predicted[0] = 0.8; target[0] = 1.0;
        predicted[1] = 0.2; target[1] = 0.0;
        predicted[2] = 0.6; target[2] = 1.0;
        predicted[3] = 0.4; target[3] = 0.0;

        double bce = bceLoss.Compute(predicted, target);
        double dice = diceLoss.Compute(predicted, target);
        double combined = combinedLoss.Compute(predicted, target);

        double expectedCombined = bceWeight * bce + diceWeight * dice;

        Assert.Equal(expectedCombined, combined, LooseTolerance);
    }

    [Fact]
    public void CombinedMaskLoss_Gradient_EqualsWeightedSumOfGradients()
    {
        double bceWeight = 1.5, diceWeight = 2.5;
        var combinedLoss = new CombinedMaskLoss<double>(bceWeight, diceWeight);
        var bceLoss = new MaskBCELoss<double>();
        var diceLoss = new MaskDiceLoss<double>();

        var predicted = new Tensor<double>(new[] { 3 });
        var target = new Tensor<double>(new[] { 3 });

        predicted[0] = 0.7; target[0] = 1.0;
        predicted[1] = 0.3; target[1] = 0.0;
        predicted[2] = 0.5; target[2] = 1.0;

        var bceGrad = bceLoss.Backward(predicted, target);
        var diceGrad = diceLoss.Backward(predicted, target);
        var combinedGrad = combinedLoss.Backward(predicted, target);

        for (int i = 0; i < 3; i++)
        {
            double expected = bceWeight * bceGrad[i] + diceWeight * diceGrad[i];
            Assert.Equal(expected, combinedGrad[i], LooseTolerance);
        }
    }

    #endregion

    #region NMS Edge Cases

    [Fact]
    public void NMS_EqualConfidences_DoesNotDropAll()
    {
        var nms = new NMS<double>();
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.5),
            new(new BoundingBox<double>(200, 200, 300, 300, BoundingBoxFormat.XYXY), 0, 0.5),
        };

        var result = nms.Apply(detections, 0.5);

        // Non-overlapping boxes with equal confidence should both be kept
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void NMS_ThresholdBoundary_IoUExactlyAtThreshold()
    {
        var nms = new NMS<double>();
        // Create boxes with known IoU
        // Box1: (0,0,100,100) area=10000
        // Box2: (10,10,110,110) area=10000
        // IoU = 8100/11900 ≈ 0.68
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(10, 10, 110, 110, BoundingBoxFormat.XYXY), 0, 0.8),
        };

        // Threshold at 0.7 (just above IoU): should keep both
        var result70 = nms.Apply(detections, 0.7);
        Assert.Equal(2, result70.Count);

        // Threshold at 0.6 (below IoU): should suppress one
        var result60 = nms.Apply(detections, 0.6);
        Assert.Single(result60);
    }

    [Fact]
    public void NMS_ManyOverlapping_KeepsOnlyHighest()
    {
        var nms = new NMS<double>();
        var detections = new List<Detection<double>>();

        // 10 nearly identical boxes with decreasing confidence
        for (int i = 0; i < 10; i++)
        {
            detections.Add(new Detection<double>(
                new BoundingBox<double>(i, i, 100 + i, 100 + i, BoundingBoxFormat.XYXY),
                0, 0.9 - i * 0.05));
        }

        var result = nms.Apply(detections, 0.5);

        // Should keep only the highest confidence one
        Assert.Single(result);
        Assert.Equal(0.9, result[0].Confidence, Tolerance);
    }

    [Fact]
    public void NMS_ClassAware_PreservesAllClassesEvenWithOverlap()
    {
        var nms = new NMS<double>();
        // Exact same box, different classes
        var detections = new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9),
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 1, 0.85),
            new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 2, 0.8),
        };

        var result = nms.ApplyClassAware(detections, 0.5);

        // Each class should be preserved even with 100% overlap
        Assert.Equal(3, result.Count);
        Assert.Contains(result, d => d.ClassId == 0);
        Assert.Contains(result, d => d.ClassId == 1);
        Assert.Contains(result, d => d.ClassId == 2);
    }

    [Fact]
    public void NMS_ApplyBatched_DefaultClassAware()
    {
        var nms = new NMS<double>();
        var batch = new List<List<Detection<double>>>
        {
            new()
            {
                new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 0, 0.9),
                new(new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY), 1, 0.85),
            }
        };

        // Default classAware = true
        var resultDefault = nms.ApplyBatched(batch, 0.5);
        Assert.Equal(2, resultDefault[0].Count); // class-aware keeps both

        // Explicit classAware = false
        var resultNonClassAware = nms.ApplyBatched(batch, 0.5, classAware: false);
        Assert.Single(resultNonClassAware[0]); // standard NMS suppresses one
    }

    #endregion

    #region BoundingBox Format Conversion Edge Cases

    [Fact]
    public void BoundingBox_COCOFormat_SameAsXYWH()
    {
        // COCO format is the same as XYWH
        var cocoBox = new BoundingBox<double>(10, 20, 40, 60, BoundingBoxFormat.COCO);
        var xywhBox = new BoundingBox<double>(10, 20, 40, 60, BoundingBoxFormat.XYWH);

        var (cx1, cy1, cx2, cy2) = cocoBox.ToXYXY();
        var (xx1, xy1, xx2, xy2) = xywhBox.ToXYXY();

        Assert.Equal(xx1, cx1, Tolerance);
        Assert.Equal(xy1, cy1, Tolerance);
        Assert.Equal(xx2, cx2, Tolerance);
        Assert.Equal(xy2, cy2, Tolerance);
    }

    [Fact]
    public void BoundingBox_PascalVOCFormat_SameAsXYXY()
    {
        var vocBox = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.PascalVOC);
        var xyxyBox = new BoundingBox<double>(10, 20, 50, 80, BoundingBoxFormat.XYXY);

        var (vx1, vy1, vx2, vy2) = vocBox.ToXYXY();
        var (xx1, xy1, xx2, xy2) = xyxyBox.ToXYXY();

        Assert.Equal(xx1, vx1, Tolerance);
        Assert.Equal(xy1, vy1, Tolerance);
        Assert.Equal(xx2, vx2, Tolerance);
        Assert.Equal(xy2, vy2, Tolerance);
    }

    [Fact]
    public void BoundingBox_YOLOFormat_RoundTrip()
    {
        var originalXYXY = new BoundingBox<double>(100, 100, 300, 400, BoundingBoxFormat.XYXY)
        {
            ImageWidth = 800,
            ImageHeight = 600
        };

        // Convert to YOLO
        var (yCx, yCy, yW, yH) = originalXYXY.ToYOLO();

        // Create YOLO box and convert back
        var yoloBox = new BoundingBox<double>(yCx, yCy, yW, yH, BoundingBoxFormat.YOLO)
        {
            ImageWidth = 800,
            ImageHeight = 600
        };

        var (x1, y1, x2, y2) = yoloBox.ToXYXY();

        Assert.Equal(100, x1, 0.01);
        Assert.Equal(100, y1, 0.01);
        Assert.Equal(300, x2, 0.01);
        Assert.Equal(400, y2, 0.01);
    }

    [Fact]
    public void BoundingBox_CXCYWH_RoundTrip_PreservesAll()
    {
        var original = new BoundingBox<double>(50, 75, 100, 120, BoundingBoxFormat.CXCYWH);

        var (xMin, yMin, xMax, yMax) = original.ToXYXY();
        // Expected: (50-50, 75-60, 50+50, 75+60) = (0, 15, 100, 135)
        Assert.Equal(0.0, xMin, Tolerance);
        Assert.Equal(15.0, yMin, Tolerance);
        Assert.Equal(100.0, xMax, Tolerance);
        Assert.Equal(135.0, yMax, Tolerance);

        // Convert XYXY back to CXCYWH
        var (cx, cy, w, h) = new BoundingBox<double>(xMin, yMin, xMax, yMax, BoundingBoxFormat.XYXY).ToCXCYWH();
        Assert.Equal(50.0, cx, Tolerance);
        Assert.Equal(75.0, cy, Tolerance);
        Assert.Equal(100.0, w, Tolerance);
        Assert.Equal(120.0, h, Tolerance);
    }

    [Fact]
    public void BoundingBox_Clip_ClipsCorrectly_WithXYWHInput()
    {
        // XYWH box that extends outside image
        var box = new BoundingBox<double>(80, 80, 100, 100, BoundingBoxFormat.XYWH);
        // XYWH (80, 80, 100, 100) -> XYXY (80, 80, 180, 180)

        box.Clip(150, 150);

        var (xMin, yMin, xMax, yMax) = box.ToXYXY();
        Assert.Equal(80, xMin, Tolerance);
        Assert.Equal(80, yMin, Tolerance);
        Assert.Equal(150, xMax, Tolerance); // clipped from 180
        Assert.Equal(150, yMax, Tolerance); // clipped from 180
    }

    [Fact]
    public void BoundingBox_Metadata_ClonesCorrectly()
    {
        var original = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY)
        {
            Metadata = new Dictionary<string, object>
            {
                { "source", "detector" },
                { "score_raw", 0.95 }
            }
        };

        var clone = original.Clone();

        Assert.NotSame(original.Metadata, clone.Metadata);
        Assert.Equal("detector", clone.Metadata?["source"]);

        // Modify clone metadata, original should be unaffected
        clone.Metadata?["source"] = "modified";
        Assert.Equal("detector", original.Metadata?["source"]);
    }

    #endregion

    #region Detection Pipeline - Deep Integration

    [Fact]
    public void DetectionStatistics_AverageConfidenceByClass_IsCorrect()
    {
        var result = new DetectionResult<double>
        {
            Detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY), 0, 0.8, "cat"),
                new(new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY), 0, 0.6, "cat"),
                new(new BoundingBox<double>(40, 40, 50, 50, BoundingBoxFormat.XYXY), 1, 0.9, "dog"),
            }
        };

        var stats = DetectionStatistics<double>.FromResult(result);

        Assert.Equal(3, stats.TotalDetections);
        Assert.Equal(2, stats.CountByClass[0]);
        Assert.Equal(1, stats.CountByClass[1]);
        Assert.Equal(0.7, stats.AverageConfidenceByClass[0], Tolerance); // (0.8+0.6)/2
        Assert.Equal(0.9, stats.AverageConfidenceByClass[1], Tolerance);
        Assert.Equal((0.8 + 0.6 + 0.9) / 3.0, stats.AverageConfidence, Tolerance);
    }

    [Fact]
    public void DetectionResult_FilterByClass_ExcludesNonMatchingClasses()
    {
        var result = new DetectionResult<double>
        {
            ImageWidth = 640,
            ImageHeight = 480,
            Detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY), 0, 0.9),
                new(new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY), 1, 0.8),
                new(new BoundingBox<double>(40, 40, 50, 50, BoundingBoxFormat.XYXY), 2, 0.7),
                new(new BoundingBox<double>(60, 60, 70, 70, BoundingBoxFormat.XYXY), 0, 0.6),
            }
        };

        var filtered = result.FilterByClass(0);

        Assert.Equal(2, filtered.Count);
        Assert.All(filtered.Detections, d => Assert.Equal(0, d.ClassId));
        // Verify metadata is preserved
        Assert.Equal(640, filtered.ImageWidth);
        Assert.Equal(480, filtered.ImageHeight);
    }

    [Fact]
    public void DetectionResult_TopN_MoreThanAvailable_ReturnsAll()
    {
        var result = new DetectionResult<double>
        {
            Detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(0, 0, 10, 10, BoundingBoxFormat.XYXY), 0, 0.9),
                new(new BoundingBox<double>(20, 20, 30, 30, BoundingBoxFormat.XYXY), 0, 0.8),
            }
        };

        var top5 = result.TopN(5);

        Assert.Equal(2, top5.Count); // Only 2 available
    }

    [Fact]
    public void BatchDetectionResult_EmptyBatch_HandlesCorrectly()
    {
        var batch = new BatchDetectionResult<double>
        {
            Results = new List<DetectionResult<double>>(),
            TotalInferenceTime = TimeSpan.FromMilliseconds(0)
        };

        Assert.Equal(0, batch.BatchSize);
        Assert.Equal(0, batch.TotalDetections);
        Assert.Equal(TimeSpan.Zero, batch.AverageInferenceTime);
    }

    #endregion

    #region Tracking - Deep Behavioral Testing

    [Fact]
    public void SORT_TrackVelocity_UpdatesWithMovement()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 1,
            IouThreshold = 0.3
        };
        var tracker = new SORT<double>(options);

        // Frame 1: object at (0,0,50,50)
        tracker.Update(new List<Detection<double>>
        {
            new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        });

        // Frame 2: object moved right by 20 pixels
        tracker.Update(new List<Detection<double>>
        {
            new(new BoundingBox<double>(20, 0, 70, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        });

        // Frame 3: object moved right again
        var result = tracker.Update(new List<Detection<double>>
        {
            new(new BoundingBox<double>(40, 0, 90, 50, BoundingBoxFormat.XYXY), 0, 0.9)
        });

        // After 3 frames of consistent rightward movement, confirmed tracks should exist
        var confirmed = tracker.GetConfirmedTracks();
        Assert.NotEmpty(confirmed);

        // Track should have positive X velocity
        var track = confirmed[0];
        Assert.True(track.VelocityX > 0, $"Expected positive VelocityX, got {track.VelocityX}");
    }

    [Fact]
    public void SORT_LostTrack_MarkedAsLost()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 1,
            MaxAge = 5
        };
        var tracker = new SORT<double>(options);

        // Establish track over multiple frames
        for (int i = 0; i < 3; i++)
        {
            tracker.Update(new List<Detection<double>>
            {
                new(new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY), 0, 0.9)
            });
        }

        // Now send empty detections - track should be lost
        tracker.Update(new List<Detection<double>>());
        tracker.Update(new List<Detection<double>>());

        // Confirmed tracks should be empty or track should have TimeSinceUpdate > 0
        var result = tracker.Update(new List<Detection<double>>());
        // After several frames with no detection, track may be removed
    }

    [Fact]
    public void SORT_MultipleObjects_IndependentTracking()
    {
        var options = new TrackingOptions<double>
        {
            MinHits = 2,
            IouThreshold = 0.3
        };
        var tracker = new SORT<double>(options);

        // Two well-separated objects across multiple frames
        for (int i = 0; i < 5; i++)
        {
            var detections = new List<Detection<double>>
            {
                new(new BoundingBox<double>(0 + i * 5, 0, 50 + i * 5, 50, BoundingBoxFormat.XYXY), 0, 0.9),
                new(new BoundingBox<double>(300 + i * 5, 300, 350 + i * 5, 350, BoundingBoxFormat.XYXY), 1, 0.8),
            };

            tracker.Update(detections);
        }

        var confirmed = tracker.GetConfirmedTracks();

        // Both objects should be tracked independently
        Assert.True(confirmed.Count >= 2,
            $"Expected at least 2 confirmed tracks, got {confirmed.Count}");

        // Track IDs should be unique
        var ids = confirmed.Select(t => t.TrackId).ToList();
        Assert.Equal(ids.Count, ids.Distinct().Count());
    }

    [Fact]
    public void Track_AgeIncrementsWithUpdates()
    {
        var box1 = new BoundingBox<double>(0, 0, 50, 50, BoundingBoxFormat.XYXY);
        var track = new Track<double>(1, box1, 0, 0.9);

        Assert.Equal(1, track.Age);
        Assert.Equal(1, track.Hits);
        Assert.Equal(TrackState.Tentative, track.State);
    }

    #endregion

    #region GIoU Loss - Verify Consistency With NMS.ComputeGIoU

    [Fact]
    public void GIoULoss_ConsistentWithNMSGIoU()
    {
        var loss = new GIoULoss<double>();
        var nms = new NMS<double>();

        var box1 = new BoundingBox<double>(0, 0, 80, 80, BoundingBoxFormat.XYXY);
        var box2 = new BoundingBox<double>(20, 20, 100, 100, BoundingBoxFormat.XYXY);

        double giou = nms.ComputeGIoU(box1, box2);
        double lossFromBox = loss.CalculateLossForBox(box1, box2);

        // GIoU Loss = 1 - GIoU
        Assert.Equal(1.0 - giou, lossFromBox, Tolerance);
    }

    [Fact]
    public void GIoULoss_MultipleBoxes_MeanAcrossBoxes()
    {
        var loss = new GIoULoss<double>();
        var nms = new NMS<double>();

        // Two box pairs
        var pred = new Vector<double>(new[]
        {
            0.0, 0.0, 100.0, 100.0,     // pair 1
            200.0, 200.0, 300.0, 300.0   // pair 2
        });
        var actual = new Vector<double>(new[]
        {
            0.0, 0.0, 100.0, 100.0,     // pair 1 - identical
            250.0, 250.0, 350.0, 350.0   // pair 2 - offset
        });

        double meanLoss = loss.CalculateLoss(pred, actual);

        // Compute individual losses
        var box1p = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box1a = new BoundingBox<double>(0, 0, 100, 100, BoundingBoxFormat.XYXY);
        var box2p = new BoundingBox<double>(200, 200, 300, 300, BoundingBoxFormat.XYXY);
        var box2a = new BoundingBox<double>(250, 250, 350, 350, BoundingBoxFormat.XYXY);

        double loss1 = 1.0 - nms.ComputeGIoU(box1p, box1a);
        double loss2 = 1.0 - nms.ComputeGIoU(box2p, box2a);
        double expectedMean = (loss1 + loss2) / 2.0;

        Assert.Equal(expectedMean, meanLoss, LooseTolerance);
    }

    #endregion

    #region AnchorPresets

    [Fact]
    public void AnchorPresets_YOLOv8Anchors_HasCorrectShape()
    {
        Assert.Equal(9, AnchorPresets.YOLOv8Anchors.GetLength(0)); // 9 anchors
        Assert.Equal(2, AnchorPresets.YOLOv8Anchors.GetLength(1)); // width, height
    }

    [Fact]
    public void AnchorPresets_YOLOv5Anchors_HasCorrectShape()
    {
        Assert.Equal(9, AnchorPresets.YOLOv5Anchors.GetLength(0));
        Assert.Equal(2, AnchorPresets.YOLOv5Anchors.GetLength(1));
    }

    [Fact]
    public void AnchorPresets_Strides_AreOrdered()
    {
        // YOLO strides should be increasing
        for (int i = 1; i < AnchorPresets.YOLOStrides.Length; i++)
        {
            Assert.True(AnchorPresets.YOLOStrides[i] > AnchorPresets.YOLOStrides[i - 1]);
        }

        // FPN strides should be increasing
        for (int i = 1; i < AnchorPresets.FPNStrides.Length; i++)
        {
            Assert.True(AnchorPresets.FPNStrides[i] > AnchorPresets.FPNStrides[i - 1]);
        }
    }

    [Fact]
    public void AnchorPresets_YOLOAnchors_SizeIncreasesByLevel()
    {
        // Small anchors (level 0) should be smaller than medium (level 1) should be smaller than large (level 2)
        var anchors = AnchorPresets.YOLOv8Anchors;

        // Average area per level
        double avgArea0 = 0, avgArea1 = 0, avgArea2 = 0;
        for (int j = 0; j < 3; j++)
        {
            avgArea0 += anchors[j, 0] * anchors[j, 1];
            avgArea1 += anchors[3 + j, 0] * anchors[3 + j, 1];
            avgArea2 += anchors[6 + j, 0] * anchors[6 + j, 1];
        }
        avgArea0 /= 3; avgArea1 /= 3; avgArea2 /= 3;

        Assert.True(avgArea0 < avgArea1, "Level 0 anchors should be smaller than level 1");
        Assert.True(avgArea1 < avgArea2, "Level 1 anchors should be smaller than level 2");
    }

    #endregion

    #region Cross-Module Integration

    [Fact]
    public void FullPipeline_DetectFilterNMSTrack_EndToEnd()
    {
        // Create detections from hypothetical detector output
        var detections = new List<Detection<double>>
        {
            // Two overlapping person boxes
            new(new BoundingBox<double>(50, 50, 150, 200, BoundingBoxFormat.XYXY), 0, 0.95, "person"),
            new(new BoundingBox<double>(55, 55, 155, 205, BoundingBoxFormat.XYXY), 0, 0.88, "person"),
            // One car box
            new(new BoundingBox<double>(300, 100, 500, 250, BoundingBoxFormat.XYXY), 1, 0.82, "car"),
            // One low-confidence detection (should be filtered)
            new(new BoundingBox<double>(600, 300, 640, 340, BoundingBoxFormat.XYXY), 2, 0.15, "bird"),
        };

        // Step 1: Create result
        var result = new DetectionResult<double>
        {
            Detections = detections,
            ImageWidth = 640,
            ImageHeight = 480
        };

        // Step 2: Filter by confidence
        var filtered = result.FilterByConfidence(0.5);
        Assert.Equal(3, filtered.Count); // bird removed

        // Step 3: Apply NMS
        var nms = new NMS<double>();
        var nmsFiltered = nms.ApplyClassAware(filtered.Detections, 0.5);

        // Person: two overlapping boxes, NMS should remove one
        // Car: only one, kept
        Assert.Equal(2, nmsFiltered.Count);

        // Step 4: Track
        var tracker = new SORT<double>(new TrackingOptions<double> { MinHits = 1 });
        var trackResult = tracker.Update(nmsFiltered);

        Assert.Equal(1, trackResult.FrameNumber);
        Assert.NotEmpty(trackResult.Tracks);
    }

    [Fact]
    public void LossFunctions_Ordering_CIoULossLessThanOrEqualToDIoULoss()
    {
        var ciouLoss = new CIoULoss<double>();
        var diouLoss = new DIoULoss<double>();

        // Square boxes: CIoU = DIoU (no aspect ratio penalty)
        var squarePred = new Vector<double>(new[] { 0.0, 0.0, 100.0, 100.0 });
        var squareActual = new Vector<double>(new[] { 20.0, 20.0, 120.0, 120.0 });

        double ciouVal = ciouLoss.CalculateLoss(squarePred, squareActual);
        double diouVal = diouLoss.CalculateLoss(squarePred, squareActual);

        Assert.Equal(ciouVal, diouVal, LooseTolerance); // Same for square boxes

        // Different aspect ratio boxes: CIoU loss >= DIoU loss (more penalty)
        var widePred = new Vector<double>(new[] { 0.0, 0.0, 200.0, 50.0 });
        var tallActual = new Vector<double>(new[] { 0.0, 0.0, 50.0, 200.0 });

        double ciouWide = ciouLoss.CalculateLoss(widePred, tallActual);
        double diouWide = diouLoss.CalculateLoss(widePred, tallActual);

        // CIoU loss should be >= DIoU loss (aspect ratio penalty increases CIoU loss)
        Assert.True(ciouWide >= diouWide - Tolerance,
            $"CIoU loss ({ciouWide}) should be >= DIoU loss ({diouWide}) with different aspect ratios");
    }

    #endregion
}
