using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics
{
    /// <summary>
    /// Verifies COCO/Pascal-VOC Average Precision against hand-computed values.
    /// </summary>
    /// <remarks>
    /// The 101-point interpolation makes some of these numbers non-obvious, so each test states
    /// the derivation. The key one to understand is the half-recall case: with recall capped at
    /// 0.5, exactly the recall samples 0.00 through 0.50 - that is 51 of the 101 - find a
    /// precision to inherit, so AP is 51/101 rather than the 0.5 an integral would give.
    /// </remarks>
    public class ObjectDetectionMetricsTests
    {
        private static Detection<double> Det(double x1, double y1, double x2, double y2, int classId, double confidence)
            => new Detection<double>(new BoundingBox<double>(x1, y1, x2, y2), classId, confidence);

        private static IReadOnlyList<IReadOnlyList<Detection<double>>> OneImage(params Detection<double>[] detections)
            => new List<IReadOnlyList<Detection<double>>> { detections };

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_PerfectPrediction_IsOne()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 10, 0, 0.9));

            Assert.Equal(1.0, metrics.AveragePrecision(predicted, truth, 0), 12);
            Assert.Equal(1.0, metrics.MeanAveragePrecision(predicted, truth), 12);
            Assert.Equal(1.0, metrics.MeanAveragePrecisionRange(predicted, truth), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_NoOverlap_IsZero()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(100, 100, 110, 110, 0, 0.9));

            Assert.Equal(0.0, metrics.AveragePrecision(predicted, truth, 0), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_NoPredictions_IsZero()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage();

            Assert.Equal(0.0, metrics.AveragePrecision(predicted, truth, 0), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_ClassWithNoGroundTruth_IsNotANumber()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 10, 7, 0.9));

            // Class 7 never occurs in the ground truth, so its recall - and therefore its AP -
            // is undefined rather than zero.
            Assert.True(double.IsNaN(metrics.AveragePrecision(predicted, truth, 7)));
        }

        [Fact(Timeout = 60000)]
        public async Task MeanAveragePrecision_ExcludesUndefinedClassesFromTheAverage()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));

            // A perfect class-0 prediction plus a hallucinated class-7 box. Class 7 has no ground
            // truth so it is not averaged in; mAP stays 1.0 for the class that does exist.
            var predicted = OneImage(
                Det(0, 0, 10, 10, 0, 0.9),
                Det(50, 50, 60, 60, 7, 0.8));

            Assert.Equal(1.0, metrics.MeanAveragePrecision(predicted, truth), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_HalfTheObjectsFound_ReflectsCappedRecall()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(
                Det(0, 0, 10, 10, 0, 1.0),
                Det(100, 100, 110, 110, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 10, 0, 0.9));

            // Precision 1.0 at recall 0.5, nothing beyond. Recall samples 0.00..0.50 inclusive
            // is 51 of the 101 sample points, each inheriting precision 1.0.
            Assert.Equal(51.0 / 101.0, metrics.AveragePrecision(predicted, truth, 0), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_DuplicateDetection_DoesNotReduceScore()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));

            // Two boxes on the same object: the higher-confidence one claims the ground truth and
            // the second is a false positive. Because full recall is already reached at rank 1,
            // the interpolated curve still integrates to 1.0 - matching pycocotools.
            var predicted = OneImage(
                Det(0, 0, 10, 10, 0, 0.9),
                Det(0, 0, 10, 10, 0, 0.8));

            Assert.Equal(1.0, metrics.AveragePrecision(predicted, truth, 0), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task AveragePrecision_GroundTruthIsClaimedOnlyOnce()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));

            var predicted = OneImage(
                Det(0, 0, 10, 10, 0, 0.9),
                Det(0, 0, 10, 10, 0, 0.8));

            // Rank 1 is the true positive, rank 2 cannot re-claim the same object: precision
            // falls to 1/2 while recall stays at 1.0.
            var (precision, recall) = metrics.PrecisionRecallCurve(predicted, truth, 0);

            Assert.Equal(new[] { 1.0, 0.5 }, precision);
            Assert.Equal(new[] { 1.0, 1.0 }, recall);
        }

        [Fact(Timeout = 60000)]
        public async Task PrecisionRecallCurve_RanksByConfidenceNotInputOrder()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));

            // The correct box is listed second but is the more confident, so it must be scored
            // first and the curve must open at precision 1.0.
            var predicted = OneImage(
                Det(500, 500, 510, 510, 0, 0.2),
                Det(0, 0, 10, 10, 0, 0.95));

            var (precision, recall) = metrics.PrecisionRecallCurve(predicted, truth, 0);

            Assert.Equal(new[] { 1.0, 0.5 }, precision);
            Assert.Equal(new[] { 1.0, 1.0 }, recall);
        }

        [Fact(Timeout = 60000)]
        public async Task MeanAveragePrecisionRange_CountsOnlyThresholdsTheOverlapClears()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();

            // Ground truth is 10x10 (area 100); the prediction is 10x7.2 (area 72) fully inside
            // it, so intersection 72 over union 100 gives IoU 0.72 exactly. Of the ten COCO
            // thresholds 0.50 .. 0.95, five (0.50, 0.55, 0.60, 0.65, 0.70) are cleared.
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 7.2, 0, 0.9));

            Assert.Equal(0.72, predicted[0][0].Box.IoU(truth[0][0].Box), 12);
            Assert.Equal(0.5, metrics.MeanAveragePrecisionRange(predicted, truth), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task MeanAveragePrecisionRange_UsesTenThresholdsInclusiveOfTheLast()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();

            // A perfect box clears every threshold including 0.95, so the average is 1.0. This
            // pins that the final threshold is not dropped by floating-point drift.
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 10, 0, 0.9));

            Assert.Equal(1.0, metrics.MeanAveragePrecisionRange(predicted, truth), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task ClassesAreScoredIndependently()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();

            // Two objects in the same place but different classes. A prediction of class 0 must
            // not be credited against the class-1 ground truth.
            var truth = OneImage(
                Det(0, 0, 10, 10, 0, 1.0),
                Det(100, 100, 110, 110, 1, 1.0));
            var predicted = OneImage(Det(100, 100, 110, 110, 0, 0.9));

            Assert.Equal(0.0, metrics.AveragePrecision(predicted, truth, 0), 12);
            Assert.Equal(0.0, metrics.AveragePrecision(predicted, truth, 1), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task MatchingIsScopedToTheImageThePredictionCameFrom()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();

            // The object is in image 0 but the detector reported it in image 1. Nothing matches.
            var truth = new List<IReadOnlyList<Detection<double>>>
            {
                new[] { Det(0, 0, 10, 10, 0, 1.0) },
                Array.Empty<Detection<double>>(),
            };
            var predicted = new List<IReadOnlyList<Detection<double>>>
            {
                Array.Empty<Detection<double>>(),
                new[] { Det(0, 0, 10, 10, 0, 0.9) },
            };

            Assert.Equal(0.0, metrics.AveragePrecision(predicted, truth, 0), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task MisalignedImageCounts_AreRejected()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = new List<IReadOnlyList<Detection<double>>>
            {
                Array.Empty<Detection<double>>(),
                Array.Empty<Detection<double>>(),
            };

            Assert.Throws<ArgumentException>(() => metrics.MeanAveragePrecision(predicted, truth));
        }

        [Fact(Timeout = 60000)]
        public async Task MeanAveragePrecisionRange_RejectsAnUnusableThresholdRange()
        {
            await Task.Yield();

            var metrics = new ObjectDetectionMetrics<double>();
            var truth = OneImage(Det(0, 0, 10, 10, 0, 1.0));
            var predicted = OneImage(Det(0, 0, 10, 10, 0, 0.9));

            Assert.Throws<ArgumentOutOfRangeException>(
                () => metrics.MeanAveragePrecisionRange(predicted, truth, step: 0.0));
            Assert.Throws<ArgumentOutOfRangeException>(
                () => metrics.MeanAveragePrecisionRange(predicted, truth, minIoU: 0.9, maxIoU: 0.5));
        }
    }
}
