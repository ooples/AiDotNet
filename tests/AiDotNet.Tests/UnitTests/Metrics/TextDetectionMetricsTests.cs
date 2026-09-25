using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Metrics;
using Xunit;

namespace AiDotNetTests.UnitTests.Metrics
{
    /// <summary>
    /// Verifies polygon geometry and the ICDAR precision / recall / H-mean protocol against
    /// hand-computed values.
    /// </summary>
    public class TextDetectionMetricsTests
    {
        private static List<(double X, double Y)> Rect(double x1, double y1, double x2, double y2)
            => new List<(double X, double Y)> { (x1, y1), (x2, y1), (x2, y2), (x1, y2) };

        private static TextRegion<double> Region(double x1, double y1, double x2, double y2, double confidence = 1.0)
            => new TextRegion<double>(new BoundingBox<double>(x1, y1, x2, y2), confidence);

        private static TextRegion<double> PolygonRegion(List<(double X, double Y)> polygon, double confidence = 1.0)
            => TextRegion<double>.FromPolygon(polygon, confidence);

        private static IReadOnlyList<IReadOnlyList<TextRegion<double>>> OneImage(params TextRegion<double>[] regions)
            => new List<IReadOnlyList<TextRegion<double>>> { regions };

        [Fact(Timeout = 60000)]
        public async Task PolygonArea_UnitSquare_IsOne()
        {
            await Task.Yield();

            Assert.Equal(1.0, TextDetectionMetrics<double>.PolygonArea(Rect(0, 0, 1, 1)), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonArea_Rectangle_IsWidthTimesHeight()
        {
            await Task.Yield();

            Assert.Equal(6.0, TextDetectionMetrics<double>.PolygonArea(Rect(0, 0, 2, 3)), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonArea_IsIndependentOfWindingDirection()
        {
            await Task.Yield();

            var clockwise = new List<(double X, double Y)> { (0, 0), (0, 3), (2, 3), (2, 0) };

            Assert.Equal(6.0, TextDetectionMetrics<double>.PolygonArea(clockwise), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonArea_Triangle_IsHalfBaseTimesHeight()
        {
            await Task.Yield();

            var triangle = new List<(double X, double Y)> { (0, 0), (4, 0), (0, 3) };

            Assert.Equal(6.0, TextDetectionMetrics<double>.PolygonArea(triangle), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonArea_DegeneratePolygon_IsZero()
        {
            await Task.Yield();

            Assert.Equal(0.0, TextDetectionMetrics<double>.PolygonArea(new List<(double X, double Y)>()), 12);
            Assert.Equal(
                0.0,
                TextDetectionMetrics<double>.PolygonArea(new List<(double X, double Y)> { (0, 0), (1, 1) }),
                12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_IdenticalPolygons_IsOne()
        {
            await Task.Yield();

            Assert.Equal(1.0, TextDetectionMetrics<double>.PolygonIoU(Rect(0, 0, 10, 10), Rect(0, 0, 10, 10)), 10);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_DisjointPolygons_IsZero()
        {
            await Task.Yield();

            Assert.Equal(0.0, TextDetectionMetrics<double>.PolygonIoU(Rect(0, 0, 1, 1), Rect(5, 5, 6, 6)), 12);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_HalfOverlappingSquares_IsOneThird()
        {
            await Task.Yield();

            // Two unit squares offset by half a unit: intersection 0.5, union 1 + 1 - 0.5 = 1.5.
            double iou = TextDetectionMetrics<double>.PolygonIoU(Rect(0, 0, 1, 1), Rect(0.5, 0, 1.5, 1));

            Assert.Equal(1.0 / 3.0, iou, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_ContainedPolygon_IsAreaRatio()
        {
            await Task.Yield();

            // A 5x5 square inside a 10x10 square: intersection 25, union 100.
            double iou = TextDetectionMetrics<double>.PolygonIoU(Rect(0, 0, 5, 5), Rect(0, 0, 10, 10));

            Assert.Equal(0.25, iou, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_IsSymmetric()
        {
            await Task.Yield();

            var first = Rect(0, 0, 10, 10);
            var second = Rect(3, 3, 13, 13);

            Assert.Equal(
                TextDetectionMetrics<double>.PolygonIoU(first, second),
                TextDetectionMetrics<double>.PolygonIoU(second, first),
                10);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_RotatedQuadrilateral_MatchesTheAnalyticArea()
        {
            await Task.Yield();

            // A diamond with diagonals of 2 (area 2) inscribed in the 2x2 square (area 4). The
            // diamond lies entirely inside, so intersection 2 over union 4.
            var diamond = new List<(double X, double Y)> { (1, 0), (2, 1), (1, 2), (0, 1) };
            var square = Rect(0, 0, 2, 2);

            Assert.Equal(2.0, TextDetectionMetrics<double>.PolygonArea(diamond), 10);
            Assert.Equal(0.5, TextDetectionMetrics<double>.PolygonIoU(diamond, square), 10);
        }

        [Fact(Timeout = 60000)]
        public async Task PolygonIoU_HandlesOppositeWindingOrders()
        {
            await Task.Yield();

            // The same two squares, one wound clockwise and one counter-clockwise. Winding is a
            // representation detail and must not change the overlap.
            var counterClockwise = Rect(0, 0, 10, 10);
            var clockwise = new List<(double X, double Y)> { (0, 0), (0, 10), (10, 10), (10, 0) };

            Assert.Equal(1.0, TextDetectionMetrics<double>.PolygonIoU(counterClockwise, clockwise), 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_PerfectDetection_ScoresOneAcross()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));
            var predicted = OneImage(Region(0, 0, 10, 10, 0.9));

            var (precision, recall, hmean) = metrics.Evaluate(predicted, truth);

            Assert.Equal(1.0, precision, 10);
            Assert.Equal(1.0, recall, 10);
            Assert.Equal(1.0, hmean, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_MissedRegion_LowersRecallOnly()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10), Region(100, 100, 110, 110));
            var predicted = OneImage(Region(0, 0, 10, 10, 0.9));

            var (precision, recall, hmean) = metrics.Evaluate(predicted, truth);

            Assert.Equal(1.0, precision, 10);
            Assert.Equal(0.5, recall, 10);
            Assert.Equal(2.0 / 3.0, hmean, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_SpuriousRegion_LowersPrecisionOnly()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));
            var predicted = OneImage(Region(0, 0, 10, 10, 0.9), Region(100, 100, 110, 110, 0.8));

            var (precision, recall, hmean) = metrics.Evaluate(predicted, truth);

            Assert.Equal(0.5, precision, 10);
            Assert.Equal(1.0, recall, 10);
            Assert.Equal(2.0 / 3.0, hmean, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_MatchesOneToOne_SoDuplicatesCostPrecision()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));

            // Two boxes on the same word. Only one can claim it.
            var predicted = OneImage(Region(0, 0, 10, 10, 0.9), Region(0, 0, 10, 10, 0.8));

            var (precision, recall, _) = metrics.Evaluate(predicted, truth);

            Assert.Equal(0.5, precision, 10);
            Assert.Equal(1.0, recall, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_OverlapBelowThreshold_IsNotAMatch()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));

            // Contained 5x5 box gives IoU 0.25, under the 0.5 protocol threshold.
            var predicted = OneImage(Region(0, 0, 5, 5, 0.9));

            var (precision, recall, hmean) = metrics.Evaluate(predicted, truth);

            Assert.Equal(0.0, precision, 10);
            Assert.Equal(0.0, recall, 10);
            Assert.Equal(0.0, hmean, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_UsesPolygonWhenPresent()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();

            // Ground truth is a diamond, the prediction the square that encloses it. FromPolygon
            // derives each box from the polygon extent, so both carry the SAME bounding box -
            // comparing boxes would give IoU 1.0 and match at every threshold. The polygons
            // overlap at exactly 0.5, so a polygon comparison matches at the 0.5 protocol
            // threshold and stops matching at 0.6. That split is only observable if the polygon
            // is what gets compared.
            var truth = OneImage(PolygonRegion(new List<(double X, double Y)> { (1, 0), (2, 1), (1, 2), (0, 1) }));
            var predicted = OneImage(PolygonRegion(new List<(double X, double Y)> { (0, 0), (2, 0), (2, 2), (0, 2) }, 0.9));

            var (precisionAtHalf, recallAtHalf, _) = metrics.Evaluate(predicted, truth, 0.5);
            Assert.Equal(1.0, precisionAtHalf, 10);
            Assert.Equal(1.0, recallAtHalf, 10);

            var (precisionAtSixTenths, recallAtSixTenths, _) = metrics.Evaluate(predicted, truth, 0.6);
            Assert.Equal(0.0, precisionAtSixTenths, 10);
            Assert.Equal(0.0, recallAtSixTenths, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_RanksByConfidenceSoTheBetterBoxClaimsTheRegion()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));

            // The exact box is listed second but is more confident, so it claims the region and
            // the loose box becomes the false positive rather than the other way round.
            var predicted = OneImage(Region(0, 0, 12, 12, 0.4), Region(0, 0, 10, 10, 0.95));

            var (precision, recall, _) = metrics.Evaluate(predicted, truth);

            Assert.Equal(0.5, precision, 10);
            Assert.Equal(1.0, recall, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_MatchingIsScopedToTheImage()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = new List<IReadOnlyList<TextRegion<double>>>
            {
                new[] { Region(0, 0, 10, 10) },
                Array.Empty<TextRegion<double>>(),
            };
            var predicted = new List<IReadOnlyList<TextRegion<double>>>
            {
                Array.Empty<TextRegion<double>>(),
                new[] { Region(0, 0, 10, 10, 0.9) },
            };

            var (precision, recall, _) = metrics.Evaluate(predicted, truth);

            Assert.Equal(0.0, precision, 10);
            Assert.Equal(0.0, recall, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_NothingPredictedAndNothingToFind_ScoresOne()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage();
            var predicted = OneImage();

            var (precision, recall, hmean) = metrics.Evaluate(predicted, truth);

            Assert.Equal(1.0, precision, 10);
            Assert.Equal(1.0, recall, 10);
            Assert.Equal(1.0, hmean, 10);
        }

        [Fact(Timeout = 60000)]
        public async Task Evaluate_MisalignedImageCounts_AreRejected()
        {
            await Task.Yield();

            var metrics = new TextDetectionMetrics<double>();
            var truth = OneImage(Region(0, 0, 10, 10));
            var predicted = new List<IReadOnlyList<TextRegion<double>>>
            {
                Array.Empty<TextRegion<double>>(),
                Array.Empty<TextRegion<double>>(),
            };

            Assert.Throws<ArgumentException>(() => metrics.Evaluate(predicted, truth));
        }
    }
}
