namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// An independent double-precision evaluation of task-aligned YOLO training, written from the papers:
/// TOOD's assignment and normalization (Feng et al. 2021), GFL's distribution focal loss (Li et al. 2020),
/// CIoU (Zheng et al. 2020, with the engine's documented 1e-7 stabilizers and detached alpha) and the
/// YOLOv9/YOLOv10 box/class/DFL gains.
/// </summary>
internal static class TaskAlignedDetectionOracle
{
    internal sealed class Level
    {
        internal Level(double[] classLogits, double[] distribution, int height, int width, int stride)
        {
            ClassLogits = classLogits;
            Distribution = distribution;
            Height = height;
            Width = width;
            Stride = stride;
        }

        /// <summary>[batch, classes, height, width], row-major.</summary>
        internal double[] ClassLogits { get; }
        /// <summary>[batch, 4 * regMax, height, width], row-major.</summary>
        internal double[] Distribution { get; }
        internal int Height { get; }
        internal int Width { get; }
        internal int Stride { get; }
        internal int Cells => Height * Width;
    }

    internal sealed class Gold
    {
        internal Gold(int classId, double centerX, double centerY, double width, double height)
        {
            ClassId = classId;
            CenterX = centerX;
            CenterY = centerY;
            Width = width;
            Height = height;
        }

        internal int ClassId { get; }
        internal double CenterX { get; }
        internal double CenterY { get; }
        internal double Width { get; }
        internal double Height { get; }
    }

    internal sealed class Positive
    {
        internal int Image { get; set; }
        internal int Level { get; set; }
        internal int Cell { get; set; }
        internal int ClassId { get; set; }
        internal double Target { get; set; }
        internal double[] GoldPixels { get; set; } = Array.Empty<double>();
        internal double AnchorX { get; set; }
        internal double AnchorY { get; set; }
        /// <summary>CIoU's alpha at the evaluation point; the engine detaches it.</summary>
        internal double CiouAlpha { get; set; }
    }

    internal static List<Positive> Assign(IReadOnlyList<Level> levels, int batch, int classes, int regMax,
        IReadOnlyList<IReadOnlyList<Gold>> gold, int imageHeight, int imageWidth, int topK, double alpha = 0.5, double beta = 6.0)
    {
        var positives = new List<Positive>();
        for (int image = 0; image < batch; image++)
        {
            var anchors = new List<(int Level, int Cell, double X, double Y, double[] Box)>();
            for (int level = 0; level < levels.Count; level++)
                for (int cell = 0; cell < levels[level].Cells; cell++)
                {
                    int stride = levels[level].Stride;
                    double x = (cell % levels[level].Width + 0.5) * stride;
                    double y = (cell / levels[level].Width + 0.5) * stride;
                    var d = Distances(levels[level], image, cell, regMax);
                    anchors.Add((level, cell, x, y, new[] { x - d[0] * stride, y - d[1] * stride, x + d[2] * stride, y + d[3] * stride }));
                }

            var objects = gold[image];
            var boxes = objects.Select(g => new[]
            {
                (g.CenterX - g.Width / 2) * imageWidth, (g.CenterY - g.Height / 2) * imageHeight,
                (g.CenterX + g.Width / 2) * imageWidth, (g.CenterY + g.Height / 2) * imageHeight
            }).ToArray();
            var owner = Enumerable.Repeat(-1, anchors.Count).ToArray();
            var ownerIoU = new double[anchors.Count];
            var ownerMetric = new double[anchors.Count];
            for (int g = 0; g < objects.Count; g++)
            {
                var ranked = new List<(int Anchor, double Metric, double IoU)>();
                for (int a = 0; a < anchors.Count; a++)
                {
                    var anchor = anchors[a];
                    double inside = Math.Min(Math.Min(anchor.X - boxes[g][0], anchor.Y - boxes[g][1]),
                        Math.Min(boxes[g][2] - anchor.X, boxes[g][3] - anchor.Y));
                    if (inside <= 1e-9) continue;
                    var level = levels[anchor.Level];
                    double score = Sigmoid(level.ClassLogits[(image * classes + objects[g].ClassId) * level.Cells + anchor.Cell]);
                    double iou = PlainIoU(anchor.Box, boxes[g]);
                    ranked.Add((a, Math.Pow(score, alpha) * Math.Pow(iou, beta), iou));
                }
                foreach (var candidate in ranked.OrderByDescending(item => item.Metric).ThenBy(item => item.Anchor).Take(topK))
                {
                    if (owner[candidate.Anchor] >= 0 && candidate.IoU <= ownerIoU[candidate.Anchor]) continue;
                    owner[candidate.Anchor] = g;
                    ownerIoU[candidate.Anchor] = candidate.IoU;
                    ownerMetric[candidate.Anchor] = candidate.Metric;
                }
            }

            for (int a = 0; a < anchors.Count; a++)
            {
                int g = owner[a];
                if (g < 0) continue;
                double maxMetric = Enumerable.Range(0, anchors.Count).Where(i => owner[i] == g).Max(i => ownerMetric[i]);
                double maxIoU = Enumerable.Range(0, anchors.Count).Where(i => owner[i] == g).Max(i => ownerIoU[i]);
                var anchor = anchors[a];
                int stride = levels[anchor.Level].Stride;
                var predicted = anchor.Box.Select(value => value / stride).ToArray();
                var target = boxes[g].Select(value => value / stride).ToArray();
                positives.Add(new Positive
                {
                    Image = image, Level = anchor.Level, Cell = anchor.Cell, ClassId = objects[g].ClassId,
                    Target = ownerMetric[a] * maxIoU / (maxMetric + 1e-9), GoldPixels = boxes[g],
                    AnchorX = anchor.X, AnchorY = anchor.Y, CiouAlpha = Ciou(predicted, target, null).Alpha
                });
            }
        }
        return positives;
    }

    /// <summary>The total objective with the assignment, targets and CIoU alphas held fixed.</summary>
    internal static double Loss(IReadOnlyList<Level> levels, int batch, int classes, int regMax, IReadOnlyList<Positive> positives,
        double boxGain = 7.5, double classGain = 0.5, double dflGain = 1.5)
    {
        double mass = positives.Sum(positive => positive.Target);
        double normalizer = Math.Max(1, mass);
        double classification = 0;
        for (int level = 0; level < levels.Count; level++)
        {
            var targets = new double[levels[level].ClassLogits.Length];
            foreach (var positive in positives.Where(p => p.Level == level))
                targets[(positive.Image * classes + positive.ClassId) * levels[level].Cells + positive.Cell] = positive.Target;
            for (int index = 0; index < targets.Length; index++)
                classification += Softplus(levels[level].ClassLogits[index]) - targets[index] * levels[level].ClassLogits[index];
        }

        double box = 0;
        double dfl = 0;
        foreach (var positive in positives)
        {
            var level = levels[positive.Level];
            double stride = level.Stride;
            double gridX = positive.AnchorX / stride;
            double gridY = positive.AnchorY / stride;
            var d = Distances(level, positive.Image, positive.Cell, regMax);
            var predicted = new[] { gridX - d[0], gridY - d[1], gridX + d[2], gridY + d[3] };
            var gold = positive.GoldPixels.Select(value => value / stride).ToArray();
            box += positive.Target * (1 - Ciou(predicted, gold, positive.CiouAlpha).Value);

            var sides = new[] { gridX - gold[0], gridY - gold[1], gold[2] - gridX, gold[3] - gridY };
            for (int side = 0; side < 4; side++)
            {
                double y = Math.Min(Math.Max(sides[side], 0), regMax - 1 - 0.01);
                int lower = (int)Math.Floor(y);
                var logProbabilities = LogSoftmax(level, positive.Image, side, positive.Cell, regMax);
                dfl -= positive.Target / 4 * ((lower + 1 - y) * logProbabilities[lower] + (y - lower) * logProbabilities[lower + 1]);
            }
        }
        return classGain * classification / normalizer + boxGain * box / normalizer + dflGain * dfl / normalizer;
    }

    private static double[] Distances(Level level, int image, int cell, int regMax)
    {
        var result = new double[4];
        for (int side = 0; side < 4; side++)
        {
            var logProbabilities = LogSoftmax(level, image, side, cell, regMax);
            for (int bin = 0; bin < regMax; bin++) result[side] += Math.Exp(logProbabilities[bin]) * bin;
        }
        return result;
    }

    private static double[] LogSoftmax(Level level, int image, int side, int cell, int regMax)
    {
        var logits = Enumerable.Range(0, regMax)
            .Select(bin => level.Distribution[(image * 4 * regMax + side * regMax + bin) * level.Cells + cell]).ToArray();
        double maximum = logits.Max();
        double logSum = maximum + Math.Log(logits.Sum(value => Math.Exp(value - maximum)));
        return logits.Select(value => value - logSum).ToArray();
    }

    /// <summary>Engine CIoU: IoU - rho^2/c^2 - alpha v, stabilized by 1e-7; alpha is fixed when supplied.</summary>
    private static (double Value, double Alpha) Ciou(double[] p, double[] t, double? fixedAlpha)
    {
        const double eps = 1e-7;
        double Relu(double x) => Math.Max(0, x);
        double intersection = Relu(Math.Min(p[2], t[2]) - Math.Max(p[0], t[0])) * Relu(Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = Relu(p[2] - p[0]) * Relu(p[3] - p[1]) + Relu(t[2] - t[0]) * Relu(t[3] - t[1]) - intersection + eps;
        double iou = intersection / union;
        double dx = (p[0] + p[2]) / 2 - (t[0] + t[2]) / 2;
        double dy = (p[1] + p[3]) / 2 - (t[1] + t[3]) / 2;
        double diagonal = Math.Pow(Math.Max(p[2], t[2]) - Math.Min(p[0], t[0]), 2) + Math.Pow(Math.Max(p[3], t[3]) - Math.Min(p[1], t[1]), 2) + eps;
        double aspect = Math.Atan((Relu(t[2] - t[0]) + eps) / (Relu(t[3] - t[1]) + eps)) - Math.Atan((Relu(p[2] - p[0]) + eps) / (Relu(p[3] - p[1]) + eps));
        double v = 4 / (Math.PI * Math.PI) * aspect * aspect;
        double alpha = fixedAlpha ?? v / (1 - iou + v + eps);
        return (iou - (dx * dx + dy * dy) / diagonal - alpha * v, alpha);
    }

    private static double PlainIoU(double[] p, double[] t)
    {
        double intersection = Math.Max(0, Math.Min(p[2], t[2]) - Math.Max(p[0], t[0])) * Math.Max(0, Math.Min(p[3], t[3]) - Math.Max(p[1], t[1]));
        double union = (p[2] - p[0]) * (p[3] - p[1]) + (t[2] - t[0]) * (t[3] - t[1]) - intersection;
        return union > 0 ? Math.Max(0, intersection / union) : 0;
    }

    internal static double Sigmoid(double x) => 1 / (1 + Math.Exp(-x));

    private static double Softplus(double x) => x > 0 ? x + Math.Log(1 + Math.Exp(-x)) : Math.Log(1 + Math.Exp(x));
}
