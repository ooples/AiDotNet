using AiDotNet.Helpers;
using AiDotNet.Solvers.Assignment;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>
/// The boxes one image's prompt grounds: per box its centre and size (<c>cx, cy, w, h</c>, normalized by the image size)
/// and the phrase it names, the index of a period-separated phrase of the prompt.
/// </summary>
internal sealed record GroundingTarget(double[][] Boxes, int[] Phrases);

/// <summary>
/// Grounding DINO's training objective (Liu et al. 2023, section 3.5): each object query is scored against the prompt's
/// tokens, and queries are matched to the ground-truth boxes by the Hungarian algorithm on a cost of the focal token
/// score, L1 and GIoU. The matched queries then take the contrastive loss, a sigmoid focal loss over every scored
/// token whose target is the matched phrase's tokens, plus L1 and GIoU box losses. The same loss applies to every decoder
/// layer's output and to the encoder's selected queries. Weights and costs are those of the released training code (as
/// in Hugging Face's port): classification 2, L1 5, GIoU 2; matching costs 1, 5 and 2; focal alpha 0.25 and gamma 2.
/// </summary>
internal sealed class GroundingDinoLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>The tokens that separate phrases: [CLS], [SEP], "." and "?" in the BERT vocabulary, and padding.</summary>
    internal static readonly int[] Delimiters = { 101, 102, 1012, 1029, 0 };

    private readonly IEngine _engine;

    public GroundingDinoLoss(IEngine engine) => _engine = engine;

    public double ClassCost { get; init; } = 1.0;

    public double BoxCost { get; init; } = 5.0;

    public double GIoUCost { get; init; } = 2.0;

    public double ClassWeight { get; init; } = 2.0;

    public double BoxWeight { get; init; } = 5.0;

    public double GIoUWeight { get; init; } = 2.0;

    public double FocalAlpha { get; init; } = 0.25;

    public double FocalGamma { get; init; } = 2.0;

    /// <summary>
    /// Per image, one token map per phrase (1 on the phrase's tokens, over <paramref name="width"/> scored tokens): the
    /// runs of tokens between delimiters, in prompt order.
    /// </summary>
    public static List<double[]>[] PhraseMaps(int[,] inputIds, int width)
    {
        int b = inputIds.GetLength(0), l = inputIds.GetLength(1);
        var maps = new List<double[]>[b];
        for (int i = 0; i < b; i++)
        {
            maps[i] = new List<double[]>();
            double[]? current = null;
            for (int j = 0; j < l; j++)
            {
                if (Array.IndexOf(Delimiters, inputIds[i, j]) >= 0)
                {
                    current = null;
                    continue;
                }
                if (current is null)
                {
                    current = new double[width];
                    maps[i].Add(current);
                }
                if (j < width) current[j] = 1;
            }
        }
        return maps;
    }

    /// <summary>
    /// The weighted loss over the final layer, every earlier decoder layer and the encoder's selected queries. When
    /// <paramref name="terms"/> is given it receives each unweighted term under the released code's names
    /// (<c>loss_ce</c>, <c>loss_bbox</c>, <c>loss_giou</c>, with <c>_0</c>… for earlier layers and <c>_enc</c>).
    /// </summary>
    public Tensor<T> Compute(GroundingDinoOutput<T> output, int[,] inputIds, bool[,] tokenMask, IReadOnlyList<GroundingTarget> targets,
        IDictionary<string, double>? terms = null)
    {
        int b = inputIds.GetLength(0);
        if (targets.Count != b) throw new ArgumentException("One target per image.", nameof(targets));
        int width = output.Logits[^1].Shape[2];
        var maps = PhraseMaps(inputIds, width);
        for (int i = 0; i < b; i++)
        {
            var t = targets[i];
            if (t.Boxes.Length != t.Phrases.Length) throw new ArgumentException($"Image {i} needs one phrase per box.", nameof(targets));
            foreach (int phrase in t.Phrases)
                if (phrase < 0 || phrase >= maps[i].Count)
                    throw new ArgumentException($"Image {i} names phrase {phrase}; its prompt has {maps[i].Count}.", nameof(targets));
            foreach (var box in t.Boxes)
                if (box.Length != 4 || box[2] <= 0 || box[3] <= 0)
                    throw new ArgumentException($"Image {i} has a box that is not (cx, cy, w, h) with positive size.", nameof(targets));
        }
        double boxCount = Math.Max(1, targets.Sum(t => t.Boxes.Length));
        var valid = new bool[b, width];
        for (int i = 0; i < b; i++)
            for (int j = 0; j < Math.Min(width, tokenMask.GetLength(1)); j++)
                valid[i, j] = tokenMask[i, j];

        int layers = output.Logits.Length;
        var total = Terms(output.Logits[layers - 1], output.Boxes[layers - 1], maps, valid, targets, boxCount, "", terms);
        for (int layer = 0; layer < layers - 1; layer++)
            total = _engine.TensorAdd(total, Terms(output.Logits[layer], output.Boxes[layer], maps, valid, targets, boxCount, $"_{layer}", terms));
        return _engine.TensorAdd(total, Terms(output.EncoderLogits, output.EncoderBoxes, maps, valid, targets, boxCount, "_enc", terms));
    }

    private Tensor<T> Terms(Tensor<T> logits, Tensor<T> boxes, List<double[]>[] maps, bool[,] valid, IReadOnlyList<GroundingTarget> targets,
        double boxCount, string suffix, IDictionary<string, double>? terms)
    {
        int b = logits.Shape[0], queries = logits.Shape[1], width = logits.Shape[2];

        // Valid scored token positions depend only on the prompt's token mask; their targets and the matched query rows
        // depend on each step's live Hungarian assignment and are recomputed per replay via CompiledStepHostValue<T>,
        // with the step's phrase maps and gold boxes carried in an external-leaf supervision tensor soClosureObjectiveGuard
        // refreshes both image and prompt/box supervision on every replay.
        var indices = new List<int>();
        for (int i = 0; i < b; i++)
            for (int q = 0; q < queries; q++)
                for (int j = 0; j < width; j++)
                    if (valid[i, j])
                        indices.Add((i * queries + q) * width + j);

        int n = indices.Count;
        int totalBoxes = 0;
        var gold = new List<double>();
        for (int i = 0; i < b; i++)
        {
            totalBoxes += targets[i].Boxes.Length;
            for (int t = 0; t < targets[i].Boxes.Length; t++)
                gold.AddRange(targets[i].Boxes[t]);
        }

        double focalAlpha = FocalAlpha;
        double focalGamma = FocalGamma;
        double boxCost = BoxCost;
        double classCost = ClassCost;
        double giouCost = GIoUCost;
        var packedSupervision = PackSupervision(b, width, maps, valid, targets);
        var y = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "GroundingDinoClassTarget" + suffix,
            new[] { logits, boxes, packedSupervision },
            new[] { n },
            (inputs, dst) => FillMatchedClassTargets(inputs[0], inputs[1], inputs[2], b, queries, width,
                focalAlpha, focalGamma, boxCost, classCost, giouCost, dst));

        var scored = CvSelect(_engine.Reshape(logits, new[] { b * queries * width }), indices.ToArray());
        var notY = _engine.TensorAddScalar(_engine.TensorNegate(y), NumOps.One);
        // Binary cross-entropy with logits: log p = -softplus(-x), log(1 - p) = -softplus(x).
        var crossEntropy = _engine.TensorAdd(_engine.TensorMultiply(y, _engine.Softplus(_engine.TensorNegate(scored))),
            _engine.TensorMultiply(notY, _engine.Softplus(scored)));
        var probability = _engine.Sigmoid(scored);
        // 1 - p_t = p (1 - y) + (1 - p) y.
        var oneMinusPt = _engine.TensorAdd(_engine.TensorMultiply(probability, notY),
            _engine.TensorMultiply(_engine.TensorAddScalar(_engine.TensorNegate(probability), NumOps.One), y));
        var alpha = _engine.TensorAddScalar(
            _engine.TensorMultiplyScalar(y, NumOps.FromDouble((2.0 * FocalAlpha) - 1.0)),
            NumOps.FromDouble(1.0 - FocalAlpha));
        var focal = _engine.TensorMultiply(alpha, _engine.TensorMultiply(crossEntropy, _engine.TensorPower(oneMinusPt, NumOps.FromDouble(FocalGamma))));
        var classLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(focal, null), NumOps.FromDouble(1.0 / boxCount));

        // L1 and GIoU on the matched boxes.
        Tensor<T> boxLoss, giouLoss;
        if (totalBoxes == 0)
        {
            var zero = _engine.TensorMultiplyScalar(_engine.ReduceSum(boxes, null), NumOps.Zero);
            boxLoss = zero;
            giouLoss = zero;
        }
        else
        {
            var matchedQueries = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
                "GroundingDinoMatchedQueries" + suffix,
                new[] { logits, boxes, packedSupervision },
                new[] { totalBoxes },
                (inputs, dst) => FillMatchedQueryIndices(inputs[0], inputs[1], inputs[2], b, queries, width,
                    focalAlpha, focalGamma, boxCost, classCost, giouCost, dst));
            var matched = _engine.TensorEmbeddingLookupFromFloatIndices(_engine.Reshape(boxes, new[] { b * queries, 4 }), matchedQueries);
            var actual = new Tensor<double>(gold.ToArray(), new[] { totalBoxes, 4 }).Cast<T>();
            boxLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorAbs(_engine.TensorSubtract(matched, actual)), null),
                NumOps.FromDouble(1.0 / boxCount));
            giouLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(OneMinusGeneralizedIoU(Corners(matched), Corners(actual)), null),
                NumOps.FromDouble(1.0 / boxCount));
        }
        if (terms is not null)
        {
            terms["loss_ce" + suffix] = NumOps.ToDouble(classLoss[0]);
            terms["loss_bbox" + suffix] = NumOps.ToDouble(boxLoss[0]);
            terms["loss_giou" + suffix] = NumOps.ToDouble(giouLoss[0]);
        }
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(classLoss, NumOps.FromDouble(ClassWeight)),
            _engine.TensorAdd(_engine.TensorMultiplyScalar(boxLoss, NumOps.FromDouble(BoxWeight)),
                _engine.TensorMultiplyScalar(giouLoss, NumOps.FromDouble(GIoUWeight))));
    }

    private static Tensor<T> PackSupervision(int b, int width, List<double[]>[] maps, bool[,] valid, IReadOnlyList<GroundingTarget> targets)
    {
        var values = new List<double> { b, width };
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < width; j++)
                values.Add(valid[i, j] ? 1.0 : 0.0);
            values.Add(maps[i].Count);
            for (int m = 0; m < maps[i].Count; m++)
                for (int j = 0; j < width; j++)
                    values.Add(maps[i][m][j]);
            int boxCount = targets[i].Boxes.Length;
            values.Add(boxCount);
            for (int t = 0; t < boxCount; t++)
            {
                values.Add(targets[i].Phrases[t]);
                for (int k = 0; k < 4; k++)
                    values.Add(targets[i].Boxes[t][k]);
            }
        }
        return new Tensor<double>(values.ToArray(), new[] { values.Count }).Cast<T>();
    }

    private static void UnpackSupervision(Tensor<T> packed, out bool[,] valid, out List<double[]>[] maps, out GroundingTarget[] targets)
    {
        var span = packed.Contiguous().Cast<double>().ToArray();
        int cursor = 0;
        int b = (int)Math.Round(span[cursor++]);
        int width = (int)Math.Round(span[cursor++]);
        valid = new bool[b, width];
        maps = new List<double[]>[b];
        targets = new GroundingTarget[b];
        for (int i = 0; i < b; i++)
        {
            for (int j = 0; j < width; j++)
                valid[i, j] = span[cursor++] > 0.5;
            int mapCount = (int)Math.Round(span[cursor++]);
            maps[i] = new List<double[]>(mapCount);
            for (int m = 0; m < mapCount; m++)
            {
                var row = new double[width];
                for (int j = 0; j < width; j++)
                    row[j] = span[cursor++];
                maps[i].Add(row);
            }
            int boxCount = (int)Math.Round(span[cursor++]);
            var phrases = new int[boxCount];
            var boxes = new double[boxCount][];
            for (int t = 0; t < boxCount; t++)
            {
                phrases[t] = (int)Math.Round(span[cursor++]);
                boxes[t] = new double[4];
                for (int k = 0; k < 4; k++)
                    boxes[t][k] = span[cursor++];
            }
            targets[i] = new GroundingTarget(boxes, phrases);
        }
    }

    private static void FillMatchedClassTargets(Tensor<T> logits, Tensor<T> boxes, Tensor<T> packedSupervision, int b, int queries, int width,
        double focalAlpha, double focalGamma, double boxCost, double classCost, double giouCost, Tensor<T> destination)
    {
        var logitData = logits.Contiguous().Cast<double>().ToArray();
        var boxData = boxes.Contiguous().Cast<double>().ToArray();
        UnpackSupervision(packedSupervision, out var valid, out var maps, out var targets);
        var values = new double[destination.Length];
        int cursor = 0;
        for (int i = 0; i < b; i++)
        {
        var assignment = GroundingDinoHostMatcher.Match(logitData, boxData, i, queries, width, maps[i], targets[i],
                focalAlpha, focalGamma, boxCost, classCost, giouCost);
            var phraseOf = new int[queries];
            for (int q = 0; q < queries; q++) phraseOf[q] = -1;
            for (int t = 0; t < assignment.Length; t++) phraseOf[assignment[t]] = targets[i].Phrases[t];
            for (int q = 0; q < queries; q++)
                for (int j = 0; j < width; j++)
                {
                    if (!valid[i, j]) continue;
                    values[cursor++] = phraseOf[q] >= 0 ? maps[i][phraseOf[q]][j] : 0.0;
                }
        }
        var typed = new Tensor<double>(values, destination.Shape.ToArray()).Cast<T>();
        typed.AsSpan().CopyTo(destination.AsWritableSpan());
    }

    private static void FillMatchedQueryIndices(Tensor<T> logits, Tensor<T> boxes, Tensor<T> packedSupervision, int b, int queries, int width,
        double focalAlpha, double focalGamma, double boxCost, double classCost, double giouCost, Tensor<T> destination)
    {
        var logitData = logits.Contiguous().Cast<double>().ToArray();
        var boxData = boxes.Contiguous().Cast<double>().ToArray();
        UnpackSupervision(packedSupervision, out _, out var maps, out var targets);
        var values = new double[destination.Length];
        int cursor = 0;
        for (int i = 0; i < b; i++)
        {
        var assignment = GroundingDinoHostMatcher.Match(logitData, boxData, i, queries, width, maps[i], targets[i],
                focalAlpha, focalGamma, boxCost, classCost, giouCost);
            for (int t = 0; t < assignment.Length; t++)
                values[cursor++] = (i * queries) + assignment[t];
        }
        var typed = new Tensor<double>(values, destination.Shape.ToArray()).Cast<T>();
        typed.AsSpan().CopyTo(destination.AsWritableSpan());
    }

    /// <summary>Generalized IoU (Rezatofighi et al. 2019) of two (cx, cy, w, h) boxes.</summary>
    internal static double GeneralizedIoU(double[] a, double[] b) => GroundingDinoHostMatcher.GeneralizedIoU(a, b);

    /// <summary>1 − GIoU of corner boxes <c>[N, 4]</c>, on the tape, as the reference computes it (no epsilon; the
    /// engine's GIoU loss adds 1e-7 to the union and the hull).</summary>
    private Tensor<T> OneMinusGeneralizedIoU(Tensor<T> a, Tensor<T> b)
    {
        int n = a.Shape[0];
        Tensor<T> Column(Tensor<T> x, int k) => _engine.TensorSlice(x, new[] { 0, k }, new[] { n, 1 });
        Tensor<T> Min(Tensor<T> x, Tensor<T> y) => _engine.TensorNegate(_engine.TensorMax(_engine.TensorNegate(x), _engine.TensorNegate(y)));
        var (ax1, ay1, ax2, ay2) = (Column(a, 0), Column(a, 1), Column(a, 2), Column(a, 3));
        var (bx1, by1, bx2, by2) = (Column(b, 0), Column(b, 1), Column(b, 2), Column(b, 3));
        var inter = _engine.TensorMultiply(_engine.ReLU(_engine.TensorSubtract(Min(ax2, bx2), _engine.TensorMax(ax1, bx1))),
            _engine.ReLU(_engine.TensorSubtract(Min(ay2, by2), _engine.TensorMax(ay1, by1))));
        var areaA = _engine.TensorMultiply(_engine.TensorSubtract(ax2, ax1), _engine.TensorSubtract(ay2, ay1));
        var areaB = _engine.TensorMultiply(_engine.TensorSubtract(bx2, bx1), _engine.TensorSubtract(by2, by1));
        var union = _engine.TensorSubtract(_engine.TensorAdd(areaA, areaB), inter);
        var hull = _engine.TensorMultiply(_engine.TensorSubtract(_engine.TensorMax(ax2, bx2), Min(ax1, bx1)),
            _engine.TensorSubtract(_engine.TensorMax(ay2, by2), Min(ay1, by1)));
        var giou = _engine.TensorSubtract(_engine.TensorDivide(inter, union), _engine.TensorDivide(_engine.TensorSubtract(hull, union), hull));
        return _engine.Reshape(_engine.ScalarMinusTensor(NumOps.One, giou), new[] { n });
    }

    private Tensor<T> Corners(Tensor<T> boxes)
    {
        int count = boxes.Shape[0];
        var centers = _engine.TensorSlice(boxes, new[] { 0, 0 }, new[] { count, 2 });
        var half = _engine.TensorMultiplyScalar(_engine.TensorSlice(boxes, new[] { 0, 2 }, new[] { count, 2 }), NumOps.FromDouble(0.5));
        return _engine.TensorConcatenate(new[] { _engine.TensorSubtract(centers, half), _engine.TensorAdd(centers, half) }, 1);
    }

    private static Tensor<T> CvSelect(Tensor<T> source, int[] rows) => AiDotNet.ComputerVision.CvTensorOps<T>.Select(source, rows, 0);
}

internal static class GroundingDinoHostMatcher
{
    public static int[] Match(double[] logits, double[] boxes, int image, int queries, int width, List<double[]> maps, GroundingTarget target,
        double focalAlpha, double focalGamma, double boxCost, double classCostWeight, double giouCost)
    {
        int count = target.Boxes.Length;
        if (count == 0) return Array.Empty<int>();
        if (count > queries) throw new ArgumentException($"Image {image} has {count} boxes for {queries} queries.");
        var cost = new Matrix<double>(queries, count);
        var costSpan = cost.AsWritableSpan();
        for (int q = 0; q < queries; q++)
        {
            int row = image * queries + q;
            var box = new double[4];
            for (int k = 0; k < 4; k++) box[k] = boxes[row * 4 + k];
            for (int t = 0; t < count; t++)
            {
                var map = maps[target.Phrases[t]];
                double tokens = map.Sum();
                double classCost = 0;
                for (int j = 0; j < width; j++)
                {
                    if (map[j] == 0) continue;
                    double p = 1 / (1 + Math.Exp(-logits[row * width + j]));
                    double negative = (1 - focalAlpha) * Math.Pow(p, focalGamma) * -Math.Log(1 - p + 1e-8);
                    double positive = focalAlpha * Math.Pow(1 - p, focalGamma) * -Math.Log(p + 1e-8);
                    classCost += (positive - negative) * map[j] / tokens;
                }
                var gold = target.Boxes[t];
                double l1 = 0;
                for (int k = 0; k < 4; k++) l1 += Math.Abs(box[k] - gold[k]);
                double c = boxCost * l1 + classCostWeight * classCost - giouCost * GeneralizedIoU(box, gold);
                costSpan[(q * count) + t] = double.IsNaN(c) || double.IsInfinity(c) ? double.MaxValue : c;
            }
        }
        var assignment = new LinearAssignmentSolver<double>().Solve(cost);
        var assignSpan = assignment.AsSpan();
        var queryOf = new int[count];
        for (int q = 0; q < queries; q++)
            if (assignSpan[q] >= 0) queryOf[assignSpan[q]] = q;
        return queryOf;
    }

    public static double GeneralizedIoU(double[] a, double[] b)
    {
        double ax1 = a[0] - a[2] / 2, ay1 = a[1] - a[3] / 2, ax2 = a[0] + a[2] / 2, ay2 = a[1] + a[3] / 2;
        double bx1 = b[0] - b[2] / 2, by1 = b[1] - b[3] / 2, bx2 = b[0] + b[2] / 2, by2 = b[1] + b[3] / 2;
        double inter = Math.Max(0, Math.Min(ax2, bx2) - Math.Max(ax1, bx1)) * Math.Max(0, Math.Min(ay2, by2) - Math.Max(ay1, by1));
        double union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
        double hull = (Math.Max(ax2, bx2) - Math.Min(ax1, bx1)) * (Math.Max(ay2, by2) - Math.Min(ay1, by1));
        return inter / union - (hull - union) / hull;
    }
}
