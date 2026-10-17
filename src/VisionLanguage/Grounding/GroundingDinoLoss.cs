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
        var logitData = logits.ToArray().Select(NumOps.ToDouble).ToArray();
        var boxData = boxes.ToArray().Select(NumOps.ToDouble).ToArray();
        var assignments = new int[b][];
        for (int i = 0; i < b; i++) assignments[i] = Match(logitData, boxData, i, queries, width, maps[i], targets[i]);

        // Contrastive focal loss over the scored tokens of every query; a matched query's target is its phrase's tokens.
        var indices = new List<int>();
        var target = new List<double>();
        for (int i = 0; i < b; i++)
        {
            var phraseOf = new int[queries];
            for (int q = 0; q < queries; q++) phraseOf[q] = -1;
            for (int t = 0; t < assignments[i].Length; t++) phraseOf[assignments[i][t]] = targets[i].Phrases[t];
            for (int q = 0; q < queries; q++)
                for (int j = 0; j < width; j++)
                {
                    if (!valid[i, j]) continue;
                    indices.Add((i * queries + q) * width + j);
                    target.Add(phraseOf[q] >= 0 ? maps[i][phraseOf[q]][j] : 0);
                }
        }
        var scored = CvSelect(_engine.Reshape(logits, new[] { b * queries * width }), indices.ToArray());
        int n = indices.Count;
        var y = new Tensor<T>(target.Select(v => NumOps.FromDouble(v)).ToArray(), new[] { n });
        var notY = new Tensor<T>(target.Select(v => NumOps.FromDouble(1 - v)).ToArray(), new[] { n });
        // Binary cross-entropy with logits: log p = -softplus(-x), log(1 - p) = -softplus(x).
        var crossEntropy = _engine.TensorAdd(_engine.TensorMultiply(y, _engine.Softplus(_engine.TensorNegate(scored))),
            _engine.TensorMultiply(notY, _engine.Softplus(scored)));
        var probability = _engine.Sigmoid(scored);
        // 1 - p_t = p (1 - y) + (1 - p) y.
        var oneMinusPt = _engine.TensorAdd(_engine.TensorMultiply(probability, notY),
            _engine.TensorMultiply(_engine.TensorAddScalar(_engine.TensorNegate(probability), NumOps.One), y));
        var alpha = new Tensor<T>(target.Select(v => NumOps.FromDouble(FocalAlpha * v + (1 - FocalAlpha) * (1 - v))).ToArray(), new[] { n });
        var focal = _engine.TensorMultiply(alpha, _engine.TensorMultiply(crossEntropy, _engine.TensorPower(oneMinusPt, NumOps.FromDouble(FocalGamma))));
        var classLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(focal, null), NumOps.FromDouble(1.0 / boxCount));

        // L1 and GIoU on the matched boxes.
        var rows = new List<int>();
        var gold = new List<T>();
        for (int i = 0; i < b; i++)
            for (int t = 0; t < assignments[i].Length; t++)
            {
                rows.Add(i * queries + assignments[i][t]);
                gold.AddRange(targets[i].Boxes[t].Select(v => NumOps.FromDouble(v)));
            }
        Tensor<T> boxLoss, giouLoss;
        if (rows.Count == 0)
        {
            var zero = _engine.TensorMultiplyScalar(_engine.ReduceSum(boxes, null), NumOps.Zero);
            boxLoss = zero;
            giouLoss = zero;
        }
        else
        {
            var matched = CvSelect(_engine.Reshape(boxes, new[] { b * queries, 4 }), rows.ToArray());
            var actual = new Tensor<T>(gold.ToArray(), new[] { rows.Count, 4 });
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

    /// <summary>The query matched to each of one image's boxes: the minimum total of the class cost (the focal positive
    /// minus negative cost, averaged over the phrase's tokens), the L1 distance and the negative GIoU.</summary>
    private int[] Match(double[] logits, double[] boxes, int image, int queries, int width, List<double[]> maps, GroundingTarget target)
    {
        int count = target.Boxes.Length;
        if (count == 0) return Array.Empty<int>();
        if (count > queries) throw new ArgumentException($"Image {image} has {count} boxes for {queries} queries.");
        var cost = new Matrix<double>(queries, count);
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
                    double negative = (1 - FocalAlpha) * Math.Pow(p, FocalGamma) * -Math.Log(1 - p + 1e-8);
                    double positive = FocalAlpha * Math.Pow(1 - p, FocalGamma) * -Math.Log(p + 1e-8);
                    classCost += (positive - negative) * map[j] / tokens;
                }
                var gold = target.Boxes[t];
                double l1 = 0;
                for (int k = 0; k < 4; k++) l1 += Math.Abs(box[k] - gold[k]);
                double c = BoxCost * l1 + ClassCost * classCost - GIoUCost * GeneralizedIoU(box, gold);
                cost[q, t] = double.IsNaN(c) || double.IsInfinity(c) ? double.MaxValue : c;
            }
        }
        var assignment = new LinearAssignmentSolver<double>().Solve(cost);
        var queryOf = new int[count];
        for (int q = 0; q < queries; q++)
            if (assignment[q] >= 0) queryOf[assignment[q]] = q;
        return queryOf;
    }

    /// <summary>Generalized IoU (Rezatofighi et al. 2019) of two (cx, cy, w, h) boxes.</summary>
    internal static double GeneralizedIoU(double[] a, double[] b)
    {
        double ax1 = a[0] - a[2] / 2, ay1 = a[1] - a[3] / 2, ax2 = a[0] + a[2] / 2, ay2 = a[1] + a[3] / 2;
        double bx1 = b[0] - b[2] / 2, by1 = b[1] - b[3] / 2, bx2 = b[0] + b[2] / 2, by2 = b[1] + b[3] / 2;
        double inter = Math.Max(0, Math.Min(ax2, bx2) - Math.Max(ax1, bx1)) * Math.Max(0, Math.Min(ay2, by2) - Math.Max(ay1, by1));
        double union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
        double hull = (Math.Max(ax2, bx2) - Math.Min(ax1, bx1)) * (Math.Max(ay2, by2) - Math.Min(ay1, by1));
        return inter / union - (hull - union) / hull;
    }

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
