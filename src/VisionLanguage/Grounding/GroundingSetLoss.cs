using AiDotNet.Helpers;
using AiDotNet.Solvers.Assignment;

namespace AiDotNet.VisionLanguage.Grounding;

/// <summary>The terms of a <see cref="GroundingSetLoss{T}"/> for one image.</summary>
/// <param name="Total">The weighted sum of the terms, a one-element tensor.</param>
/// <param name="Classification">The sigmoid focal loss over every query's logits, per ground-truth box.</param>
/// <param name="Box">The L1 distance of the matched boxes, per ground-truth box.</param>
/// <param name="GeneralizedIoU">One minus the GIoU of the matched boxes, per ground-truth box.</param>
/// <param name="QueryOfTarget">The query the Hungarian matching assigned to each ground-truth box.</param>
internal sealed record GroundingSetLossTerms<T>(Tensor<T> Total, Tensor<T> Classification, Tensor<T> Box,
    Tensor<T> GeneralizedIoU, int[] QueryOfTarget);

/// <summary>The loss of a DETR-style detector over all its outputs.</summary>
/// <param name="Total">The sum of every output's weighted loss, a one-element tensor.</param>
/// <param name="Final">The last decoder layer's terms.</param>
/// <param name="Auxiliary">The earlier decoder layers' terms, first to last.</param>
/// <param name="Encoder">The two-stage encoder proposals' terms.</param>
internal sealed record GroundingDetectorLoss<T>(Tensor<T> Total, GroundingSetLossTerms<T> Final,
    IReadOnlyList<GroundingSetLossTerms<T>> Auxiliary, GroundingSetLossTerms<T> Encoder);

/// <summary>
/// The set-prediction loss of the DETR-family grounding detectors (DETR, Carion et al. 2020, section 3.1; Deformable
/// DETR / DINO's focal variant; Grounding DINO, Liu et al. 2023, section 3.5; OWL-ViT, Minderer et al. 2022,
/// section 3.2): queries and ground-truth boxes are matched one-to-one by the Hungarian algorithm on a cost of the focal
/// class cost, the L1 box distance and the negative GIoU; every query then takes a sigmoid focal loss against its
/// target (its matched box's positive logits, or none), and the matched queries an L1 and a GIoU box loss, each summed
/// and divided by the number of ground-truth boxes.
/// </summary>
/// <remarks>
/// <para>Logit columns are whatever the head scores a query against: text tokens for Grounding DINO's contrastive head
/// (the target is the token span of the box's phrase), text queries for OWL-ViT (the box's query), or one objectness
/// column for a class-agnostic proposer. Grounding DINO's defaults: matching costs class 1, L1 5, GIoU 2; loss weights
/// class 2, L1 5, GIoU 2; focal alpha 0.25 and gamma 2.</para>
/// <para>The cost and the matching are not differentiated (the assignment is a constant of the loss); the loss terms
/// are engine operations on the predictions and so carry gradients to the head.</para>
/// </remarks>
internal sealed class GroundingSetLoss<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // The reference's epsilon inside the focal matching cost's logarithms.
    private const double ReferenceCostEpsilon = 1e-8;

    // Half's smallest normal value, 2^-14.
    private const double HalfSmallestNormal = 6.103515625e-5;

    /// <summary>
    /// The epsilon inside the matching cost's logarithms: the reference's 1e-8, or, for a type that rounds 1e-8 to zero
    /// (Half), its smallest normal value, so a saturated sigmoid still gives a finite cost the assignment can use.
    /// </summary>
    internal static double CostEpsilon(INumericOperations<T> numOps)
        => numOps.ToDouble(numOps.FromDouble(ReferenceCostEpsilon)) > 0 ? ReferenceCostEpsilon : HalfSmallestNormal;

    private readonly IEngine _engine;

    public GroundingSetLoss(IEngine engine) => _engine = engine ?? throw new ArgumentNullException(nameof(engine));

    public double ClassCost { get; init; } = 1.0;
    public double BoxCost { get; init; } = 5.0;
    public double GIoUCost { get; init; } = 2.0;
    public double ClassWeight { get; init; } = 2.0;
    public double BoxWeight { get; init; } = 5.0;
    public double GIoUWeight { get; init; } = 2.0;
    public double FocalAlpha { get; init; } = 0.25;
    public double FocalGamma { get; init; } = 2.0;

    /// <summary>
    /// The full training loss of a <see cref="GroundingQueryDecoder{T}"/> on one image, as DINO and Grounding DINO
    /// train it: the set loss of the last decoder layer, plus the auxiliary set loss of every earlier decoder layer,
    /// plus the set loss of the encoder's two-stage proposals; each output is matched to the targets on its own.
    /// </summary>
    public GroundingDetectorLoss<T> Compute(GroundingQueryPrediction<T> prediction, Tensor<T> targetBoxes, Tensor<T> targetLabels)
    {
        if (prediction is null) throw new ArgumentNullException(nameof(prediction));
        var final = Compute(prediction.Boxes, prediction.Logits, targetBoxes, targetLabels);
        var auxiliary = new List<GroundingSetLossTerms<T>>();
        for (int layer = 0; layer < prediction.LayerBoxes.Count - 1; layer++)
            auxiliary.Add(Compute(prediction.LayerBoxes[layer], prediction.LayerLogits[layer], targetBoxes, targetLabels));
        var encoder = Compute(prediction.EncoderBoxes, prediction.EncoderLogits, targetBoxes, targetLabels);
        var total = auxiliary.Aggregate(_engine.TensorAdd(final.Total, encoder.Total), (sum, terms) => _engine.TensorAdd(sum, terms.Total));
        return new GroundingDetectorLoss<T>(total, final, auxiliary, encoder);
    }

    /// <summary>The loss of one image's predictions.</summary>
    /// <param name="boxes">Predicted boxes <c>[Q, 4]</c>, normalized (cx, cy, w, h).</param>
    /// <param name="logits">Predicted logits <c>[Q, C]</c>.</param>
    /// <param name="targetBoxes">Ground-truth boxes <c>[G, 4]</c>, normalized (cx, cy, w, h); <c>G</c> may be 0.</param>
    /// <param name="targetLabels">Each ground-truth box's positive logit columns <c>[G, C]</c> as 0/1.</param>
    public GroundingSetLossTerms<T> Compute(Tensor<T> boxes, Tensor<T> logits, Tensor<T> targetBoxes, Tensor<T> targetLabels)
    {
        if (boxes is null) throw new ArgumentNullException(nameof(boxes));
        if (logits is null) throw new ArgumentNullException(nameof(logits));
        if (targetBoxes is null) throw new ArgumentNullException(nameof(targetBoxes));
        if (targetLabels is null) throw new ArgumentNullException(nameof(targetLabels));
        int q = boxes.Shape[0], c = logits.Shape[1], g = targetBoxes.Shape[0];
        if (boxes.Rank != 2 || boxes.Shape[1] != 4 || logits.Rank != 2 || logits.Shape[0] != q)
            throw new ArgumentException("The predictions must be [Q, 4] boxes and [Q, C] logits.", nameof(boxes));
        if (targetBoxes.Rank != 2 || targetBoxes.Shape[1] != 4 || targetLabels.Rank != 2 || targetLabels.Shape[0] != g || targetLabels.Shape[1] != c)
            throw new ArgumentException("The targets must be [G, 4] boxes and [G, C] labels.", nameof(targetBoxes));
        if (g > q) throw new ArgumentException($"{g} boxes cannot be matched one-to-one by {q} queries.", nameof(targetBoxes));

        var queryOfTarget = Match(boxes, logits, targetBoxes, targetLabels);
        double boxCount = Math.Max(1, g);
        var inverseCount = NumOps.FromDouble(1.0 / boxCount);

        // Each query's class target: its matched box's labels, or zeros (an extra all-zero target row).
        var targetOfQuery = Enumerable.Repeat(g, q).ToArray();
        for (int t = 0; t < g; t++) targetOfQuery[queryOfTarget[t]] = t;
        var labelsWithNone = _engine.TensorConcatenate(new[] { targetLabels, new Tensor<T>(new[] { 1, c }) }, 0);
        var y = _engine.TensorIndexSelect(labelsWithNone, new Tensor<int>(new[] { q }, Vector<int>.FromArray(targetOfQuery)), 0);
        var classLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(FocalLoss(logits, y), new[] { 0, 1 }, false), inverseCount);

        Tensor<T> boxLoss, giouLoss;
        if (g == 0)
        {
            var zero = _engine.TensorMultiplyScalar(_engine.ReduceSum(boxes, new[] { 0, 1 }, false), NumOps.Zero);
            boxLoss = zero;
            giouLoss = zero;
        }
        else
        {
            var matched = _engine.TensorIndexSelect(boxes, new Tensor<int>(new[] { g }, Vector<int>.FromArray(queryOfTarget)), 0);
            boxLoss = _engine.TensorMultiplyScalar(
                _engine.ReduceSum(_engine.TensorAbs(_engine.TensorSubtract(matched, targetBoxes)), new[] { 0, 1 }, false), inverseCount);
            var pairGIoU = Diagonal(PairwiseGeneralizedIoU(matched, targetBoxes));
            giouLoss = _engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.ScalarMinusTensor(NumOps.One, pairGIoU), new[] { 0 }, false),
                inverseCount);
        }

        var total = _engine.TensorAdd(_engine.TensorMultiplyScalar(classLoss, NumOps.FromDouble(ClassWeight)),
            _engine.TensorAdd(_engine.TensorMultiplyScalar(boxLoss, NumOps.FromDouble(BoxWeight)),
                _engine.TensorMultiplyScalar(giouLoss, NumOps.FromDouble(GIoUWeight))));
        return new GroundingSetLossTerms<T>(total, classLoss, boxLoss, giouLoss, queryOfTarget);
    }

    /// <summary>
    /// The Hungarian matching: the query assigned to each ground-truth box, minimizing the total of the focal class
    /// cost (positive minus negative focal cost, averaged over the box's positive columns), the L1 distance and the
    /// negative GIoU.
    /// </summary>
    internal int[] Match(Tensor<T> boxes, Tensor<T> logits, Tensor<T> targetBoxes, Tensor<T> targetLabels)
    {
        int q = boxes.Shape[0], g = targetBoxes.Shape[0], c = logits.Shape[1];
        if (g == 0) return Array.Empty<int>();
        Tensor<T> cost;
        using (new Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            var p = _engine.TensorSigmoid(logits);
            var oneMinusP = _engine.ScalarMinusTensor(NumOps.One, p);
            var eps = NumOps.FromDouble(CostEpsilon(NumOps));
            var gamma = NumOps.FromDouble(FocalGamma);
            var negative = _engine.TensorMultiplyScalar(_engine.TensorMultiply(_engine.TensorPow(p, gamma),
                _engine.TensorNegate(_engine.TensorLog(_engine.TensorAddScalar(oneMinusP, eps)))), NumOps.FromDouble(1 - FocalAlpha));
            var positive = _engine.TensorMultiplyScalar(_engine.TensorMultiply(_engine.TensorPow(oneMinusP, gamma),
                _engine.TensorNegate(_engine.TensorLog(_engine.TensorAddScalar(p, eps)))), NumOps.FromDouble(FocalAlpha));
            // [Q, C] x [C, G], each column divided by its box's positive-column count.
            var counts = _engine.TensorMax(_engine.ReduceSum(targetLabels, new[] { 1 }, false), NumOps.One).Reshape(new[] { 1, g });
            var classCost = _engine.TensorDivide(_engine.TensorMatMul(_engine.TensorSubtract(positive, negative), _engine.TensorTranspose(targetLabels)),
                _engine.TensorBroadcastTo(counts, new[] { q, g }));

            var predicted = _engine.TensorBroadcastTo(boxes.Reshape(new[] { q, 1, 4 }), new[] { q, g, 4 });
            var actual = _engine.TensorBroadcastTo(targetBoxes.Reshape(new[] { 1, g, 4 }), new[] { q, g, 4 });
            var l1 = _engine.ReduceSum(_engine.TensorAbs(_engine.TensorSubtract(predicted, actual)), new[] { 2 }, false);

            cost = _engine.TensorAdd(_engine.TensorMultiplyScalar(classCost, NumOps.FromDouble(ClassCost)),
                _engine.TensorSubtract(_engine.TensorMultiplyScalar(l1, NumOps.FromDouble(BoxCost)),
                    _engine.TensorMultiplyScalar(PairwiseGeneralizedIoU(boxes, targetBoxes), NumOps.FromDouble(GIoUCost))));
        }
        var assignment = new LinearAssignmentSolver<T>().Solve(cost.ToMatrix()).ToArray();
        var queryOfTarget = new int[g];
        for (int query = 0; query < q; query++)
            if (assignment[query] >= 0) queryOfTarget[assignment[query]] = query;
        return queryOfTarget;
    }

    // Sigmoid focal loss (Lin et al. 2017) per element: alpha_t (1 - p_t)^gamma BCE(x, y).
    private Tensor<T> FocalLoss(Tensor<T> logits, Tensor<T> y)
    {
        var notY = _engine.ScalarMinusTensor(NumOps.One, y);
        // log p = -softplus(-x), log(1 - p) = -softplus(x).
        var crossEntropy = _engine.TensorAdd(_engine.TensorMultiply(y, _engine.Softplus(_engine.TensorNegate(logits))),
            _engine.TensorMultiply(notY, _engine.Softplus(logits)));
        var p = _engine.TensorSigmoid(logits);
        var oneMinusPt = _engine.TensorAdd(_engine.TensorMultiply(p, notY), _engine.TensorMultiply(_engine.ScalarMinusTensor(NumOps.One, p), y));
        var alpha = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(y, NumOps.FromDouble(2 * FocalAlpha - 1)), NumOps.FromDouble(1 - FocalAlpha));
        return _engine.TensorMultiply(alpha, _engine.TensorMultiply(crossEntropy, _engine.TensorPow(oneMinusPt, NumOps.FromDouble(FocalGamma))));
    }

    /// <summary>The GIoU (Rezatofighi et al. 2019) of every pair of (cx, cy, w, h) boxes <c>[A, 4]</c> x <c>[B, 4]</c>: <c>[A, B]</c>.</summary>
    internal Tensor<T> PairwiseGeneralizedIoU(Tensor<T> a, Tensor<T> b)
    {
        int na = a.Shape[0], nb = b.Shape[0];
        var shape = new[] { na, nb };
        var half = NumOps.FromDouble(0.5);
        Tensor<T> Column(Tensor<T> x, int k) => _engine.TensorSlice(x, new[] { 0, k }, new[] { x.Shape[0], 1 });
        Tensor<T> Rows(Tensor<T> column) => _engine.TensorBroadcastTo(column, shape);
        Tensor<T> Columns(Tensor<T> column) => _engine.TensorBroadcastTo(column.Reshape(new[] { 1, nb }), shape);
        (Tensor<T> X1, Tensor<T> Y1, Tensor<T> X2, Tensor<T> Y2) Corners(Tensor<T> x, Func<Tensor<T>, Tensor<T>> spread)
        {
            var halfW = _engine.TensorMultiplyScalar(Column(x, 2), half);
            var halfH = _engine.TensorMultiplyScalar(Column(x, 3), half);
            return (spread(_engine.TensorSubtract(Column(x, 0), halfW)), spread(_engine.TensorSubtract(Column(x, 1), halfH)),
                spread(_engine.TensorAdd(Column(x, 0), halfW)), spread(_engine.TensorAdd(Column(x, 1), halfH)));
        }
        var (ax1, ay1, ax2, ay2) = Corners(a, Rows);
        var (bx1, by1, bx2, by2) = Corners(b, Columns);
        var inter = _engine.TensorMultiply(_engine.ReLU(_engine.TensorSubtract(_engine.TensorMin(ax2, bx2), _engine.TensorMax(ax1, bx1))),
            _engine.ReLU(_engine.TensorSubtract(_engine.TensorMin(ay2, by2), _engine.TensorMax(ay1, by1))));
        var areaA = _engine.TensorMultiply(_engine.TensorSubtract(ax2, ax1), _engine.TensorSubtract(ay2, ay1));
        var areaB = _engine.TensorMultiply(_engine.TensorSubtract(bx2, bx1), _engine.TensorSubtract(by2, by1));
        var union = _engine.TensorSubtract(_engine.TensorAdd(areaA, areaB), inter);
        var hull = _engine.TensorMultiply(_engine.TensorSubtract(_engine.TensorMax(ax2, bx2), _engine.TensorMin(ax1, bx1)),
            _engine.TensorSubtract(_engine.TensorMax(ay2, by2), _engine.TensorMin(ay1, by1)));
        return _engine.TensorSubtract(_engine.TensorDivide(inter, union), _engine.TensorDivide(_engine.TensorSubtract(hull, union), hull));
    }

    // The diagonal [n] of a square [n, n] matrix.
    private Tensor<T> Diagonal(Tensor<T> square)
    {
        int n = square.Shape[0];
        var identity = _engine.TensorOneHot<T>(new Tensor<int>(new[] { n }, Vector<int>.FromArray(Enumerable.Range(0, n).ToArray())), n);
        return _engine.ReduceSum(_engine.TensorMultiply(square, identity), new[] { 1 }, false);
    }
}
