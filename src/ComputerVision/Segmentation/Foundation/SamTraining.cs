using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// SAM's training objective (Kirillov et al. 2023, Appendix A, "Training algorithm"), shared by every model built on
/// <see cref="SamNetwork{T}"/>.
/// </summary>
/// <remarks>
/// <para>For an image and one ground-truth mask, the image is encoded once and the mask decoder run 11 times. The first
/// prompt is a foreground point drawn uniformly from the mask or, with equal probability, the mask's box with noise
/// (each coordinate moved by N(0, (0.1·side)²), at most 20 pixels). Each of the next 8 rounds adds a point drawn
/// uniformly from the error region of the previous prediction (foreground where it missed the mask, background where it
/// spilled over it); 2 more rounds add no point (one at a random place among those 8, one at the end). Every round after
/// the first also receives the previous round's unthresholded low-resolution logits as its mask prompt.</para>
/// <para>A single-point prompt is ambiguous, so it gets the three masks and only the lowest loss back-propagates. A
/// round's loss is focal:dice at 20:1 (the sigmoid focal loss of Lin et al. with α 0.25 and γ 2, and the dice loss) plus
/// the mean squared error between the predicted IoU and the IoU of the predicted mask with the ground truth. The
/// objective is the mean over the 11 rounds.</para>
/// </remarks>
internal sealed class SamTraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly SamNetwork<T> _network;
    private readonly double _focalWeight, _diceWeight, _gamma, _alpha;

    public SamTraining(IEngine engine, SamNetwork<T> network, double focalWeight, double diceWeight, double gamma, double alpha)
    {
        _engine = engine;
        _network = network;
        _focalWeight = focalWeight;
        _diceWeight = diceWeight;
        _gamma = gamma;
        _alpha = alpha;
    }

    /// <summary>
    /// The initial prompt and sampling draws are fixed once per training step. Re-evaluating the
    /// objective uses those same draws against the current predictions, without consuming random state.
    /// </summary>
    public Func<Tensor<T>> Objective(Tensor<T> image, Tensor<T> groundTruth, Random random)
    {
        int h = image.Shape[2], w = image.Shape[3];
        if (groundTruth.Length != h * w)
            throw new ArgumentException($"The ground truth is one {h}x{w} mask.", nameof(groundTruth));
        var values = groundTruth.ToArray();
        var gt = new double[h * w];
        for (int i = 0; i < gt.Length; i++) gt[i] = NumOps.ToDouble(values[i]) > 0.5 ? 1.0 : 0.0;
        var truth = new Tensor<T>(new[] { 1, 1, h, w }, new Vector<T>(gt.Select(NumOps.FromDouble).ToArray()));
        int noPromptRound = 1 + random.Next(9);
        Tensor<T>? points = null, labels = null, box = null;
        if (random.NextDouble() < 0.5)
            box = Box();
        else if (Sample(gt, w, random, v => v > 0.5) is { } first)
        {
            points = new Tensor<T>(new[] { NumOps.FromDouble(first.X), NumOps.FromDouble(first.Y) }, new[] { 1, 1, 2 });
            labels = Tensor<T>.CreateDefault(new[] { 1, 1 }, NumOps.One);
        }
        else box = Box();
        Tensor<T> Box() => new(new[] { 1, 4 }, new Vector<T>(NoisyBox(gt, h, w, random).Select(NumOps.FromDouble).ToArray()));

        // Double inputs preserve the sampling resolution even when the model uses Half.
        var draws = new Tensor<double>[11];
        for (int round = 0; round < draws.Length; round++)
            draws[round] = new Tensor<double>(new[] { round > 0 && round != noPromptRound && round != 10 ? random.NextDouble() : 0.0 }, new[] { 1 });
        return () => Loss(image, truth, points, labels, box, noPromptRound, draws, h, w);
    }

    private Tensor<T> Loss(Tensor<T> image, Tensor<T> truth, Tensor<T>? points, Tensor<T>? labels,
        Tensor<T>? box, int noPromptRound, Tensor<double>[] draws, int h, int w)
    {
        var net = _network;
        var embedding = net.EncodeImage(net.Preprocess(image));
        var positional = net.DensePositional();
        Tensor<T>? previous = null;
        Tensor<T>? total = null;
        for (int round = 0; round < 11; round++)
        {
            if (round > 0 && round != noPromptRound && round != 10)
            {
                var predicted = _engine.TensorGreaterThan(net.Postprocess(previous!, h, w, h, w), NumOps.Zero);
                var error = _engine.Reshape(_engine.TensorSubtract(truth, predicted), new[] { h * w }).Cast<double>();
                var sampled = SampleErrorPoint(_engine, error, draws[round], w).Cast<T>();
                var point = _engine.TensorSlice(sampled, new[] { 0, 0, 0 }, new[] { 1, 1, 2 });
                var label = _engine.Reshape(_engine.TensorSlice(sampled, new[] { 0, 0, 2 }, new[] { 1, 1, 1 }), new[] { 1, 1 });
                points = points is null ? point : _engine.TensorConcatenate(new[] { points, point }, 1);
                labels = labels is null ? label : _engine.TensorConcatenate(new[] { labels, label }, 1);
            }
            var (sparse, dense) = net.EncodePrompts(points, labels, box, previous);
            bool ambiguous = round == 0 && box is null;
            var (low, iou) = net.DecodeMasks(embedding, positional, sparse, dense, multimask: ambiguous,
                sparseValidity: net.SparsePromptValidity(labels, box is not null));
            var full = net.Postprocess(low, h, w, h, w);
            Tensor<T>? best = null, chosen = null;
            for (int m = 0; m < low.Shape[1]; m++)
            {
                var logits = _engine.TensorSlice(full, new[] { 0, m, 0, 0 }, new[] { 1, 1, h, w });
                var predictedIou = _engine.Reshape(_engine.TensorSlice(iou, new[] { 0, m }, new[] { 1, 1 }), new[] { 1 });
                var difference = _engine.TensorSubtract(predictedIou, _engine.StopGradient(ActualIou(_engine, logits, truth)));
                var loss = _engine.TensorAdd(MaskLoss(logits, truth), _engine.TensorMultiply(difference, difference));
                var candidate = _engine.TensorSlice(low, new[] { 0, m, 0, 0 }, new[] { 1, 1, low.Shape[2], low.Shape[3] });
                if (best is null) { best = loss; chosen = candidate; }
                else
                {
                    // Strict comparison keeps the first minimum, including exact ties, just as argmin.
                    var better = _engine.TensorLessThan(loss, best);
                    best = _engine.TensorWhere(better, loss, best);
                    chosen = _engine.TensorWhere(_engine.TensorBroadcastTo(
                        _engine.Reshape(better, new[] { 1, 1, 1, 1 }), candidate.Shape.ToArray()), candidate, chosen!);
                }
            }
            total = total is null ? best! : _engine.TensorAdd(total, best!);
            previous = _engine.StopGradient(chosen!);
        }
        return _engine.TensorMultiplyScalar(total!, NumOps.FromDouble(1.0 / 11));
    }


    /// <summary>focalWeight·focal + diceWeight·dice on mask logits <c>[1, 1, H, W]</c>, the focal loss averaged over
    /// pixels and the dice loss smoothed by 1.</summary>
    private Tensor<T> MaskLoss(Tensor<T> logits, Tensor<T> truth)
    {
        var shape = logits.Shape.ToArray();
        var axes = Enumerable.Range(0, shape.Length).ToArray();
        var one = new Tensor<T>(shape);
        one.AsWritableSpan().Fill(NumOps.One);
        var p = _engine.TensorSigmoid(logits);
        // Binary cross-entropy with logits: softplus(x) − x·t.
        var ce = _engine.TensorSubtract(_engine.Softplus(logits), _engine.TensorMultiply(logits, truth));
        var pt = _engine.TensorAdd(_engine.TensorMultiply(p, truth), _engine.TensorMultiply(_engine.TensorSubtract(one, p), _engine.TensorSubtract(one, truth)));
        var modulating = _engine.TensorPower(_engine.TensorSubtract(one, pt), NumOps.FromDouble(_gamma));
        var alphaT = _engine.TensorAdd(_engine.TensorMultiplyScalar(truth, NumOps.FromDouble(_alpha)),
            _engine.TensorMultiplyScalar(_engine.TensorSubtract(one, truth), NumOps.FromDouble(1 - _alpha)));
        var focal = _engine.ReduceMean(_engine.TensorMultiply(alphaT, _engine.TensorMultiply(modulating, ce)), axes, false);
        var scalarOne = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.One }));
        var intersection = _engine.ReduceSum(_engine.TensorMultiply(p, truth), axes, false);
        var sums = _engine.TensorAdd(_engine.ReduceSum(p, axes, false), _engine.ReduceSum(truth, axes, false));
        var dice = _engine.TensorSubtract(scalarOne, _engine.TensorDivide(
            _engine.TensorAdd(_engine.TensorMultiplyScalar(intersection, NumOps.FromDouble(2)), scalarOne), _engine.TensorAdd(sums, scalarOne)));
        return _engine.TensorAdd(_engine.TensorMultiplyScalar(focal, NumOps.FromDouble(_focalWeight)),
            _engine.TensorMultiplyScalar(dice, NumOps.FromDouble(_diceWeight)));
    }

    internal static Tensor<T> ActualIou(IEngine engine, Tensor<T> logits, Tensor<T> truth)
    {
        // Count in double: image-sized masks can exceed Half's finite range.
        var predicted = engine.TensorGreaterThan(logits, NumOps.Zero).Cast<double>();
        var target = truth.Cast<double>();
        var intersection = engine.ReduceSum(engine.TensorMultiply(predicted, target), null);
        var union = engine.ReduceSum(engine.TensorMax(predicted, target), null);
        return engine.TensorWhere(engine.TensorGreaterThan(union, 0.0),
            engine.TensorDivide(intersection, engine.TensorClampMin(union, 1.0)),
            Tensor<double>.CreateDefault(union.Shape.ToArray(), 1.0)).Cast<T>();
    }

    /// <summary>A discrete, input-bound sampling node. Its output is data, so its derivative is zero.</summary>
    internal static Tensor<double> SampleErrorPoint(IEngine engine, Tensor<double> error, Tensor<double> draw, int width)
    {
        if (error.Rank != 1 || width <= 0 || error.Length % width != 0 || draw.Length != 1)
            throw new ArgumentException("Expected flattened pixel errors, an image width and one uniform draw.");
        if (AiDotNet.Tensors.Engines.Compilation.GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            return scope.RecordBinary(AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom, "SamErrorPoint",
                error, draw, new[] { 1, 1, 3 },
                (replayEngine, output) => SelectErrorPoint(replayEngine, error, draw, width, output),
                static (_, _, _, _, _, _) => { });
        }
        var result = new Tensor<double>(new[] { 1, 1, 3 });
        SelectErrorPoint(engine, error, draw, width, result);
        return result;
    }

    private static void SelectErrorPoint(IEngine engine, Tensor<double> error, Tensor<double> draw, int width, Tensor<double> output)
    {
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<double>();
        double uniform = draw[0];
        if (double.IsNaN(uniform) || double.IsInfinity(uniform) || uniform < 0.0 || uniform >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(draw), "A sampling draw must be in [0, 1).");
        int count = engine.TensorCountNonzero(error);
        output.AsWritableSpan().Clear();
        if (count == 0) { output[2] = -2.0; return; }
        var candidates = engine.TensorNonzero(error);
        int picked = candidates[Math.Min(count - 1, (int)(uniform * count)), 0];
        output[0] = picked % width;
        output[1] = picked / width;
        output[2] = error[picked] > 0.0 ? 1.0 : 0.0;
    }

    /// <summary>A pixel (x, y) drawn uniformly from those satisfying <paramref name="where"/>, or null if none does.</summary>
    private static (double X, double Y)? Sample(double[] map, int w, Random random, Func<double, bool> where)
    {
        var candidates = new List<int>();
        for (int i = 0; i < map.Length; i++) if (where(map[i])) candidates.Add(i);
        if (candidates.Count == 0) return null;
        int pick = candidates[random.Next(candidates.Count)];
        int column = pick % w, row = pick / w;   // the pixel's integer coordinates
        return (column, row);
    }

    /// <summary>The mask's box with noise: each coordinate moved by N(0, (0.1·side)²), at most 20 pixels.</summary>
    private static double[] NoisyBox(double[] gt, int h, int w, Random random)
    {
        int x0 = int.MaxValue, y0 = int.MaxValue, x1 = -1, y1 = -1;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                if (gt[y * w + x] > 0.5) { x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y); }
        if (x1 < 0) return new double[] { 0, 0, w - 1, h - 1 };
        double Normal() => Math.Sqrt(-2.0 * Math.Log(1.0 - random.NextDouble())) * Math.Cos(2.0 * Math.PI * random.NextDouble());
        double Noise(double side) => Math.Max(-20.0, Math.Min(20.0, 0.1 * side * Normal()));
        double bw = x1 - x0, bh = y1 - y0;
        return new[]
        {
            Math.Max(0, x0 + Noise(bw)), Math.Max(0, y0 + Noise(bh)), Math.Min(w - 1, x1 + Noise(bw)), Math.Min(h - 1, y1 + Noise(bh)),
        };
    }
}
