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
    /// The objective for an image <c>[1, C, H, W]</c> (pixel values, H and W at most the encoder size) and its ground
    /// truth (<c>H·W</c> values; above 0.5 is the mask). The prompts are drawn when the objective is evaluated.
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
        return () => Loss(image, gt, truth, h, w, random);
    }

    private Tensor<T> Loss(Tensor<T> image, double[] gt, Tensor<T> truth, int h, int w, Random random)
    {
        var net = _network;
        var embedding = net.EncodeImage(net.Preprocess(image));
        var positional = net.DensePositional();
        int noPromptRound = 1 + random.Next(9);
        var points = new List<(double X, double Y, double Label)>();
        double[]? box = null;
        Tensor<T>? previous = null;
        Tensor<T>? total = null;
        for (int round = 0; round < 11; round++)
        {
            if (round == 0)
            {
                if (random.NextDouble() < 0.5) box = NoisyBox(gt, h, w, random);
                else if (Sample(gt, w, random, v => v > 0.5) is { } first) points.Add((first.X, first.Y, 1.0));
                else box = NoisyBox(gt, h, w, random);
            }
            else if (round != noPromptRound && round != 10 && previous is not null)
            {
                var predicted = net.Postprocess(previous, h, w, h, w).ToArray();
                var error = new double[h * w];
                for (int i = 0; i < error.Length; i++)
                {
                    bool on = NumOps.ToDouble(predicted[i]) > 0;
                    error[i] = gt[i] > 0.5 && !on ? 1.0 : gt[i] <= 0.5 && on ? -1.0 : 0.0;
                }
                if (Sample(error, w, random, v => v != 0) is { } p)
                    points.Add((p.X, p.Y, error[(int)p.Y * w + (int)p.X] > 0 ? 1.0 : 0.0));
            }
            Tensor<T>? pointTensor = null, labelTensor = null, boxTensor = null;
            if (points.Count > 0)
            {
                pointTensor = new Tensor<T>(new[] { 1, points.Count, 2 });
                labelTensor = new Tensor<T>(new[] { 1, points.Count });
                for (int i = 0; i < points.Count; i++)
                {
                    pointTensor[0, i, 0] = NumOps.FromDouble(points[i].X);
                    pointTensor[0, i, 1] = NumOps.FromDouble(points[i].Y);
                    labelTensor[0, i] = NumOps.FromDouble(points[i].Label);
                }
            }
            if (box is not null) boxTensor = new Tensor<T>(new[] { 1, 4 }, new Vector<T>(box.Select(NumOps.FromDouble).ToArray()));
            var (sparse, dense) = net.EncodePrompts(pointTensor, labelTensor, boxTensor, previous);
            bool ambiguous = round == 0 && box is null;
            var (low, iou) = net.DecodeMasks(embedding, positional, sparse, dense, multimask: ambiguous);
            var full = net.Postprocess(low, h, w, h, w);
            Tensor<T>? best = null;
            double bestValue = double.PositiveInfinity;
            int bestIndex = 0;
            for (int m = 0; m < low.Shape[1]; m++)
            {
                var logits = _engine.TensorSlice(full, new[] { 0, m, 0, 0 }, new[] { 1, 1, h, w });
                var predictedIou = _engine.Reshape(_engine.TensorSlice(iou, new[] { 0, m }, new[] { 1, 1 }), new[] { 1 });
                var target = new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(ActualIou(logits.ToArray(), gt)) }));
                var difference = _engine.TensorSubtract(predictedIou, target);
                var loss = _engine.TensorAdd(MaskLoss(logits, truth), _engine.TensorMultiply(difference, difference));
                double value = NumOps.ToDouble(loss[0]);
                if (value < bestValue)
                {
                    bestValue = value;
                    best = loss;
                    bestIndex = m;
                }
            }
            total = total is null ? best! : _engine.TensorAdd(total, best!);
            // The next round's mask prompt: this round's (chosen) unthresholded logits, as data.
            var chosen = _engine.TensorSlice(low, new[] { 0, bestIndex, 0, 0 }, new[] { 1, 1, low.Shape[2], low.Shape[3] });
            previous = new Tensor<T>(chosen.Shape.ToArray(), new Vector<T>(chosen.ToArray()));
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

    private static double ActualIou(T[] logits, double[] gt)
    {
        double inter = 0, union = 0;
        for (int i = 0; i < gt.Length; i++)
        {
            bool on = NumOps.ToDouble(logits[i]) > 0, truth = gt[i] > 0.5;
            if (on && truth) inter++;
            if (on || truth) union++;
        }
        return union == 0 ? 1.0 : inter / union;
    }

    /// <summary>A pixel (x, y) drawn uniformly from those satisfying <paramref name="where"/>, or null if none does.</summary>
    private static (double X, double Y)? Sample(double[] map, int w, Random random, Func<double, bool> where)
    {
        var candidates = new List<int>();
        for (int i = 0; i < map.Length; i++) if (where(map[i])) candidates.Add(i);
        if (candidates.Count == 0) return null;
        int pick = candidates[random.Next(candidates.Count)];
        return (pick % w, pick / w);
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
