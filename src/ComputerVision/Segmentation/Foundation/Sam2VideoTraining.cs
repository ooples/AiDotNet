using AiDotNet.Helpers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>
/// The prompt simulation and loss weights of SAM 2's training (Ravi et al. 2024, Appendix D.2.2 and Table 12b). The
/// paper fixes the values it states; facebookresearch/sam2's <c>SAM2Train</c> and its training config fill the rest.
/// </summary>
internal sealed record Sam2TrainingSettings
{
    /// <summary>The chance the first prompts are clicks or boxes rather than the ground-truth mask (paper: mask 50%).</summary>
    public double PointProbability { get; init; } = 0.5;

    /// <summary>The chance clicks are a box instead (paper: click 25%, box 25%).</summary>
    public double BoxProbability { get; init; } = 0.5;

    /// <summary>The most prompted frames (the first and up to one more, drawn uniformly: <c>SAM2Train</c>).</summary>
    public int InitialFrames { get; init; } = 2;

    /// <summary>The most frames given corrective clicks, including the first (paper: "up to 2").</summary>
    public int CorrectedFrames { get; init; } = 2;

    /// <summary>Corrective clicks per corrected frame (paper: 7).</summary>
    public int CorrectionPoints { get; init; } = 7;

    /// <summary>The chance a corrective click is drawn from the ground truth rather than the error (paper: 10%).</summary>
    public double GroundTruthClickProbability { get; init; } = 0.1;

    /// <summary>The chance the sequence plays in reverse (paper: 50%).</summary>
    public double ReverseProbability { get; init; } = 0.5;

    /// <summary>The most a box corner is moved, in pixels (SAM's 20); it is also capped at 10% of the box side.</summary>
    public double BoxNoiseBound { get; init; } = 20.0;

    /// <summary>Whether a frame given corrective clicks joins the prompted frames in the memory (MedSAM2's recipe).</summary>
    public bool CorrectedFramesAsPrompted { get; init; }

    public double FocalWeight { get; init; } = 20.0;

    public double DiceWeight { get; init; } = 1.0;

    public double IouWeight { get; init; } = 1.0;

    public double OcclusionWeight { get; init; } = 1.0;

    public double FocalAlpha { get; init; } = 0.25;

    public double FocalGamma { get; init; } = 2.0;
}

/// <summary>
/// SAM 2's training objective on a video and one object's masks (<c>SAM2Train.forward</c> and
/// <c>MultiStepMultiMasksAndIous</c>): prompts are simulated, every frame is tracked with the memory, corrected frames get
/// iterative clicks from the error region (each prediction feeding the next as its mask input), and every prediction
/// (initial and corrected, on every frame) adds the focal and dice losses of its best mask (lowest 20·focal + dice), the
/// ℓ1 IoU loss of all its masks and, when the object is present, the occlusion cross-entropy; mask, dice and IoU losses
/// count only when the object is present.
/// </summary>
internal sealed class Sam2VideoTraining<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly Sam2Network<T> _net;
    private readonly Sam2TrainingSettings _settings;

    public Sam2VideoTraining(IEngine engine, Sam2Network<T> network, Sam2TrainingSettings settings)
    {
        _engine = engine;
        _net = network;
        _settings = settings;
    }

    /// <summary>A frame's prompt plan, drawn before the objective runs so a re-evaluation replays it.</summary>
    internal sealed record Plan(int[] Order, List<int> Prompted, Dictionary<int, Sam2Prompt<T>> Prompts, HashSet<int> Corrected, bool Reverse);

    /// <summary>
    /// Draws the prompts for normalized frames and their masks (<c>H·W</c> values each, ImageSize square; above 0.5 is the
    /// object), as <c>SAM2Train.prepare_prompt_inputs</c> does.
    /// </summary>
    public Plan Draw(IReadOnlyList<double[]> masks, Random random)
    {
        var s = _settings;
        int frames = masks.Count;
        bool reverse = frames > 1 && random.NextDouble() < s.ReverseProbability;
        var order = Enumerable.Range(0, frames).ToArray();
        if (reverse) Array.Reverse(order);
        double pointProbability = s.PointProbability;
        int corrected = s.CorrectedFrames, initial = s.InitialFrames;
        if (frames == 1)
        {
            pointProbability = 1.0;
            corrected = 1;
            initial = 1;
        }
        initial = Math.Min(initial, frames);
        corrected = Math.Min(corrected, frames);
        bool usePoints = random.NextDouble() < pointProbability;
        if (initial > 1) initial = random.Next(1, initial + 1);
        if (usePoints && corrected > initial) corrected = random.Next(initial, corrected + 1);

        // Positions in the (possibly reversed) sequence: the first, then distinct others.
        var prompted = new List<int> { 0 };
        prompted.AddRange(Choose(Enumerable.Range(1, frames - 1).ToList(), initial - 1, random));
        var prompts = new Dictionary<int, Sam2Prompt<T>>();
        foreach (int position in prompted)
        {
            var gt = masks[order[position]];
            if (!usePoints) prompts[position] = new Sam2Prompt<T>(null, null, MaskTensor(gt));
            else if (random.NextDouble() < s.BoxProbability) prompts[position] = Box(gt, random);
            else
            {
                var (x, y, label) = ErrorPoint(gt, null, random);
                prompts[position] = Points(new[] { (x, y, label) });
            }
        }
        var correctedSet = new HashSet<int>();
        if (usePoints)
        {
            foreach (int position in prompted) correctedSet.Add(position);
            var others = Enumerable.Range(0, frames).Where(t => !prompted.Contains(t)).ToList();
            foreach (int position in Choose(others, Math.Max(0, corrected - initial), random)) correctedSet.Add(position);
        }
        return new Plan(order, prompted, prompts, correctedSet, reverse);
    }

    private static List<int> Choose(List<int> pool, int count, Random random)
    {
        var copy = pool.ToList();
        var chosen = new List<int>();
        for (int i = 0; i < count && copy.Count > 0; i++)
        {
            int index = random.Next(copy.Count);
            chosen.Add(copy[index]);
            copy.RemoveAt(index);
        }
        return chosen;
    }

    private Tensor<T> MaskTensor(double[] gt)
    {
        int size = _net.Configuration.ImageSize;
        return new Tensor<T>(new[] { 1, 1, size, size }, new Vector<T>(gt.Select(v => v > 0.5 ? NumOps.One : NumOps.Zero).ToArray()));
    }

    private static Sam2Prompt<T> Points(IReadOnlyList<(double X, double Y, int Label)> points)
    {
        var coordinates = new Tensor<T>(new[] { 1, points.Count, 2 });
        var labels = new Tensor<T>(new[] { 1, points.Count });
        for (int i = 0; i < points.Count; i++)
        {
            coordinates[0, i, 0] = NumOps.FromDouble(points[i].X);
            coordinates[0, i, 1] = NumOps.FromDouble(points[i].Y);
            labels[0, i] = NumOps.FromDouble(points[i].Label);
        }
        return new Sam2Prompt<T>(coordinates, labels, null);
    }

    /// <summary>The official <c>sample_box_points</c>: the mask's box (its pixel extent; an empty mask gives (W, H, −1, −1))
    /// with each coordinate moved uniformly by up to 10% of the box side, at most <see cref="Sam2TrainingSettings.BoxNoiseBound"/>
    /// pixels, clamped to the image, as two corner points (labels 2 and 3).</summary>
    private Sam2Prompt<T> Box(double[] gt, Random random)
    {
        int size = _net.Configuration.ImageSize;
        double x0 = size, y0 = size, x1 = -1, y1 = -1;
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                if (gt[y * size + x] <= 0.5) continue;
                x0 = Math.Min(x0, x); y0 = Math.Min(y0, y); x1 = Math.Max(x1, x); y1 = Math.Max(y1, y);
            }
        double bound = _settings.BoxNoiseBound;
        double dx = Math.Min((x1 - x0) * 0.1, bound), dy = Math.Min((y1 - y0) * 0.1, bound);
        double Noise(double v, double max, double bound) => Math.Min(Math.Max(v + (2 * random.NextDouble() - 1) * max, 0), bound);
        double bx0 = Noise(x0, dx, size - 1), by0 = Noise(y0, dy, size - 1), bx1 = Noise(x1, dx, size - 1), by1 = Noise(y1, dy, size - 1);
        return Points(new[] { (bx0, by0, 2), (bx1, by1, 3) });
    }

    /// <summary>The official <c>sample_random_points_from_errors</c>: a pixel drawn uniformly from the error region (a
    /// positive click where the prediction misses the mask, a negative one where it spills over); with no error, a negative
    /// click from the background (the image corner, labelled negative, when there is none).</summary>
    private (double X, double Y, int Label) ErrorPoint(double[] gt, bool[]? predicted, Random random)
    {
        int size = _net.Configuration.ImageSize;
        var candidates = new List<int>();
        bool allCorrect = true;
        for (int i = 0; i < gt.Length; i++)
            if ((gt[i] > 0.5) != (predicted?[i] ?? false)) { allCorrect = false; break; }
        for (int i = 0; i < gt.Length; i++)
        {
            bool truth = gt[i] > 0.5, pred = predicted?[i] ?? false;
            if (truth != pred || (allCorrect && !truth)) candidates.Add(i);
        }
        if (candidates.Count == 0) return (0, 0, 0);
        int pick = candidates[random.Next(candidates.Count)];
        bool positive = gt[pick] > 0.5 && !(predicted?[pick] ?? false);
        return (pick % size, pick / size, positive ? 1 : 0);
    }

    /// <summary>
    /// The objective for normalized frames <c>[1, 3, S, S]</c> and their masks under <paramref name="plan"/>; corrective
    /// clicks are drawn from <paramref name="random"/> as the predictions arrive, or given by <paramref name="clicks"/>
    /// (the sequence position and the click's index on it).
    /// </summary>
    public Tensor<T> Loss(IReadOnlyList<Tensor<T>> images, IReadOnlyList<double[]> masks, Plan plan, Random random,
        Func<int, int, (double X, double Y, int Label)>? clicks = null)
    {
        int frames = images.Count;
        var tracker = new Sam2VideoTracker<T>(_engine, _net) { Frames = frames };
        var processing = plan.Prompted.Concat(Enumerable.Range(0, frames).Where(t => !plan.Prompted.Contains(t))).ToList();
        Tensor<T>? total = null;
        foreach (int position in processing)
        {
            int frame = plan.Order[position];
            var gt = masks[frame];
            var truth = MaskTensor(gt);
            bool prompted = plan.Prompted.Contains(position);
            plan.Prompts.TryGetValue(position, out var prompt);
            var features = _net.EncodeFeatures(images[frame]);
            var (heads, conditioned) = tracker.Step(position, prompted, features, prompt, reverse: false);
            var predictions = new List<Sam2Heads<T>> { heads };
            if (plan.Corrected.Contains(position))
            {
                var points = prompt;
                for (int i = 0; i < _settings.CorrectionPoints; i++)
                {
                    (double X, double Y, int Label) click;
                    if (clicks is not null)
                    {
                        click = clicks(position, i);
                    }
                    else
                    {
                        bool fromTruth = random.NextDouble() < _settings.GroundTruthClickProbability;
                        bool[]? predicted = fromTruth ? null : heads.High.ToArray().Select(v => NumOps.ToDouble(v) > 0).ToArray();
                        click = ErrorPoint(gt, predicted, random);
                    }
                    points = Append(points, click);
                    heads = tracker.Heads(conditioned, features.HighRes0, features.HighRes1, points!.Points, points.Labels, heads.Low,
                        Sam2VideoTracker<T>.UseMultimask(points.Labels));
                    predictions.Add(heads);
                }
                prompt = points;
            }
            var output = new Sam2FrameOutput<T> { LowRes = heads.Low, HighRes = heads.High, Pointer = heads.Pointer, ObjectScore = heads.Score(NumOps) };
            output.Memory = tracker.EncodeMemory(features.Features, heads.High, output.ObjectScore, prompt?.Points is not null);
            bool conditioning = prompted || (_settings.CorrectedFramesAsPrompted && plan.Corrected.Contains(position));
            (conditioning ? tracker.Conditioning : tracker.Tracked)[position] = output;
            foreach (var prediction in predictions)
            {
                var loss = PredictionLoss(prediction, truth);
                total = total is null ? loss : _engine.TensorAdd(total, loss);
            }
        }
        return total!;
    }

    private static Sam2Prompt<T> Append(Sam2Prompt<T>? prompt, (double X, double Y, int Label) click)
    {
        int old = prompt?.Points?.Shape[1] ?? 0;
        var coordinates = new Tensor<T>(new[] { 1, old + 1, 2 });
        var labels = new Tensor<T>(new[] { 1, old + 1 });
        for (int i = 0; i < old; i++)
        {
            coordinates[0, i, 0] = prompt!.Points![0, i, 0];
            coordinates[0, i, 1] = prompt.Points[0, i, 1];
            labels[0, i] = prompt.Labels![0, i];
        }
        coordinates[0, old, 0] = NumOps.FromDouble(click.X);
        coordinates[0, old, 1] = NumOps.FromDouble(click.Y);
        labels[0, old] = NumOps.FromDouble(click.Label);
        return new Sam2Prompt<T>(coordinates, labels, null);
    }

    /// <summary>One prediction's loss against the ground truth <c>[1, 1, S, S]</c> (<c>MultiStepMultiMasksAndIous</c>).</summary>
    public Tensor<T> PredictionLoss(Sam2Heads<T> heads, Tensor<T> truth)
    {
        var s = _settings;
        var masks = heads.MultiHigh;
        int count = masks.Shape[1], size = masks.Shape[2], pixels = size * size;
        var truthValues = truth.ToArray().Select(NumOps.ToDouble).ToArray();
        bool present = truthValues.Any(v => v > 0);
        var focal = new Tensor<T>[count];
        var dice = new Tensor<T>[count];
        var iou = new Tensor<T>[count];
        var truthMap = _engine.Reshape(truth, new[] { 1, pixels });
        for (int k = 0; k < count; k++)
        {
            var logits = _engine.Reshape(_engine.TensorSlice(masks, new[] { 0, k, 0, 0 }, new[] { 1, 1, size, size }), new[] { 1, pixels });
            focal[k] = Focal(logits, truthMap, s.FocalAlpha, s.FocalGamma);
            var probability = _engine.Sigmoid(logits);
            var numerator = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(_engine.ReduceSum(_engine.TensorMultiply(probability, truthMap), new[] { 1 }, keepDims: false),
                NumOps.FromDouble(2.0)), NumOps.One);
            var denominator = _engine.TensorAddScalar(_engine.ReduceSum(probability, new[] { 1 }, keepDims: false), NumOps.FromDouble(truthValues.Sum() + 1.0));
            dice[k] = _engine.ScalarMinusTensor(NumOps.One, _engine.TensorDivide(numerator, denominator));
            // The IoU of the thresholded mask with the truth is a target, not differentiated.
            var predicted = logits.ToArray();
            int intersection = 0, union = 0;
            for (int i = 0; i < pixels; i++)
            {
                bool a = NumOps.ToDouble(predicted[i]) > 0, b = truthValues[i] > 0;
                if (a && b) intersection++;
                if (a || b) union++;
            }
            var actual = new Tensor<T>(new[] { 1 });
            actual[0] = NumOps.FromDouble(intersection / (double)Math.Max(union, 1));
            iou[k] = _engine.TensorAbs(_engine.TensorSubtract(_engine.Reshape(_engine.TensorSlice(heads.Iou, new[] { 0, k }, new[] { 1, 1 }), new[] { 1 }), actual));
        }
        int best = 0;
        if (count > 1)
        {
            double lowest = double.PositiveInfinity;
            for (int k = 0; k < count; k++)
            {
                double combined = s.FocalWeight * NumOps.ToDouble(focal[k][0]) + s.DiceWeight * NumOps.ToDouble(dice[k][0]);
                if (combined < lowest) { lowest = combined; best = k; }
            }
        }
        var iouLoss = iou[0];
        for (int k = 1; k < count; k++) iouLoss = _engine.TensorAdd(iouLoss, iou[k]);
        iouLoss = _engine.TensorMultiplyScalar(iouLoss, NumOps.FromDouble(1.0 / count));

        var objectTarget = new Tensor<T>(new[] { 1 });
        objectTarget[0] = present ? NumOps.One : NumOps.Zero;
        var occlusion = Focal(_engine.Reshape(heads.ObjectScore, new[] { 1, 1 }), _engine.Reshape(objectTarget, new[] { 1, 1 }), -1, 0);

        var total = _engine.TensorMultiplyScalar(occlusion, NumOps.FromDouble(s.OcclusionWeight));
        if (present)
        {
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(focal[best], NumOps.FromDouble(s.FocalWeight)));
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(dice[best], NumOps.FromDouble(s.DiceWeight)));
            total = _engine.TensorAdd(total, _engine.TensorMultiplyScalar(iouLoss, NumOps.FromDouble(s.IouWeight)));
        }
        return total;
    }

    /// <summary>The sigmoid focal loss of logits <c>[1, n]</c> (Lin et al.; α &lt; 0 skips the class weighting, γ 0 is the
    /// cross-entropy) averaged over the n values, with the cross-entropy computed stably from the logits.</summary>
    private Tensor<T> Focal(Tensor<T> logits, Tensor<T> target, double alpha, double gamma)
    {
        // softplus(x) − x·t, softplus(x) = max(x, 0) + log(1 + exp(−|x|)).
        var softplus = _engine.TensorAdd(_engine.ReLU(logits),
            _engine.TensorLog(_engine.TensorAddScalar(_engine.TensorExp(_engine.TensorMultiplyScalar(_engine.TensorAbs(logits), NumOps.FromDouble(-1.0))), NumOps.One)));
        var loss = _engine.TensorSubtract(softplus, _engine.TensorMultiply(logits, target));
        if (gamma != 0)
        {
            var p = _engine.Sigmoid(logits);
            // 1 − p_t = p + t − 2·p·t.
            var miss = _engine.TensorSubtract(_engine.TensorAdd(p, target), _engine.TensorMultiplyScalar(_engine.TensorMultiply(p, target), NumOps.FromDouble(2.0)));
            var modulation = gamma == 2.0 ? _engine.TensorMultiply(miss, miss) : _engine.TensorPower(miss, NumOps.FromDouble(gamma));
            loss = _engine.TensorMultiply(loss, modulation);
        }
        if (alpha >= 0)
        {
            // α·t + (1 − α)·(1 − t).
            var weight = _engine.TensorAddScalar(_engine.TensorMultiplyScalar(target, NumOps.FromDouble(2 * alpha - 1)), NumOps.FromDouble(1 - alpha));
            loss = _engine.TensorMultiply(loss, weight);
        }
        return _engine.Reshape(_engine.ReduceMean(loss, new[] { 1 }, keepDims: false), new[] { 1 });
    }
}
