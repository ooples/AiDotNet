namespace AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;

/// <summary>Shared target conversion for training the two-stage detectors.</summary>
internal static class TwoStageTargets
{
    /// <summary>A normalized center-format target as corner coordinates in input pixels.</summary>
    internal static double[] PixelCorners<T>(DetectionTrainingTarget<T> target, int width, int height)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        double cx = ops.ToDouble(target.CenterX) * width;
        double cy = ops.ToDouble(target.CenterY) * height;
        double halfWidth = ops.ToDouble(target.Width) * width / 2;
        double halfHeight = ops.ToDouble(target.Height) * height / 2;
        return new[] { cx - halfWidth, cy - halfHeight, cx + halfWidth, cy + halfHeight };
    }

    /// <summary>The first image's rows of a [batch, rows, width] head output, as [rows, width].</summary>
    internal static Tensor<T> FirstImage<T>(Tensor<T> head)
    {
        if (head.Rank != 3 || head.Shape[0] != 1)
            throw new InvalidOperationException("Two-stage training expects the proposal outputs of exactly one image.");
        return AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(head, new[] { head.Shape[1], head.Shape[2] });
    }

    /// <summary>Appends constant corner boxes to detached proposal boxes [proposals, 4].</summary>
    internal static Tensor<T> AppendBoxes<T>(Tensor<T> proposals, IReadOnlyList<double[]> boxes)
    {
        if (boxes.Count == 0) return proposals.Clone();
        var flat = new double[boxes.Count * 4];
        for (int b = 0; b < boxes.Count; b++)
            Array.Copy(boxes[b], 0, flat, b * 4, 4);
        var appended = CvTensorOps<T>.FromDoubleArray(flat, new[] { boxes.Count, 4 });
        if (proposals.Length == 0) return appended;
        var existing = AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Reshape(proposals, new[] { proposals.Length / 4, 4 });
        return AiDotNet.Tensors.Engines.AiDotNetEngine.Current.Concat(new[] { existing, appended }, 0);
    }
}

/// <summary>
/// Whole-tensor R-CNN box decoding shared by the two-stage detectors: softmax class scores and the
/// <c>(dx, dy, dw, dh)</c> deltas applied to proposal boxes, as engine ops over every row at once.
/// </summary>
internal static class TwoStageBoxDecoding<T>
{
    private static AiDotNet.Tensors.Engines.IEngine Engine => AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

    private static AiDotNet.Tensors.Interfaces.INumericOperations<T> NumOps => AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>Decoded boxes <c>[M]</c> per coordinate, plus the proposal widths and heights they came from.</summary>
    internal readonly struct Boxes
    {
        public Boxes(Tensor<T> x1, Tensor<T> y1, Tensor<T> x2, Tensor<T> y2, Tensor<T> proposalWidth, Tensor<T> proposalHeight)
        {
            X1 = x1; Y1 = y1; X2 = x2; Y2 = y2; ProposalWidth = proposalWidth; ProposalHeight = proposalHeight;
        }

        public Tensor<T> X1 { get; }
        public Tensor<T> Y1 { get; }
        public Tensor<T> X2 { get; }
        public Tensor<T> Y2 { get; }
        public Tensor<T> ProposalWidth { get; }
        public Tensor<T> ProposalHeight { get; }
    }

    /// <summary>Row-wise softmax of class logits <c>[N, C]</c> (max-subtracted).</summary>
    internal static Tensor<T> ClassProbabilities(Tensor<T> logits) => Engine.Softmax(logits, 1);

    /// <summary>
    /// Applies deltas <c>[M, 4]</c> to proposals <c>[M, 4]</c> (XYXY): centre shift <c>d / weight * size</c>, size
    /// scale <c>exp(min(d / weight, clamp))</c>, then the corners scaled by <c>(scaleX, scaleY)</c> and clipped
    /// to <c>[0, imageWidth] x [0, imageHeight]</c>.
    /// </summary>
    internal static Boxes Decode(Tensor<T> proposals, Tensor<T> deltas, double[] weights, double clamp,
        double scaleX, double scaleY, int imageWidth, int imageHeight)
    {
        int rows = proposals.Shape[0];
        Tensor<T> Column(Tensor<T> source, int index)
            => Engine.Reshape(Engine.TensorSlice(source, new[] { 0, index }, new[] { rows, 1 }), new[] { rows });

        var half = NumOps.FromDouble(0.5);
        var px1 = Column(proposals, 0);
        var py1 = Column(proposals, 1);
        var pw = Engine.TensorSubtract(Column(proposals, 2), px1);
        var ph = Engine.TensorSubtract(Column(proposals, 3), py1);
        var pcx = Engine.TensorAdd(px1, Engine.TensorDivideScalar(pw, NumOps.FromDouble(2)));
        var pcy = Engine.TensorAdd(py1, Engine.TensorDivideScalar(ph, NumOps.FromDouble(2)));

        var dx = Engine.TensorDivideScalar(Column(deltas, 0), NumOps.FromDouble(weights[0]));
        var dy = Engine.TensorDivideScalar(Column(deltas, 1), NumOps.FromDouble(weights[1]));
        var dw = Engine.TensorDivideScalar(Column(deltas, 2), NumOps.FromDouble(weights[2]));
        var dh = Engine.TensorDivideScalar(Column(deltas, 3), NumOps.FromDouble(weights[3]));

        var cx = Engine.TensorAdd(pcx, Engine.TensorMultiply(dx, pw));
        var cy = Engine.TensorAdd(pcy, Engine.TensorMultiply(dy, ph));
        var halfW = Engine.TensorMultiplyScalar(Engine.TensorMultiply(pw, Engine.TensorExp(Engine.TensorMin(dw, NumOps.FromDouble(clamp)))), half);
        var halfH = Engine.TensorMultiplyScalar(Engine.TensorMultiply(ph, Engine.TensorExp(Engine.TensorMin(dh, NumOps.FromDouble(clamp)))), half);

        var sx = NumOps.FromDouble(scaleX);
        var sy = NumOps.FromDouble(scaleY);
        var x1 = Engine.TensorMax(Engine.TensorMultiplyScalar(Engine.TensorSubtract(cx, halfW), sx), NumOps.Zero);
        var y1 = Engine.TensorMax(Engine.TensorMultiplyScalar(Engine.TensorSubtract(cy, halfH), sy), NumOps.Zero);
        var x2 = Engine.TensorMin(Engine.TensorMultiplyScalar(Engine.TensorAdd(cx, halfW), sx), NumOps.FromDouble(imageWidth));
        var y2 = Engine.TensorMin(Engine.TensorMultiplyScalar(Engine.TensorAdd(cy, halfH), sy), NumOps.FromDouble(imageHeight));
        return new Boxes(x1, y1, x2, y2, pw, ph);
    }

    /// <summary>
    /// Per row of a probability table <c>[N, C]</c>, the first class from 1 up whose probability is strictly
    /// greater than every earlier one, starting from class 0 at score 0 - so a row with no positive foreground
    /// probability keeps class 0 - and that class's probability.
    /// </summary>
    internal static (int[] Classes, Tensor<T> Scores) BestForegroundClass(Tensor<T> probabilities)
    {
        int rows = probabilities.Shape[0], classes = probabilities.Shape[1];
        var zeroColumn = new Tensor<T>(new[] { rows, 1 });
        var candidates = classes > 1
            ? Engine.Concat(new[] { zeroColumn, Engine.TensorSlice(probabilities, new[] { 0, 1 }, new[] { rows, classes - 1 }) }, 1)
            : zeroColumn;
        var best = Engine.TensorArgMax(candidates, 1);
        var bestClasses = best.ToArray();
        var flatIndex = new int[rows];
        for (int i = 0; i < rows; i++) flatIndex[i] = i * candidates.Shape[1] + bestClasses[i];
        var scores = CvTensorOps<T>.Select(Engine.Reshape(candidates, new[] { candidates.Length }), flatIndex, 0);
        return (bestClasses, scores);
    }

    /// <summary>Row <c>i</c> of <c>deltas.Reshape([N, C, 4])</c> at class <c>classes[i]</c>, as <c>[N, 4]</c>.</summary>
    internal static Tensor<T> ClassDeltas(Tensor<T> deltas, int[] classes, int numClasses)
    {
        int rows = classes.Length;
        var flatIndex = new int[rows];
        for (int i = 0; i < rows; i++) flatIndex[i] = i * numClasses + classes[i];
        return CvTensorOps<T>.Select(Engine.Reshape(deltas, new[] { rows * numClasses, 4 }), flatIndex, 0);
    }
    /// <summary>
    /// One candidate per RoI for the single-label two-stage heads (Faster and Cascade R-CNN): the best
    /// foreground class (see <see cref="BestForegroundClass"/>) when its probability reaches
    /// <paramref name="confidenceThreshold"/>, with its box decoded from <paramref name="deltas"/> <c>[N, 4]</c>;
    /// empty boxes are dropped. Class ids drop the background offset.
    /// </summary>
    internal static List<AiDotNet.ComputerVision.Detection.ObjectDetection.Detection<T>> BestClassCandidates(
        int[] bestClasses, Tensor<T> bestScores, Tensor<T> proposals, Tensor<T> deltas,
        double[] weights, double clamp, double scaleX, double scaleY, int imageWidth, int imageHeight,
        double confidenceThreshold, IReadOnlyList<string> classNames)
    {
        var boxes = Decode(proposals, deltas, weights, clamp, scaleX, scaleY, imageWidth, imageHeight);
        var scoreValues = CvTensorOps<T>.ToDoubleArray(bestScores);
        var scores = bestScores.ToArray();
        var x1 = boxes.X1.ToArray(); var y1 = boxes.Y1.ToArray(); var x2 = boxes.X2.ToArray(); var y2 = boxes.Y2.ToArray();
        var x1d = CvTensorOps<T>.ToDoubleArray(boxes.X1); var y1d = CvTensorOps<T>.ToDoubleArray(boxes.Y1);
        var x2d = CvTensorOps<T>.ToDoubleArray(boxes.X2); var y2d = CvTensorOps<T>.ToDoubleArray(boxes.Y2);

        var candidates = new List<AiDotNet.ComputerVision.Detection.ObjectDetection.Detection<T>>();
        for (int i = 0; i < bestClasses.Length; i++)
        {
            if (scoreValues[i] < confidenceThreshold) continue;
            if (x2d[i] <= x1d[i] || y2d[i] <= y1d[i]) continue;

            int classId = bestClasses[i] - 1;
            var box = new AiDotNet.Augmentation.Image.BoundingBox<T>(x1[i], y1[i], x2[i], y2[i]);
            candidates.Add(new AiDotNet.ComputerVision.Detection.ObjectDetection.Detection<T>(
                box, classId, scores[i], classId < classNames.Count ? classNames[classId] : null));
        }

        return candidates;
    }
}
