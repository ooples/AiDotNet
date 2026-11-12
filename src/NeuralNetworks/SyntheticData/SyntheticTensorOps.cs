using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Whole-tensor engine helpers shared by the tabular GAN generators (CTGAN, CTAB-GAN+, CopulaGAN,
/// DP-CTGAN, CausalGAN, TableGAN, ...), replacing the per-element loops each generator used to carry.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal static class SyntheticTensorOps<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// x where x &gt; 0, else 0. A NaN entry compares false and becomes 0.
    /// </summary>
    public static Tensor<T> ReLU(Tensor<T> input)
    {
        var engine = Engine;
        return engine.TensorWhere(engine.TensorGreaterThan(input, NumOps.Zero), input, new Tensor<T>(input._shape));
    }

    /// <summary>
    /// x where x &gt; 0, else slope * x.
    /// </summary>
    public static Tensor<T> LeakyReLU(Tensor<T> input, double slope)
    {
        var engine = Engine;
        return engine.TensorWhere(engine.TensorGreaterThan(input, NumOps.Zero), input,
            engine.TensorMultiplyScalar(input, NumOps.FromDouble(slope)));
    }

    /// <summary>
    /// The elements of <paramref name="a"/> followed by those of <paramref name="b"/>, as a rank-1 tensor.
    /// </summary>
    public static Tensor<T> Concat1D(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate(new[] { a.Reshape(new[] { a.Length }), b.Reshape(new[] { b.Length }) }, 0);
    }

    /// <summary>
    /// The elements of <paramref name="a"/> followed by those of <paramref name="b"/>.
    /// </summary>
    public static Vector<T> ConcatVectors(Vector<T> a, Vector<T> b)
    {
        return Concat1D(Tensor<T>.FromVector(a), Tensor<T>.FromVector(b)).ToVector();
    }

    /// <summary>
    /// A rank-1 copy of <paramref name="v"/> with exactly <paramref name="length"/> entries:
    /// truncated when longer, zero-padded when shorter.
    /// </summary>
    public static Tensor<T> FitLength(Tensor<T> v, int length)
    {
        var flat = v.Reshape(new[] { v.Length });
        if (v.Length == length)
        {
            return flat.Clone();
        }

        if (v.Length > length)
        {
            return Engine.TensorSlice(flat, new[] { 0 }, new[] { length });
        }

        return Engine.TensorConcatenate(new[] { flat, new Tensor<T>(new[] { length - v.Length }) }, 0);
    }

    /// <summary>
    /// Stacks <paramref name="rows"/> into a [rows.Count, width] tensor, truncating or zero-padding each row.
    /// </summary>
    public static Tensor<T> StackRows(IReadOnlyList<Tensor<T>> rows, int width)
    {
        if (rows.Count == 0)
        {
            return new Tensor<T>(new[] { 0, width });
        }

        var fitted = new Tensor<T>[rows.Count];
        int index = 0;
        foreach (var row in rows)
        {
            fitted[index++] = FitLength(row, width);
        }

        return Engine.TensorStack(fitted, 0);
    }

    /// <summary>
    /// Stacks <paramref name="rows"/> into a [rows.Count, width] tensor, truncating or zero-padding each row.
    /// </summary>
    public static Tensor<T> StackRows(IReadOnlyList<Vector<T>> rows, int width)
    {
        var tensors = new List<Tensor<T>>(rows.Count);
        foreach (var row in rows)
        {
            tensors.Add(Tensor<T>.FromVector(row));
        }

        return StackRows(tensors, width);
    }

    /// <summary>
    /// The rows <paramref name="rowIndices"/> of <paramref name="data"/>, first <paramref name="width"/> columns
    /// (zero-padded when the data is narrower), as a [rowIndices.Length, width] tensor.
    /// </summary>
    public static Tensor<T> GatherRows(Tensor<T> data, int[] rowIndices, int width)
    {
        int columns = data.Shape[1];
        var picked = Engine.TensorIndexSelect(data, new Tensor<int>(rowIndices, new[] { rowIndices.Length }), 0);
        if (columns == width)
        {
            return picked;
        }

        if (columns > width)
        {
            return Engine.TensorSlice(picked, new[] { 0, 0 }, new[] { rowIndices.Length, width });
        }

        return Engine.TensorConcatenate(new[] { picked, new Tensor<T>(new[] { rowIndices.Length, width - columns }) }, 1);
    }

    /// <summary>
    /// Each entry truncated toward zero and converted to int, as <c>(int)value</c> does.
    /// </summary>
    public static int[] TruncateToInt(Vector<T> values)
    {
        return Engine.TensorTrunc(Tensor<T>.FromVector(values)).Cast<int>().ToArray();
    }

    /// <summary>
    /// Per-sample output activation for the VGM layout (Xu et al. 2019 §4.2): tanh on each continuous
    /// column's normalized value, softmax over its mode indicators and over each categorical block.
    /// Entries past the last column stay 0.
    /// </summary>
    public static Tensor<T> ApplyOutputActivations(Tensor<T> output, TabularDataTransformer<T> transformer, int columnCount)
    {
        var engine = Engine;
        var flat = output.Reshape(new[] { output.Length });
        int total = flat.Length;
        var blocks = new List<Tensor<T>>(columnCount * 2);
        int idx = 0;

        for (int col = 0; col < columnCount && idx < total; col++)
        {
            var transform = transformer.GetTransformInfo(col);
            int softmaxWidth = transform.Width;
            if (transform.IsContinuous)
            {
                blocks.Add(engine.TensorTanh(engine.TensorSlice(flat, new[] { idx }, new[] { 1 })));
                idx++;
                softmaxWidth = transform.Width - 1;
            }

            int count = Math.Min(softmaxWidth, total - idx);
            if (count > 0)
            {
                blocks.Add(engine.Softmax(engine.TensorSlice(flat, new[] { idx }, new[] { count }), -1));
                idx += count;
            }
        }

        if (idx < total)
        {
            blocks.Add(new Tensor<T>(new[] { total - idx }));
        }

        var activated = blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 0);
        return activated.Reshape(output._shape);
    }

    /// <summary>
    /// Standard Gumbel noise -log(-log U), U ~ Uniform(1e-10, 1 - 1e-10), drawn with one seed from <paramref name="random"/>.
    /// </summary>
    public static Tensor<T> GumbelNoise(int[] shape, Random random)
    {
        var engine = Engine;
        var u = engine.TensorRandomUniformRange<T>(shape, NumOps.FromDouble(1e-10), NumOps.FromDouble(1.0 - 1e-10), random.Next());
        return engine.TensorNegate(engine.TensorLog(engine.TensorNegate(engine.TensorLog(u))));
    }

    /// <summary>
    /// Batched output activation for the VGM layout (Xu et al. 2019 §4.2 and sdv-dev/CTGAN <c>_apply_activate</c>):
    /// tanh on each continuous column's normalized value and a softmax over each mode block and categorical block.
    /// With <paramref name="gumbelNoise"/> ([batch, width], the same shape as <paramref name="output"/>) the softmax
    /// blocks are Gumbel-softmax samples softmax((logits + g) / <paramref name="temperature"/>), as the reference
    /// code's <c>gumbel_softmax(tau=0.2)</c>; without it they are the plain softmax of the logits.
    /// Columns past the transformer's layout get tanh. Without a transformer the whole output gets tanh.
    /// </summary>
    public static Tensor<T> ApplyOutputActivationsBatched(Tensor<T> output, TabularDataTransformer<T>? transformer,
        int columnCount, Tensor<T>? gumbelNoise, double temperature)
    {
        var engine = Engine;
        if (transformer is null) return engine.TensorTanh(output);

        int batch = output.Shape[0];
        int totalWidth = output.Shape[1];
        var logits = output;
        if (gumbelNoise is not null)
        {
            logits = engine.TensorMultiplyScalar(engine.TensorAdd(output, gumbelNoise), NumOps.FromDouble(1.0 / temperature));
        }

        var blocks = new List<Tensor<T>>(columnCount * 2);
        int idx = 0;
        for (int col = 0; col < columnCount && idx < totalWidth; col++)
        {
            var transform = transformer.GetTransformInfo(col);
            int softmaxWidth = transform.Width;
            if (transform.IsContinuous)
            {
                blocks.Add(engine.TensorTanh(engine.TensorSlice(output, new[] { 0, idx }, new[] { batch, 1 })));
                idx++;
                softmaxWidth = transform.Width - 1;
            }

            int count = Math.Min(softmaxWidth, totalWidth - idx);
            if (count > 0)
            {
                blocks.Add(engine.Softmax(engine.TensorSlice(logits, new[] { 0, idx }, new[] { batch, count }), axis: 1));
                idx += count;
            }
        }

        if (idx < totalWidth)
        {
            blocks.Add(engine.TensorTanh(engine.TensorSlice(output, new[] { 0, idx }, new[] { batch, totalWidth - idx })));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), axis: 1);
    }

    /// <summary>
    /// Replaces every mode block and categorical block of each generated row ([rows, width]) by a one-hot sample
    /// from that block's probabilities. sdv-dev/CTGAN samples with <c>gumbel_softmax</c> and then takes the argmax,
    /// which is the Gumbel-max trick: a draw from the categorical distribution. Taking the argmax of the plain
    /// softmax instead would always pick the most likely mode or category.
    /// </summary>
    public static Tensor<T> SampleCategoricalBlocks(Tensor<T> rows, TabularDataTransformer<T> transformer, int columnCount, Random random)
    {
        var engine = Engine;
        int batch = rows.Shape[0];
        int totalWidth = rows.Shape[1];
        var blocks = new List<Tensor<T>>(columnCount * 2);
        int idx = 0;
        for (int col = 0; col < columnCount && idx < totalWidth; col++)
        {
            var transform = transformer.GetTransformInfo(col);
            int categoricalWidth = transform.Width;
            if (transform.IsContinuous)
            {
                blocks.Add(engine.TensorSlice(rows, new[] { 0, idx }, new[] { batch, 1 }));
                idx++;
                categoricalWidth = transform.Width - 1;
            }

            int count = Math.Min(categoricalWidth, totalWidth - idx);
            if (count > 0)
            {
                var probabilities = engine.TensorSlice(rows, new[] { 0, idx }, new[] { batch, count });
                blocks.Add(engine.TensorCategoricalSample(probabilities, axis: 1, seed: random.Next()));
                idx += count;
            }
        }

        if (idx < totalWidth)
        {
            blocks.Add(engine.TensorSlice(rows, new[] { 0, idx }, new[] { batch, totalWidth - idx }));
        }

        return blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), axis: 1);
    }

    /// <summary>
    /// Offset and width of each categorical column's block in the generated row, in categorical-column order
    /// (the order the conditional vector uses).
    /// </summary>
    public static List<(int Offset, int Width)> CategoricalOutputBlocks(TabularDataTransformer<T>? transformer, int columnCount)
    {
        var blocks = new List<(int Offset, int Width)>();
        if (transformer is null) return blocks;
        for (int col = 0; col < columnCount; col++)
        {
            var info = transformer.GetTransformInfo(col);
            if (!info.IsContinuous) blocks.Add((info.StartOffset, info.Width));
        }

        return blocks;
    }

    /// <summary>
    /// Conditional generator loss (Xu et al. 2019 §4.3, sdv-dev/CTGAN <c>_cond_loss</c>): the cross-entropy of each
    /// conditioned column's category logits against the requested one-hot, masked to that column and averaged
    /// over the batch.
    /// </summary>
    public static Tensor<T> ConditionalCrossEntropy(Tensor<T> generatorLogits, Tensor<T> condBatch, Tensor<T> maskBatch,
        IReadOnlyList<(int Offset, int Width)> categoricalBlocks)
    {
        var engine = Engine;
        int batch = generatorLogits.Shape[0];
        var blocks = new Tensor<T>[categoricalBlocks.Count];
        int index = 0;
        foreach (var (offset, width) in categoricalBlocks)
        {
            blocks[index++] = engine.TensorLogSoftmax(
                engine.TensorSlice(generatorLogits, new[] { 0, offset }, new[] { batch, width }), 1);
        }

        var logp = blocks.Length == 1 ? blocks[0] : engine.TensorConcatenate(blocks, axis: 1);
        var masked = engine.TensorMultiply(engine.TensorMultiply(maskBatch, condBatch), logp);
        var perRow = engine.ReduceSum(masked, new[] { 1 }, keepDims: false);
        return engine.TensorNegate(engine.ReduceMean(perRow, new[] { 0 }, keepDims: false));
    }

    /// <summary>
    /// A batch of training-by-sampling conditions with their column masks, each [batchSize, condWidth].
    /// </summary>
    public static (Tensor<T> Cond, Tensor<T> Mask) SampleCondMaskBatch(CTGANDataSampler<T>? sampler, int batchSize, int condWidth)
    {
        if (sampler is null || condWidth == 0)
            return (new Tensor<T>(new[] { batchSize, condWidth }), new Tensor<T>(new[] { batchSize, condWidth }));

        var conds = new List<Vector<T>>(batchSize);
        var masks = new List<Vector<T>>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, mv, _, _, _) = sampler.SampleCondVecWithMask();
            conds.Add(cv);
            masks.Add(mv);
        }

        return (StackRows(conds, condWidth), StackRows(masks, condWidth));
    }

    /// <summary>
    /// A batch of training-by-sampling conditions [batchSize, condWidth] with the real row drawn for each.
    /// </summary>
    public static (Tensor<T> CondBatch, int[] RowIndices) SampleConditionsWithRows(CTGANDataSampler<T>? sampler, int batchSize, int condWidth)
    {
        var rowIndices = new int[batchSize];
        if (sampler is null) return (new Tensor<T>(new[] { batchSize, condWidth }), rowIndices);

        var conds = new List<Vector<T>>(batchSize);
        for (int b = 0; b < batchSize; b++)
        {
            var (cv, rowIndex) = sampler.SampleConditionAndRow();
            conds.Add(cv);
            rowIndices[b] = rowIndex;
        }

        return (StackRows(conds, condWidth), rowIndices);
    }

    /// <summary>
    /// Interleaves two Box-Muller halves z1[0], z2[0], z1[1], ... into a [batchSize, width] tensor, dropping the
    /// spare value when batchSize * width is odd.
    /// </summary>
    public static Tensor<T> InterleaveNormalPairs(Tensor<T> z1, Tensor<T> z2, int batchSize, int width)
    {
        int total = batchSize * width;
        int half = z1.Length;
        var interleaved = Engine.TensorStack(new[] { z1, z2 }, 1).Reshape(new[] { 2 * half });
        var noise = 2 * half == total ? interleaved : Engine.TensorSlice(interleaved, new[] { 0 }, new[] { total });
        return noise.Reshape(new[] { batchSize, width });
    }

    /// <summary>
    /// Gaussian-copula forward transform of one column: z = Phi^-1(u) with u = (midRank + 1) / (n + 1) clipped to
    /// [1e-6, 1 - 1e-6], where midRank is the mean sorted position of each value's ties. Returns z and the sorted
    /// column, which <see cref="GaussianCopulaInverse"/> interpolates.
    /// </summary>
    public static (Tensor<T> Z, Tensor<T> Sorted) GaussianCopulaForward(Tensor<T> values)
    {
        var engine = Engine;
        int n = values.Length;
        T one = NumOps.One;
        var sorted = engine.TensorSort(values, 0, false).Values;

        var first = engine.TensorSearchSorted(sorted, values, right: false).Cast<T>();
        var last = engine.TensorSubtractScalar(engine.TensorSearchSorted(sorted, values, right: true).Cast<T>(), one);
        var midRank = engine.TensorMultiplyScalar(engine.TensorAdd(first, last), NumOps.FromDouble(0.5));

        var u = engine.TensorClamp(
            engine.TensorMultiplyScalar(engine.TensorAddScalar(midRank, one), NumOps.FromDouble(1.0 / (n + 1.0))),
            NumOps.FromDouble(1e-6), NumOps.FromDouble(1.0 - 1e-6));
        return (StandardNormalQuantile(u), sorted);
    }

    /// <summary>
    /// Inverse of <see cref="GaussianCopulaForward"/>: u = Phi(z), rank = u (n + 1) - 1 clipped to [0, n - 1], and
    /// the value linearly interpolated between the sorted entries around that rank.
    /// </summary>
    public static Tensor<T> GaussianCopulaInverse(Tensor<T> z, Tensor<T> sorted)
    {
        var engine = Engine;
        int n = sorted.Length;
        T zero = NumOps.Zero;
        T one = NumOps.One;
        T lastIndex = NumOps.FromDouble(n - 1);

        var u = engine.TensorClamp(StandardNormalCdf(z), zero, one);
        var rank = engine.TensorClamp(
            engine.TensorSubtractScalar(engine.TensorMultiplyScalar(u, NumOps.FromDouble(n + 1.0)), one), zero, lastIndex);
        var lo = engine.TensorFloor(rank);
        var hi = engine.TensorMin(engine.TensorAddScalar(lo, one), lastIndex);
        var frac = engine.TensorSubtract(rank, lo);

        var below = engine.TensorGather(sorted, lo.Cast<int>(), 0);
        var above = engine.TensorGather(sorted, hi.Cast<int>(), 0);
        return engine.TensorAdd(below, engine.TensorMultiply(engine.TensorSubtract(above, below), frac));
    }

    /// <summary>Phi^-1(u) = sqrt(2) erfinv(2u - 1) (scipy.stats.norm.ppf).</summary>
    public static Tensor<T> StandardNormalQuantile(Tensor<T> u)
    {
        var engine = Engine;
        var centered = engine.TensorSubtractScalar(engine.TensorMultiplyScalar(u, NumOps.FromDouble(2.0)), NumOps.One);
        return engine.TensorMultiplyScalar(engine.TensorErfinv(centered), NumOps.FromDouble(Math.Sqrt(2.0)));
    }

    /// <summary>Phi(z) = (1 + erf(z / sqrt 2)) / 2 (scipy.stats.norm.cdf).</summary>
    public static Tensor<T> StandardNormalCdf(Tensor<T> z)
    {
        var engine = Engine;
        var erf = engine.TensorErf(engine.TensorMultiplyScalar(z, NumOps.FromDouble(1.0 / Math.Sqrt(2.0))));
        return engine.TensorMultiplyScalar(engine.TensorAddScalar(erf, NumOps.One), NumOps.FromDouble(0.5));
    }

    /// <summary>
    /// Matrix exponential e^A of a square [d, d] tensor by scaling and squaring (Higham 2005): A is scaled by 2^-s
    /// so its 1-norm is at most 1/2, a degree-12 Taylor polynomial is evaluated in Horner form (truncation error
    /// below 1e-13 at that norm), and the result is squared s times. Matches scipy.linalg.expm to double precision
    /// for the small, non-negative matrices NOTEARS uses.
    /// </summary>
    public static Tensor<T> MatrixExponential(Tensor<T> a)
    {
        var engine = Engine;
        int d = a.Shape[0];
        var identity = engine.TensorEye<T>(d);
        if (d == 0) return identity;

        // 1-norm: the largest absolute column sum.
        double norm = NumOps.ToDouble(engine.TensorMaxValue(engine.ReduceSum(engine.TensorAbs(a), new[] { 0 }, keepDims: false)));
        int squarings = norm > 0.5 ? (int)Math.Ceiling(Math.Log(norm / 0.5, 2.0)) : 0;
        var x = squarings > 0 ? engine.TensorMultiplyScalar(a, NumOps.FromDouble(Math.Pow(2.0, -squarings))) : a;

        // Horner: e^X ~ I + X (I + X/2 (I + X/3 ( ... (I + X/12)))).
        Tensor<T> Level(Tensor<T> inner, double inverseK)
            => engine.TensorAdd(identity, engine.TensorMultiplyScalar(engine.TensorMatMul(x, inner), NumOps.FromDouble(inverseK)));
        var e = engine.TensorAdd(identity, engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.0 / 12.0)));
        e = Level(e, 1.0 / 11.0);
        e = Level(e, 1.0 / 10.0);
        e = Level(e, 1.0 / 9.0);
        e = Level(e, 1.0 / 8.0);
        e = Level(e, 1.0 / 7.0);
        e = Level(e, 1.0 / 6.0);
        e = Level(e, 1.0 / 5.0);
        e = Level(e, 1.0 / 4.0);
        e = Level(e, 1.0 / 3.0);
        e = Level(e, 1.0 / 2.0);
        e = Level(e, 1.0);

        for (int i = 0; i < squarings; i++)
        {
            e = engine.TensorMatMul(e, e);
        }

        return e;
    }
}
