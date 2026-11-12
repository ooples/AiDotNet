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
}
