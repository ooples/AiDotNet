using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// A linear layer that owns its weight <c>[in, out]</c> and bias <c>[out]</c>, initialized like
/// <c>torch.nn.Linear</c> (uniform in <c>+-1/sqrt(in)</c>) from the model's seed scope, so DINO can apply the
/// reference's explicit re-initializations (zeroed box-head output, prior-probability class bias).
/// </summary>
internal sealed class DetrLinear<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public DetrLinear(int inFeatures, int outFeatures)
    {
        if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
        if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
        InFeatures = inFeatures;
        OutFeatures = outFeatures;
        Weight = new Tensor<T>(new[] { inFeatures, outFeatures });
        Bias = new Tensor<T>(new[] { outFeatures });
        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        double bound = 1.0 / Math.Sqrt(inFeatures);
        for (int i = 0; i < Weight.Length; i++) Weight[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
        for (int i = 0; i < Bias.Length; i++) Bias[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
    }

    public int InFeatures { get; }
    public int OutFeatures { get; }
    public Tensor<T> Weight { get; }
    public Tensor<T> Bias { get; }

    /// <summary>Xavier-uniform weight and zero bias (the reference's attention and projection init).</summary>
    public void XavierUniformZeroBias()
    {
        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        double bound = Math.Sqrt(6.0 / (InFeatures + OutFeatures));
        for (int i = 0; i < Weight.Length; i++) Weight[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
        Bias.Fill(_numOps.Zero);
    }

    /// <summary>
    /// Xavier-uniform weight, bias unchanged: DeformableTransformer._reset_parameters re-initializes every
    /// parameter with more than one dimension this way and leaves Linear biases at their defaults.
    /// </summary>
    public void XavierUniformWeight()
    {
        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        double bound = Math.Sqrt(6.0 / (InFeatures + OutFeatures));
        for (int i = 0; i < Weight.Length; i++) Weight[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
    }
    /// <summary>Zero weight and bias.</summary>
    public void Zero()
    {
        Weight.Fill(_numOps.Zero);
        Bias.Fill(_numOps.Zero);
    }

    /// <summary>Copies another linear layer's values (the reference deep-copies its heads).</summary>
    public void CopyFrom(DetrLinear<T> other)
    {
        for (int i = 0; i < Weight.Length; i++) Weight[i] = other.Weight[i];
        for (int i = 0; i < Bias.Length; i++) Bias[i] = other.Bias[i];
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var engine = AiDotNetEngine.Current;
        int rows = input.Length / InFeatures;
        var outShape = (int[])input._shape.Clone();
        outShape[outShape.Length - 1] = OutFeatures;
        var product = engine.TensorMatMul(engine.Reshape(input, new[] { rows, InFeatures }), Weight);
        var biased = engine.TensorAdd(product, engine.TensorBroadcastTo(engine.Reshape(Bias, new[] { 1, OutFeatures }), new[] { rows, OutFeatures }));
        return engine.Reshape(biased, outShape);
    }

    public void Write(BinaryWriter writer)
    {
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) writer.Write(_numOps.ToDouble(tensor[i]));
    }

    public void Read(BinaryReader reader)
    {
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) tensor[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return Weight;
        yield return Bias;
    }
}

/// <summary>The reference <c>MLP</c>: linear layers with ReLU between them (none after the last).</summary>
internal sealed class DetrMlp<T> : CvParameterModule<T>
{
    public DetrMlp(int inputDim, int hiddenDim, int outputDim, int numLayers)
    {
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        Layers = new List<DetrLinear<T>>();
        for (int i = 0; i < numLayers; i++)
            Layers.Add(new DetrLinear<T>(i == 0 ? inputDim : hiddenDim, i == numLayers - 1 ? outputDim : hiddenDim));
    }

    public List<DetrLinear<T>> Layers { get; }

    public Tensor<T> Forward(Tensor<T> input)
    {
        var x = input;
        for (int i = 0; i < Layers.Count; i++)
        {
            x = Layers[i].Forward(x);
            if (i < Layers.Count - 1) x = AiDotNetEngine.Current.ReLU(x);
        }
        return x;
    }

    public void CopyFrom(DetrMlp<T> other)
    {
        for (int i = 0; i < Layers.Count; i++) Layers[i].CopyFrom(other.Layers[i]);
    }

    public void Write(BinaryWriter writer) { foreach (var layer in Layers) layer.Write(writer); }

    public void Read(BinaryReader reader) { foreach (var layer in Layers) layer.Read(reader); }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Layers;
}

/// <summary>GroupNorm over NCHW with learnable affine parameters (32 groups in DINO's input projection).</summary>
internal sealed class DinoGroupNorm<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _groups;

    public DinoGroupNorm(int groups, int channels)
    {
        if (channels % groups != 0) throw new ArgumentException($"{channels} channels do not split into {groups} groups.", nameof(groups));
        _groups = groups;
        Gamma = new Tensor<T>(new[] { channels });
        Beta = new Tensor<T>(new[] { channels });
        Gamma.Fill(_numOps.One);
    }

    public Tensor<T> Gamma { get; }
    public Tensor<T> Beta { get; }

    public Tensor<T> Forward(Tensor<T> input) => AiDotNetEngine.Current.GroupNorm(input, _groups, Gamma, Beta, 1e-5, out _, out _);

    public void Write(BinaryWriter writer)
    {
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) writer.Write(_numOps.ToDouble(tensor[i]));
    }

    public void Read(BinaryReader reader)
    {
        foreach (var tensor in OwnParameterTensors())
            for (int i = 0; i < tensor.Length; i++) tensor[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren() => Array.Empty<IParameterSource<T>?>();

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return Gamma;
        yield return Beta;
    }
}

/// <summary>
/// The reference <c>DeformableTransformerEncoderLayer</c> (post-norm): deformable self-attention over the
/// multi-scale tokens, then a ReLU feed-forward block, each with a residual and LayerNorm. DINO trains
/// without dropout.
/// </summary>
internal sealed class DeformableEncoderLayer<T> : CvParameterModule<T>
{
    private readonly MultiScaleDeformableAttention<T> _selfAttention;
    private readonly LayerNorm<T> _norm1;
    private readonly DetrLinear<T> _linear1;
    private readonly DetrLinear<T> _linear2;
    private readonly LayerNorm<T> _norm2;

    public DeformableEncoderLayer(int dModel, int feedForward, int numLevels, int numHeads, int numPoints)
    {
        _selfAttention = new MultiScaleDeformableAttention<T>(dModel, numLevels, numHeads, numPoints);
        _norm1 = new LayerNorm<T>(dModel, 1e-5);
        _linear1 = new DetrLinear<T>(dModel, feedForward);
        _linear2 = new DetrLinear<T>(feedForward, dModel);
        _linear1.XavierUniformWeight();
        _linear2.XavierUniformWeight();
        _norm2 = new LayerNorm<T>(dModel, 1e-5);
    }

    /// <summary>The layer's LayerNorms, in forward order.</summary>
    internal IEnumerable<LayerNorm<T>> Norms() { yield return _norm1; yield return _norm2; }

    public Tensor<T> Forward(Tensor<T> source, Tensor<T> position, Tensor<T> referencePoints, int[][] spatialShapes, int[] levelStarts)
    {
        var engine = AiDotNetEngine.Current;
        var attended = _selfAttention.Forward(engine.TensorAdd(source, position), referencePoints, source, spatialShapes, levelStarts);
        var x = _norm1.Forward(engine.TensorAdd(source, attended));
        var ffn = _linear2.Forward(engine.ReLU(_linear1.Forward(x)));
        return _norm2.Forward(engine.TensorAdd(x, ffn));
    }

    public void Write(BinaryWriter writer)
    {
        _selfAttention.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _linear1.Write(writer);
        _linear2.Write(writer);
        _norm2.WriteParameters(writer);
    }

    public void Read(BinaryReader reader)
    {
        _selfAttention.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _linear1.Read(reader);
        _linear2.Read(reader);
        _norm2.ReadParameters(reader);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _selfAttention;
        yield return _norm1;
        yield return _linear1;
        yield return _linear2;
        yield return _norm2;
    }
}

/// <summary>
/// The reference DINO <c>DeformableTransformerDecoderLayer</c> in its default order: self-attention among the
/// queries (with the denoising group mask), deformable cross-attention into the encoder memory at the query's
/// reference box, then the feed-forward block. Each has a residual and LayerNorm.
/// </summary>
internal sealed class DeformableDecoderLayer<T> : CvParameterModule<T>
{
    private readonly int _dModel;
    private readonly int _numHeads;
    private readonly DetrLinear<T> _query;
    private readonly DetrLinear<T> _key;
    private readonly DetrLinear<T> _value;
    private readonly DetrLinear<T> _selfOutput;
    private readonly LayerNorm<T> _norm2;
    private readonly MultiScaleDeformableAttention<T> _crossAttention;
    private readonly LayerNorm<T> _norm1;
    private readonly DetrLinear<T> _linear1;
    private readonly DetrLinear<T> _linear2;
    private readonly LayerNorm<T> _norm3;

    public DeformableDecoderLayer(int dModel, int feedForward, int numLevels, int numHeads, int numPoints)
    {
        _dModel = dModel;
        _numHeads = numHeads;
        // nn.MultiheadAttention: Xavier-uniform in-projections with zero bias, zero out-projection bias.
        _query = new DetrLinear<T>(dModel, dModel);
        _key = new DetrLinear<T>(dModel, dModel);
        _value = new DetrLinear<T>(dModel, dModel);
        _query.XavierUniformZeroBias();
        _key.XavierUniformZeroBias();
        _value.XavierUniformZeroBias();
        _selfOutput = new DetrLinear<T>(dModel, dModel);
        _selfOutput.XavierUniformZeroBias();
        _norm2 = new LayerNorm<T>(dModel, 1e-5);
        _crossAttention = new MultiScaleDeformableAttention<T>(dModel, numLevels, numHeads, numPoints);
        _norm1 = new LayerNorm<T>(dModel, 1e-5);
        _linear1 = new DetrLinear<T>(dModel, feedForward);
        _linear2 = new DetrLinear<T>(feedForward, dModel);
        _linear1.XavierUniformWeight();
        _linear2.XavierUniformWeight();
        _norm3 = new LayerNorm<T>(dModel, 1e-5);
    }

    /// <summary>The layer's LayerNorms, in forward order (self-attention, cross-attention, feed-forward).</summary>
    internal IEnumerable<LayerNorm<T>> Norms() { yield return _norm2; yield return _norm1; yield return _norm3; }

    /// <param name="target">Queries <c>[batch, queries, d]</c>.</param>
    /// <param name="queryPosition">Positional queries from the reference boxes, <c>[batch, queries, d]</c>.</param>
    /// <param name="referenceBoxes">Reference boxes per level, <c>[batch, queries, levels, 4]</c>.</param>
    /// <param name="memory">Encoder memory <c>[batch, tokens, d]</c>.</param>
    /// <param name="attendMask"><c>[batch, heads, queries, queries]</c>, true where a query may attend; null for none.</param>
    public Tensor<T> Forward(Tensor<T> target, Tensor<T> queryPosition, Tensor<T> referenceBoxes, Tensor<T> memory,
        int[][] spatialShapes, int[] levelStarts, Tensor<bool>? attendMask)
    {
        var engine = AiDotNetEngine.Current;
        int batch = target.Shape[0], queries = target.Shape[1], headDim = _dModel / _numHeads;

        var withPosition = engine.TensorAdd(target, queryPosition);
        Tensor<T> Heads(Tensor<T> x) => engine.TensorPermute(engine.Reshape(x, new[] { batch, queries, _numHeads, headDim }), new[] { 0, 2, 1, 3 });
        var context = engine.ScaledDotProductAttention(
            Heads(_query.Forward(withPosition)), Heads(_key.Forward(withPosition)), Heads(_value.Forward(target)),
            attendMask, 1.0 / Math.Sqrt(headDim), out _);
        var merged = engine.Reshape(engine.TensorPermute(context, new[] { 0, 2, 1, 3 }), new[] { batch, queries, _dModel });
        var x = _norm2.Forward(engine.TensorAdd(target, _selfOutput.Forward(merged)));

        var crossed = _crossAttention.Forward(engine.TensorAdd(x, queryPosition), referenceBoxes, memory, spatialShapes, levelStarts);
        x = _norm1.Forward(engine.TensorAdd(x, crossed));

        var ffn = _linear2.Forward(engine.ReLU(_linear1.Forward(x)));
        return _norm3.Forward(engine.TensorAdd(x, ffn));
    }

    public void Write(BinaryWriter writer)
    {
        _query.Write(writer); _key.Write(writer); _value.Write(writer); _selfOutput.Write(writer);
        _norm2.WriteParameters(writer);
        _crossAttention.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _linear1.Write(writer); _linear2.Write(writer);
        _norm3.WriteParameters(writer);
    }

    public void Read(BinaryReader reader)
    {
        _query.Read(reader); _key.Read(reader); _value.Read(reader); _selfOutput.Read(reader);
        _norm2.ReadParameters(reader);
        _crossAttention.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _linear1.Read(reader); _linear2.Read(reader);
        _norm3.ReadParameters(reader);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        yield return _query; yield return _key; yield return _value; yield return _selfOutput;
        yield return _norm2;
        yield return _crossAttention;
        yield return _norm1;
        yield return _linear1; yield return _linear2;
        yield return _norm3;
    }
}

/// <summary>Sine embeddings used by DINO, computed on the host for inputs the reference detaches.</summary>
internal static class DetrEmbeddings
{
    /// <summary>
    /// <c>PositionEmbeddingSineHW</c> with normalize = true, scale 2 pi and the given temperature, for an
    /// unpadded <c>height x width</c> map: <c>[height * width, 2 * numPosFeats]</c>, y features then x.
    /// </summary>
    public static double[] SinePositionalEncoding(int height, int width, int numPosFeats, double temperature)
    {
        var result = new double[height * width * numPosFeats * 2];
        const double eps = 1e-6, scale = 2 * Math.PI;
        for (int y = 0; y < height; y++)
        {
            double yEmbed = (y + 1) / (height + eps) * scale;
            for (int x = 0; x < width; x++)
            {
                double xEmbed = (x + 1) / (width + eps) * scale;
                int row = ((y * width) + x) * numPosFeats * 2;
                for (int i = 0; i < numPosFeats; i++)
                {
                    double dim = Math.Pow(temperature, 2.0 * (i / 2) / numPosFeats);
                    result[row + i] = (i % 2 == 0) ? Math.Sin(yEmbed / dim) : Math.Cos(yEmbed / dim);
                    result[row + numPosFeats + i] = (i % 2 == 0) ? Math.Sin(xEmbed / dim) : Math.Cos(xEmbed / dim);
                }
            }
        }
        return result;
    }

    /// <summary>
    /// <c>gen_sineembed_for_position</c> for boxes (cx, cy, w, h) in [0, 1]: 128 features per coordinate with
    /// temperature 10000, concatenated as (y, x, w, h) - 512 features per box.
    /// </summary>
    public static double[] BoxSineEmbedding(double[] boxes, int count)
    {
        const int feats = 128;
        var result = new double[count * feats * 4];
        for (int b = 0; b < count; b++)
        {
            double cx = boxes[b * 4], cy = boxes[(b * 4) + 1], w = boxes[(b * 4) + 2], h = boxes[(b * 4) + 3];
            double[] order = { cy, cx, w, h };
            for (int part = 0; part < 4; part++)
            {
                double embed = order[part] * 2 * Math.PI;
                int offset = (b * feats * 4) + (part * feats);
                for (int i = 0; i < feats; i++)
                {
                    double dim = Math.Pow(10000, 2.0 * (i / 2) / feats);
                    result[offset + i] = (i % 2 == 0) ? Math.Sin(embed / dim) : Math.Cos(embed / dim);
                }
            }
        }
        return result;
    }

    /// <summary>The reference <c>inverse_sigmoid</c> with eps 1e-5, on the host.</summary>
    public static double InverseSigmoid(double x)
    {
        const double eps = 1e-5;
        x = Math.Min(Math.Max(x, 0), 1);
        return Math.Log(Math.Max(x, eps) / Math.Max(1 - x, eps));
    }
}
