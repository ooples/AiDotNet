using System.IO;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// RT-DETR's decoder (reference <c>RTDETRTransformer</c>). It runs these stages:
/// <list type="bullet">
/// <item>A 1x1 ConvNorm projection of the hybrid encoder's three levels, flattened into the memory.</item>
/// <item>Token-centered anchors of size 0.05 * 2^level.</item>
/// <item>IoU-aware query selection: the top-K tokens by encoder class score. Their encoder features, detached,
/// are the content queries, and their refined anchors, detached, are the reference boxes.</item>
/// <item>The decoder layers: masked self-attention, deformable cross-attention and an FFN, with
/// <c>query_pos</c> an MLP of the raw box.</item>
/// <item>Per-layer class and box heads with iterative refinement. During training, each layer's box uses the
/// previous layer's undetached box ("look forward twice").</item>
/// </list>
/// </summary>
internal sealed class RtdetrDecoderTransformer<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _d;
    private readonly int _numHeads;
    private readonly int _numLevels;
    private readonly int _numQueries;
    private readonly int _numClasses;

    private readonly List<RtdetrConvNorm<T>> _inputProjections = new();
    private readonly DetrLinear<T> _encoderOutput;
    private readonly LayerNorm<T> _encoderOutputNorm;
    private readonly DetrLinear<T> _encoderScore;
    private readonly DetrMlp<T> _encoderBox;
    private readonly Tensor<T> _labelEmbed;
    private readonly List<DeformableDecoderLayer<T>> _layers = new();
    private readonly DetrMlp<T> _queryPosHead;
    private readonly List<DetrLinear<T>> _scoreHeads = new();
    private readonly List<DetrMlp<T>> _boxHeads = new();

    public RtdetrDecoderTransformer(int encoderHidden, int dModel, int numHeads, int numLayers, int feedForward,
        int numQueries, int numPoints, int numClasses)
    {
        _d = dModel;
        _numHeads = numHeads;
        _numLevels = 3;
        _numQueries = numQueries;
        _numClasses = numClasses;
        double prior = -Math.Log((1 - 0.01) / 0.01);

        for (int l = 0; l < _numLevels; l++) _inputProjections.Add(new RtdetrConvNorm<T>(encoderHidden, dModel, 1));
        _encoderOutput = new DetrLinear<T>(dModel, dModel);
        _encoderOutput.XavierUniformWeight();
        _encoderOutputNorm = new LayerNorm<T>(dModel, 1e-5);
        _encoderScore = new DetrLinear<T>(dModel, numClasses);
        _encoderScore.Bias.Fill(_numOps.FromDouble(prior));
        _encoderBox = new DetrMlp<T>(dModel, dModel, 4, 3);
        _encoderBox.Layers[_encoderBox.Layers.Count - 1].Zero();

        // denoising_class_embed: nn.Embedding(num_classes + 1, d, padding_idx=num_classes), N(0, 1), padding row 0.
        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        _labelEmbed = new Tensor<T>(new[] { numClasses + 1, dModel });
        for (int i = 0; i < numClasses * dModel; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            _labelEmbed[i] = _numOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new DeformableDecoderLayer<T>(dModel, feedForward, _numLevels, numHeads, numPoints, transformerWideXavier: false));
            var score = new DetrLinear<T>(dModel, numClasses);
            score.Bias.Fill(_numOps.FromDouble(prior));
            _scoreHeads.Add(score);
            var box = new DetrMlp<T>(dModel, dModel, 4, 3);
            box.Layers[box.Layers.Count - 1].Zero();
            _boxHeads.Add(box);
        }
        _queryPosHead = new DetrMlp<T>(4, 2 * dModel, dModel, 2);
        foreach (var layer in _queryPosHead.Layers) layer.XavierUniformWeight();
    }

    public int NumHeads => _numHeads;

    // Test access for controlled-weight fixtures.
    internal IEnumerable<Tensor<T>> InputProjectionBetas() => _inputProjections.Select(projection => projection.NormBeta);
    internal IReadOnlyList<DeformableDecoderLayer<T>> Layers => _layers;
    internal DetrMlp<T> QueryPosHead => _queryPosHead;
    internal IReadOnlyList<DetrLinear<T>> ScoreHeads => _scoreHeads;
    internal IReadOnlyList<DetrMlp<T>> BoxHeads => _boxHeads;
    internal IEnumerable<Tensor<T>> LayerNormGammas()
    {
        yield return _encoderOutputNorm.Gamma;
        foreach (var layer in _layers) foreach (var norm in layer.Norms()) yield return norm.Gamma;
    }

    public DetrPass<T> Forward(IReadOnlyList<Tensor<T>> encoderLevels, ContrastiveDenoisingPlan<T>? denoising)
    {
        var engine = AiDotNetEngine.Current;
        if (encoderLevels.Count != _numLevels) throw new ArgumentException("RT-DETR's decoder takes three levels.", nameof(encoderLevels));
        int batch = encoderLevels[0].Shape[0];
        var shapes = new int[_numLevels][];
        var starts = new int[_numLevels];
        var flattened = new Tensor<T>[_numLevels];
        int tokens = 0;
        for (int l = 0; l < _numLevels; l++)
        {
            var map = _inputProjections[l].Forward(encoderLevels[l]);
            int h = map.Shape[2], w = map.Shape[3];
            shapes[l] = new[] { h, w };
            starts[l] = tokens;
            tokens += h * w;
            flattened[l] = engine.TensorPermute(engine.Reshape(map, new[] { batch, _d, h * w }), new[] { 0, 2, 1 });
        }
        var memory = engine.TensorConcatenate(flattened, 1);

        // _generate_anchors: grid centers with size 0.05 * 2^level, valid inside (0.01, 0.99), logit space. The
        // reference's +inf for an invalid anchor is a large finite logit here: sigmoid is exactly 1, gradient 0.
        var anchors = new T[batch * tokens * 4];
        var valid = new T[batch * tokens * _d];
        for (int l = 0; l < _numLevels; l++)
        {
            int h = shapes[l][0], w = shapes[l][1];
            double size = 0.05 * Math.Pow(2, l);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double[] anchor = { (x + 0.5) / w, (y + 0.5) / h, size, size };
                    bool isValid = anchor.All(v => v > 0.01 && v < 1 - 0.01);
                    for (int b = 0; b < batch; b++)
                    {
                        int token = (b * tokens) + starts[l] + (y * w) + x;
                        for (int coord = 0; coord < 4; coord++)
                            anchors[(token * 4) + coord] = _numOps.FromDouble(isValid ? Math.Log(anchor[coord] / (1 - anchor[coord])) : 1e4);
                        for (int c = 0; c < _d; c++) valid[(token * _d) + c] = isValid ? _numOps.One : _numOps.Zero;
                    }
                }
        }
        var outputMemory = _encoderOutputNorm.Forward(_encoderOutput.Forward(
            engine.TensorMultiply(memory, new Tensor<T>(valid, new[] { batch, tokens, _d }))));
        var encoderClassAll = _encoderScore.Forward(outputMemory);
        var encoderBoxAll = engine.TensorAdd(_encoderBox.Forward(outputMemory), new Tensor<T>(anchors, new[] { batch, tokens, 4 }));

        int k = Math.Min(_numQueries, tokens);
        int numClasses = _numClasses;
        var rows = AiDotNet.Training.CompiledStepHostValue<T>.Compute(
            "RtdetrQuerySelection",
            new[] { encoderClassAll },
            new[] { batch * k },
            (inputs, dst) => SelectTopKQueries(inputs[0], batch, tokens, numClasses, k, dst));
        var referenceUndetached = engine.Reshape(
            engine.TensorEmbeddingLookupFromFloatIndices(engine.Reshape(encoderBoxAll, new[] { batch * tokens, 4 }), rows),
            new[] { batch, k, 4 });
        var pass = new DetrPass<T>
        {
            EncoderClasses = engine.Reshape(
                engine.TensorEmbeddingLookupFromFloatIndices(engine.Reshape(encoderClassAll, new[] { batch * tokens, _numClasses }), rows),
                new[] { batch, k, _numClasses }),
            EncoderBoxes = engine.Sigmoid(referenceUndetached),
            EncoderBoxLogits = referenceUndetached,
            Denoising = denoising
        };

        // Content queries: the selected encoder features, detached (learnt_init_query = False).
        var content = engine.StopGradient(engine.Reshape(
            engine.TensorEmbeddingLookupFromFloatIndices(engine.Reshape(outputMemory, new[] { batch * tokens, _d }), rows),
            new[] { batch, k, _d }));
        int pad = denoising?.PadSize ?? 0;
        int total = pad + k;
        var referenceSigmoidTensor = engine.StopGradient(engine.Sigmoid(referenceUndetached));
        var target = content;
        Tensor<bool>? mask = null;
        if (denoising is not null)
        {
            var labelRows = new int[batch * pad];
            var occupied = new T[batch * pad * _d];
            for (int slot = 0; slot < batch * pad; slot++)
            {
                // Padding slots use the zero padding row (padding_idx = num_classes).
                labelRows[slot] = denoising.Occupied[slot] ? denoising.Labels[slot] : _numClasses;
                for (int c = 0; c < _d; c++) occupied[(slot * _d) + c] = denoising.Occupied[slot] ? _numOps.One : _numOps.Zero;
            }
            var labelQueries = engine.TensorMultiply(
                engine.Reshape(CvTensorOps<T>.Select(_labelEmbed, labelRows, 0), new[] { batch, pad, _d }),
                new Tensor<T>(occupied, new[] { batch, pad, _d }));
            target = engine.TensorConcatenate(new[] { labelQueries, content }, 1);
            var dnSigmoid = new double[batch * pad * 4];
            for (int idx = 0; idx < dnSigmoid.Length; idx++)
                dnSigmoid[idx] = Sigmoid(denoising.UnsigmoidBoxes[idx]);
            referenceSigmoidTensor = engine.TensorConcatenate(
                new[] { new Tensor<double>(dnSigmoid, new[] { batch, pad, 4 }).Cast<T>(), referenceSigmoidTensor }, 1);
            mask = ContrastiveDenoising<T>.AttendMask(batch, _numHeads, total, denoising);
        }

        var output = target;
        Tensor<T>? previous = null;
        for (int i = 0; i < _layers.Count; i++)
        {
            var referenceBoxes = engine.TensorBroadcastTo(
                engine.Reshape(referenceSigmoidTensor, new[] { batch, total, 1, 4 }),
                new[] { batch, total, _numLevels, 4 });
            var queryPosition = _queryPosHead.Forward(referenceSigmoidTensor);
            output = _layers[i].Forward(output, queryPosition, referenceBoxes,
                memory, shapes, starts, mask);

            var delta = _boxHeads[i].Forward(output);
            var detachedInvSigmoid = DetrHeads<T>.InverseSigmoid(referenceSigmoidTensor);
            var inter = engine.Sigmoid(engine.TensorAdd(delta, detachedInvSigmoid));
            var boxLogits = previous is null
                ? engine.TensorAdd(delta, detachedInvSigmoid)
                : engine.TensorAdd(delta, DetrHeads<T>.InverseSigmoid(previous));
            var classes = _scoreHeads[i].Forward(output);

            var matchingBoxLogits = engine.TensorSlice(boxLogits, new[] { 0, pad, 0 }, new[] { batch, k, 4 });
            pass.Classes.Add(engine.TensorSlice(classes, new[] { 0, pad, 0 }, new[] { batch, k, _numClasses }));
            pass.BoxLogits.Add(matchingBoxLogits);
            pass.Boxes.Add(engine.Sigmoid(matchingBoxLogits));
            if (pad > 0)
            {
                pass.DenoisingClasses.Add(engine.TensorSlice(classes, new[] { 0, 0, 0 }, new[] { batch, pad, _numClasses }));
                pass.DenoisingBoxes.Add(engine.Sigmoid(engine.TensorSlice(boxLogits, new[] { 0, 0, 0 }, new[] { batch, pad, 4 })));
            }
            if (i == _layers.Count - 1) pass.FinalBoxLogits = matchingBoxLogits;

            previous = inter;
            referenceSigmoidTensor = engine.StopGradient(inter);
        }

        return pass;
    }

    private static void SelectTopKQueries(Tensor<T> encoderClassAll, int batch, int tokens, int numClasses, int k, Tensor<T> dst)
    {
        var classValues = encoderClassAll.Contiguous().Cast<double>().ToArray();
        var rows = new double[batch * k];
        for (int b = 0; b < batch; b++)
        {
            var best = new double[tokens];
            for (int t = 0; t < tokens; t++)
            {
                double m = double.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                    m = Math.Max(m, classValues[(((b * tokens) + t) * numClasses) + c]);
                best[t] = m;
            }
            var order = Enumerable.Range(0, tokens).OrderByDescending(t => best[t]).ThenBy(t => t).Take(k).ToArray();
            for (int i = 0; i < k; i++)
                rows[(b * k) + i] = (b * tokens) + order[i];
        }
        var typed = new Tensor<double>(dst._shape, new Vector<double>(rows)).Cast<T>();
        typed.AsSpan().CopyTo(dst.AsWritableSpan());
    }

    public void SetTrainingMode(bool training)
    {
        foreach (var projection in _inputProjections) projection.SetTrainingMode(training);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public void Write(BinaryWriter writer)
    {
        writer.Write(_d); writer.Write(_numHeads); writer.Write(_numQueries); writer.Write(_numClasses); writer.Write(_layers.Count);
        foreach (var projection in _inputProjections) projection.Write(writer);
        _encoderOutput.Write(writer); _encoderOutputNorm.WriteParameters(writer); _encoderScore.Write(writer); _encoderBox.Write(writer);
        for (int i = 0; i < _labelEmbed.Length; i++) writer.Write(_numOps.ToDouble(_labelEmbed[i]));
        foreach (var layer in _layers) layer.Write(writer);
        _queryPosHead.Write(writer);
        foreach (var head in _scoreHeads) head.Write(writer);
        foreach (var head in _boxHeads) head.Write(writer);
    }

    public void Read(BinaryReader reader)
    {
        int d = reader.ReadInt32(), heads = reader.ReadInt32(), queries = reader.ReadInt32(), classes = reader.ReadInt32(), layers = reader.ReadInt32();
        if (d != _d || heads != _numHeads || queries != _numQueries || classes != _numClasses || layers != _layers.Count)
            throw new InvalidOperationException("RT-DETR decoder configuration mismatch in the weight file.");
        foreach (var projection in _inputProjections) projection.Read(reader);
        _encoderOutput.Read(reader); _encoderOutputNorm.ReadParameters(reader); _encoderScore.Read(reader); _encoderBox.Read(reader);
        for (int i = 0; i < _labelEmbed.Length; i++) _labelEmbed[i] = _numOps.FromDouble(reader.ReadDouble());
        foreach (var layer in _layers) layer.Read(reader);
        _queryPosHead.Read(reader);
        foreach (var head in _scoreHeads) head.Read(reader);
        foreach (var head in _boxHeads) head.Read(reader);
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        foreach (var projection in _inputProjections) yield return projection;
        yield return _encoderOutput;
        yield return _encoderOutputNorm;
        yield return _encoderScore;
        yield return _encoderBox;
        foreach (var layer in _layers) yield return layer;
        yield return _queryPosHead;
        foreach (var head in _scoreHeads) yield return head;
        foreach (var head in _boxHeads) yield return head;
    }

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return _labelEmbed;
    }
}
