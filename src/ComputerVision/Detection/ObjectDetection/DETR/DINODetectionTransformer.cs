using System.IO;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>Every head output of one DETR-family forward (DINO, RT-DETR). Boxes are sigmoid (cx, cy, w, h) in [0, 1].</summary>
internal sealed class DetrPass<T>
{
    public List<Tensor<T>> Classes { get; } = new();
    public List<Tensor<T>> Boxes { get; } = new();
    /// <summary>Pre-sigmoid box logits of every layer's matching queries (Predict's box format).</summary>
    public List<Tensor<T>> BoxLogits { get; } = new();
    public List<Tensor<T>> DenoisingClasses { get; } = new();
    public List<Tensor<T>> DenoisingBoxes { get; } = new();
    public Tensor<T> EncoderClasses { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> EncoderBoxes { get; set; } = new Tensor<T>(new[] { 0 });
    /// <summary>Pre-sigmoid encoder proposal boxes of the selected queries.</summary>
    public Tensor<T> EncoderBoxLogits { get; set; } = new Tensor<T>(new[] { 0 });
    public Tensor<T> FinalBoxLogits { get; set; } = new Tensor<T>(new[] { 0 });
    public ContrastiveDenoisingPlan<T>? Denoising { get; set; }

    public List<Tensor<T>> All()
    {
        var all = new List<Tensor<T>>();
        all.AddRange(Classes);
        all.AddRange(Boxes);
        all.AddRange(DenoisingClasses);
        all.AddRange(DenoisingBoxes);
        all.Add(EncoderClasses);
        all.Add(EncoderBoxes);
        return all;
    }
}

/// <summary>
/// DINO's deformable transformer and heads (reference <c>DeformableTransformer</c> with two-stage mixed query
/// selection, plus the DINO heads):
/// <list type="bullet">
/// <item>Input projection: a 1x1 conv + GroupNorm(32) per backbone level, plus a 3x3 stride-2 conv +
/// GroupNorm on the last level.</item>
/// <item>A 2-D sine positional encoding with a learnable level embedding.</item>
/// <item>The deformable encoder.</item>
/// <item>Encoder proposals scored by their own class/box heads. The top-K positions become the anchors,
/// and the content queries are learnable.</item>
/// <item>A six-layer decoder with iterative box refinement, where each prediction uses the previous
/// layer's undetached box ("look forward twice").</item>
/// <item>Shared class and box heads.</item>
/// </list>
/// </summary>
internal sealed class DinoDetectionTransformer<T> : CvParameterModule<T>
{
    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();
    private readonly int _d;
    private readonly int _numLevels;
    private readonly int _numHeads;
    private readonly int _numQueries;
    private readonly int _numClasses;
    private readonly double _temperature;

    private readonly List<Conv2D<T>> _projections = new();
    private readonly List<DinoGroupNorm<T>> _projectionNorms = new();
    private readonly Tensor<T> _levelEmbed;
    private readonly List<DeformableEncoderLayer<T>> _encoder = new();
    private readonly DetrLinear<T> _encoderOutput;
    private readonly LayerNorm<T> _encoderOutputNorm;
    private readonly DetrLinear<T> _encoderClass;
    private readonly DetrMlp<T> _encoderBox;
    private readonly Tensor<T> _targetEmbed;
    private readonly Tensor<T> _labelEmbed;
    private readonly List<DeformableDecoderLayer<T>> _decoder = new();
    private readonly DetrMlp<T> _refPointHead;
    private readonly LayerNorm<T> _decoderNorm;
    private readonly DetrLinear<T> _classHead;
    private readonly DetrMlp<T> _boxHead;

    public DinoDetectionTransformer(DINOOptionsView options, IReadOnlyList<int> backboneChannels, int numClasses)
    {
        _d = options.HiddenDimension;
        _numHeads = options.NumHeads;
        _numQueries = options.NumQueries;
        _numClasses = numClasses;
        _temperature = options.PositionalTemperature;
        _numLevels = backboneChannels.Count + 1;
        if (_d % 2 != 0) throw new ArgumentException("The hidden dimension must be even for the sine encoding.", nameof(options));

        for (int l = 0; l < backboneChannels.Count; l++)
        {
            _projections.Add(new Conv2D<T>(backboneChannels[l], _d, 1));
            _projectionNorms.Add(new DinoGroupNorm<T>(32, _d));
        }
        _projections.Add(new Conv2D<T>(backboneChannels[backboneChannels.Count - 1], _d, 3, 2, 1));
        _projectionNorms.Add(new DinoGroupNorm<T>(32, _d));

        var random = AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope.NextRandom();
        _levelEmbed = Normal(new[] { _numLevels, _d }, random);

        for (int i = 0; i < options.NumEncoderLayers; i++)
            _encoder.Add(new DeformableEncoderLayer<T>(_d, options.FeedForwardDimension, _numLevels, _numHeads, options.NumSamplingPoints));

        _encoderOutput = new DetrLinear<T>(_d, _d);
        _encoderOutput.XavierUniformWeight();
        _encoderOutputNorm = new LayerNorm<T>(_d, 1e-5);

        // tgt_embed is created with normal_ and then re-initialized by _reset_parameters (Xavier), so Xavier.
        _targetEmbed = new Tensor<T>(new[] { _numQueries, _d });
        double bound = Math.Sqrt(6.0 / (_numQueries + _d));
        for (int i = 0; i < _targetEmbed.Length; i++) _targetEmbed[i] = _numOps.FromDouble(((random.NextDouble() * 2) - 1) * bound);
        // label_enc: nn.Embedding(dn_labelbook_size + 1, d), N(0, 1).
        _labelEmbed = Normal(new[] { numClasses + 1, _d }, random);

        for (int i = 0; i < options.NumDecoderLayers; i++)
            _decoder.Add(new DeformableDecoderLayer<T>(_d, options.FeedForwardDimension, _numLevels, _numHeads, options.NumSamplingPoints));
        _refPointHead = new DetrMlp<T>(2 * _d, _d, _d, 2);
        foreach (var layer in _refPointHead.Layers) layer.XavierUniformWeight();
        _decoderNorm = new LayerNorm<T>(_d, 1e-5);

        // Shared heads: class bias = -log((1 - 0.01) / 0.01); the box MLP's last layer starts at zero.
        _classHead = new DetrLinear<T>(_d, numClasses);
        _classHead.Bias.Fill(_numOps.FromDouble(-Math.Log((1 - 0.01) / 0.01)));
        _boxHead = new DetrMlp<T>(_d, _d, 4, 3);
        _boxHead.Layers[_boxHead.Layers.Count - 1].Zero();
        // The encoder heads are deep copies of the decoder heads (two_stage_*_embed_share = False).
        _encoderClass = new DetrLinear<T>(_d, numClasses);
        _encoderClass.CopyFrom(_classHead);
        _encoderBox = new DetrMlp<T>(_d, _d, 4, 3);
        _encoderBox.CopyFrom(_boxHead);
    }

    /// <summary>Feature levels: the backbone taps plus the extra stride-2 level.</summary>
    public int NumLevels => _numLevels;

    public int NumHeads => _numHeads;

    /// <summary>Every LayerNorm gamma in forward order (test access for controlled-weight fixtures).</summary>
    internal IEnumerable<Tensor<T>> LayerNormGammas()
    {
        foreach (var layer in _encoder) foreach (var norm in layer.Norms()) yield return norm.Gamma;
        yield return _encoderOutputNorm.Gamma;
        foreach (var layer in _decoder) foreach (var norm in layer.Norms()) yield return norm.Gamma;
        yield return _decoderNorm.Gamma;
    }

    /// <summary>The learnable content queries, <c>[numQueries, d]</c>.</summary>
    internal Tensor<T> ContentQueries => _targetEmbed;

    /// <summary>The shared class head.</summary>
    internal DetrLinear<T> ClassHead => _classHead;

    /// <summary>The shared box-refinement MLP.</summary>
    internal DetrMlp<T> BoxHead => _boxHead;

    /// <summary>Runs the transformer on the backbone levels; <paramref name="denoising"/> is null at inference.</summary>
    public DetrPass<T> Forward(IReadOnlyList<Tensor<T>> backboneLevels, ContrastiveDenoisingPlan<T>? denoising)
    {
        var engine = AiDotNetEngine.Current;
        if (backboneLevels.Count != _numLevels - 1)
            throw new ArgumentException($"Expected {_numLevels - 1} backbone levels.", nameof(backboneLevels));

        // Input projection and flattening, level by level in the reference's order.
        var maps = new List<Tensor<T>>();
        for (int l = 0; l < backboneLevels.Count; l++) maps.Add(_projectionNorms[l].Forward(_projections[l].Forward(backboneLevels[l])));
        maps.Add(_projectionNorms[_numLevels - 1].Forward(_projections[_numLevels - 1].Forward(backboneLevels[backboneLevels.Count - 1])));

        int batch = maps[0].Shape[0];
        var shapes = new int[_numLevels][];
        var starts = new int[_numLevels];
        var flattened = new Tensor<T>[_numLevels];
        int tokens = 0;
        for (int l = 0; l < _numLevels; l++)
        {
            int h = maps[l].Shape[2], w = maps[l].Shape[3];
            shapes[l] = new[] { h, w };
            starts[l] = tokens;
            tokens += h * w;
            flattened[l] = engine.TensorPermute(engine.Reshape(maps[l], new[] { batch, _d, h * w }), new[] { 0, 2, 1 });
        }
        var source = engine.TensorConcatenate(flattened, 1);

        // Positional encoding: sine (host constant) + the level's learnable embedding.
        var sine = new T[batch * tokens * _d];
        var levelOfToken = new int[tokens];
        for (int l = 0; l < _numLevels; l++)
        {
            var pe = DetrEmbeddings.SinePositionalEncoding(shapes[l][0], shapes[l][1], _d / 2, _temperature);
            for (int t = 0; t < shapes[l][0] * shapes[l][1]; t++)
            {
                levelOfToken[starts[l] + t] = l;
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < _d; c++)
                        sine[(((b * tokens) + starts[l] + t) * _d) + c] = _numOps.FromDouble(pe[(t * _d) + c]);
            }
        }
        var levelEmbedding = engine.TensorBroadcastTo(
            engine.Reshape(CvTensorOps<T>.Select(_levelEmbed, levelOfToken, 0), new[] { 1, tokens, _d }), new[] { batch, tokens, _d });
        var position = engine.TensorAdd(new Tensor<T>(sine, new[] { batch, tokens, _d }), levelEmbedding);

        // Encoder reference points: every token's pixel center, the same on every level (no padding).
        var encoderReference = new T[batch * tokens * _numLevels * 2];
        for (int l = 0; l < _numLevels; l++)
        {
            int h = shapes[l][0], w = shapes[l][1];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    for (int b = 0; b < batch; b++)
                        for (int level = 0; level < _numLevels; level++)
                        {
                            int index = ((((b * tokens) + starts[l] + (y * w) + x) * _numLevels) + level) * 2;
                            encoderReference[index] = _numOps.FromDouble((x + 0.5) / w);
                            encoderReference[index + 1] = _numOps.FromDouble((y + 0.5) / h);
                        }
        }
        var encoderReferenceTensor = new Tensor<T>(encoderReference, new[] { batch, tokens, _numLevels, 2 });

        var memory = source;
        foreach (var layer in _encoder) memory = layer.Forward(memory, position, encoderReferenceTensor, shapes, starts);

        // Two-stage proposals (gen_encoder_output_proposals): token-centered boxes of size 0.05 * 2^level.
        // An invalid proposal (any coordinate outside (0.01, 0.99)) has its memory zeroed and its logit set to
        // a large value standing in for the reference's +inf, so its sigmoid is exactly 1 with zero gradient.
        var proposal = new T[batch * tokens * 4];
        var valid = new T[batch * tokens * _d];
        for (int l = 0; l < _numLevels; l++)
        {
            int h = shapes[l][0], w = shapes[l][1];
            double size = 0.05 * Math.Pow(2, l);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double[] box = { (x + 0.5) / w, (y + 0.5) / h, size, size };
                    bool isValid = box.All(v => v > 0.01 && v < 0.99);
                    for (int b = 0; b < batch; b++)
                    {
                        int token = (b * tokens) + starts[l] + (y * w) + x;
                        for (int coord = 0; coord < 4; coord++)
                            proposal[(token * 4) + coord] = _numOps.FromDouble(isValid ? Math.Log(box[coord] / (1 - box[coord])) : 1e4);
                        for (int c = 0; c < _d; c++) valid[(token * _d) + c] = isValid ? _numOps.One : _numOps.Zero;
                    }
                }
        }
        var outputMemory = _encoderOutputNorm.Forward(_encoderOutput.Forward(
            engine.TensorMultiply(memory, new Tensor<T>(valid, new[] { batch, tokens, _d }))));
        var encoderClassAll = _encoderClass.Forward(outputMemory);
        var encoderBoxAll = engine.TensorAdd(_encoderBox.Forward(outputMemory), new Tensor<T>(proposal, new[] { batch, tokens, 4 }));

        // Top-K tokens by their best class logit (ties keep the lower index, like a stable sort).
        int k = Math.Min(_numQueries, tokens);
        var classValues = encoderClassAll.ToArray();
        var rows = new int[batch * k];
        for (int b = 0; b < batch; b++)
        {
            var best = new double[tokens];
            for (int t = 0; t < tokens; t++)
            {
                double m = double.NegativeInfinity;
                for (int c = 0; c < _numClasses; c++) m = Math.Max(m, _numOps.ToDouble(classValues[(((b * tokens) + t) * _numClasses) + c]));
                best[t] = m;
            }
            var order = Enumerable.Range(0, tokens).OrderByDescending(t => best[t]).ThenBy(t => t).Take(k).ToArray();
            for (int i = 0; i < k; i++) rows[(b * k) + i] = (b * tokens) + order[i];
        }
        var referenceUndetached = engine.Reshape(CvTensorOps<T>.Select(engine.Reshape(encoderBoxAll, new[] { batch * tokens, 4 }), rows, 0), new[] { batch, k, 4 });
        var pass = new DetrPass<T>
        {
            EncoderClasses = engine.Reshape(CvTensorOps<T>.Select(engine.Reshape(encoderClassAll, new[] { batch * tokens, _numClasses }), rows, 0), new[] { batch, k, _numClasses }),
            EncoderBoxes = engine.Sigmoid(referenceUndetached),
            EncoderBoxLogits = referenceUndetached,
            Denoising = denoising
        };

        // Queries: learnable content (mixed query selection) with detached encoder anchors; denoising first.
        int pad = denoising?.PadSize ?? 0;
        int total = pad + k;
        var content = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorSlice(_targetEmbed, new[] { 0, 0 }, new[] { k, _d }), new[] { 1, k, _d }), new[] { batch, k, _d });
        var anchorValues = referenceUndetached.ToArray();
        var referenceSigmoid = new double[batch * total * 4];
        for (int b = 0; b < batch; b++)
            for (int q = 0; q < k; q++)
                for (int c = 0; c < 4; c++)
                    referenceSigmoid[(((b * total) + pad + q) * 4) + c] = Sigmoid(_numOps.ToDouble(anchorValues[(((b * k) + q) * 4) + c]));
        var target = content;
        Tensor<bool>? mask = null;
        if (denoising is not null)
        {
            var labelRows = new int[batch * pad];
            var occupied = new T[batch * pad * _d];
            for (int slot = 0; slot < batch * pad; slot++)
            {
                labelRows[slot] = denoising.Occupied[slot] ? denoising.Labels[slot] : 0;
                for (int c = 0; c < _d; c++) occupied[(slot * _d) + c] = denoising.Occupied[slot] ? _numOps.One : _numOps.Zero;
            }
            var labelQueries = engine.TensorMultiply(
                engine.Reshape(CvTensorOps<T>.Select(_labelEmbed, labelRows, 0), new[] { batch, pad, _d }),
                new Tensor<T>(occupied, new[] { batch, pad, _d }));
            target = engine.TensorConcatenate(new[] { labelQueries, content }, 1);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < pad; s++)
                    for (int c = 0; c < 4; c++)
                        referenceSigmoid[(((b * total) + s) * 4) + c] = Sigmoid(denoising.UnsigmoidBoxes[(((b * pad) + s) * 4) + c]);
            mask = ContrastiveDenoising<T>.AttendMask(batch, _numHeads, total, denoising);
        }

        // Decoder with iterative refinement. References entering a layer are detached; each layer's refined
        // box stays on the tape for the next layer's prediction (look forward twice).
        var output = target;
        var refined = new List<Tensor<T>>();
        var hidden = new List<Tensor<T>>();
        var firstReference = (double[])referenceSigmoid.Clone();
        for (int i = 0; i < _decoder.Count; i++)
        {
            var referenceBoxes = new T[batch * total * _numLevels * 4];
            for (int q = 0; q < batch * total; q++)
                for (int l = 0; l < _numLevels; l++)
                    for (int c = 0; c < 4; c++)
                        referenceBoxes[(((q * _numLevels) + l) * 4) + c] = _numOps.FromDouble(referenceSigmoid[(q * 4) + c]);
            var sineQuery = DetrEmbeddings.BoxSineEmbedding(referenceSigmoid, batch * total);
            var queryPosition = _refPointHead.Forward(new Tensor<T>(sineQuery.Select(v => _numOps.FromDouble(v)).ToArray(), new[] { batch, total, 2 * _d }));

            output = _decoder[i].Forward(output, queryPosition, new Tensor<T>(referenceBoxes, new[] { batch, total, _numLevels, 4 }),
                memory, shapes, starts, mask);

            var newReference = engine.Sigmoid(engine.TensorAdd(_boxHead.Forward(output), InverseSigmoidConstant(referenceSigmoid, batch, total)));
            refined.Add(newReference);
            var newValues = newReference.ToArray();
            for (int j = 0; j < referenceSigmoid.Length; j++) referenceSigmoid[j] = _numOps.ToDouble(newValues[j]);
            hidden.Add(_decoderNorm.Forward(output));
        }

        for (int i = 0; i < _decoder.Count; i++)
        {
            var previous = i == 0 ? InverseSigmoidConstant(firstReference, batch, total) : InverseSigmoidTape(refined[i - 1]);
            var boxLogits = engine.TensorAdd(_boxHead.Forward(hidden[i]), previous);
            var classes = _classHead.Forward(hidden[i]);
            pass.Classes.Add(engine.TensorSlice(classes, new[] { 0, pad, 0 }, new[] { batch, k, _numClasses }));
            var matchingBoxLogits = engine.TensorSlice(boxLogits, new[] { 0, pad, 0 }, new[] { batch, k, 4 });
            pass.BoxLogits.Add(matchingBoxLogits);
            pass.Boxes.Add(engine.Sigmoid(matchingBoxLogits));
            if (pad > 0)
            {
                pass.DenoisingClasses.Add(engine.TensorSlice(classes, new[] { 0, 0, 0 }, new[] { batch, pad, _numClasses }));
                pass.DenoisingBoxes.Add(engine.Sigmoid(engine.TensorSlice(boxLogits, new[] { 0, 0, 0 }, new[] { batch, pad, 4 })));
            }
            if (i == _decoder.Count - 1) pass.FinalBoxLogits = matchingBoxLogits;
        }

        return pass;
    }

    private Tensor<T> InverseSigmoidConstant(double[] sigmoid, int batch, int total)
        => new Tensor<T>(sigmoid.Select(v => _numOps.FromDouble(DetrEmbeddings.InverseSigmoid(v))).ToArray(), new[] { batch, total, 4 });

    /// <summary>The reference <c>inverse_sigmoid</c> (eps 1e-5) on the tape.</summary>
    private Tensor<T> InverseSigmoidTape(Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        var eps = _numOps.FromDouble(1e-5);
        var numerator = engine.TensorMax(x, eps);
        var denominator = engine.TensorMax(engine.TensorAddScalar(engine.TensorNegate(x), _numOps.One), eps);
        return engine.TensorSubtract(engine.TensorLog(numerator), engine.TensorLog(denominator));
    }

    private Tensor<T> Normal(int[] shape, Random random)
    {
        var tensor = new Tensor<T>(shape);
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            tensor[i] = _numOps.FromDouble(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return tensor;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    public void Write(BinaryWriter writer)
    {
        writer.Write(_d); writer.Write(_numLevels); writer.Write(_numHeads); writer.Write(_numQueries); writer.Write(_numClasses);
        writer.Write(_encoder.Count); writer.Write(_decoder.Count);
        foreach (var conv in _projections) conv.WriteParameters(writer);
        foreach (var norm in _projectionNorms) norm.Write(writer);
        WriteTensor(writer, _levelEmbed);
        foreach (var layer in _encoder) layer.Write(writer);
        _encoderOutput.Write(writer); _encoderOutputNorm.WriteParameters(writer); _encoderClass.Write(writer); _encoderBox.Write(writer);
        WriteTensor(writer, _targetEmbed); WriteTensor(writer, _labelEmbed);
        foreach (var layer in _decoder) layer.Write(writer);
        _refPointHead.Write(writer); _decoderNorm.WriteParameters(writer); _classHead.Write(writer); _boxHead.Write(writer);
    }

    public void Read(BinaryReader reader)
    {
        int d = reader.ReadInt32(), levels = reader.ReadInt32(), heads = reader.ReadInt32(), queries = reader.ReadInt32(), classes = reader.ReadInt32();
        int encoderLayers = reader.ReadInt32(), decoderLayers = reader.ReadInt32();
        if (d != _d || levels != _numLevels || heads != _numHeads || queries != _numQueries || classes != _numClasses
            || encoderLayers != _encoder.Count || decoderLayers != _decoder.Count)
            throw new InvalidOperationException("DINO transformer configuration mismatch in the weight file.");
        foreach (var conv in _projections) conv.ReadParameters(reader);
        foreach (var norm in _projectionNorms) norm.Read(reader);
        ReadTensor(reader, _levelEmbed);
        foreach (var layer in _encoder) layer.Read(reader);
        _encoderOutput.Read(reader); _encoderOutputNorm.ReadParameters(reader); _encoderClass.Read(reader); _encoderBox.Read(reader);
        ReadTensor(reader, _targetEmbed); ReadTensor(reader, _labelEmbed);
        foreach (var layer in _decoder) layer.Read(reader);
        _refPointHead.Read(reader); _decoderNorm.ReadParameters(reader); _classHead.Read(reader); _boxHead.Read(reader);
    }

    private void WriteTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++) writer.Write(_numOps.ToDouble(tensor[i]));
    }

    private void ReadTensor(BinaryReader reader, Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++) tensor[i] = _numOps.FromDouble(reader.ReadDouble());
    }

    protected override IEnumerable<IParameterSource<T>?> ParameterChildren()
    {
        foreach (var conv in _projections) yield return conv;
        foreach (var norm in _projectionNorms) yield return norm;
        foreach (var layer in _encoder) yield return layer;
        yield return _encoderOutput;
        yield return _encoderOutputNorm;
        yield return _encoderClass;
        yield return _encoderBox;
        foreach (var layer in _decoder) yield return layer;
        yield return _refPointHead;
        yield return _decoderNorm;
        yield return _classHead;
        yield return _boxHead;
    }

    protected override IEnumerable<Tensor<T>> OwnParameterTensors()
    {
        yield return _levelEmbed;
        yield return _targetEmbed;
        yield return _labelEmbed;
    }
}

/// <summary>The DINO hyperparameters the transformer reads, independent of the numeric type.</summary>
internal readonly record struct DINOOptionsView(int HiddenDimension, int NumHeads, int NumEncoderLayers, int NumDecoderLayers,
    int FeedForwardDimension, int NumQueries, int NumSamplingPoints, double PositionalTemperature);
