namespace AiDotNet.TextToSpeech.FrontEnd;

/// <summary>
/// Thinc's <c>Tok2Vec.v2</c> (<c>MultiHashEmbed.v2</c> + <c>MaxoutWindowEncoder.v2</c>) and a linear tagger head as spaCy
/// serializes them (srsly MessagePack of each model's nodes and float32 parameters), run in float32 on the engine.
/// </summary>
internal sealed class SpacyTok2Vec
{
    private static readonly IEngine Engine = AiDotNetEngine.Current;
    private const double LayerNormEpsilon = 1e-8;
    private const int EncoderPadding = 4;

    private sealed record Embedding(Tensor<float> Table, uint Seed);
    private sealed record Maxout(Tensor<float> Weight, Tensor<float> Bias, int Outputs, int Pieces);
    private sealed record Norm(Tensor<float> Gain, Tensor<float> Bias);

    private readonly Embedding[] _embeddings;
    private readonly (Maxout Maxout, Norm Norm) _reduce;
    private readonly (Maxout Maxout, Norm Norm)[] _blocks;
    private readonly Tensor<float> _headWeight, _headBias;
    private readonly int _width;

    private SpacyTok2Vec(Embedding[] embeddings, (Maxout, Norm) reduce, (Maxout, Norm)[] blocks, Tensor<float> headWeight, Tensor<float> headBias)
    {
        _embeddings = embeddings;
        _reduce = reduce;
        _blocks = blocks;
        _headWeight = headWeight;
        _headBias = headBias;
        _width = reduce.Item1.Outputs;
    }

    private static Tensor<float> Tensor((int[] Shape, float[] Values) array) => new(array.Shape, new Vector<float>(array.Values));

    // A parameter vector as one row [1, n].
    private static Tensor<float> Row((int[] Shape, float[] Values) array) => new(new[] { 1, array.Values.Length }, new Vector<float>(array.Values));

    /// <summary>Loads a tok2vec model file and a tagger (softmax) model file.</summary>
    public static SpacyTok2Vec Load(string tok2vecPath, string headPath)
    {
        var model = (Dictionary<object, object?>)MessagePackReader.Read(File.ReadAllBytes(tok2vecPath))!;
        var nodes = ((List<object?>)model["nodes"]!).Cast<Dictionary<object, object?>>().ToList();
        var attrs = ((List<object?>)model["attrs"]!).Cast<Dictionary<object, object?>>().ToList();
        var parameters = ((List<object?>)model["params"]!).Cast<Dictionary<object, object?>>().ToList();
        var embeddings = new List<Embedding>();
        var maxouts = new List<Maxout>();
        var norms = new List<Norm>();
        for (int i = 0; i < nodes.Count; i++)
        {
            switch ((string)nodes[i]["name"]!)
            {
                case "hashembed":
                {
                    // Thinc serializes each attribute as its own MessagePack value.
                    var seed = Convert.ToUInt32(MessagePackReader.Read((byte[])attrs[i]["seed"]!));
                    embeddings.Add(new Embedding(Tensor(MessagePackReader.Float32Array(parameters[i]["E"])), seed));
                    break;
                }
                case "maxout":
                {
                    var w = MessagePackReader.Float32Array(parameters[i]["W"]);                 // [nO, nP, nI]
                    int outputs = w.Shape[0], pieces = w.Shape[1], inputs = w.Shape[2];
                    var weight = Engine.TensorTranspose(Engine.Reshape(Tensor(w), new[] { outputs * pieces, inputs }));   // [nI, nO·nP]
                    var bias = Row(MessagePackReader.Float32Array(parameters[i]["b"]));
                    maxouts.Add(new Maxout(weight, bias, outputs, pieces));
                    break;
                }
                case "layernorm":
                    norms.Add(new Norm(Row(MessagePackReader.Float32Array(parameters[i]["G"])), Row(MessagePackReader.Float32Array(parameters[i]["b"]))));
                    break;
            }
        }
        if (embeddings.Count == 0 || maxouts.Count < 1 || maxouts.Count != norms.Count)
            throw new InvalidDataException("The tok2vec model does not have the MultiHashEmbed and MaxoutWindowEncoder layers expected.");

        var head = (Dictionary<object, object?>)MessagePackReader.Read(File.ReadAllBytes(headPath))!;
        var headNodes = ((List<object?>)head["nodes"]!).Cast<Dictionary<object, object?>>().ToList();
        var headParams = ((List<object?>)head["params"]!).Cast<Dictionary<object, object?>>().ToList();
        int softmax = headNodes.FindIndex(n => (string)n["name"]! == "softmax");
        if (softmax < 0) throw new InvalidDataException("The tagger model has no softmax layer.");
        var headWeight = Engine.TensorTranspose(Tensor(MessagePackReader.Float32Array(headParams[softmax]["W"])));   // [nI, nO]
        var headBias = Row(MessagePackReader.Float32Array(headParams[softmax]["b"]));
        return new SpacyTok2Vec(embeddings.ToArray(), (maxouts[0], norms[0]),
            maxouts.Skip(1).Zip(norms.Skip(1), (m, n) => (m, n)).ToArray(), headWeight, headBias);
    }

    /// <summary>Thinc's <c>MurmurHash3_x86_128_uint64</c>: the four 32-bit hashes of one 64-bit key.</summary>
    internal static uint[] Hash(ulong value, uint seed)
    {
        ulong h1 = value * 0x87c37b91114253d5UL;
        h1 = (h1 << 31) | (h1 >> 33);
        h1 *= 0x4cf5ad432745937fUL;
        h1 ^= seed;
        h1 ^= 8;
        ulong h2 = seed;
        h2 ^= 8;
        h1 += h2;
        h2 += h1;
        static ulong Mix(ulong h)
        {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdUL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53UL;
            h ^= h >> 33;
            return h;
        }
        h1 = Mix(h1);
        h2 = Mix(h2);
        h1 += h2;
        h2 += h1;
        return new[] { (uint)h1, (uint)(h1 >> 32), (uint)h2, (uint)(h2 >> 32) };
    }

    private static Tensor<float> MaxoutForward(Maxout layer, Tensor<float> x)
    {
        int n = x.Shape[0];
        var y = Engine.TensorAdd(Engine.TensorMatMul(x, layer.Weight), Engine.TensorBroadcastTo(layer.Bias, new[] { n, layer.Outputs * layer.Pieces }));
        return Engine.ReduceMax(Engine.Reshape(y, new[] { n, layer.Outputs, layer.Pieces }), new[] { 2 }, keepDims: false);
    }

    // Thinc's LayerNorm: (x − mean) / sqrt(var + 1e-8), then gain and bias.
    private static Tensor<float> NormForward(Norm layer, Tensor<float> x)
    {
        var shape = x.Shape.ToArray();
        var centered = Engine.TensorSubtract(x, Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { 1 }, keepDims: true), shape));
        var variance = Engine.TensorAddScalar(Engine.ReduceMean(Engine.TensorMultiply(centered, centered), new[] { 1 }, keepDims: true), (float)LayerNormEpsilon);
        var normalized = Engine.TensorDivide(centered, Engine.TensorBroadcastTo(Engine.TensorSqrt(variance), shape));
        return Engine.TensorAdd(Engine.TensorMultiply(normalized, Engine.TensorBroadcastTo(layer.Gain, shape)), Engine.TensorBroadcastTo(layer.Bias, shape));
    }

    // expand_window(1): each row with its predecessor and successor, zeros past the ends: [N, 3W].
    private Tensor<float> Window(Tensor<float> x)
    {
        int n = x.Shape[0];
        var zero = new Tensor<float>(new[] { 1, _width });
        var previous = n > 1 ? Engine.TensorConcatenate(new[] { zero, Engine.TensorSlice(x, new[] { 0, 0 }, new[] { n - 1, _width }) }, 0) : zero;
        var next = n > 1 ? Engine.TensorConcatenate(new[] { Engine.TensorSlice(x, new[] { 1, 0 }, new[] { n - 1, _width }), zero }, 0) : zero;
        return Engine.TensorConcatenate(new[] { previous, x, next }, 1);
    }

    /// <summary>The tok2vec output <c>[tokens, width]</c> of each token's six attribute values.</summary>
    internal Tensor<float> Encode(IReadOnlyList<ulong[]> features)
    {
        int n = features.Count;
        var columns = new Tensor<float>[_embeddings.Length];
        for (int c = 0; c < _embeddings.Length; c++)
        {
            var e = _embeddings[c];
            int rows = e.Table.Shape[0];
            var index = new Tensor<int>(new[] { n * 4 });
            for (int t = 0; t < n; t++)
            {
                var keys = Hash(features[t][c], e.Seed);
                for (int k = 0; k < 4; k++) index[t * 4 + k] = (int)(keys[k] % (uint)rows);
            }
            var gathered = Engine.Reshape(Engine.TensorIndexSelect(e.Table, index, 0), new[] { n, 4, e.Table.Shape[1] });
            columns[c] = Engine.ReduceSum(gathered, new[] { 1 }, keepDims: false);
        }
        var x = NormForward(_reduce.Norm, MaxoutForward(_reduce.Maxout, Engine.TensorConcatenate(columns, 1)));
        // The encoder runs under with_array(pad=4): the sequence between four zero rows on each side, which pass
        // through the residual blocks too, then the sequence's own rows.
        var padding = new Tensor<float>(new[] { EncoderPadding, _width });
        x = Engine.TensorConcatenate(new[] { padding, x, padding }, 0);
        foreach (var (maxout, norm) in _blocks) x = Engine.TensorAdd(x, NormForward(norm, MaxoutForward(maxout, Window(x))));
        return Engine.TensorSlice(x, new[] { EncoderPadding, 0 }, new[] { n, _width });
    }

    /// <summary>The tagger's label (arg-max of its scores; the first on ties) for each token.</summary>
    public int[] Predict(IReadOnlyList<ulong[]> features)
    {
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<float>();
        var states = Encode(features);
        int n = states.Shape[0], labels = _headBias.Shape[1];
        var scores = Engine.TensorAdd(Engine.TensorMatMul(states, _headWeight), Engine.TensorBroadcastTo(_headBias, new[] { n, labels })).ToArray();
        var result = new int[n];
        for (int t = 0; t < n; t++)
        {
            int best = 0;
            for (int k = 1; k < labels; k++)
                if (scores[t * labels + k] > scores[t * labels + best]) best = k;
            result[t] = best;
        }
        return result;
    }
}
