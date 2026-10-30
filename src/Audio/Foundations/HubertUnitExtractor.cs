using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.Audio.Foundations;

/// <summary>
/// Reads a scikit-learn k-means model saved by joblib (<c>joblib.dump(kmeans, path)</c>): its <c>cluster_centers_</c>,
/// which joblib stores as a <c>NumpyArrayWrapper</c> in the pickle with the array's bytes written right after the
/// wrapper (after a padding byte and padding when the wrapper records <c>numpy_array_alignment_bytes</c>).
/// </summary>
internal static class JoblibKMeans
{
    private static int IndexOf(byte[] data, byte[] pattern, int start)
    {
        for (int i = start; i <= data.Length - pattern.Length; i++)
        {
            int j = 0;
            while (j < pattern.Length && data[i + j] == pattern[j]) j++;
            if (j == pattern.Length) return i;
        }
        return -1;
    }

    // A pickle SHORT_BINUNICODE string: 0x8c, its length, its bytes.
    private static byte[] Key(string name) => new byte[] { 0x8c, (byte)name.Length }.Concat(System.Text.Encoding.ASCII.GetBytes(name)).ToArray();

    /// <summary>The centroids <c>[clusters, features]</c> of a joblib-pickled k-means model.</summary>
    public static double[,] ReadClusterCenters(string path)
    {
        if (!File.Exists(path)) throw new FileNotFoundException($"k-means model not found: {path}", path);
        var data = File.ReadAllBytes(path);
        int at = IndexOf(data, Key("cluster_centers_"), 0);
        if (at < 0) throw new InvalidDataException("The joblib file has no cluster_centers_.");
        int wrapper = IndexOf(data, System.Text.Encoding.ASCII.GetBytes("NumpyArrayWrapper"), at);
        int shapeKey = IndexOf(data, Key("shape"), at);
        int mmapKey = IndexOf(data, Key("allow_mmap"), at);
        if (wrapper < 0 || shapeKey < 0 || mmapKey < 0) throw new InvalidDataException("cluster_centers_ is not a joblib NumpyArrayWrapper.");

        // The shape tuple: integers (K: 1 byte, M: 2, J: 4, little-endian) up to a TUPLE opcode, memo PUTs skipped.
        var shape = new List<int>();
        int p = shapeKey + Key("shape").Length;
        while (p < data.Length)
        {
            byte op = data[p];
            if (op == 0x94) { p++; continue; }                                                              // MEMOIZE
            if (op == (byte)'K') { shape.Add(data[p + 1]); p += 2; }
            else if (op == (byte)'M') { shape.Add(data[p + 1] | (data[p + 2] << 8)); p += 3; }
            else if (op == (byte)'J') { shape.Add(BitConverter.ToInt32(data, p + 1)); p += 5; }
            else if (op is 0x85 or 0x86 or 0x87 or (byte)'t' or (byte)'(') { if (op == (byte)'(') { p++; continue; } break; }
            else throw new InvalidDataException($"Unexpected pickle opcode 0x{op:x2} in the centroids' shape.");
        }
        if (shape.Count != 2) throw new InvalidDataException($"The centroids are {shape.Count}-dimensional; k-means centroids are 2-dimensional.");

        // SETITEMS then BUILD close the wrapper; newer joblib then writes a padding-length byte and the padding.
        int alignmentKey = IndexOf(data, Key("numpy_array_alignment_bytes"), mmapKey);
        int build = IndexOf(data, new[] { (byte)'u', (byte)'b' }, mmapKey);
        if (build < 0) throw new InvalidDataException("The centroids' wrapper does not end.");
        int start = build + 2;
        if (alignmentKey >= 0 && alignmentKey < build) start += 1 + data[start];
        int rows = shape[0], columns = shape[1];

        // The dtype: written out ("f4"/"f8") in the wrapper, or a memo reference to one pickled earlier. Then the element
        // size is the one after whose bytes the pickle resumes (an opcode such as FRAME or a string), within the file.
        int dtype = IndexOf(data, Key("dtype"), at);
        if (IndexOf(data, Key(">"), dtype) is int big && big >= 0 && big < mmapKey) throw new InvalidDataException("Big-endian centroids are not supported.");
        bool Resumes(int elementSize)
        {
            long end = start + (long)rows * columns * elementSize;
            return end == data.Length || (end < data.Length && data[end] is 0x95 or 0x8c or 0x94 or 0x75 or 0x62 or 0x2e or 0x68 or 0x71 or 0x73);
        }
        int size = IndexOf(data, Key("f8"), dtype) is int f8 && f8 >= 0 && f8 < mmapKey ? 8
            : IndexOf(data, Key("f4"), dtype) is int f4 && f4 >= 0 && f4 < mmapKey ? 4
            : Resumes(4) && !Resumes(8) ? 4
            : Resumes(8) && !Resumes(4) ? 8
            : throw new InvalidDataException("The centroids' dtype cannot be told from the file; they must be float32 or float64.");
        if (start + (long)rows * columns * size > data.Length) throw new InvalidDataException("The joblib file ends inside the centroids.");
        var centers = new double[rows, columns];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < columns; c++)
            {
                int offset = start + (r * columns + c) * size;
                centers[r, c] = size == 4 ? BitConverter.ToSingle(data, offset) : BitConverter.ToDouble(data, offset);
            }
        return centers;
    }
}

/// <summary>
/// Discrete speech units from a HuBERT layer and k-means (Lakhotia et al. 2021; fairseq's
/// <c>examples/hubert/simple_kmeans</c> <c>HubertFeatureReader</c> and <c>ApplyKmeans</c>): the waveform's features at one
/// layer, in chunks of at most 1.6 M samples, each frame labelled with its nearest centroid. LLaMA-Omni's and SpeechGPT's
/// targets are mHuBERT's layer 11 in 1000 clusters, consecutive repeats merged.
/// </summary>
/// <remarks>Frozen: the models that use it read units from a released HuBERT and k-means model. The nearest centroid is
/// found in double precision; scikit-learn's float32 distances can differ on a near-tie.</remarks>
internal sealed class HubertUnitExtractor<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private const int MaxChunk = 1600000;

    public HubertUnitExtractor(IEngine engine, List<LayerBase<T>> layers, HubertConfiguration configuration, int layer, int clusters)
    {
        if (layer < 1 || layer > configuration.Layers) throw new ArgumentOutOfRangeException(nameof(layer), $"The network builds {configuration.Layers} layers.");
        Network = new HubertNetwork<T>(engine, layers, configuration);
        Layer = layer;
        Centroids = new double[clusters, configuration.HiddenSize];
    }

    /// <summary>mHuBERT Base (<c>mhubert_base_vp_en_es_fr_it3</c>): group norm on the first convolution, 768 wide,
    /// post-norm layers, built up to <paramref name="layers"/>.</summary>
    public static HubertConfiguration MHubertBase(int layers = 11) => new()
    {
        LayerNormFeatureEncoder = false, HiddenSize = 768, Layers = layers, Heads = 12, IntermediateSize = 3072, LayerNormFirst = false,
    };

    public HubertNetwork<T> Network { get; }

    /// <summary>The layer read (1-based: the output of that many encoder layers).</summary>
    public int Layer { get; }

    /// <summary>The k-means centroids <c>[clusters, hidden]</c>.</summary>
    public double[,] Centroids { get; set; }

    /// <summary>The units of 16 kHz audio, consecutive repeats merged when <paramref name="deduplicate"/>.</summary>
    public int[] Units(IReadOnlyList<double> audio, bool deduplicate)
    {
        if (audio.Count == 0) throw new ArgumentException("The audio is empty.", nameof(audio));
        int clusters = Centroids.GetLength(0), width = Centroids.GetLength(1);
        var norms = new double[clusters];
        for (int k = 0; k < clusters; k++)
            for (int i = 0; i < width; i++) norms[k] += Centroids[k, i] * Centroids[k, i];
        var units = new List<int>();
        using var _ = new NoGradScope<T>();
        for (int start = 0; start < audio.Count; start += MaxChunk)
        {
            int count = Math.Min(MaxChunk, audio.Count - start);
            var chunk = new Tensor<T>(new[] { count });
            for (int i = 0; i < count; i++) chunk[i] = NumOps.FromDouble(audio[start + i]);
            var features = Network.Forward(chunk, Layer)[Layer];                                         // [frames, hidden]
            int frames = features.Shape[0];
            for (int f = 0; f < frames; f++)
            {
                int best = 0;
                double bestDistance = double.PositiveInfinity;
                for (int k = 0; k < clusters; k++)
                {
                    double dot = 0;
                    for (int i = 0; i < width; i++) dot += NumOps.ToDouble(features[f, i]) * Centroids[k, i];
                    double distance = norms[k] - 2 * dot;
                    if (distance < bestDistance) { bestDistance = distance; best = k; }
                }
                if (!deduplicate || units.Count == 0 || units[units.Count - 1] != best) units.Add(best);
            }
        }
        return units.ToArray();
    }

    /// <summary>
    /// Loads a fairseq HuBERT checkpoint's <c>model</c> state (<c>feature_extractor.conv_layers.{i}.0</c>,
    /// <c>post_extract_proj</c>, <c>encoder.pos_conv.0</c>, <c>encoder.layers.{i}.self_attn</c>/<c>fc1</c>/<c>fc2</c>, …)
    /// under the names <see cref="HubertNetwork{T}"/> reads; <paramref name="take"/> returns null for a missing tensor.
    /// </summary>
    public void LoadFairseqWeights(Func<string, double[]?> take)
    {
        string? Fairseq(string name)
        {
            if (name.StartsWith("hubert.", StringComparison.Ordinal) || name.Contains(".parametrizations.")) return null;
            var m = System.Text.RegularExpressions.Regex.Match(name, @"^feature_extractor\.conv_layers\.(\d+)\.conv\.weight$");
            if (m.Success) return $"feature_extractor.conv_layers.{m.Groups[1].Value}.0.weight";
            m = System.Text.RegularExpressions.Regex.Match(name, @"^feature_extractor\.conv_layers\.(\d+)\.layer_norm\.(weight|bias)$");
            if (m.Success) return $"feature_extractor.conv_layers.{m.Groups[1].Value}.2.{m.Groups[2].Value}";
            if (name.StartsWith("feature_projection.layer_norm.", StringComparison.Ordinal)) return "layer_norm." + name.Substring("feature_projection.layer_norm.".Length);
            if (name.StartsWith("feature_projection.projection.", StringComparison.Ordinal)) return "post_extract_proj." + name.Substring("feature_projection.projection.".Length);
            if (name.StartsWith("encoder.pos_conv_embed.conv.", StringComparison.Ordinal)) return "encoder.pos_conv.0." + name.Substring("encoder.pos_conv_embed.conv.".Length);
            m = System.Text.RegularExpressions.Regex.Match(name, @"^encoder\.layers\.(\d+)\.(.+)$");
            if (!m.Success) return name;
            string rest = m.Groups[2].Value;
            rest = rest.StartsWith("attention.", StringComparison.Ordinal) ? "self_attn." + rest.Substring("attention.".Length)
                : rest.StartsWith("layer_norm.", StringComparison.Ordinal) ? "self_attn_layer_norm." + rest.Substring("layer_norm.".Length)
                : rest.StartsWith("feed_forward.intermediate_dense.", StringComparison.Ordinal) ? "fc1." + rest.Substring("feed_forward.intermediate_dense.".Length)
                : rest.StartsWith("feed_forward.output_dense.", StringComparison.Ordinal) ? "fc2." + rest.Substring("feed_forward.output_dense.".Length)
                : rest;
            return $"encoder.layers.{m.Groups[1].Value}.{rest}";
        }
        Network.LoadTorchWeights(name => Fairseq(name) is { } mapped ? take(mapped) : null);
    }
}
