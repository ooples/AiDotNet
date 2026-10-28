namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Dia<T>
{
    /// <summary>
    /// Loads Dia's weights (<c>nari-labs/Dia-1.6B-0626</c>): the release's <c>dia-v1.pth</c> (<c>DiaModel</c>'s names and
    /// <c>DenseGeneral</c> kernels), transformers' <c>model.safetensors</c> (one file, or the shards its
    /// <c>model.safetensors.index.json</c> lists), or a directory holding either.
    /// </summary>
    public void LoadReferenceWeights(string path)
    {
        ThrowIfDisposed();
        if (!HasPaperLayers) throw NotNative();
        if (path is null) throw new ArgumentNullException(nameof(path));
        var files = new List<string>();
        if (Directory.Exists(path))
        {
            string index = Path.Combine(path, "model.safetensors.index.json");
            if (File.Exists(index))
            {
                using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(index));
                files.AddRange(document.RootElement.GetProperty("weight_map").EnumerateObject().Select(p => p.Value.GetString()!).Distinct()
                    .Select(f => Path.Combine(path, f)));
            }
            else
            {
                var file = new[] { "model.safetensors", "dia-v1.pth" }.Select(f => Path.Combine(path, f)).FirstOrDefault(File.Exists)
                    ?? throw new FileNotFoundException($"No model.safetensors, model.safetensors.index.json or dia-v1.pth in {path}.");
                files.Add(file);
            }
        }
        else files.Add(path);
        var missing = files.FirstOrDefault(f => !File.Exists(f));
        if (missing is not null) throw new FileNotFoundException($"The checkpoint file {missing} does not exist.", missing);

        var state = new Dictionary<string, Tensor<float>>(StringComparer.Ordinal);
        foreach (var file in files)
            foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(file)) state[entry.Key] = entry.Value;
        // Each tensor is widened when it is read rather than the whole checkpoint at once.
        double[]? Read(string name) => state.TryGetValue(name, out var v) ? Array.ConvertAll(v.ToArray(), x => (double)x) : null;
        var used = state.ContainsKey("logits_dense.weight")
            ? Network.LoadTorchWeights((name, shape) => Need(Read, name, shape))
            : LoadReleaseWeights(Read);
        var unused = state.Keys.Where(k => !used.Contains(k)).ToList();
        if (unused.Count > 0) throw new InvalidDataException($"The checkpoint has {unused.Count} tensors this model has no place for (first: '{unused[0]}').");
    }

    private static double[] Need(Func<string, double[]?> read, string name, int[] shape)
    {
        var values = read(name) ?? throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
        int count = shape.Aggregate(1, (a, s) => a * s);
        if (values.Length != count) throw new InvalidDataException($"'{name}' has {values.Length} values; this model expects {count}.");
        return values;
    }

    /// <summary>
    /// Loads a state dict in the release's names: each <c>DenseGeneral</c> kernel <c>[in…, out…]</c> becomes the
    /// <c>[out, in]</c> weight it multiplies by (the fused SwiGLU kernel <c>[in, 2, intermediate]</c> its gate and up
    /// projections), the channels' tables are stacked and the head <c>[width, channels, vocabulary]</c> transposed. Returns
    /// the names read.
    /// </summary>
    internal HashSet<string> LoadReleaseWeights(Func<string, double[]?> read)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        var c = Network.Configuration;
        // A kernel [rows, columns] (its leading axes flattened to rows) as [columns, rows].
        static double[] Transposed(double[] kernel, int rows, int columns)
        {
            var result = new double[kernel.Length];
            for (int r = 0; r < rows; r++)
                for (int k = 0; k < columns; k++) result[k * rows + r] = kernel[r * columns + k];
            return result;
        }
        double[] Release(string name, int count)
        {
            used.Add(name);
            return Need(read, name, new[] { count });
        }
        double[] Port(string name, int[] shape)
        {
            int count = shape.Aggregate(1, (a, s) => a * s);
            if (name == "logits_dense.weight") return Transposed(Release("decoder.logits_dense.weight", count), c.DecoderHidden, c.Channels * c.AudioVocab);
            if (name == "model.decoder.embeddings.embed.weight")
                return Enumerable.Range(0, c.Channels).SelectMany(k => Release($"decoder.embeddings.{k}.weight", c.AudioVocab * c.DecoderHidden)).ToArray();
            string release = name.Substring("model.".Length).Replace("mlp.gate_up_proj", "mlp.wi_fused").Replace("mlp.down_proj", "mlp.wo");
            // transformers' weights are [out, in]; the release's kernels are [in…, out…].
            bool kernel = name.EndsWith("proj.weight", StringComparison.Ordinal);
            var values = Release(release, count);
            return kernel ? Transposed(values, shape[1], shape[0]) : values;
        }
        Network.LoadTorchWeights(Port);
        return used;
    }
}
