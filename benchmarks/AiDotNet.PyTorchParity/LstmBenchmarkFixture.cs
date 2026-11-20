using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.ModelLoading;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PyTorchParity;
using AiDotNet.Tensors.LinearAlgebra;

internal sealed partial class AiDotNetLstmModel
{
    private LstmBenchmarkFixture? _fixture;
    internal NeuralNetworkBase<float> Classifier => Network;

    internal LstmBenchmarkFixture AttachFixture(BenchmarkOptions options)
        => _fixture = new LstmBenchmarkFixture(options, Network, PyTorchModulePrefixes);

    public override void LoadSyntheticBatch(int batchSize)
    {
        if (_fixture is null) base.LoadSyntheticBatch(batchSize);
        else CurrentBatch = _fixture.Next(batchSize);
    }

    public override void Backward()
    {
        base.Backward();
        _fixture?.Trained(CurrentBatch);
    }
}

/// <summary>Loads, binds and observes the existing classifier and its ordinary training and inference entry points.</summary>
internal sealed class LstmBenchmarkFixture
{
    private readonly record struct Raw(int[] Shape, float[] Values);
    private const string NumericalSchema = "lstm-classification-numerical-v1";
    private readonly BenchmarkOptions _options;
    private readonly string _receipt;
    private readonly JsonObject _manifest;
    private readonly string _manifestHash;
    private readonly string _initialHash;
    private readonly string _inputsHash;
    private readonly Dictionary<string, object> _initial;
    private readonly (Tensor<float> Input, Tensor<float> Label)[] _training = new (Tensor<float>, Tensor<float>)[20];
    private readonly Dictionary<int, Tensor<float>> _inference = [];
    private readonly List<int> _updates = new(60);
    private readonly List<int> _inferred = new(3);
    private int _loaded;

    internal LstmBenchmarkFixture(BenchmarkOptions options, NeuralNetworkBase<float> network, IReadOnlyList<string> prefixes)
    {
        _options = options;
        if (network.MaxGradNormValue != 1.0) throw new InvalidDataException("Unexpected model gradient clip norm.");
        string root = Path.GetFullPath(options.LstmFixture
            ?? throw new InvalidOperationException("An LSTM fixture directory is required."));
        string path = Path.Combine(root, "manifest.json");
        _receipt = Path.GetFullPath(options.LstmWorkloadReceipt ?? (options.LstmNumerics is null
            ? options.OutputPath + ".workload.json" : Path.Combine(options.LstmNumerics, "workload.json")));
        if (File.Exists(_receipt)) throw new IOException("Preserve prior workload receipt: " + _receipt);
        _manifest = (JsonNode.Parse(File.ReadAllBytes(path))
            ?? throw new InvalidDataException("The LSTM fixture manifest is empty.")).AsObject();
        var protocol = Req(_manifest["protocol"], "protocol");
        if (Req(_manifest["schema"], "schema").GetValue<string>() != "lstm-classification-shared-v1" || Req(_manifest["model"], "model").GetValue<string>() != "lstm"
            || options.Epochs != 3 || options.TrainBatches != 20 || options.BatchSize != 64 || options.Seed != 1234
            || !options.InferenceBatchSizes.SequenceEqual(new[] { 1, 8, 16 })
            || Req(protocol["epochs"], "epochs").GetValue<int>() != options.Epochs || Req(protocol["train_batches"], "train_batches").GetValue<int>() != options.TrainBatches
            || Req(protocol["batch_size"], "batch_size").GetValue<int>() != options.BatchSize || Req(protocol["seed"], "seed").GetValue<int>() != options.Seed
            || !Parsed<int[]>(Req(protocol["inference_batches"], "inference_batches")).SequenceEqual(options.InferenceBatchSizes))
            throw new InvalidDataException("Unexpected shared LSTM protocol.");
        var optimizer = Req(_manifest["optimizer"], "optimizer");
        if (Req(optimizer["name"], "name").GetValue<string>() != "AdamW" || Req(optimizer["lr"], "lr").GetValue<double>() != 0.001
            || Req(optimizer["eps"], "eps").GetValue<double>() != 1e-8 || Req(optimizer["weight_decay"], "weight_decay").GetValue<double>() != 0.01
            || Req(optimizer["max_grad_norm"], "max_grad_norm").GetValue<double>() != 1.0 || !Parsed<double[]>(Req(optimizer["betas"], "betas")).SequenceEqual(new[] { 0.9, 0.999 }))
            throw new InvalidDataException("Unexpected shared LSTM optimizer.");
        _manifestHash = HashFile(path);
        string initialFile = Path.Combine(root, "initial.safetensors");
        string inputsFile = Path.Combine(root, "inputs.safetensors");
        _initialHash = HashFile(initialFile);
        _inputsHash = HashFile(inputsFile);
        if (_initialHash != (string?)_manifest["initial_state_sha256"] || _inputsHash != (string?)_manifest["inputs_sha256"])
            throw new InvalidDataException("Fixture file hash mismatch.");
        var state = SafetensorsReader.Read(File.ReadAllBytes(initialFile));
        if (state.Names.Count != 6 || Read(state, "lstm.bias_hh_l0", [256]).Any(v => v != 0))
            throw new InvalidDataException("Expected one fixed zero secondary LSTM bias.");
        PyTorchStateDictImporter.Import(network, state, prefixes);
        _initial = Fields(network).ToDictionary(k => k.Key, k => Describe(k.Value));
        Verify("initial", _initial);
        using var stream = File.OpenRead(inputsFile);
        var inputs = SafetensorsReader.Read(stream);
        using var source = TorchCheckpointFile.Open(inputsFile, strictAxes: true);
        if (inputs.Names.Count != 43) throw new InvalidDataException("Expected exactly 43 input tensors.");
        Tensor<float> Input(string name, int[] shape)
        {
            if (inputs.Get(name)?.DataType != "F32") throw new InvalidDataException("Expected F32 " + name);
            var tensor = new Tensor<float>(shape);
            source.CopyTo(name, shape, tensor);
            return tensor;
        }
        for (int i = 0; i < _training.Length; i++)
        {
            float[] classes = Read(inputs, $"train/{i:D2}/labels", [64]);
            float[] oneHot = new float[64 * 10];
            for (int row = 0; row < classes.Length; row++)
            {
                int index = (int)classes[row];
                if (classes[row] != index || index < 0 || index >= 10) throw new InvalidDataException("Invalid class index.");
                oneHot[row * 10 + index] = 1f;
            }
            _training[i] = (Input($"train/{i:D2}/input", [64, 32, 32]),
                new Tensor<float>(oneHot, [64, 10]));
        }
        foreach (int batch in options.InferenceBatchSizes)
            _inference.Add(batch, Input($"inference/{batch}/input", [batch, 32, 32]));
        Verify("inputs", Inputs());
    }

    private static float[] Read(SafetensorsFile file, string name, int[] shape)
    {
        var info = file.Get(name);
        if (info?.DataType != "F32" || !info.Shape.SequenceEqual(shape.Select(d => (long)d)))
            throw new InvalidDataException("Missing or invalid tensor: " + name);
        var values = MemoryMarshal.Cast<byte, float>(file.GetRawBytes(name)).ToArray();
        if (values.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite " + name);
        return values;
    }

    internal (Tensor<float> Input, Tensor<float> Label) Next(int batch)
    {
        if (batch == 64) return _training[_loaded++];
        var value = _inference[batch];
        _inferred.Add(batch);
        return (value, Tensor<float>.Empty());
    }

    internal void Trained((Tensor<float> Input, Tensor<float> Label) actual)
    {
        int index = _updates.Count % _training.Length;
        if (!ReferenceEquals(actual.Input, _training[index].Input) || !ReferenceEquals(actual.Label, _training[index].Label))
            throw new InvalidDataException("Unexpected training input or label after update.");
        _updates.Add(index);
    }

    private static object Describe(Raw value) => Describe(value.Shape, value.Values);

    private static object Describe(int[] shape, ReadOnlySpan<float> values) => new { dtype = "F32", shape,
        sha256 = Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values))).ToLowerInvariant() };

    private static object DescribeInput(Tensor<float> tensor)
    {
        var result = Describe(tensor.Shape.ToArray(), ParityTensorStorage.Read(tensor));
        GC.KeepAlive(tensor);
        return result;
    }

    private static Raw Snapshot(Tensor<float> t) => new(t.Shape.ToArray(), t.ToVector().ToArray());

    private Dictionary<string, object> Inputs()
    {
        var result = new Dictionary<string, object>();
        for (int i = 0; i < _training.Length; i++)
        {
            result.Add($"train/{i:D2}/input", DescribeInput(_training[i].Input));
            ReadOnlySpan<float> oneHot = ParityTensorStorage.Read(_training[i].Label);
            var indices = new float[64];
            for (int row = 0; row < indices.Length; row++)
            {
                int active = -1;
                for (int col = 0; col < 10; col++)
                {
                    float value = oneHot[row * 10 + col];
                    if (value == 0f) continue;
                    if (value != 1f || active >= 0) throw new InvalidDataException("Invalid one-hot labels.");
                    active = col;
                }
                if (active < 0) throw new InvalidDataException("Missing class.");
                indices[row] = active;
            }
            result.Add($"train/{i:D2}/labels", Describe(new Raw([64], indices)));
            GC.KeepAlive(_training[i].Label);
        }
        foreach (var (b, x) in _inference) result.Add($"inference/{b}/input", DescribeInput(x));
        return result;
    }

    private void Verify(string name, Dictionary<string, object> fields)
    {
        if (!JsonNode.DeepEquals(_manifest[name], JsonSerializer.SerializeToNode(fields)))
            throw new InvalidDataException("Observed LSTM " + name + " differs from the fixture.");
    }

    private static Dictionary<string, Raw> Fields(NeuralNetworkBase<float> net, float[]? gradients = null)
    {
        var layout = new List<(Tensor<float>? Tensor, int Length, bool IsBuffer)>();
        foreach (var layer in net.Layers.OfType<LayerBase<float>>()) layer.AppendFlatParameterLayout(layout);
        var offsets = new Dictionary<Tensor<float>, int>(ReferenceEqualityComparer.Instance);
        int offset = 0;
        foreach (var entry in layout)
        {
            if (entry.Tensor is not null && !entry.IsBuffer) offsets.Add(entry.Tensor, offset);
            offset += entry.Length;
        }
        if (offset != 25482 || gradients is not null && gradients.Length != offset)
            throw new InvalidDataException("Unexpected LSTM parameter layout.");
        var result = new Dictionary<string, Raw>();
        void Field(string name, Tensor<float> tensor, bool transpose = false)
        {
            int at = offsets[tensor];
            var shape = tensor.Shape.ToArray();
            var values = gradients is null ? tensor.ToVector().ToArray() : gradients.AsSpan(at, tensor.Length).ToArray();
            if (transpose)
            {
                var reordered = new float[values.Length];
                for (int r = 0; r < shape[0]; r++)
                    for (int c = 0; c < shape[1]; c++) reordered[c * shape[0] + r] = values[r * shape[1] + c];
                values = reordered;
                shape = [shape[1], shape[0]];
            }
            result.Add(name, new Raw(shape, values));
        }
        var lstm = net.Layers.OfType<LSTMLayer<float>>().Single();
        void Gate(string name, Tensor<float> input, Tensor<float> recurrent, Tensor<float> bias)
        {
            Field($"lstm/{name}/weight_ih", input);
            Field($"lstm/{name}/weight_hh", recurrent);
            Field($"lstm/{name}/bias", bias);
        }
        Gate("i", lstm.WeightsIi, lstm.WeightsIh, lstm.BiasI);
        Gate("f", lstm.WeightsFi, lstm.WeightsFh, lstm.BiasF);
        Gate("g", lstm.WeightsCi, lstm.WeightsCh, lstm.BiasC);
        Gate("o", lstm.WeightsOi, lstm.WeightsOh, lstm.BiasO);
        var head = net.Layers.OfType<DenseLayer<float>>().Single().GetTrainableParameters();
        Field("head/weight", head[0], transpose: true);
        Field("head/bias", head[1]);
        return result;
    }

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private Dictionary<string, object> Receipt(string schema) => new()
    {
        ["schema"] = schema, ["model"] = "lstm", ["framework"] = "aidotnet", ["device"] = BenchStats.DeviceName,
        ["pid"] = Environment.ProcessId, ["fixture_manifest_sha256"] = _manifestHash, ["initial_state_sha256"] = _initialHash,
        ["inputs_sha256"] = _inputsHash, ["source_sha256"] = new Dictionary<string, string>
        {
            ["AiDotNet.PyTorchParity"] = HashFile(typeof(LstmBenchmarkFixture).Assembly.Location),
            ["AiDotNet"] = HashFile(typeof(NeuralNetworkBase<float>).Assembly.Location),
            ["Tensors"] = HashFile(typeof(Tensor<float>).Assembly.Location)
        }
    };

    internal void Complete()
    {
        bool numeric = _options.LstmNumerics is not null;
        int count = numeric ? 3 : 60;
        if (_updates.Count != count || !_inferred.SequenceEqual(_options.InferenceBatchSizes))
            throw new InvalidDataException("Incomplete LSTM workload.");
        var inputs = Inputs();
        Verify("inputs", inputs);
        var receipt = Receipt("lstm-classification-consumption-v1");
        receipt["mode"] = numeric ? "numeric" : "timing";
        receipt["train_order"] = _updates;
        receipt["inference_batches"] = _inferred;
        receipt["inputs"] = inputs;
        receipt["initial"] = _initial;
        WriteJson(_receipt, receipt);
    }

    private static JsonNode Req(JsonNode? node, string name) => node
        ?? throw new InvalidDataException("The LSTM fixture lacks '" + name + "'.");

    private static TValue Parsed<TValue>(JsonNode node) => node.Deserialize<TValue>()
        ?? throw new InvalidDataException("An LSTM fixture value is null.");

    internal void WriteNumerics(AiDotNetLstmModel model)
    {
        var directory = Directory.CreateDirectory(_options.LstmNumerics
            ?? throw new InvalidOperationException("Numerical receipts require a numerics directory."));
        // The numerical receipt names workload.json by basename, so the two must be one directory. Fold case only
        // where the filesystem does: /tmp/Run and /tmp/run are different directories elsewhere.
        var pathComparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        if (!string.Equals(Path.GetFullPath(directory.FullName), Path.GetDirectoryName(_receipt), pathComparison))
            throw new ArgumentException("Numeric receipts must share one directory.");
        string observations = Path.Combine(directory.FullName, "observations.safetensors");
        string receiptPath = Path.Combine(directory.FullName, "numeric-receipt.json");
        if (File.Exists(observations) || File.Exists(receiptPath)) throw new IOException("Preserve existing numerical evidence.");
        var raw = new Dictionary<string, Raw>();
        var preClipNorms = new List<double>(3);
        var net = model.Classifier;
        for (int step = 1; step <= 3; step++)
        {
            model.LoadSyntheticBatch(64);
            string prefix = $"train/step{step}/";
            raw.Add(prefix + "logits", Snapshot(net.Predict(model.CurrentBatch.Input)));
            model.Backward();
            raw.Add(prefix + "loss", new Raw([], [net.GetLastLoss()]));
            // The public gradient surface precedes clipping. On this frozen trajectory the clip is
            // inactive, making the published values exactly the values AdamW applied.
            float[] gradients = net.GetParameterGradients().ToArray();
            double norm = Math.Sqrt(gradients.Sum(v => (double)v * v));
            if (!double.IsFinite(norm) || norm + 1e-6 >= 1.0)
                throw new InvalidDataException("The canonical LSTM numerical path requires a finite pre-clip gradient norm below the "
                    + "clipping threshold of 1: the exported gradients are the applied ones only while the clip is inactive.");
            preClipNorms.Add(norm);
            foreach (var (name, v) in Fields(net, gradients)) raw.Add(prefix + "gradient/" + name, v);
            foreach (var (name, v) in Fields(net)) raw.Add(prefix + "state/" + name, v);
        }
        foreach (int batch in _options.InferenceBatchSizes)
        {
            model.LoadSyntheticBatch(batch);
            raw.Add($"inference/b{batch}/logits", Snapshot(net.Predict(model.CurrentBatch.Input)));
        }
        Complete();
        WriteTensors(observations, raw);
        var receipt = Receipt(NumericalSchema);
        receipt["completed_updates"] = 3;
        receipt["gradient_definition"] = "post_global_norm_clip_1";
        receipt["pre_clip_global_norms"] = preClipNorms;
        receipt["observations"] = new { file = Path.GetFileName(observations), sha256 = HashFile(observations) };
        receipt["workload"] = new { file = Path.GetFileName(_receipt), sha256 = HashFile(_receipt) };
        WriteJson(receiptPath, receipt);
    }

    private static void WriteTensors(string path, Dictionary<string, Raw> tensors)
    {
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, value) in tensors)
        {
            long end = offset + sizeof(float) * (long)value.Values.Length;
            header[name] = new { dtype = "F32", shape = value.Shape, data_offsets = new[] { offset, end } };
            offset = end;
        }
        var json = JsonSerializer.SerializeToUtf8Bytes(header);
        int padded = (json.Length + 7) & ~7;
        using var file = new BinaryWriter(new FileStream(path, FileMode.CreateNew));
        file.Write((long)padded);
        file.Write(json);
        for (int i = json.Length; i < padded; i++) file.Write((byte)' ');
        foreach (var value in tensors.Values) file.Write(MemoryMarshal.AsBytes(value.Values.AsSpan()));
    }

    private static void WriteJson(string path, Dictionary<string, object> receipt)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)
            ?? throw new InvalidOperationException("The receipt path has no directory."));
        using var stream = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(stream, receipt);
    }
}
