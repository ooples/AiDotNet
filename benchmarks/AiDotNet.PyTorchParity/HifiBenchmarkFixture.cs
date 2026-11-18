using System.Buffers.Binary;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.ModelLoading;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.Vocoders;
using Newtonsoft.Json.Linq;

namespace AiDotNet.PyTorchParity;

// Explicit benchmark sampling/state input control. The production model, objectives and optimizers are unchanged.
internal sealed class HifiBenchmarkFixture
{
    // Existing read-only storage access: no payload clone, version write or escape mark. The owner stays alive
    // through each SHA operation; no span is stored in the fixture or returned to a benchmark caller.
    private delegate ReadOnlySpan<float> TensorReadSpan(TensorBase<float> owner);
    private static readonly TensorReadSpan s_readSpan = (typeof(TensorBase<float>)
        .GetMethod("AsSpanUnmarked", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new MissingMethodException("The validated read-only tensor storage contract changed."))
        .CreateDelegate<TensorReadSpan>();

    private readonly string _directory, _receiptPath, _manifestHash;
    private readonly JObject _manifest;
    private readonly (Tensor<float> Input, Tensor<float> Label)[] _pool = new (Tensor<float>, Tensor<float>)[3];
    private readonly Dictionary<int, Tensor<float>> _inference = new();
    private readonly Dictionary<string, Tensor<float>> _inputs = new(StringComparer.Ordinal);
    private readonly JObject _actualInputs = new();
    private readonly JArray _actualSteps = new();
    private readonly FixtureCropRandom _crop;
    private int _dataLoads;
    private JObject? _preflight;

    internal HifiBenchmarkFixture(string directory, int epochs, int batches, int batchSize, int seed,
        int[] inferenceBatches, string receiptPath)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("The fixture requires little-endian FP32.");
        _directory = Path.GetFullPath(directory);
        _receiptPath = Path.GetFullPath(receiptPath);
        if (File.Exists(_receiptPath)) throw new IOException("Preserve prior workload identity evidence.");
        string manifestPath = Path.Combine(_directory, "manifest.json");
        _manifest = JObject.Parse(File.ReadAllText(manifestPath));
        _manifestHash = HashFile(manifestPath);
        var protocol = JObject.FromObject(new { epochs, train_batches = batches, batch_size = batchSize, seed,
            inference_batches = inferenceBatches, utterance_frames = 256, segment_frames = 32, hop = 256, mel_channels = 80 });
        if ((string?)_manifest["schema"] != "hifigan-shared-workload-v1" || !JToken.DeepEquals(_manifest["protocol"], protocol)
            || epochs != 3 || batches != 3 || batchSize != 1 || !inferenceBatches.SequenceEqual(new[] { 1, 8, 16 }))
            throw new InvalidDataException("Fixture protocol differs from the actual3x3/b1,b8,b16 invocation.");
        if ((long?)_manifest["learned_parameters"] != 84_660_721 || (int?)_manifest["state_tensor_count"] != 404)
            throw new InvalidDataException("Full V1 shared404 active fields are required.");
        RequireKeys(Object(_manifest, "files_sha256"), new[] { "initial.safetensors", "inputs.safetensors" });
        ValidateFiles();
        ValidateTensorMetadata(Path.Combine(_directory, "initial.safetensors"), Object(_manifest, "state"));
        ValidateTensorMetadata(Path.Combine(_directory, "inputs.safetensors"), Object(_manifest, "inputs"));
        var steps = _manifest["training_steps"] as JArray ?? throw new InvalidDataException("Missing nine training steps.");
        if (steps.Count != 9) throw new InvalidDataException("Exactly nine training records required.");
        var starts = new int[9];
        for (int index = 0; index < 9; index++)
        {
            var step = steps[index];
            if ((int?)step["index"] != index || (int?)step["pool_index"] != index % 3 || (int?)step["epoch"] != index / 3)
                throw new InvalidDataException("Actual pool/epoch order differs.");
            starts[index] = (int?)step["crop_start"] ?? -1;
            if (starts[index] < 0 || starts[index] > 224) throw new InvalidDataException("Crop is outside the full utterance.");
        }
        _crop = new FixtureCropRandom(starts);
        var requiredInputs = Enumerable.Range(0, 3).SelectMany(index => new[] { $"train/{index}/mel", $"train/{index}/audio" })
            .Concat(new[] { "inference/1/mel", "inference/8/mel", "inference/16/mel" }).ToArray();
        RequireKeys(Object(_manifest, "inputs"), requiredInputs);
        using (var source = TorchCheckpointFile.Open(Path.Combine(_directory, "inputs.safetensors"), strictAxes: true))
        {
            if (!new HashSet<string>(requiredInputs, StringComparer.Ordinal).SetEquals(source.Names))
                throw new InvalidDataException("Missing/extra actual input payloads.");
            foreach (string name in requiredInputs)
            {
                int[] shape = source.ShapeOf(name)?.Select(value => checked((int)value)).ToArray()
                    ?? throw new InvalidDataException("Missing shape: " + name);
                var value = new Tensor<float>(shape);
                source.CopyTo(name, shape, value);
                _actualInputs[name] = CheckDescriptor(name, value, Object(_manifest, "inputs")[name]!);
                _inputs.Add(name, value);
            }
        }
        for (int index = 0; index < 3; index++)
        {
            _pool[index] = (_inputs[$"train/{index}/mel"], _inputs[$"train/{index}/audio"]);
            RequireShape(_pool[index].Input, new[] { 1, 80, 256 });
            RequireShape(_pool[index].Label, new[] { 1, 65536 });
        }
        foreach (int batch in new[] { 1, 8, 16 })
        {
            var value = _inputs[$"inference/{batch}/mel"];
            RequireShape(value, new[] { batch, 80, 256 });
            _inference.Add(batch, value);
        }
        foreach (var step in steps)
        {
            int pool = (int)step["pool_index"]!, start = (int)step["crop_start"]!;
            ReadOnlySpan<float> mel = s_readSpan(_pool[pool].Input), audio = s_readSpan(_pool[pool].Label);
            var melDescription = DescribeMelCrop(mel, start);
            var audioDescription = Describe(audio.Slice(start * 256, 8192), new[] { 1, 8192 });
            GC.KeepAlive(_pool[pool].Input);
            GC.KeepAlive(_pool[pool].Label);
            if (!JToken.DeepEquals(step["mel"], Object(_manifest, "inputs")[$"train/{pool}/mel"])
                || !JToken.DeepEquals(step["audio"], Object(_manifest, "inputs")[$"train/{pool}/audio"])
                || !JToken.DeepEquals(step["cropped_mel"], melDescription) || !JToken.DeepEquals(step["cropped_audio"], audioDescription))
                throw new InvalidDataException("Actual loaded cropped inputs differ.");
            var actualStep = (JObject)step.DeepClone();
            actualStep["cropped_mel"] = melDescription;
            actualStep["cropped_audio"] = audioDescription;
            _actualSteps.Add(actualStep);
        }
    }

    internal (Tensor<float> Input, Tensor<float> Label) WarmupBatch => (_inference[1], Tensor<float>.Empty());

    internal void ImportState(HiFiGAN<float> model)
        => HifiGanReferenceVerifier.ImportSharedBenchmarkState(model, Path.Combine(_directory, "initial.safetensors"));

    internal void RestoreAfterWarmupAndEmit(HiFiGAN<float> model)
    {
        ImportState(model);
        var fields = HifiGanReferenceVerifier.SharedBenchmarkState(model).OrderBy(field => field.Name, StringComparer.Ordinal).ToArray();
        if (fields.Length != 404 || model.TrainableParameterCount != 84_660_721) throw new InvalidDataException("Full imported state required.");
        RequireKeys(Object(_manifest, "state"), fields.Select(field => field.Name));
        var actual = new JObject();
        using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        foreach (var field in fields)
        {
            long version = field.Value.Version;
            ReadOnlySpan<float> values = s_readSpan(field.Value);
            int[] shape = field.Value.Shape.ToArray();
            var entry = Describe(values, shape);
            if (!JToken.DeepEquals(entry, Object(_manifest, "state")[field.Name]))
                throw new InvalidDataException("Actual imported404 field differs: " + field.Name);
            actual[field.Name] = entry;
            digest.AppendData(Encoding.UTF8.GetBytes(field.Name + "|" + string.Join(",", shape) + "|"));
            digest.AppendData(MemoryMarshal.AsBytes(values));
            GC.KeepAlive(field.Value);
            if (field.Value.Version != version) throw new InvalidDataException("The state identity observer mutated its source.");
        }
        string aggregate = Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
        if (aggregate != (string?)_manifest["state_sha256"]) throw new InvalidDataException("Imported404 aggregate differs.");
        // An instance Random field is the production crop source. Only its benchmark sampling stream is replaced;
        // AlignedSegment and the actual discriminator/generator Train path continue to run unchanged.
        var random = typeof(GanVocoderBase<float>).GetField("_trainingRandom", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException("The validated crop sampling contract changed.");
        if (random.FieldType != typeof(Random)) throw new InvalidDataException("The crop source contract changed.");
        random.SetValue(model, _crop);
        _preflight = new JObject {
            ["schema"]="hifigan-shared-workload-consumption-v1",["framework"]="aidotnet",["pass"]=false,
            ["pid"]=Environment.ProcessId,["state_tensor_count"]=fields.Length,
            ["preflight_pass"]=true,["fixture_manifest_sha256"]=_manifestHash,["state_sha256"]=aggregate,
            ["state"]=actual,["inputs"]=_actualInputs.DeepClone(),["training_steps"]=_actualSteps.DeepClone(),
            ["consumed_steps"]=new JArray(),["inference_batches_consumed"]=new JArray(),["state_checkpoint_retained"]=false,
            ["hash_observer"]=new JObject { ["cpu_payload_clone"]=false,["source_version_unchanged"]=true,
                ["gpu_readback"]="existing canonical tensor host storage; no separate observer payload buffer" },
        };
        SaveReceipt();
    }

    internal (Tensor<float> Input, Tensor<float> Label) NextData(int batchSize)
    {
        int load = _dataLoads++;
        if (load < 3)
        {
            if (batchSize != 1) throw new InvalidDataException("Unexpected actual training batch size.");
            return _pool[load];
        }
        int inference = load - 3;
        if (inference >= 3 || new[] { 1, 8, 16 }[inference] != batchSize)
            throw new InvalidDataException("Unexpected actual inference data load/order.");
        return (_inference[batchSize], Tensor<float>.Empty());
    }

    internal void ValidateActualTrainingBatch(Tensor<float> input, Tensor<float> label)
    {
        int step = _crop.Consumed;
        if (step >= 9 || !ReferenceEquals(input, _pool[step % 3].Input) || !ReferenceEquals(label, _pool[step % 3].Label))
            throw new InvalidDataException("Actual training received a different prevalidated batch.");
    }

    internal void Complete()
    {
        if (_preflight is null || _dataLoads != 6 || _crop.Consumed != 9)
            throw new InvalidDataException("The actual benchmark did not consume the complete canonical workload.");
        foreach (var pair in _inputs) CheckDescriptor(pair.Key, pair.Value, Object(_manifest, "inputs")[pair.Key]!);
        ValidateFiles();
        _preflight["pass"] = true;
        _preflight["consumed_steps"] = new JArray(((JArray)_manifest["training_steps"]!).Select(step => new JObject {
            ["index"]=(int)step["index"]!,["pool_index"]=(int)step["pool_index"]!,["crop_start"]=(int)step["crop_start"]!,
        }));
        _preflight["inference_batches_consumed"] = new JArray(1, 8, 16);
        SaveReceipt();
    }

    internal static void ValidateMatchedReference(string? fixtureDirectory, string referenceDirectory,
        JObject referenceManifest, TorchCheckpointFile reference)
    {
        string? requiredHash = (string?)referenceManifest["workload_manifest_sha256"];
        if (requiredHash is null)
        {
            if (fixtureDirectory is not null) throw new InvalidDataException("An old standalone reference does not prove this shared workload.");
            return;
        }
        if (fixtureDirectory is null) throw new InvalidDataException("Matched references require the explicit shared fixture path.");
        string manifestPath = Path.Combine(fixtureDirectory, "manifest.json");
        if (HashFile(manifestPath) != requiredHash) throw new InvalidDataException("Reference binds a different workload manifest.");
        var fixture = JObject.Parse(File.ReadAllText(manifestPath));
        if ((string?)fixture["schema"] != "hifigan-shared-workload-v1"
            || (string?)fixture["state_sha256"] != (string?)referenceManifest["workload_initial_state_sha256"])
            throw new InvalidDataException("Reference canonical initial state differs.");
        foreach (var entry in Object(fixture, "files_sha256").Properties())
            if (HashFile(Path.Combine(fixtureDirectory, entry.Name)) != (string?)entry.Value)
                throw new InvalidDataException("Reference fixture file changed.");
        if (HashFile(Path.Combine(referenceDirectory, "initial.safetensors")) != (string?)fixture["files_sha256"]!["initial.safetensors"])
            throw new InvalidDataException("Actual numerical reference initial404 state differs from the benchmark.");
        var recipes = Object(referenceManifest, "recipe_artifacts_sha256");
        RequireKeys(recipes, new[] { "benchmark.py", "benchmark_telemetry.py", "export_reference.py", "export_hifigan_reference.py", "hifigan_fixture.py", "export_hifigan_workload.py" });
        foreach (var entry in recipes.Properties())
            if (HashFile(Path.Combine(AppContext.BaseDirectory, "pytorch", entry.Name)) != (string?)entry.Value)
                throw new InvalidDataException("Actual consumer reference/fixture recipe source differs: " + entry.Name);
        var steps = fixture["training_steps"] as JArray ?? throw new InvalidDataException("Missing actual benchmark steps.");
        for (int index = 0; index < 3; index++)
        {
            foreach (string role in new[] { "mel", "audio" })
            {
                string key = role + "/" + index;
                int[] shape = reference.ShapeOf(key)?.Select(dim => checked((int)dim)).ToArray()
                    ?? throw new InvalidDataException("Missing matched reference input.");
                var value = new Tensor<float>(shape);
                reference.CopyTo(key, shape, value);
                int[] canonicalShape = role == "mel" ? new[] { 1, 80, 32 } : new[] { 1, 8192 };
                if (!shape.SequenceEqual(role == "mel" ? canonicalShape : new[] { 8192 }))
                    throw new InvalidDataException("Actual raw reference crop geometry differs.");
                var actual = Describe(s_readSpan(value), canonicalShape);
                GC.KeepAlive(value);
                if (!JToken.DeepEquals(actual, steps[index]["cropped_" + role]))
                    throw new InvalidDataException("Actual raw reference input differs from the benchmark crop.");
            }
        }
        foreach (int batch in new[] { 1, 8, 16 })
        {
            string key = "inference/mel/" + batch;
            var value = new Tensor<float>(new[] { batch, 80, 256 });
            reference.CopyTo(key, value.Shape.ToArray(), value);
            CheckDescriptor(key, value, fixture["inputs"]![$"inference/{batch}/mel"]!);
        }
    }

    private void SaveReceipt()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(_receiptPath)!);
        File.WriteAllText(_receiptPath, _preflight!.ToString() + "\n");
    }

    private void ValidateFiles()
    {
        foreach (var file in Object(_manifest, "files_sha256").Properties())
            if (HashFile(Path.Combine(_directory, file.Name)) != (string?)file.Value)
                throw new InvalidDataException("Shared fixture file changed: " + file.Name);
    }

    private static void ValidateTensorMetadata(string path, JObject descriptors)
    {
        using var stream = File.OpenRead(path);
        Span<byte> prefix = stackalloc byte[8];
        stream.ReadExactly(prefix);
        ulong length = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (length > 16 * 1024 * 1024) throw new InvalidDataException("Unbounded safetensors metadata.");
        var bytes = new byte[checked((int)length)];
        stream.ReadExactly(bytes);
        var header = JObject.Parse(Encoding.UTF8.GetString(bytes));
        var names = header.Properties().Where(property => property.Name != "__metadata__").Select(property => property.Name);
        RequireKeys(descriptors, names);
        foreach (var name in names)
        {
            var entry = header[name]!;
            long elements = 1;
            foreach (var dim in (JArray)entry["shape"]!) { long value = (long)dim; if (value < 0) throw new InvalidDataException("Negative dimension."); elements = checked(elements * value); }
            var offsets = (JArray)entry["data_offsets"]!;
            long begin = (long)offsets[0], end = (long)offsets[1];
            if ((string?)entry["dtype"] != "F32" || (string?)descriptors[name]!["dtype"] != "F32"
                || !JToken.DeepEquals(entry["shape"], descriptors[name]!["shape"])
                || begin < 0 || end < begin || end > stream.Length - 8 - (long)length || end - begin != checked(elements * 4))
                throw new InvalidDataException("Invalid exact FP32 tensor metadata: " + name);
        }
    }

    private static JObject Object(JObject parent, string name) => parent[name] as JObject
        ?? throw new InvalidDataException("Missing object: " + name);
    private static void RequireKeys(JObject actual, IEnumerable<string> expected)
    {
        if (!new HashSet<string>(expected, StringComparer.Ordinal).SetEquals(actual.Properties().Select(property => property.Name)))
            throw new InvalidDataException("Missing or extra shared workload fields.");
    }
    private static void RequireShape(Tensor<float> value, int[] shape)
    {
        if (!value.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("Full workload tensor geometry differs.");
    }
    private static JObject Describe(ReadOnlySpan<float> values, int[] shape) => new() {
        ["dtype"]="F32",["shape"]=new JArray(shape),["sha256"]=Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values))).ToLowerInvariant(),
    };
    private static JObject CheckDescriptor(string name, Tensor<float> value, JToken expected)
    {
        var actual = Describe(s_readSpan(value), value.Shape.ToArray());
        GC.KeepAlive(value);
        if (!JToken.DeepEquals(actual, expected))
            throw new InvalidDataException("Actual loaded shared input differs: " + name);
        return actual;
    }
    private static JObject DescribeMelCrop(ReadOnlySpan<float> values, int start)
    {
        using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (int channel = 0; channel < 80; channel++)
            digest.AppendData(MemoryMarshal.AsBytes(values.Slice(channel * 256 + start, 32)));
        return new JObject { ["dtype"]="F32",["shape"]=new JArray(1, 80, 32),
            ["sha256"]=Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant() };
    }
    private static string HashFile(string path) { using var stream=File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }

    private sealed class FixtureCropRandom(int[] starts) : Random
    {
        internal int Consumed { get; private set; }
        public override int Next(int minValue, int maxValue)
        {
            if (minValue != 0 || maxValue != 225 || Consumed >= 9)
                throw new InvalidDataException("Unexpected actual crop bound or additional training step.");
            return starts[Consumed++];
        }
        public override int Next() => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override int Next(int maxValue) => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override double NextDouble() => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override float NextSingle() => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override long NextInt64() => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override long NextInt64(long maxValue) => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override long NextInt64(long minValue, long maxValue) => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override void NextBytes(byte[] buffer) => throw new InvalidDataException("Unexpected crop RNG operation.");
        public override void NextBytes(Span<byte> buffer) => throw new InvalidDataException("Unexpected crop RNG operation.");
        protected override double Sample() => throw new InvalidDataException("Unexpected crop RNG operation.");
    }
}
