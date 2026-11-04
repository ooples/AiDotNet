using AiDotNet.Audio.Generation;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>The released MetaVoice audio decoder: eight-codebook EnCodec conditioning,
/// four independent diffusion processes, fitted processor inversion and 32-band re-equalization.</summary>
internal sealed class MetaVoiceMultibandDecoder<T>
{
    private readonly IEngine _engine;
    private readonly EnCodec<T> _codec;
    private readonly MetaVoiceDiffusionUnet<T>?[] _models = new MetaVoiceDiffusionUnet<T>?[4];
    private readonly MetaVoiceBandProcessor<T>?[] _processors = new MetaVoiceBandProcessor<T>?[4];
    private readonly Dictionary<string, double[]>?[] _processorStates = new Dictionary<string, double[]>?[4];
    private readonly MetaVoiceDiffusionSchedule<T> _schedule;
    private readonly MetaVoiceBandFilters<T> _equalizer;
    internal IReadOnlyList<LayerBase<T>> DiffusionLayers => _models.Where(x => x is not null).SelectMany(x => x!.Layers).ToArray();

    internal MetaVoiceMultibandDecoder(IEngine engine, EnCodec<T> trainedCodec)
    {
        _engine = engine; _codec = trainedCodec;
        _schedule = new MetaVoiceDiffusionSchedule<T>(engine, MetaVoiceDiffusionBetas.CreateReleaseSchedule());
        _equalizer = new MetaVoiceBandFilters<T>(engine, 32);
    }

    internal void LoadAuthorBand(string checkpoint, int band)
    {
        if (band < 0 || band >= 4) throw new ArgumentOutOfRangeException(nameof(band));
        var metadata = MetaVoiceTokenizer.ReadMetadata(checkpoint);
        if (Convert.ToInt32(metadata["n_bands"]) != 4 || Convert.ToInt32(metadata["sample_rate"]) != 24000)
            throw new InvalidDataException("MetaVoice requires the four-band 24 kHz AudioCraft checkpoint.");
        string prefix = band + ".";
        var state = new WeightLoader().LoadWeights(checkpoint, name => name.StartsWith(prefix + "model_state.", StringComparison.Ordinal) ||
            name.StartsWith(prefix + "processor_state.", StringComparison.Ordinal));
        double[] Read(string name, int[] shape)
        {
            if (!state.TryGetValue(name, out var value)) throw new InvalidDataException("Missing MetaVoice MBD tensor: " + name);
            if (!value.Shape.ToArray().SequenceEqual(shape)) throw new InvalidDataException("MBD tensor dimensions differ: " + name);
            state.Remove(name);
            return value.ToArray().Select(x => (double)x).ToArray();
        }
        var model = new MetaVoiceDiffusionUnet<T>(_engine);
        model.LoadTorch((name, shape) => Read(prefix + "model_state." + name, shape));
        MetaVoiceBandProcessor<T>? processor = null;
        if (band >= 2)
        {
            processor = new MetaVoiceBandProcessor<T>(_engine, band == 2 ? .4 : .75);
            var fitted = new Dictionary<string, double[]>(StringComparer.Ordinal);
            processor.LoadTorch((name, shape) =>
            {
                var values = Read(prefix + "processor_state." + name, shape);
                fitted.Add(name, (double[])values.Clone());
                return values;
            });
            _processorStates[band] = fitted;
        }
        if (state.Count != 0) throw new InvalidDataException("Unconsumed MetaVoice multiband checkpoint tensors: " + state.Count);
        _models[band] = model; _processors[band] = processor;
    }

    internal sealed class RuntimeState
    {
        public int[][] Configurations { get; set; } = Array.Empty<int[]>();
        public int[] LayerCounts { get; set; } = Array.Empty<int>();
        public Dictionary<string, double[]>?[] Processors { get; set; } = Array.Empty<Dictionary<string, double[]>?>();
    }

    internal string SaveRuntimeState()
    {
        if (_models.Any(model => model is null)) throw new InvalidOperationException("Load all four diffusion bands first.");
        return Newtonsoft.Json.JsonConvert.SerializeObject(new RuntimeState
        {
            Configurations = _models.Select(model => model!.Configuration).ToArray(),
            LayerCounts = _models.Select(model => model!.Layers.Count).ToArray(),
            Processors = _processorStates
        });
    }

    internal static MetaVoiceMultibandDecoder<T> RestoreRuntime(IEngine engine, EnCodec<T> codec, string json,
        IReadOnlyList<LayerBase<T>> layers)
    {
        var state = Newtonsoft.Json.JsonConvert.DeserializeObject<RuntimeState>(json)
            ?? throw new InvalidDataException("Missing multiband runtime state.");
        if (state.Configurations.Length != 4 || state.LayerCounts.Length != 4 || state.Processors.Length != 4 ||
            state.LayerCounts.Any(count => count <= 0) || state.LayerCounts.Sum() != layers.Count)
            throw new InvalidDataException("The saved decoder must describe exactly four diffusion bands.");
        var decoder = new MetaVoiceMultibandDecoder<T>(engine, codec);
        int offset = 0;
        for (int band = 0; band < 4; band++)
        {
            var c = state.Configurations[band];
            if (c.Length != 10) throw new InvalidDataException("Invalid saved diffusion architecture.");
            var model = new MetaVoiceDiffusionUnet<T>(engine, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9],
                layers.Skip(offset).Take(state.LayerCounts[band]).ToArray());
            decoder.InstallBand(band, model, state.Processors[band]);
            offset += state.LayerCounts[band];
        }
        return decoder;
    }

    internal void InstallBand(int band, MetaVoiceDiffusionUnet<T> model, Dictionary<string, double[]>? fitted)
    {
        if (band < 0 || band >= 4) throw new ArgumentOutOfRangeException(nameof(band));
        MetaVoiceBandProcessor<T>? processor = null;
        if (band >= 2)
        {
            if (fitted is null) throw new InvalidDataException("A fitted processor is required for the upper bands.");
            processor = new MetaVoiceBandProcessor<T>(_engine, band == 2 ? .4 : .75);
            processor.LoadTorch((name, shape) =>
            {
                if (!fitted.TryGetValue(name, out var values) || values.Length != shape.Aggregate(1, (a, b) => checked(a * b)))
                    throw new InvalidDataException("Invalid saved fitted processor tensor: " + name);
                return values;
            });
        }
        _models[band] = model;
        _processors[band] = processor;
        _processorStates[band] = fitted is null ? null : fitted.ToDictionary(pair => pair.Key,
            pair => (double[])pair.Value.Clone(), StringComparer.Ordinal);
    }

    internal Tensor<T> Decode(int[,] codes, Func<int, int[], Tensor<T>> initialGaussian,
        Func<int, int, int[], Tensor<T>> stepGaussian, IReadOnlyList<int>? steps = null)
    {
        if (codes.GetLength(0) != 8 || codes.GetLength(1) <= 0)
            throw new ArgumentException("MetaVoice requires eight EnCodec hierarchies with at least one frame.", nameof(codes));
        if (_models.Any(model => model is null)) throw new InvalidOperationException("Load all four released multiband denoisers before decoding.");
        using var noGrad = new NoGradScope<T>();
        var reference = _codec.Decode(codes);
        var condition = _codec.DecodeCodesToEmbedding(codes);
        var output = new Tensor<T>(reference.Shape.ToArray());
        for (int band = 0; band < 4; band++)
        {
            var initial = initialGaussian(band, reference.Shape.ToArray());
            if (!initial.Shape.ToArray().SequenceEqual(reference.Shape.ToArray())) throw new ArgumentException("Initial diffusion noise has the wrong shape.", nameof(initialGaussian));
            var model = _models[band] ?? throw new InvalidOperationException("Missing diffusion band.");
            var trajectory = _schedule.Generate(initial, condition,
                (current, step, embedding) => model.Forward(current, Enumerable.Repeat(step, current.Shape[0]).ToArray(), embedding),
                index => stepGaussian(band, index, reference.Shape.ToArray()), steps);
            var waveform = trajectory[trajectory.Count - 1];
            if (_processors[band] is { } processor) waveform = processor.ReturnSample(waveform);
            output = _engine.TensorAdd(output, waveform);
        }
        return _equalizer.MatchEqualization(output, reference);
    }
}
