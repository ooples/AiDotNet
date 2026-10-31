using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The files a <see cref="MARS5TTS{T}"/> loads.</summary>
public sealed class Mars5Checkpoints
{
    /// <summary>The AR model (<c>mars5_ar.safetensors</c>, Hugging Face <c>CAMB-AI/MARS5-TTS</c>).</summary>
    public required string Autoregressive { get; init; }

    /// <summary>The NAR model (<c>mars5_nar.safetensors</c>).</summary>
    public required string NonAutoregressive { get; init; }

    /// <summary>EnCodec's 24 kHz checkpoint (<c>encodec_24khz-d7cc33bc.th</c>).</summary>
    public required string Codec { get; init; }

    /// <summary>Vocos's EnCodec model (<c>charactr/vocos-encodec-24khz</c>'s <c>pytorch_model.bin</c>).</summary>
    public required string Vocoder { get; init; }
}

/// <summary>Reads a safetensors file's <c>__metadata__</c> (MARS5 stores its tokenizers and training configs there).</summary>
internal static class Mars5Checkpoint
{
    public static Dictionary<string, string> Metadata(string path)
    {
        using var stream = File.OpenRead(path);
        var lengthBytes = new byte[8];
        if (stream.Read(lengthBytes, 0, 8) != 8) throw new InvalidDataException($"'{path}' is not a safetensors file.");
        long length = BitConverter.ToInt64(lengthBytes, 0);
        if (length <= 0 || length > 100_000_000) throw new InvalidDataException($"'{path}' has an invalid safetensors header length.");
        var header = new byte[length];
        int read = 0;
        while (read < length)
        {
            int n = stream.Read(header, read, (int)length - read);
            if (n <= 0) throw new InvalidDataException($"'{path}' ends inside its safetensors header.");
            read += n;
        }
        var json = JObject.Parse(System.Text.Encoding.UTF8.GetString(header));
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (json["__metadata__"] is JObject metadata)
            foreach (var property in metadata.Properties()) result[property.Name] = (string?)property.Value ?? "";
        return result;
    }
}

public partial class MARS5TTS<T>
{
    /// <summary>
    /// Loads the release's weights: the AR and NAR models (their F16 tensors read as the model's type), EnCodec's 24 kHz
    /// checkpoint and Vocos's EnCodec model. Build the options with <see cref="MARS5TTSOptions.FromCheckpoint"/> so the
    /// tokenizers, and the vocabulary sizes, match the checkpoints.
    /// </summary>
    public void LoadReferenceWeights(Mars5Checkpoints checkpoints)
    {
        ThrowIfDisposed();
        if (checkpoints is null) throw new ArgumentNullException(nameof(checkpoints));
        if (!HasPaperLayers) throw NotNative();
        Func<string, double[]?> Reader(string path)
        {
            var file = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path);
            return name =>
            {
                if (!file.TryGetValue(name, out var tensor)) return null;
                file.Remove(name);
                var values = new double[tensor.Length];
                for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
                return values;
            };
        }
        Autoregressive.Load(Reader(checkpoints.Autoregressive));
        NonAutoregressive.Load(Reader(checkpoints.NonAutoregressive));
        Codec.LoadPretrainedWeights(checkpoints.Codec);
        var vocos = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoints.Vocoder);
        Vocos.LoadTorchWeights((name, shape) =>
        {
            if (!vocos.TryGetValue(name, out var tensor)) throw new InvalidDataException($"Vocos's checkpoint has no tensor '{name}'.");
            return tensor.ToVector().Select(v => (double)v).ToArray();
        });
        _optimizers.Clear();
    }
}
