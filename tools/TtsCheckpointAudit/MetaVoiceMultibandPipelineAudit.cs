using System.Text.Json;
using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.VoiceCloning;

internal static class MetaVoiceMultibandPipelineAudit
{
    internal static void Run(string fixture)
    {
        AiDotNet.Tensors.Helpers.CpuParallelSettings.MaxDegreeOfParallelism = 1;
        AiDotNetEngine.ResetToCpu();
        using var document = JsonDocument.Parse(File.ReadAllText(fixture));
        var root = document.RootElement;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1) { RandomSeed = 2294 };
        using var codec = new EnCodec<double>(architecture, EnCodecOptions.OfficialCheckpoint24kHz());
        codec.LoadPretrainedWeights(root.GetProperty("codec_release").GetProperty("paths").GetProperty("model.safetensors").GetString()!);
        var decoder = new MetaVoiceMultibandDecoder<double>(AiDotNetEngine.Current, codec);
        for (int band = 0; band < 4; band++)
        {
            decoder.LoadAuthorBand(root.GetProperty("release").GetProperty("checkpoint_path").GetString()!, band);
            Console.WriteLine($"MetaVoice complete MBD: original PTH band{band} imported, including trained processor where present.");
        }
        int frames = root.GetProperty("codes")[0].GetArrayLength(); var codes = new int[8, frames];
        for (int q = 0; q < 8; q++) for (int t = 0; t < frames; t++) codes[q, t] = root.GetProperty("codes")[q][t].GetInt32();
        var consumed = new HashSet<int>();
        Tensor<double> Draw(int index, int[] shape)
        {
            if (!consumed.Add(index)) throw new InvalidDataException("Duplicate MBD Gaussian draw.");
            var values = root.GetProperty("gaussian_draws")[index].EnumerateArray().Select(x => x.GetDouble()).ToArray();
            return new Tensor<double>(shape, new Vector<double>(values));
        }
        var actual = decoder.Decode(codes, (band, shape) => Draw(20 * band, shape),
            (band, step, shape) => Draw(20 * band + 1 + step, shape));
        if (consumed.Count != 80) throw new InvalidDataException("MBD controller consumed the wrong number of Gaussian draws.");
        if (!actual.Shape.ToArray().SequenceEqual(root.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32())))
            throw new InvalidDataException("Complete MBD waveform axes differ.");
        double error = actual.ToArray().Zip(root.GetProperty("waveform").EnumerateArray().Select(x => x.GetDouble()), (a, b) => Math.Abs(a - b)).Max();
        Console.WriteLine($"MetaVoice complete MBD: 80 neural evaluations,80 Gaussian draws, original EnCodec+4bands+2processors+32-band re_eq; waveform error {error:R}.");
        if (!double.IsFinite(error) || error > 1e-8) throw new InvalidDataException("Complete released MetaVoice multiband pipeline mismatch.");
    }
}
