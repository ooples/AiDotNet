using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>
/// MegaTTS 3's aligner (<c>aligner_lm</c>): a Whisper that reads a prompt's 16 kHz log-mel and writes its phones, each
/// followed by the frame it ends at, from which the prompt's phones, tones and frame-to-phone map come
/// (<c>frontend_function.align</c> and <c>split_ph_timestamp</c>).
/// </summary>
internal sealed class MegaTts3Aligner<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <summary>The token that starts an alignment.</summary>
    public const int Start = 798;

    /// <summary>The token that ends an alignment.</summary>
    public const int End = 799;

    /// <summary>The frames are cut to a multiple of this (the latent's 8 mel frames).</summary>
    public const int FrameMultiple = 8;

    /// <summary>The release's sizes (<c>whisper_small.Whisper</c>): 80 mels, 3000 frames, width 512, eight heads and six layers
    /// both ways, 6800 tokens.</summary>
    public static WhisperDimensions ReleaseDimensions => new(80, 3000, 512, 8, 6, 6800, 2048, 512, 8, 6) { UntiedOutput = true, VariableAudioLength = true };

    public MegaTts3Aligner(IEngine engine, List<LayerBase<T>> layers, WhisperDimensions dimensions)
    {
        if (!dimensions.UntiedOutput || !dimensions.VariableAudioLength)
            throw new ArgumentException("The aligner has untied logits and reads audio of any length.", nameof(dimensions));
        _engine = engine;
        Network = new OpenAiWhisperNetwork<T>(engine, layers, dimensions, new Random(0));
    }

    public OpenAiWhisperNetwork<T> Network { get; }

    /// <summary>The most tokens decoded (the release's 768).</summary>
    public int MaxTokens { get; set; } = 768;

    /// <summary>Whisper's log-mel of 16 kHz audio, its frames cut to a multiple of <see cref="FrameMultiple"/>:
    /// <c>[mels, frames]</c>.</summary>
    public Tensor<T> LogMel(double[] audio)
    {
        var mel = OpenAiWhisperTranscriber<T>.LogMelSpectrogram(audio, Network.Dimensions.Mels);
        int mels = mel.GetLength(0), frames = mel.GetLength(1) / FrameMultiple * FrameMultiple;
        if (frames == 0) throw new ArgumentException("The audio is shorter than one latent frame.", nameof(audio));
        var result = new Tensor<T>(new[] { mels, frames });
        for (int m = 0; m < mels; m++)
            for (int t = 0; t < frames; t++) result[m, t] = NumOps.FromDouble(mel[m, t]);
        return result;
    }

    /// <summary>Greedy decoding from <see cref="Start"/> until <see cref="End"/> or <see cref="MaxTokens"/> steps: the tokens
    /// (with the start) and the last step's logits.</summary>
    public (List<int> Tokens, Tensor<T> LastLogits) Decode(Tensor<T> mel)
    {
        var audio = Network.Encode(mel);
        var cache = new OpenAiWhisperNetwork<T>.Cache();
        var tokens = new List<int> { Start };
        Tensor<T> logits = null!;
        for (int step = 0; step < MaxTokens; step++)
        {
            logits = Network.Logits(new[] { tokens[^1] }, audio, cache);
            int best = 0;
            for (int v = 1; v < logits.Shape[1]; v++)
                if (NumOps.GreaterThan(logits[0, v], logits[0, best])) best = v;
            tokens.Add(best);
            if (best == End) break;
        }
        return (tokens, _engine.Reshape(logits, new[] { logits.Shape[1] }));
    }

    /// <summary>
    /// <c>split_ph_timestamp</c> on the decoded tokens without the first and last, then <c>align</c>'s fitting of the
    /// durations to the prompt's <paramref name="frames"/> (the shortfall added to the last phone, an excess taken one
    /// frame a phone from the first on) and the length regulator: the phones, tones (merged Chinese phones split into the
    /// phone and tones 4 or 11–15, others tone 3) and the 1-based phone of each frame.
    /// </summary>
    public static (int[] Phones, int[] Tones, int[] FrameToPhone) Split(IReadOnlyList<int> tokens, int frames)
    {
        if (tokens.Count < 2) throw new ArgumentException("An alignment has a start and an end token.", nameof(tokens));
        var body = tokens.Skip(1).Take(tokens.Count - 2).Select(t => t >= 800 ? t - 800 : t).ToArray();
        if (body.Length % 2 != 0)
            throw new InvalidOperationException($"The aligner wrote {(body.Length + 1) / 2} phones but {body.Length / 2} timestamps.");
        int count = body.Length / 2;
        if (count == 0) throw new InvalidOperationException("The aligner wrote no phones.");
        var phones = new int[count];
        var tones = new int[count];
        var durations = new int[count];
        int current = 0;
        for (int i = 0; i < count; i++)
        {
            int item = body[2 * i];
            (phones[i], tones[i]) = SplitPhone(item);
            durations[i] = body[2 * i + 1] - current;
            current = body[2 * i + 1];
        }
        int sum = durations.Sum();
        if (sum < frames) durations[count - 1] += frames - sum;
        else if (sum > frames)
        {
            int excess = sum - frames;
            for (int i = 0; excess > 0; i = (i + 1) % count, excess--) durations[i]--;
        }
        var map = LengthRegulate(durations);
        return (phones, tones, map.Take(map.Length / FrameMultiple * FrameMultiple).ToArray());
    }

    /// <summary>A merged token's phone and tone (<c>split_ph</c>): 200–788 are Chinese phones with their tones
    /// (<c>(t − 201) // 6 + 3</c>, tone 4 or 11–15), others keep their id with tone 3.</summary>
    public static (int Phone, int Tone) SplitPhone(int token)
    {
        if (token < 200 || token > 788) return (token, 3);
        int offset = token - 201;
        int phone = FloorDiv(offset, 6) + 3, tone = offset - 6 * FloorDiv(offset, 6) + 1;
        return (phone, tone == 1 ? 4 : tone + 9);
    }

    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    /// <summary>The length regulator: frame t belongs to every phone i (1-based, summed) whose span
    /// [cumsum<sub>i−1</sub>, cumsum<sub>i</sub>) holds it, over as many frames as the durations add to.</summary>
    public static int[] LengthRegulate(IReadOnlyList<int> durations)
    {
        int total = Math.Max(0, durations.Sum());
        var map = new int[total];
        int previous = 0;
        for (int i = 0; i < durations.Count; i++)
        {
            int end = previous + durations[i];
            for (int t = Math.Max(previous, 0); t < Math.Min(end, total); t++) map[t] += i + 1;
            previous = end;
        }
        return map;
    }

    /// <summary><c>align</c> on 16 kHz audio: the phones, tones and frame-to-phone map.</summary>
    public (int[] Phones, int[] Tones, int[] FrameToPhone) Align(double[] audio)
    {
        var mel = LogMel(audio);
        return Split(Decode(mel).Tokens, mel.Shape[1]);
    }

    /// <summary>Loads the release's <c>aligner_lm</c> checkpoint state.</summary>
    public void LoadTorch(Func<string, double[]> read) => Network.LoadTorchWeights((name, _) => read(name));
}
