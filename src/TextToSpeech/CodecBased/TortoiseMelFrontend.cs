using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>The distinct TorchMelSpectrogram and TacotronSTFT front ends in the TorToiSe release.</summary>
internal sealed class TortoiseMelFrontend<T>
{
    private readonly IEngine _engine;
    private readonly CenteredLogMel<T> _autoregressive, _diffusion;
    private readonly int _arBands, _diffusionBands;
    private double[] _normalization;

    // Exact float values from neonbjb/tortoise-tts/tortoise/data/mel_norms.pth.
    private static readonly double[] ReleasedNormalization =
    [
        -7.00946044921875, -6.08323335647583, -4.643990993499756, -3.3562171459198,
        -2.4547812938690186, -2.009686231613159, -1.6035902500152588, -1.8640903234481812,
        -2.3727996349334717, -2.3455493450164795, -2.594703197479248, -2.669504404067993,
        -2.712865114212036, -2.8554766178131104, -3.0250935554504395, -3.0889248847961426,
        -3.426147699356079, -3.675870656967163, -4.077994346618652, -4.462376117706299,
        -4.781159400939941, -5.007452487945557, -5.128354072570801, -5.271664619445801,
        -5.400559902191162, -5.499321937561035, -5.531040668487549, -5.587804317474365,
        -5.67262601852417, -5.7015767097473145, -5.794291973114014, -5.88307523727417,
        -5.953695774078369, -5.9988579750061035, -6.0305256843566895, -6.053944110870361,
        -6.074769496917725, -6.116333961486816, -6.148137092590332, -6.24762487411499,
        -6.319519996643066, -6.445708274841309, -6.537699222564697, -6.610982418060303,
        -6.648107051849365, -6.667131423950195, -6.6539483070373535, -6.649946689605713,
        -6.679383754730225, -6.783320903778076, -6.930696487426758, -7.081806182861328,
        -7.18935489654541, -7.243902683258057, -7.316774845123291, -7.377870559692383,
        -7.449100017547607, -7.523320198059082, -7.622361660003662, -7.747303485870361,
        -7.899416446685791, -8.060430526733398, -8.218083381652832, -8.399794578552246,
        -8.555578231811523, -8.716107368469238, -8.848074913024902, -8.958160400390625,
        -9.037120819091797, -9.086742401123047, -9.154592514038086, -9.203839302062988,
        -9.233386993408203, -9.229235649108887, -9.230441093444824, -9.267979621887207,
        -9.315550804138184, -9.371553421020508, -9.416544914245605, -9.482237815856934
    ];

    internal TortoiseMelFrontend(IEngine engine, TortoiseTTSOptions options)
    {
        _engine = engine; _arBands = options.AutoregressiveMelChannels; _diffusionBands = options.MelChannels;
        _autoregressive = new CenteredLogMel<T>(engine, options.TokenizerSampleRate, options.AutoregressiveFftSize,
            options.AutoregressiveHopSize, options.AutoregressiveFftSize, _arBands, 0,
            Math.Min(8000, options.TokenizerSampleRate / 2.0), 1e-5, htkScale: true, naturalLog: true,
            spectrumPower: 2, htkAreaNormalization: true);
        _diffusion = new CenteredLogMel<T>(engine, options.SampleRate, options.FftSize, options.HopSize,
            options.FftSize, _diffusionBands, 0, options.SampleRate / 2.0, 1e-5, naturalLog: true);
        _normalization = _arBands == ReleasedNormalization.Length
            ? (double[])ReleasedNormalization.Clone() : Enumerable.Repeat(1.0, _arBands).ToArray();
        if (options.AutoregressiveMelNormalization is not null) SetNormalization(options.AutoregressiveMelNormalization);
    }

    internal void SetNormalization(double[] values)
    {
        Validation.Guard.NotNull(values);
        if (values.Length != _arBands || values.Any(v => double.IsNaN(v) || double.IsInfinity(v) || v == 0))
            throw new ArgumentException($"Expected {_arBands} finite nonzero mel normalization values.", nameof(values));
        _normalization = (double[])values.Clone();
    }

    private Tensor<T> Rows(Tensor<T> mel, int bands) =>
        _engine.TensorTranspose(_engine.Reshape(mel, new[] { bands, mel.Shape[2] }));

    internal Tensor<T> Autoregressive(Tensor<T> audio) =>
        Rows(_autoregressive.Forward(audio, new double[_arBands], _normalization), _arBands);

    internal Tensor<T> Diffusion(Tensor<T> audio) => Rows(_diffusion.Forward(_engine.TensorClamp(audio,
        TortoiseOps<T>.NumOps.FromDouble(-1), TortoiseOps<T>.NumOps.One)), _diffusionBands);
}
