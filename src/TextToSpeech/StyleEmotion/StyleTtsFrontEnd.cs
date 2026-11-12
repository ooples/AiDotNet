using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.Vocoders;

namespace AiDotNet.TextToSpeech.StyleEmotion;

/// <summary>
/// The text and audio front end StyleTTS and StyleTTS 2 share (yl4579/StyleTTS <c>text_utils.py</c> and
/// <c>meldataset.py</c>): the 178-symbol phoneme table and the normalized log-mel spectrogram.
/// </summary>
internal sealed class StyleTtsFrontEnd<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;
    private readonly StyleTtsBaseOptions _options;
    private CenteredComplexStft<T>? _stft;
    private Tensor<T>? _melBasis;

    public StyleTtsFrontEnd(IEngine engine, StyleTtsBaseOptions options)
    {
        _engine = engine;
        _options = options;
    }

    // The released symbol table (text_utils.py): pad, punctuation, letters, IPA. A repeated symbol keeps its last index,
    // as the reference's dictionary does.
    internal static readonly IReadOnlyDictionary<char, int> SymbolIds = BuildSymbols();

    private static Dictionary<char, int> BuildSymbols()
    {
        const string pad = "$";
        const string punctuation = ";:,.!?¡¿—…\"«»“” ";
        const string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        const string ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";
        var symbols = pad + punctuation + letters + ipa;
        var ids = new Dictionary<char, int>();
        for (int i = 0; i < symbols.Length; i++) ids[symbols[i]] = i;
        return ids;
    }

    /// <summary>The number of entries in the released symbol table (178).</summary>
    public const int SymbolCount = 178;

    /// <summary>The symbol ids of IPA phonemes framed by the pad symbol; characters outside the table are skipped, as the
    /// reference skips them.</summary>
    public static int[] EncodePhonemes(string phonemes)
    {
        if (phonemes is null) throw new ArgumentNullException(nameof(phonemes));
        var ids = new List<int> { 0 };
        foreach (char c in phonemes)
            if (SymbolIds.TryGetValue(c, out int id)) ids.Add(id);
        ids.Add(0);
        return ids.ToArray();
    }

    /// <summary>The normalized log-mel spectrogram <c>[mels, frames]</c> of a waveform (torchaudio's centred, periodic-Hann,
    /// power-2, HTK-scale <c>MelSpectrogram</c>, then <c>(log(1e-5 + mel) − mean) / std</c>).</summary>
    public Tensor<T> ComputeMel(Tensor<T> audio)
    {
        if (audio is null) throw new ArgumentNullException(nameof(audio));
        var o = _options;
        _stft ??= new CenteredComplexStft<T>(_engine, o.FftSize, o.HopSize, o.WindowLength);
        if (_melBasis is null)
        {
            var basis = MelFilterbanks.Htk(o.SampleRate, o.FftSize, o.MelChannels, 0.0, Math.Floor(o.SampleRate / 2.0));
            _melBasis = new Tensor<T>(new[] { o.MelChannels, o.FftSize / 2 + 1 });
            for (int m = 0; m < o.MelChannels; m++)
                for (int k = 0; k < o.FftSize / 2 + 1; k++) _melBasis[m, k] = NumOps.FromDouble(basis[m, k]);
        }
        using var _ = new NoGradScope<T>();
        var (re, im) = _stft.Forward(_engine.Reshape(audio, new[] { audio.Length }));
        int bins = re.Shape[1], frames = re.Shape[2];
        var power = _engine.TensorAdd(_engine.TensorMultiply(re, re), _engine.TensorMultiply(im, im));
        var mel = _engine.TensorMatMul(_melBasis, _engine.Reshape(power, new[] { bins, frames }));
        var result = new Tensor<T>(new[] { o.MelChannels, frames });
        for (int i = 0; i < result.Length; i++)
            result[i] = NumOps.FromDouble((Math.Log(1e-5 + NumOps.ToDouble(mel[i])) - o.MelMean) / o.MelStd);
        return result;
    }

    /// <summary>The energy of each frame of a normalized mel spectrogram <c>[mels, frames]</c>: the log of the L2 norm of
    /// the de-normalized mel magnitudes (the reference's <c>log_norm</c>).</summary>
    internal Tensor<T> LogNorm(Tensor<T> mel)
    {
        int mels = mel.Shape[0], frames = mel.Shape[1];
        var result = new Tensor<T>(new[] { 1, frames });
        for (int f = 0; f < frames; f++)
        {
            double sum = 0;
            for (int m = 0; m < mels; m++)
            {
                double v = Math.Exp(NumOps.ToDouble(mel[m, f]) * _options.MelStd + _options.MelMean);
                sum += v * v;
            }
            result[0, f] = NumOps.FromDouble(Math.Log(Math.Sqrt(sum)));
        }
        return result;
    }
}
