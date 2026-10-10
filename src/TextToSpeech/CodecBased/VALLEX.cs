using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.FrontEnd;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// VALL-E X: VALL-E trained on English and Mandarin with a language ID, which speaks a language in the voice of a prompt
/// recorded in another.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Reference: "Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling" (Zhang et al.,
/// 2023). Microsoft released no code; the runnable reproduction is Plachtaa/VALL-E-X, whose Mandarin front end
/// <see cref="MandarinG2P"/> reproduces. The networks are VALL-E's (<see cref="VallEModelBase{T}"/>).
/// </para>
/// <para>
/// <b>Languages</b> (§3.3): a language embedding added to the AR model's acoustic-token embeddings — the prompt's codes
/// in the prompt's language, the generated ones in the text's — guides the speaking style.
/// <see cref="VALLEXOptions.LanguagePlacement"/> selects the reference's placement instead (the phoneme embeddings of
/// both models).
/// </para>
/// <para>
/// <b>Training</b> (§3.2, §5.2): the AR model as VALL-E's; the NAR model, at a uniformly drawn stage, reads the
/// phonemes, an acoustic prompt from another sentence of the same speaker (all codebooks, Eq. 2) and the lower
/// codebooks. Each model is optimized on its own.
/// </para>
/// <para>
/// <b>Synthesis</b> (§3.4): the prompt's transcript and the text precede the prompt's first-codebook codes in the AR
/// model (Eq. 3, sampled to the end token); the NAR reads the text alone and the prompt's codes (Eq. 4, greedy);
/// EnCodec decodes. Text is phonemized by script: Chinese characters through <see cref="MandarinG2P"/>, the rest
/// through <see cref="EnglishG2P"/>, so mixed sentences work.
/// </para>
/// <para>
/// <b>Training data</b>: <see cref="TtsTrainingSample{T}.Tokens"/> (phoneme ids,
/// <see cref="VallEModelBase{T}.EncodePhonemes"/>), <see cref="TtsTrainingSample{T}.CodecTokens"/> (or the recording),
/// <see cref="TtsTrainingSample{T}.PromptCodecTokens"/> (another sentence of the same speaker) and
/// <see cref="TtsTrainingSample{T}.LanguageId"/> (<see cref="VallEXLanguage"/>).
/// </para>
/// <para><b>For Beginners:</b> Record a few seconds of yourself in English, and VALL-E X can speak Chinese in your
/// voice — or the other way round.</para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(InputType.OneDimensional,
///     NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
/// var vallex = new VALLEX&lt;float&gt;(architecture, new VALLEXOptions());
/// vallex.Voice = vallex.CreateVoice(englishPrompt24kHz, "the prompt's transcript");
/// var audio = vallex.Synthesize("你好，世界。");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling",
    "https://arxiv.org/abs/2303.03926",
    Year = 2023,
    Authors = "Zhang et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 8_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "autoregressive", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Section 5.2 states the maximum learning rate 5e-4, 8,000 warm-up steps and 800k steps, but not the "
             + "optimizer; VALL-E (Wang et al. 2023 §5.1), which the paper extends, uses AdamW with linear decay "
             + "(PyTorch's default betas and weight decay).")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 5e-4, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
    Schedule = LearningRateSchedulerType.LinearWarmup, WarmupSteps = 8_000,
    PostWarmupDecay = LinearWarmupScheduler.DecayMode.Linear,
    Component = "non-autoregressive", Provenance = RecipeProvenance.DerivedFromCitedWork,
    Source = "Section 5.2 states the maximum learning rate 5e-4, 8,000 warm-up steps and 800k steps, but not the "
             + "optimizer; VALL-E (Wang et al. 2023 §5.1), which the paper extends, uses AdamW with linear decay "
             + "(PyTorch's default betas and weight decay).")]
public partial class VALLEX<T> : VallEModelBase<T>
{
    private static readonly Lazy<string[]> Table = new(() =>
        LibriTtsPhonemeTable.Symbols.Concat(MandarinG2P.Symbols).Distinct(StringComparer.Ordinal).ToArray(), isThreadSafe: true);

    private static readonly Lazy<(HashSet<string> English, HashSet<string> Mandarin)> LanguageOnly = new(() =>
    {
        var english = new HashSet<string>(LibriTtsPhonemeTable.Symbols, StringComparer.Ordinal);
        var mandarin = new HashSet<string>(MandarinG2P.Symbols, StringComparer.Ordinal);
        return (new HashSet<string>(english.Except(mandarin), StringComparer.Ordinal),
            new HashSet<string>(mandarin.Except(english), StringComparer.Ordinal));
    }, isThreadSafe: true);

    /// <summary>Creates an ONNX-backed VALL-E X for inference.</summary>
    public VALLEX(NeuralNetworkArchitecture<T> architecture, string modelPath, VALLEXOptions? options = null)
        : base(architecture, modelPath, options ?? new VALLEXOptions())
    {
    }

    /// <summary>Creates a native, trainable VALL-E X.</summary>
    public VALLEX(NeuralNetworkArchitecture<T> architecture, VALLEXOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, options ?? new VALLEXOptions(), optimizer)
    {
    }

    private VALLEXOptions XOptions => (VALLEXOptions)Settings;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => Settings;

    /// <inheritdoc />
    /// <remarks>The English espeak table and <see cref="MandarinG2P.Symbols"/>, in one table (symbols both languages use,
    /// such as punctuation and the word separator, once).</remarks>
    protected override IReadOnlyList<string> PhonemeTable => Table.Value;

    /// <inheritdoc />
    protected override int LanguageCount => 2;

    /// <inheritdoc />
    protected override VallELanguagePlacement LanguagePlacement => ((VALLEXOptions)Settings).LanguagePlacement;

    /// <inheritdoc />
    /// <remarks>Symbols outside the table are skipped, as the reference's tokenizer skips them.</remarks>
    protected override bool RejectsUnknownSymbols => false;

    /// <inheritdoc />
    protected override TtsSupervision RequiredSupervision =>
        TtsSupervision.CodecTokens | TtsSupervision.PromptCodecTokens | TtsSupervision.LanguageId;

    /// <inheritdoc />
    /// <remarks>Runs of Chinese characters (and Chinese punctuation) through <see cref="MandarinG2P"/>, the rest through
    /// <see cref="EnglishG2P"/>, joined by word separators.</remarks>
    protected override IReadOnlyList<string> PhonemizeText(string text)
    {
        var symbols = new List<string>();
        foreach (var (mandarin, run) in ScriptRuns(text))
        {
            var phonemes = mandarin ? MandarinG2P.Default.Phonemize(run) : EnglishG2P.Default.Phonemize(run);
            if (phonemes.Count == 0) continue;
            if (symbols.Count > 0 && symbols[symbols.Count - 1] != "_" && phonemes[0] != "_") symbols.Add("_");
            symbols.AddRange(phonemes);
        }
        return symbols;
    }

    private static bool IsChinese(char c) =>
        c is >= '一' and <= '鿿' or >= '㐀' and <= '䶿' or '，' or '。' or '！' or '？' or '、' or '；' or '：' or '—';

    private static IEnumerable<(bool Mandarin, string Text)> ScriptRuns(string text)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        int start = 0;
        for (int i = 1; i <= text.Length; i++)
        {
            if (i == text.Length || IsChinese(text[i]) != IsChinese(text[start]))
            {
                string run = text.Substring(start, i - start);
                if (run.Trim().Length > 0) yield return (IsChinese(text[start]), run);
                start = i;
            }
        }
    }

    /// <summary>
    /// The language of each token, decided word by word (words are separated by <c>_</c>): a word speaks the language
    /// most of its single-language symbols belong to (an English multi-letter phoneme, a Mandarin tone or IPA character);
    /// a word with none takes the previous word's language, else the next one's, else English. Separators and the frame
    /// tokens take the previous word's.
    /// </summary>
    internal int[] TokenLanguages(IReadOnlyList<int> ids)
    {
        var (english, mandarin) = LanguageOnly.Value;
        // Words: maximal runs of tokens other than the separator and the frame tokens.
        var words = new List<(int Start, int End, int? Language)>();
        for (int i = 0; i < ids.Count;)
        {
            if (IsBoundary(ids[i])) { i++; continue; }
            int start = i, englishCount = 0, mandarinCount = 0;
            for (; i < ids.Count && !IsBoundary(ids[i]); i++)
            {
                string symbol = Symbol(ids[i]);
                if (english.Contains(symbol)) englishCount++;
                else if (mandarin.Contains(symbol)) mandarinCount++;
            }
            int? language = englishCount == 0 && mandarinCount == 0 ? null
                : mandarinCount > englishCount ? (int)VallEXLanguage.Mandarin : (int)VallEXLanguage.English;
            words.Add((start, i, language));
        }
        var decided = new int[words.Count];
        int? last = null;
        for (int w = 0; w < words.Count; w++)
        {
            last = words[w].Language ?? last;
            decided[w] = last ?? -1;
        }
        int? next = null;
        for (int w = words.Count - 1; w >= 0; w--)
        {
            next = words[w].Language ?? next;
            if (decided[w] < 0) decided[w] = next ?? (int)VallEXLanguage.English;
        }
        var languages = new int[ids.Count];
        int current = decided.Length > 0 ? decided[0] : (int)VallEXLanguage.English, word = 0;
        for (int i = 0; i < ids.Count; i++)
        {
            if (word < words.Count && i >= words[word].Start && i < words[word].End) current = decided[word];
            languages[i] = current;
            if (word < words.Count && i == words[word].End - 1) word++;
        }
        return languages;
    }

    private bool IsBoundary(int id) => id < 3 || Symbol(id) == "_";

    private int SampleLanguage(TtsTrainingSample<T> sample)
    {
        int language = sample.LanguageId ?? throw new ArgumentException("VALL-E X trains with the utterance's language.", nameof(sample));
        if (language is < 0 or > 1) throw new ArgumentOutOfRangeException(nameof(sample), "VALL-E X's languages are 0 (English) and 1 (Mandarin).");
        return language;
    }

    private bool OnText => LanguagePlacement == VallELanguagePlacement.TextTokens;

    /// <inheritdoc />
    /// <remarks>The utterance's codes and phonemes are in its language (<see cref="TtsTrainingSample{T}.LanguageId"/>).</remarks>
    protected override Func<Tensor<T>, Tensor<T>, Tensor<T>> AutoRegressiveObjective(TtsTrainingSample<T> sample, int[] text,
        int[,] codes, bool training, Random random)
    {
        int language = SampleLanguage(sample);
        var first = FirstCodebook(codes);
        var textLanguages = OnText ? Enumerable.Repeat(language, text.Length).ToArray() : null;
        var codeLanguages = OnText ? null : Enumerable.Repeat(language, first.Length).ToArray();
        return (_, _) => PaperCore.ArLoss(text, first, training, random, textLanguages, codeLanguages);
    }

    /// <inheritdoc />
    /// <remarks>The prompt is another sentence of the same speaker (<see cref="TtsTrainingSample{T}.PromptCodecTokens"/>,
    /// Eq. 2).</remarks>
    protected override Func<Tensor<T>, Tensor<T>, Tensor<T>> NonAutoRegressiveObjective(TtsTrainingSample<T> sample,
        int[] text, int[,] codes, bool training, Random random)
    {
        int language = SampleLanguage(sample);
        var prompt = CodesOf(sample.PromptCodecTokens ?? throw new ArgumentException(
            "VALL-E X's NAR model trains with another sentence of the same speaker as its prompt; set PromptCodecTokens.",
            nameof(sample)), null, nameof(sample));
        var textLanguages = OnText ? Enumerable.Repeat(language, text.Length).ToArray() : null;
        return (_, _) => PaperCore.NarLossWithPrompt(text, prompt, codes, training, random, textLanguages);
    }

    /// <inheritdoc />
    /// <remarks>Eq. 3 and 4: the AR model reads the prompt's transcript and the text (the prompt's codes in the prompt's
    /// language, the new ones in the text's); the NAR reads the text alone.</remarks>
    protected override int[,] Generate(int[] target, int[] enrolled, int[,] prompt, TtsVoice<T> voice, Random random)
    {
        var core = PaperCore;
        var full = JoinPrompt(enrolled, target);
        var fullLanguages = TokenLanguages(full);
        var targetLanguages = TokenLanguages(target);
        int targetLanguage = targetLanguages.Length > 2 ? targetLanguages[1] : (int)VallEXLanguage.English;
        int promptLanguage = voice.LanguageId ?? (enrolled.Length > 0 ? fullLanguages[1] : targetLanguage);
        var first = core.ArGenerate(full, FirstCodebook(prompt), Settings.Temperature, Settings.TopK,
            Settings.MaxCodesPerTextToken * full.Length + 1, random, OnText ? fullLanguages : null, promptLanguage, targetLanguage);
        return core.NarGenerate(target, prompt, first.ToArray(), random, OnText ? targetLanguages : null);
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = IsNative ? "VALL-E-X-Native" : "VALL-E-X-ONNX",
            Description = "VALL-E X: Cross-Lingual Neural Codec Language Modeling (Zhang et al., 2023)",
            FeatureCount = Settings.HiddenDim,
            Complexity = Settings.NumEncoderLayers + Settings.NumDecoderLayers,
        };
        metadata.AdditionalInfo["Architecture"] = "VALL-E X";
        metadata.AdditionalInfo["LanguagePlacement"] = XOptions.LanguagePlacement.ToString();
        metadata.AdditionalInfo["SampleRate"] = Settings.SampleRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
