using AiDotNet.Attributes;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class Voicebox<T>
{
    // The materialized dataset inventory is model state, not merely a mutable
    // constructor option. Keep it as a declared dictionary like other phone maps.
    private Dictionary<string, int>? _phoneVocabulary;

    private void InitializePhoneFrontend()
    {
        _phoneVocabulary = _options.PhoneVocabulary is null ? null
            : new Dictionary<string, int>(_options.PhoneVocabulary, StringComparer.Ordinal);
        _tokenizer = new VoiceboxPhoneTokenizer(_options.PhoneVocabularySize, _phoneVocabulary);
    }
    /// <summary>Supplies MFA-compatible phones for each word, before word-position suffixes and ghost silences are added.
    /// Pair this external frontend with the corresponding <see cref="VoiceboxOptions.PhoneVocabulary"/>.
    /// The delegate must be supplied again after external deserialization.</summary>
    [ExternalState]
    public Func<string, IReadOnlyList<IReadOnlyList<string>>>? MfaWordPhonemizer { get; set; }

    /// <summary>Sets a dataset/checkpoint's exact MFA phone inventory without changing its embedding dimensions.</summary>
    public void SetPhoneVocabulary(IReadOnlyDictionary<string, int> vocabulary)
    {
        ThrowIfDisposed();
        if (vocabulary is null) throw new ArgumentNullException(nameof(vocabulary));
        var copy = vocabulary.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);
        var tokenizer = new VoiceboxPhoneTokenizer(_options.PhoneVocabularySize, copy);
        _phoneVocabulary = copy;
        _options.PhoneVocabulary = new Dictionary<string, int>(copy, StringComparer.Ordinal);
        _tokenizer = tokenizer;
    }

    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _options.PhoneVocabulary = _phoneVocabulary is null ? null : new Dictionary<string, int>(_phoneVocabulary, StringComparer.Ordinal);
        _tokenizer = new VoiceboxPhoneTokenizer(_options.PhoneVocabularySize, _phoneVocabulary);
    }
}
