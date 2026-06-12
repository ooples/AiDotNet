using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Agentic.Models.Local;

/// <summary>
/// Bridges a full repo <see cref="ITokenizer"/> to the engine's minimal <see cref="IGenerationTokenizer"/>
/// seam, so any AiDotNet tokenizer can drive <see cref="LocalEngineChatClient{T}"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The library's tokenizers do a lot more than generation needs. This adapter
/// exposes just the three things the generation loop uses (encode, decode, end-of-sequence id), so you can
/// plug a real tokenizer into the local engine without it depending on the larger tokenizer surface.
/// </para>
/// </remarks>
public sealed class TokenizerGenerationAdapter : IGenerationTokenizer
{
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Initializes a new adapter over the given tokenizer. The EOS id is resolved from the tokenizer's
    /// <see cref="AiDotNet.Tokenization.Models.SpecialTokens"/>.
    /// </summary>
    /// <param name="tokenizer">The tokenizer to wrap.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="tokenizer"/> is <c>null</c>.</exception>
    public TokenizerGenerationAdapter(ITokenizer tokenizer)
    {
        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;

        var eosToken = tokenizer.SpecialTokens.EosToken;
        if (eosToken is null || eosToken.Trim().Length == 0)
        {
            EosTokenId = -1;
            return;
        }

        var ids = tokenizer.ConvertTokensToIds(new List<string> { eosToken });
        EosTokenId = ids.Count > 0 ? ids[0] : -1;
    }

    /// <inheritdoc/>
    public int EosTokenId { get; }

    /// <inheritdoc/>
    public IReadOnlyList<int> Encode(string text)
    {
        Guard.NotNull(text);
        return _tokenizer.Encode(text).TokenIds;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<int> tokenIds)
    {
        Guard.NotNull(tokenIds);
        return _tokenizer.Decode(new List<int>(tokenIds), skipSpecialTokens: true);
    }
}
