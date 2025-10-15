namespace AiDotNet.Enums
{
    /// <summary>
    /// Specifies the type of tokenizer to use for text processing
    /// </summary>
    public enum TokenizerType
    {
        /// <summary>
        /// Character-level tokenizer that splits text into individual characters
        /// </summary>
        CharacterLevel,

        /// <summary>
        /// Word-level tokenizer that splits text by whitespace and punctuation
        /// </summary>
        WordLevel,

        /// <summary>
        /// Byte-Pair Encoding tokenizer (used by GPT models)
        /// </summary>
        BPE,

        /// <summary>
        /// WordPiece tokenizer (used by BERT models)
        /// </summary>
        WordPiece,

        /// <summary>
        /// SentencePiece tokenizer (used by T5, ALBERT models)
        /// </summary>
        SentencePiece,

        /// <summary>
        /// Unigram tokenizer
        /// </summary>
        Unigram,

        /// <summary>
        /// Byte-level BPE tokenizer (used by GPT-2, GPT-3)
        /// </summary>
        ByteLevelBPE,

        /// <summary>
        /// TikToken tokenizer (used by GPT-4, ChatGPT)
        /// </summary>
        TikToken,

        /// <summary>
        /// Claude tokenizer for Anthropic's Claude models
        /// </summary>
        Claude,

        /// <summary>
        /// Custom tokenizer implementation
        /// </summary>
        Custom,

        /// <summary>
        /// Automatically detect tokenizer type based on model
        /// </summary>
        Auto
    }
}