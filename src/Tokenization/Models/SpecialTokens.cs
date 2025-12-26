using System.Collections.Generic;

namespace AiDotNet.Tokenization.Models
{
    /// <summary>
    /// Represents special tokens used by tokenizers.
    /// </summary>
    public class SpecialTokens
    {
        /// <summary>
        /// Gets or sets the unknown token.
        /// </summary>
        public string UnkToken { get; set; } = "[UNK]";

        /// <summary>
        /// Gets or sets the padding token.
        /// </summary>
        public string PadToken { get; set; } = "[PAD]";

        /// <summary>
        /// Gets or sets the classification token (start of sequence).
        /// </summary>
        public string ClsToken { get; set; } = "[CLS]";

        /// <summary>
        /// Gets or sets the separation token.
        /// </summary>
        public string SepToken { get; set; } = "[SEP]";

        /// <summary>
        /// Gets or sets the mask token.
        /// </summary>
        public string MaskToken { get; set; } = "[MASK]";

        /// <summary>
        /// Gets or sets the beginning of sequence token.
        /// </summary>
        public string BosToken { get; set; } = "[BOS]";

        /// <summary>
        /// Gets or sets the end of sequence token.
        /// </summary>
        public string EosToken { get; set; } = "[EOS]";

        /// <summary>
        /// Gets or sets additional special tokens.
        /// </summary>
        public List<string> AdditionalSpecialTokens { get; set; } = new List<string>();

        /// <summary>
        /// Gets all special tokens as a list.
        /// </summary>
        public List<string> GetAllSpecialTokens()
        {
            var tokens = new List<string>();

            if (!string.IsNullOrEmpty(UnkToken)) tokens.Add(UnkToken);
            if (!string.IsNullOrEmpty(PadToken)) tokens.Add(PadToken);
            if (!string.IsNullOrEmpty(ClsToken)) tokens.Add(ClsToken);
            if (!string.IsNullOrEmpty(SepToken)) tokens.Add(SepToken);
            if (!string.IsNullOrEmpty(MaskToken)) tokens.Add(MaskToken);
            if (!string.IsNullOrEmpty(BosToken)) tokens.Add(BosToken);
            if (!string.IsNullOrEmpty(EosToken)) tokens.Add(EosToken);

            tokens.AddRange(AdditionalSpecialTokens);

            return tokens;
        }

        /// <summary>
        /// Creates BERT-style special tokens.
        /// </summary>
        public static SpecialTokens Bert() => new SpecialTokens
        {
            UnkToken = "[UNK]",
            PadToken = "[PAD]",
            ClsToken = "[CLS]",
            SepToken = "[SEP]",
            MaskToken = "[MASK]",
            BosToken = string.Empty,
            EosToken = string.Empty
        };

        /// <summary>
        /// Creates GPT-style special tokens.
        /// </summary>
        public static SpecialTokens Gpt() => new SpecialTokens
        {
            UnkToken = "<|endoftext|>",
            PadToken = "<|endoftext|>",
            BosToken = "<|endoftext|>",
            EosToken = "<|endoftext|>",
            ClsToken = string.Empty,
            SepToken = string.Empty,
            MaskToken = string.Empty
        };

        /// <summary>
        /// Creates T5-style special tokens.
        /// </summary>
        public static SpecialTokens T5() => new SpecialTokens
        {
            UnkToken = "<unk>",
            PadToken = "<pad>",
            EosToken = "</s>",
            BosToken = string.Empty,
            ClsToken = string.Empty,
            SepToken = string.Empty,
            MaskToken = string.Empty
        };

        /// <summary>
        /// Creates CLIP-style special tokens.
        /// </summary>
        /// <remarks>
        /// <para>
        /// CLIP uses a specific token format:
        /// - Start of text: <|startoftext|>
        /// - End of text: <|endoftext|>
        /// </para>
        /// <para><b>For Beginners:</b> CLIP uses special markers to tell the model
        /// where text begins and ends. These are similar to GPT tokens but with
        /// a distinct start token.
        /// </para>
        /// </remarks>
        public static SpecialTokens Clip() => new SpecialTokens
        {
            UnkToken = "<|endoftext|>",
            PadToken = "<|endoftext|>",
            BosToken = "<|startoftext|>",
            EosToken = "<|endoftext|>",
            ClsToken = "<|startoftext|>",
            SepToken = string.Empty,
            MaskToken = string.Empty
        };

        /// <summary>
        /// Creates default special tokens (BERT-style).
        /// </summary>
        public static SpecialTokens Default() => Bert();
    }
}
