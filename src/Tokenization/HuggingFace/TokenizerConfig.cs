using System.Collections.Generic;
using Newtonsoft.Json;

namespace AiDotNet.Tokenization.HuggingFace
{
    /// <summary>
    /// Configuration for HuggingFace tokenizers.
    /// </summary>
    public class TokenizerConfig
    {
        /// <summary>
        /// Gets or sets the tokenizer type.
        /// </summary>
        [JsonProperty("tokenizer_class")]
        public string? TokenizerClass { get; set; }

        /// <summary>
        /// Gets or sets the model type.
        /// </summary>
        [JsonProperty("model_type")]
        public string? ModelType { get; set; }

        /// <summary>
        /// Gets or sets the vocabulary file.
        /// </summary>
        [JsonProperty("vocab_file")]
        public string? VocabFile { get; set; }

        /// <summary>
        /// Gets or sets the merges file (for BPE).
        /// </summary>
        [JsonProperty("merges_file")]
        public string? MergesFile { get; set; }

        /// <summary>
        /// Gets or sets special tokens.
        /// </summary>
        [JsonProperty("unk_token")]
        public string? UnkToken { get; set; }

        [JsonProperty("pad_token")]
        public string? PadToken { get; set; }

        [JsonProperty("cls_token")]
        public string? ClsToken { get; set; }

        [JsonProperty("sep_token")]
        public string? SepToken { get; set; }

        [JsonProperty("mask_token")]
        public string? MaskToken { get; set; }

        [JsonProperty("bos_token")]
        public string? BosToken { get; set; }

        [JsonProperty("eos_token")]
        public string? EosToken { get; set; }

        /// <summary>
        /// Gets or sets whether to lowercase input.
        /// </summary>
        [JsonProperty("do_lower_case")]
        public bool DoLowerCase { get; set; }

        /// <summary>
        /// Gets or sets the model max length.
        /// </summary>
        [JsonProperty("model_max_length")]
        public int? ModelMaxLength { get; set; }

        /// <summary>
        /// Gets or sets additional special tokens.
        /// </summary>
        [JsonProperty("additional_special_tokens")]
        public List<string>? AdditionalSpecialTokens { get; set; }
    }
}
