namespace AiDotNet.Tokenization.Configuration
{
    /// <summary>
    /// Specifies pretrained tokenizer models available from HuggingFace Hub.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are industry-standard tokenizers that have been trained
    /// on large text corpora. Each is designed for different use cases:
    ///
    /// - BERT models: Best for understanding text (classification, Q&amp;A, NER)
    /// - GPT models: Best for text generation
    /// - RoBERTa: Improved BERT with better training
    /// - T5: Versatile text-to-text model
    /// - DistilBERT: Faster, smaller BERT
    /// </remarks>
    public enum PretrainedTokenizerModel
    {
        /// <summary>
        /// BERT Base Uncased - The default choice for most NLP tasks.
        /// Vocabulary: 30,522 tokens. Case-insensitive.
        /// </summary>
        BertBaseUncased,

        /// <summary>
        /// BERT Base Cased - Preserves case information.
        /// Best when capitalization matters (e.g., named entity recognition).
        /// </summary>
        BertBaseCased,

        /// <summary>
        /// BERT Large Uncased - Larger model with better accuracy.
        /// Vocabulary: 30,522 tokens. More compute intensive.
        /// </summary>
        BertLargeUncased,

        /// <summary>
        /// BERT Large Cased - Large model preserving case.
        /// Best accuracy for case-sensitive tasks.
        /// </summary>
        BertLargeCased,

        /// <summary>
        /// GPT-2 - OpenAI's text generation model.
        /// Vocabulary: 50,257 tokens. Best for text generation.
        /// </summary>
        Gpt2,

        /// <summary>
        /// GPT-2 Medium - Larger GPT-2 variant.
        /// Better quality generation, more compute required.
        /// </summary>
        Gpt2Medium,

        /// <summary>
        /// GPT-2 Large - Even larger GPT-2 variant.
        /// High quality generation for demanding applications.
        /// </summary>
        Gpt2Large,

        /// <summary>
        /// RoBERTa Base - Robustly optimized BERT.
        /// Often outperforms BERT on benchmarks.
        /// </summary>
        RobertaBase,

        /// <summary>
        /// RoBERTa Large - Large RoBERTa model.
        /// State-of-the-art performance on many tasks.
        /// </summary>
        RobertaLarge,

        /// <summary>
        /// DistilBERT Base Uncased - Distilled BERT (40% smaller, 60% faster).
        /// Good balance of speed and accuracy.
        /// </summary>
        DistilBertBaseUncased,

        /// <summary>
        /// DistilBERT Base Cased - Distilled BERT preserving case.
        /// Fast and case-sensitive.
        /// </summary>
        DistilBertBaseCased,

        /// <summary>
        /// T5 Small - Text-to-Text Transfer Transformer (small).
        /// Versatile for many NLP tasks.
        /// </summary>
        T5Small,

        /// <summary>
        /// T5 Base - Text-to-Text Transfer Transformer (base).
        /// Good balance of performance and efficiency.
        /// </summary>
        T5Base,

        /// <summary>
        /// T5 Large - Text-to-Text Transfer Transformer (large).
        /// High performance for complex tasks.
        /// </summary>
        T5Large,

        /// <summary>
        /// ALBERT Base v2 - A Lite BERT with parameter sharing.
        /// Much smaller model size with competitive performance.
        /// </summary>
        AlbertBaseV2,

        /// <summary>
        /// XLNet Base Cased - Autoregressive pretraining.
        /// Strong performance on long-context tasks.
        /// </summary>
        XlnetBaseCased,

        /// <summary>
        /// Electra Small - Efficient pretraining approach.
        /// Very efficient for its size.
        /// </summary>
        ElectraSmall,

        /// <summary>
        /// Electra Base - Efficient pretraining approach (base size).
        /// Good accuracy with efficient training.
        /// </summary>
        ElectraBase,

        /// <summary>
        /// CodeBERT Base - BERT for programming languages.
        /// Best for code understanding tasks.
        /// </summary>
        CodeBertBase,

        /// <summary>
        /// Microsoft CodeBERT - Multi-language code model.
        /// Supports multiple programming languages.
        /// </summary>
        MicrosoftCodeBert,

        /// <summary>
        /// GraphCodeBERT - Code model with data flow.
        /// Enhanced code understanding with graph structure.
        /// </summary>
        GraphCodeBert
    }

    /// <summary>
    /// Extension methods for PretrainedTokenizerModel enum.
    /// </summary>
    public static class PretrainedTokenizerModelExtensions
    {
        /// <summary>
        /// Converts the enum value to the HuggingFace Hub model identifier.
        /// </summary>
        /// <param name="model">The pretrained model enum value.</param>
        /// <returns>The HuggingFace Hub model identifier string.</returns>
        public static string ToModelId(this PretrainedTokenizerModel model)
        {
            return model switch
            {
                PretrainedTokenizerModel.BertBaseUncased => "bert-base-uncased",
                PretrainedTokenizerModel.BertBaseCased => "bert-base-cased",
                PretrainedTokenizerModel.BertLargeUncased => "bert-large-uncased",
                PretrainedTokenizerModel.BertLargeCased => "bert-large-cased",
                PretrainedTokenizerModel.Gpt2 => "gpt2",
                PretrainedTokenizerModel.Gpt2Medium => "gpt2-medium",
                PretrainedTokenizerModel.Gpt2Large => "gpt2-large",
                PretrainedTokenizerModel.RobertaBase => "roberta-base",
                PretrainedTokenizerModel.RobertaLarge => "roberta-large",
                PretrainedTokenizerModel.DistilBertBaseUncased => "distilbert-base-uncased",
                PretrainedTokenizerModel.DistilBertBaseCased => "distilbert-base-cased",
                PretrainedTokenizerModel.T5Small => "t5-small",
                PretrainedTokenizerModel.T5Base => "t5-base",
                PretrainedTokenizerModel.T5Large => "t5-large",
                PretrainedTokenizerModel.AlbertBaseV2 => "albert-base-v2",
                PretrainedTokenizerModel.XlnetBaseCased => "xlnet-base-cased",
                PretrainedTokenizerModel.ElectraSmall => "google/electra-small-discriminator",
                PretrainedTokenizerModel.ElectraBase => "google/electra-base-discriminator",
                PretrainedTokenizerModel.CodeBertBase => "microsoft/codebert-base",
                PretrainedTokenizerModel.MicrosoftCodeBert => "microsoft/codebert-base",
                PretrainedTokenizerModel.GraphCodeBert => "microsoft/graphcodebert-base",
                _ => "bert-base-uncased" // Fallback to default
            };
        }
    }
}
