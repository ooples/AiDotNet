using System;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.FoundationModels.Models;
using AiDotNet.FoundationModels.Tokenizers;
using AiDotNet.Logging;

namespace AiDotNet.Factories
{
    /// <summary>
    /// Factory for creating specific foundation model instances
    /// </summary>
    public static class FoundationModelInstanceFactory
    {
        /// <summary>
        /// Creates a GPT model instance
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="variant">Model variant (e.g., "gpt2", "gpt2-medium", "gpt2-large")</param>
        /// <param name="tokenizer">Optional custom tokenizer</param>
        /// <param name="logger">Optional logger</param>
        /// <returns>GPT model instance</returns>
        public static async Task<GPTModel<T>> CreateGPTModelAsync<T>(
            string variant = "gpt2",
            ITokenizer? tokenizer = null,
            ILogging? logger = null)
        {
            var config = variant.ToLower() switch
            {
                "gpt2" or "gpt2-base" => GPTConfig.GPT2Base(),
                "gpt2-medium" => GPTConfig.GPT2Medium(),
                "gpt2-large" => GPTConfig.GPT2Large(),
                _ => throw new ArgumentException($"Unknown GPT variant: {variant}")
            };

            if (tokenizer == null)
            {
                tokenizer = new BPETokenizer("vocab.json", "merges.txt");
                await tokenizer.InitializeAsync();
            }

            var model = new GPTModel<T>(config, tokenizer, logger);
            // Model will initialize itself when first used via EnsureInitializedAsync()
            
            return model;
        }

        /// <summary>
        /// Creates a BERT model instance
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="variant">Model variant (e.g., "bert-base", "bert-large", "distilbert")</param>
        /// <param name="tokenizer">Optional custom tokenizer</param>
        /// <param name="logger">Optional logger</param>
        /// <returns>BERT model instance</returns>
        public static async Task<BERTModel<T>> CreateBERTModelAsync<T>(
            string variant = "bert-base",
            ITokenizer? tokenizer = null,
            ILogging? logger = null)
        {
            var config = variant.ToLower() switch
            {
                "bert-base" or "bert-base-uncased" => BERTConfig.BertBase(),
                "bert-large" or "bert-large-uncased" => BERTConfig.BertLarge(),
                "distilbert" or "distilbert-base-uncased" => BERTConfig.DistilBert(),
                _ => throw new ArgumentException($"Unknown BERT variant: {variant}")
            };

            if (tokenizer == null)
            {
                var doLowerCase = variant.Contains("uncased");
                tokenizer = new WordPieceTokenizer("vocab.txt", doLowerCase);
                await tokenizer.InitializeAsync();
            }

            var model = new BERTModel<T>(config, tokenizer, logger);
            // Model will initialize itself when first used via EnsureInitializedAsync()
            
            return model;
        }

        /// <summary>
        /// Creates a T5 model instance
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="variant">Model variant (e.g., "t5-small", "t5-base", "t5-large")</param>
        /// <param name="tokenizer">Optional custom tokenizer</param>
        /// <param name="logger">Optional logger</param>
        /// <returns>T5 model instance</returns>
        public static async Task<T5Model<T>> CreateT5ModelAsync<T>(
            string variant = "t5-base",
            ITokenizer? tokenizer = null,
            ILogging? logger = null)
        {
            var config = variant.ToLower() switch
            {
                "t5-small" => T5Config.T5Small(),
                "t5-base" => T5Config.T5Base(),
                "t5-large" => T5Config.T5Large(),
                _ => throw new ArgumentException($"Unknown T5 variant: {variant}")
            };

            if (tokenizer == null)
            {
                tokenizer = new SentencePieceTokenizer("spiece.model");
                await tokenizer.InitializeAsync();
            }

            var model = new T5Model<T>(config, tokenizer, logger);
            // Model will initialize itself when first used via EnsureInitializedAsync()
            
            return model;
        }

        /// <summary>
        /// Creates a foundation model by type and variant
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="modelType">Model type (GPT, BERT, T5)</param>
        /// <param name="variant">Model variant</param>
        /// <param name="logger">Optional logger</param>
        /// <returns>Foundation model instance</returns>
        public static async Task<IFoundationModel<T>> CreateModelAsync<T>(
            string modelType,
            string variant = "base",
            ILogging? logger = null)
        {
            return modelType.ToUpper() switch
            {
                "GPT" or "GPT2" => await CreateGPTModelAsync<T>(variant, logger: logger),
                "BERT" => await CreateBERTModelAsync<T>(variant, logger: logger),
                "T5" => await CreateT5ModelAsync<T>(variant, logger: logger),
                _ => throw new ArgumentException($"Unknown model type: {modelType}")
            };
        }

        /// <summary>
        /// Creates a model with automatic type detection from model ID
        /// </summary>
        /// <typeparam name="T">The numeric type used for calculations</typeparam>
        /// <param name="modelId">Model identifier (e.g., "gpt2-medium", "bert-base-uncased")</param>
        /// <param name="logger">Optional logger</param>
        /// <returns>Foundation model instance</returns>
        public static async Task<IFoundationModel<T>> CreateFromModelIdAsync<T>(
            string modelId,
            ILogging? logger = null)
        {
            var lowerModelId = modelId.ToLower();

            if (lowerModelId.Contains("gpt"))
            {
                return await CreateGPTModelAsync<T>(modelId, logger: logger);
            }
            else if (lowerModelId.Contains("bert"))
            {
                return await CreateBERTModelAsync<T>(modelId, logger: logger);
            }
            else if (lowerModelId.Contains("t5"))
            {
                return await CreateT5ModelAsync<T>(modelId, logger: logger);
            }
            else
            {
                throw new ArgumentException($"Cannot determine model type from ID: {modelId}");
            }
        }
    }
}