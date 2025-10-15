using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration specific to BERT models.
    /// </summary>
    public class BERTConfig : FoundationModelConfig
    {
        /// <summary>
        /// Number of transformer layers
        /// </summary>
        public int NumLayers { get; set; } = 12;

        /// <summary>
        /// Number of attention heads
        /// </summary>
        public int NumHeads { get; set; } = 12;

        /// <summary>
        /// Hidden size dimension
        /// </summary>
        public int HiddenSize { get; set; } = 768;

        /// <summary>
        /// Intermediate size in feed-forward layers
        /// </summary>
        public int IntermediateSize { get; set; } = 3072;

        /// <summary>
        /// Maximum position embeddings
        /// </summary>
        public int MaxPositionEmbeddings { get; set; } = 512;

        /// <summary>
        /// Vocabulary size
        /// </summary>
        public int VocabSize { get; set; } = 30522;

        /// <summary>
        /// Type vocabulary size (for segment embeddings)
        /// </summary>
        public int TypeVocabSize { get; set; } = 2;

        /// <summary>
        /// Hidden dropout probability
        /// </summary>
        public double HiddenDropoutProb { get; set; } = 0.1;

        /// <summary>
        /// Attention dropout probability
        /// </summary>
        public double AttentionDropoutProb { get; set; } = 0.1;

        /// <summary>
        /// Layer normalization epsilon
        /// </summary>
        public double LayerNormEps { get; set; } = 1e-12;

        /// <summary>
        /// Hidden activation function
        /// </summary>
        public string HiddenAct { get; set; } = "gelu";

        /// <summary>
        /// Creates configuration for BERT base model
        /// </summary>
        public static BERTConfig BertBase()
        {
            return new BERTConfig
            {
                ModelId = "bert-base",
                NumLayers = 12,
                NumHeads = 12,
                HiddenSize = 768,
                IntermediateSize = 3072,
                MaxPositionEmbeddings = 512,
                VocabSize = 30522,
                TypeVocabSize = 2,
                HiddenDropoutProb = 0.1,
                AttentionDropoutProb = 0.1,
                LayerNormEps = 1e-12,
                HiddenAct = "gelu",
                MaxSequenceLength = 512
            };
        }

        /// <summary>
        /// Creates configuration for BERT large model
        /// </summary>
        public static BERTConfig BertLarge()
        {
            return new BERTConfig
            {
                ModelId = "bert-large",
                NumLayers = 24,
                NumHeads = 16,
                HiddenSize = 1024,
                IntermediateSize = 4096,
                MaxPositionEmbeddings = 512,
                VocabSize = 30522,
                TypeVocabSize = 2,
                HiddenDropoutProb = 0.1,
                AttentionDropoutProb = 0.1,
                LayerNormEps = 1e-12,
                HiddenAct = "gelu",
                MaxSequenceLength = 512
            };
        }

        /// <summary>
        /// Creates configuration for DistilBERT model
        /// </summary>
        public static BERTConfig DistilBert()
        {
            return new BERTConfig
            {
                ModelId = "distilbert-base-uncased",
                NumLayers = 6,
                NumHeads = 12,
                HiddenSize = 768,
                IntermediateSize = 3072,
                MaxPositionEmbeddings = 512,
                VocabSize = 30522,
                TypeVocabSize = 0, // DistilBERT doesn't use segment embeddings
                HiddenDropoutProb = 0.1,
                AttentionDropoutProb = 0.1,
                LayerNormEps = 1e-12,
                HiddenAct = "gelu",
                MaxSequenceLength = 512
            };
        }
    }
}