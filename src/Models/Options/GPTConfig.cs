namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration for GPT models
    /// </summary>
    public class GPTConfig
    {
        /// <summary>
        /// Gets or sets the hidden size (dimension of the model)
        /// </summary>
        public int HiddenSize { get; set; } = 768;
        
        /// <summary>
        /// Gets or sets the number of transformer layers
        /// </summary>
        public int NumLayers { get; set; } = 12;
        
        /// <summary>
        /// Gets or sets the number of attention heads
        /// </summary>
        public int NumHeads { get; set; } = 12;
        
        /// <summary>
        /// Gets or sets the vocabulary size
        /// </summary>
        public int VocabSize { get; set; } = 50257;
        
        /// <summary>
        /// Gets or sets the maximum position embeddings
        /// </summary>
        public int MaxPositionEmbeddings { get; set; } = 1024;
        
        /// <summary>
        /// Gets or sets the feed-forward dimension
        /// </summary>
        public int FFDim { get; set; } = 3072;
        
        /// <summary>
        /// Gets or sets the dropout rate
        /// </summary>
        public double DropoutRate { get; set; } = 0.1;
        
        /// <summary>
        /// Gets or sets the layer normalization epsilon
        /// </summary>
        public double LayerNormEpsilon { get; set; } = 1e-5;
        
        /// <summary>
        /// Gets or sets the initializer range
        /// </summary>
        public double InitializerRange { get; set; } = 0.02;
        
        /// <summary>
        /// Creates configuration for GPT-2 base model
        /// </summary>
        public static GPTConfig GPT2Base()
        {
            return new GPTConfig
            {
                HiddenSize = 768,
                NumLayers = 12,
                NumHeads = 12,
                VocabSize = 50257,
                MaxPositionEmbeddings = 1024,
                FFDim = 3072
            };
        }
        
        /// <summary>
        /// Creates configuration for GPT-2 medium model
        /// </summary>
        public static GPTConfig GPT2Medium()
        {
            return new GPTConfig
            {
                HiddenSize = 1024,
                NumLayers = 24,
                NumHeads = 16,
                VocabSize = 50257,
                MaxPositionEmbeddings = 1024,
                FFDim = 4096
            };
        }
        
        /// <summary>
        /// Creates configuration for GPT-2 large model
        /// </summary>
        public static GPTConfig GPT2Large()
        {
            return new GPTConfig
            {
                HiddenSize = 1280,
                NumLayers = 36,
                NumHeads = 20,
                VocabSize = 50257,
                MaxPositionEmbeddings = 1024,
                FFDim = 5120
            };
        }
    }
}