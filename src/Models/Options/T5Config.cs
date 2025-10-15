namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration for T5 models
    /// </summary>
    public class T5Config
    {
        public int HiddenSize { get; set; } = 768;
        public int NumLayers { get; set; } = 12;
        public int NumHeads { get; set; } = 12;
        public int VocabSize { get; set; } = 32128;
        public int MaxPositionEmbeddings { get; set; } = 512;
        public int FFDim { get; set; } = 3072;
        public double DropoutRate { get; set; } = 0.1;
        public double LayerNormEpsilon { get; set; } = 1e-6;
        public double InitializerRange { get; set; } = 1.0;

        /// <summary>
        /// Creates configuration for T5 small model
        /// </summary>
        public static T5Config T5Small()
        {
            return new T5Config
            {
                HiddenSize = 512,
                NumLayers = 6,
                NumHeads = 8,
                FFDim = 2048,
                VocabSize = 32128
            };
        }

        /// <summary>
        /// Creates configuration for T5 base model
        /// </summary>
        public static T5Config T5Base()
        {
            return new T5Config
            {
                HiddenSize = 768,
                NumLayers = 12,
                NumHeads = 12,
                FFDim = 3072,
                VocabSize = 32128
            };
        }

        /// <summary>
        /// Creates configuration for T5 large model
        /// </summary>
        public static T5Config T5Large()
        {
            return new T5Config
            {
                HiddenSize = 1024,
                NumLayers = 24,
                NumHeads = 16,
                FFDim = 4096,
                VocabSize = 32128
            };
        }
    }
}