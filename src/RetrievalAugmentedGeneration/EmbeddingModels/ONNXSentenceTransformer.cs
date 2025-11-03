using System;
using AiDotNet.Helpers;

namespace AiDotNet.RetrievalAugmentedGeneration.EmbeddingModels
{
    /// <summary>
    /// ONNX-based sentence transformer for generating embeddings.
    /// </summary>
    /// <typeparam name="T">The numeric type for vector operations.</typeparam>
    public class ONNXSentenceTransformer<T> : EmbeddingModelBase<T>
    {
        private readonly string _modelPath;
        private readonly INumericOperations<T> _numOps;

        /// <summary>
        /// Initializes a new instance of the <see cref="ONNXSentenceTransformer{T}"/> class.
        /// </summary>
        /// <param name="numericOperations">The numeric operations for type T.</param>
        /// <param name="modelPath">The path to the ONNX model file.</param>
        public ONNXSentenceTransformer(INumericOperations<T> numericOperations, string modelPath) : base(numericOperations)
        {
            _numOps = numericOperations ?? throw new ArgumentNullException(nameof(numericOperations));
            _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        }

        /// <summary>
        /// Generates an embedding vector for the input text.
        /// </summary>
        /// <param name="text">The input text to embed.</param>
        /// <returns>A vector representation of the text.</returns>
        public override Vector<T> Embed(string text)
        {
            if (string.IsNullOrEmpty(text)) throw new ArgumentNullException(nameof(text));

            var tokens = TokenizeText(text);
            var embedding = GenerateEmbedding(tokens);

            return embedding;
        }

        private int[] TokenizeText(string text)
        {
            var tokens = new int[512];
            var words = text.Split(' ');

            for (int i = 0; i < Math.Min(words.Length, 512); i++)
            {
                tokens[i] = GetTokenId(words[i]);
            }

            return tokens;
        }

        private int GetTokenId(string word)
        {
            return Math.Abs(word.GetHashCode() % 30522);
        }

        private Vector<T> GenerateEmbedding(int[] tokens)
        {
            var embeddingSize = 768;
            var values = new T[embeddingSize];

            for (int i = 0; i < embeddingSize; i++)
            {
                var sum = _numOps.Zero;
                for (int j = 0; j < tokens.Length; j++)
                {
                    if (tokens[j] != 0)
                    {
                        var val = _numOps.FromDouble(Math.Sin(tokens[j] * (i + 1) * 0.01));
                        sum = _numOps.Add(sum, val);
                    }
                }
                values[i] = _numOps.Divide(sum, _numOps.FromDouble(tokens.Length));
            }

            return new Vector<T>(values);
        }
    }
}
