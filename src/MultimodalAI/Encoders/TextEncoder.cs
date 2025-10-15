using AiDotNet.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Encoder for text modality
    /// </summary>
    public class TextEncoder : ModalityEncoder
    {
        private readonly int _vocabSize;
        private readonly int _maxSequenceLength;
        private readonly Dictionary<string, int> _vocabulary = default!;
        private readonly bool _useTfIdf;
        private readonly Dictionary<string, double> _idfWeights = default!;

        /// <summary>
        /// Initializes a new instance of TextEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder</param>
        /// <param name="vocabSize">Size of the vocabulary</param>
        /// <param name="maxSequenceLength">Maximum sequence length</param>
        /// <param name="useTfIdf">Whether to use TF-IDF weighting</param>
        public TextEncoder(int outputDimension, int vocabSize = 10000, int maxSequenceLength = 512, bool useTfIdf = false)
            : base("text", outputDimension)
        {
            _vocabSize = vocabSize;
            _maxSequenceLength = maxSequenceLength;
            _vocabulary = new Dictionary<string, int>();
            _useTfIdf = useTfIdf;
            _idfWeights = new Dictionary<string, double>();
            
            InitializeVocabulary();
        }

        /// <summary>
        /// Encodes text input into a vector representation
        /// </summary>
        /// <param name="input">Text input as string or string[]</param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<double> Encode(object input)
        {
            if (!ValidateInput(input))
                throw new ArgumentException("Invalid input type for text encoder");

            var preprocessed = Preprocess(input);
            var tokens = preprocessed as List<string> ?? new List<string>();

            // Create bag-of-words representation
            var bowVector = new Vector<double>(_vocabSize);
            foreach (var token in tokens)
            {
                if (_vocabulary.ContainsKey(token))
                {
                    int index = _vocabulary[token];
                    bowVector[index] += 1.0;
                }
            }

            // Apply TF-IDF if enabled
            if (_useTfIdf)
            {
                ApplyTfIdf(bowVector, tokens);
            }

            // Normalize the vector
            bowVector = Normalize(bowVector);

            // Project to output dimension if needed
            if (_vocabSize != _outputDimension)
            {
                var projectionMatrix = CreateProjectionMatrix(_vocabSize, _outputDimension);
                bowVector = Project(bowVector, projectionMatrix);
            }

            return bowVector;
        }

        /// <summary>
        /// Preprocesses text input
        /// </summary>
        /// <param name="input">Raw text input</param>
        /// <returns>List of tokens</returns>
        public override object Preprocess(object input)
        {
            string text = input switch
            {
                string s => s,
                string[] arr => string.Join(" ", arr),
                _ => input?.ToString() ?? ""
            };

            // Convert to lowercase
            text = text.ToLower();

            // Remove special characters and digits
            text = Regex.Replace(text, @"[^a-zA-Z\s]", " ");

            // Tokenize
            var tokens = text.Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries)
                            .Take(_maxSequenceLength)
                            .ToList();

            return tokens;
        }

        /// <summary>
        /// Validates the input for text encoding
        /// </summary>
        /// <param name="input">Input to validate</param>
        /// <returns>True if valid</returns>
        protected override bool ValidateInput(object input)
        {
            return input is string || input is string[] || input is IEnumerable<string>;
        }

        /// <summary>
        /// Applies TF-IDF weighting to the vector
        /// </summary>
        /// <param name="vector">Bag-of-words vector</param>
        /// <param name="tokens">List of tokens</param>
        private void ApplyTfIdf(Vector<double> vector, List<string> tokens)
        {
            double totalTokens = tokens.Count;
            
            for (int i = 0; i < vector.Dimension; i++)
            {
                if (vector[i] > 0)
                {
                    // Term frequency
                    double tf = vector[i] / totalTokens;
                    
                    // Inverse document frequency (simplified - would need corpus in real implementation)
                    double idf = Math.Log(10000.0 / (1.0 + vector[i]));
                    
                    vector[i] = tf * idf;
                }
            }
        }

        /// <summary>
        /// Initializes a basic vocabulary
        /// </summary>
        private void InitializeVocabulary()
        {
            // Initialize with common English words (simplified)
            var commonWords = new[]
            {
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
                "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
                "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
                "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
                "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
                "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
                "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
                "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
                "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
                "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
            };

            for (int i = 0; i < Math.Min(commonWords.Length, _vocabSize); i++)
            {
                _vocabulary[commonWords[i]] = i;
            }

            // Fill remaining vocabulary slots
            for (int i = commonWords.Length; i < _vocabSize; i++)
            {
                _vocabulary[$"token_{i}"] = i;
            }
        }

        /// <summary>
        /// Creates a random projection matrix
        /// </summary>
        /// <param name="inputDim">Input dimension</param>
        /// <param name="outputDim">Output dimension</param>
        /// <returns>Projection matrix</returns>
        private Matrix<double> CreateProjectionMatrix(int inputDim, int outputDim)
        {
            var random = new Random(42);
            var matrix = new Matrix<double>(outputDim, inputDim);
            
            // Xavier initialization
            double scale = Math.Sqrt(2.0 / (inputDim + outputDim));
            
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    matrix[i, j] = (random.NextDouble() * 2 - 1) * scale;
                }
            }
            
            return matrix;
        }

        /// <summary>
        /// Updates the vocabulary with new words
        /// </summary>
        /// <param name="words">New words to add</param>
        public void UpdateVocabulary(IEnumerable<string> words)
        {
            foreach (var word in words)
            {
                if (!_vocabulary.ContainsKey(word) && _vocabulary.Count < _vocabSize)
                {
                    _vocabulary[word] = _vocabulary.Count;
                }
            }
        }
    }
}