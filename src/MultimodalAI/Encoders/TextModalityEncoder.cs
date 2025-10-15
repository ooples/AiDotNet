using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;

namespace AiDotNet.MultimodalAI.Encoders
{
    /// <summary>
    /// Text-specific modality encoder for processing text data
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations</typeparam>
    public class TextModalityEncoder<T> : ModalityEncoderBase<T>
    {
        private readonly int _vocabularySize;
        private readonly int _maxSequenceLength;
        private readonly bool _useTfIdf;
        private readonly bool _useCharFeatures;
        private readonly Dictionary<string, int> _vocabulary;
        private readonly Dictionary<string, T> _idfScores;
        
        /// <summary>
        /// Initializes a new instance of TextModalityEncoder
        /// </summary>
        /// <param name="outputDimension">Output dimension of the encoder</param>
        /// <param name="vocabularySize">Maximum vocabulary size (default: 10000)</param>
        /// <param name="maxSequenceLength">Maximum sequence length (default: 512)</param>
        /// <param name="useTfIdf">Whether to use TF-IDF weighting (default: true)</param>
        /// <param name="useCharFeatures">Whether to extract character-level features (default: true)</param>
        /// <param name="encoder">Optional custom neural network encoder. If null, a default encoder will be created when needed.</param>
        public TextModalityEncoder(int outputDimension = 768,
            int vocabularySize = 10000, int maxSequenceLength = 512, 
            bool useTfIdf = true, bool useCharFeatures = true,
            INeuralNetworkModel<T>? encoder = null) 
            : base("Text", outputDimension, encoder)
        {
            _vocabularySize = vocabularySize;
            _maxSequenceLength = maxSequenceLength;
            _useTfIdf = useTfIdf;
            _useCharFeatures = useCharFeatures;
            _vocabulary = new Dictionary<string, int>();
            _idfScores = new Dictionary<string, T>();
        }

        /// <summary>
        /// Encodes text data into a vector representation
        /// </summary>
        /// <param name="input">Text data as string, string[], or List<string></param>
        /// <returns>Encoded vector representation</returns>
        public override Vector<T> Encode(object input)
        {
            if (!ValidateInput(input))
            {
                throw new ArgumentException($"Invalid input type for text encoding. Expected string or string collection, got {input?.GetType()?.Name ?? "null"}");
            }

            // Preprocess the input
            var preprocessed = Preprocess(input);
            var textData = preprocessed as TextData ?? throw new InvalidOperationException("Preprocessing failed");

            // Extract features
            var features = ExtractTextFeatures(textData);
            
            // Project to output dimension if needed
            if (features.Length != OutputDimension)
            {
                features = ProjectToOutputDimension(features);
            }

            // Normalize the output
            return Normalize(features);
        }

        /// <summary>
        /// Preprocesses raw text input
        /// </summary>
        public override object Preprocess(object input)
        {
            List<string> texts;

            switch (input)
            {
                case string singleText:
                    texts = new List<string> { singleText };
                    break;
                case string[] textArray:
                    texts = textArray.ToList();
                    break;
                case List<string> textList:
                    texts = textList;
                    break;
                case IEnumerable<string> textEnumerable:
                    texts = textEnumerable.ToList();
                    break;
                default:
                    throw new ArgumentException($"Unsupported input type: {input?.GetType()?.Name ?? "null"}");
            }

            var textData = new TextData
            {
                RawTexts = texts,
                ProcessedTexts = new List<string>(),
                Tokens = new List<List<string>>()
            };

            // Process each text
            foreach (var text in texts)
            {
                var processed = CleanText(text);
                textData.ProcessedTexts.Add(processed);
                
                var tokens = Tokenize(processed);
                textData.Tokens.Add(tokens);
            }

            return textData;
        }

        /// <summary>
        /// Validates the input for text encoding
        /// </summary>
        protected override bool ValidateInput(object input)
        {
            return input is string || input is string[] || 
                   input is List<string> || input is IEnumerable<string>;
        }

        /// <summary>
        /// Extracts text features from preprocessed data
        /// </summary>
        private Vector<T> ExtractTextFeatures(TextData textData)
        {
            var features = new List<T>();

            // Bag-of-words or TF-IDF features
            var bowFeatures = ExtractBagOfWordsFeatures(textData);
            features.AddRange(bowFeatures);

            // Statistical features
            var statFeatures = ExtractStatisticalFeatures(textData);
            features.AddRange(statFeatures);

            if (_useCharFeatures)
            {
                // Character-level features
                var charFeatures = ExtractCharacterFeatures(textData);
                features.AddRange(charFeatures);
            }

            // Semantic features (simplified)
            var semanticFeatures = ExtractSemanticFeatures(textData);
            features.AddRange(semanticFeatures);

            return new Vector<T>(features.ToArray());
        }

        /// <summary>
        /// Extracts bag-of-words or TF-IDF features
        /// </summary>
        private T[] ExtractBagOfWordsFeatures(TextData textData)
        {
            // Build vocabulary if needed
            if (_vocabulary.Count == 0)
            {
                BuildVocabulary(textData);
            }

            var features = new T[Math.Min(_vocabulary.Count, _vocabularySize)];

            foreach (var tokens in textData.Tokens)
            {
                var termFrequencies = new Dictionary<string, int>();
                
                // Count term frequencies
                foreach (var token in tokens)
                {
                    if (_vocabulary.ContainsKey(token))
                    {
                        if (!termFrequencies.ContainsKey(token))
                            termFrequencies[token] = 0;
                        termFrequencies[token]++;
                    }
                }

                // Apply TF-IDF if enabled
                foreach (var kvp in termFrequencies)
                {
                    int index = _vocabulary[kvp.Key];
                    if (index < features.Length)
                    {
                        T tf = _numericOps.FromDouble(kvp.Value / (double)tokens.Count);
                        T value = _useTfIdf && _idfScores.ContainsKey(kvp.Key) 
                            ? _numericOps.Multiply(tf, _idfScores[kvp.Key]) 
                            : tf;
                        features[index] = _numericOps.Add(features[index], value);
                    }
                }
            }

            // Average over all texts
            if (textData.Tokens.Count > 0)
            {
                T divisor = _numericOps.FromDouble(textData.Tokens.Count);
                for (int i = 0; i < features.Length; i++)
                {
                    features[i] = _numericOps.Divide(features[i], divisor);
                }
            }

            return features;
        }

        /// <summary>
        /// Extracts statistical features from text
        /// </summary>
        private T[] ExtractStatisticalFeatures(TextData textData)
        {
            var features = new List<T>();

            foreach (var text in textData.ProcessedTexts)
            {
                // Text length
                features.Add(_numericOps.FromDouble(text.Length));

                // Word count
                var words = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries);
                features.Add(_numericOps.FromDouble(words.Length));

                // Average word length
                features.Add(_numericOps.FromDouble(words.Length > 0 ? words.Average(w => w.Length) : 0));

                // Sentence count (simplified)
                var sentences = Regex.Split(text, @"[.!?]+").Length - 1;
                features.Add(_numericOps.FromDouble(Math.Max(1, sentences)));

                // Punctuation ratio
                var punctuationCount = text.Count(c => char.IsPunctuation(c));
                features.Add(_numericOps.FromDouble(text.Length > 0 ? punctuationCount / (double)text.Length : 0));

                // Digit ratio
                var digitCount = text.Count(char.IsDigit);
                features.Add(_numericOps.FromDouble(text.Length > 0 ? digitCount / (double)text.Length : 0));

                // Upper case ratio
                var upperCount = text.Count(char.IsUpper);
                features.Add(_numericOps.FromDouble(text.Length > 0 ? upperCount / (double)text.Length : 0));
            }

            // Average features across all texts
            int numTexts = textData.ProcessedTexts.Count;
            int featuresPerText = 7;
            var avgFeatures = new T[featuresPerText];

            if (numTexts > 0)
            {
                for (int i = 0; i < featuresPerText; i++)
                {
                    T sum = _numericOps.Zero;
                    for (int j = 0; j < numTexts; j++)
                    {
                        sum = _numericOps.Add(sum, features[j * featuresPerText + i]);
                    }
                    avgFeatures[i] = _numericOps.Divide(sum, _numericOps.FromDouble(numTexts));
                }
            }

            return avgFeatures;
        }

        /// <summary>
        /// Extracts character-level features
        /// </summary>
        private T[] ExtractCharacterFeatures(TextData textData)
        {
            var features = new List<T>();
            var charHistogram = new T[128]; // ASCII characters
            
            // Initialize histogram
            for (int i = 0; i < charHistogram.Length; i++)
            {
                charHistogram[i] = _numericOps.Zero;
            }

            foreach (var text in textData.ProcessedTexts)
            {
                foreach (char c in text)
                {
                    if (c < 128)
                    {
                        charHistogram[c] = _numericOps.Add(charHistogram[c], _numericOps.One);
                    }
                }
            }

            // Normalize histogram
            T total = _numericOps.Zero;
            for (int i = 0; i < charHistogram.Length; i++)
            {
                total = _numericOps.Add(total, charHistogram[i]);
            }
            
            if (_numericOps.GreaterThan(total, _numericOps.Zero))
            {
                for (int i = 0; i < charHistogram.Length; i++)
                {
                    charHistogram[i] = _numericOps.Divide(charHistogram[i], total);
                }
            }

            // Add top character frequencies
            features.AddRange(charHistogram.Take(32)); // Most common ASCII range

            return features.ToArray();
        }

        /// <summary>
        /// Extracts semantic features (simplified word embeddings)
        /// </summary>
        private T[] ExtractSemanticFeatures(TextData textData)
        {
            var features = new T[64]; // Simplified semantic space
            
            // Initialize features
            for (int i = 0; i < features.Length; i++)
            {
                features[i] = _numericOps.Zero;
            }
            var random = new Random(42); // Fixed seed for consistency

            foreach (var tokens in textData.Tokens)
            {
                foreach (var token in tokens)
                {
                    // Generate pseudo-embedding based on token hash
                    int hash = token.GetHashCode();
                    var rng = new Random(hash);
                    
                    for (int i = 0; i < features.Length; i++)
                    {
                        features[i] = _numericOps.Add(features[i], 
                            _numericOps.FromDouble((rng.NextDouble() - 0.5) * 2));
                    }
                }
            }

            // Average and normalize
            if (textData.Tokens.Sum(t => t.Count) > 0)
            {
                int totalTokens = textData.Tokens.Sum(t => t.Count);
                T divisor = _numericOps.FromDouble(totalTokens);
                for (int i = 0; i < features.Length; i++)
                {
                    features[i] = _numericOps.Divide(features[i], divisor);
                }
            }

            return features;
        }

        /// <summary>
        /// Projects features to the desired output dimension
        /// </summary>
        private Vector<T> ProjectToOutputDimension(Vector<T> features)
        {
            if (features.Length == OutputDimension)
                return features;

            var result = new T[OutputDimension];

            if (features.Length > OutputDimension)
            {
                // Use random projection for dimensionality reduction
                var projectionMatrix = GenerateRandomProjectionMatrix(features.Length, OutputDimension);
                for (int i = 0; i < OutputDimension; i++)
                {
                    T sum = _numericOps.Zero;
                    for (int j = 0; j < features.Length; j++)
                    {
                        sum = _numericOps.Add(sum, 
                            _numericOps.Multiply(_numericOps.FromDouble(projectionMatrix[i, j]), features[j]));
                    }
                    result[i] = sum;
                }
            }
            else
            {
                // Pad with learned embeddings
                Array.Copy(features.ToArray(), result, features.Length);
                
                // Fill remaining with small random values
                var random = new Random(42);
                for (int i = features.Length; i < OutputDimension; i++)
                {
                    result[i] = _numericOps.FromDouble((random.NextDouble() - 0.5) * 0.1);
                }
            }

            return new Vector<T>(result);
        }

        /// <summary>
        /// Cleans and normalizes text
        /// </summary>
        private string CleanText(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return string.Empty;

            // Convert to lowercase
            text = text.ToLowerInvariant();

            // Remove extra whitespace
            text = Regex.Replace(text, @"\s+", " ");

            // Remove special characters (keep alphanumeric and basic punctuation)
            text = Regex.Replace(text, @"[^\w\s.,!?;:'-]", " ");

            return text.Trim();
        }

        /// <summary>
        /// Tokenizes text into words
        /// </summary>
        private List<string> Tokenize(string text)
        {
            // Simple word tokenization
            var tokens = text.Split(new[] { ' ', '\t', '\n', '.', ',', '!', '?', ';', ':', '-' }, 
                StringSplitOptions.RemoveEmptyEntries);

            // Filter and limit length
            return tokens
                .Where(t => t.Length > 1 && t.Length < 20)
                .Take(_maxSequenceLength)
                .ToList();
        }

        /// <summary>
        /// Builds vocabulary from text data
        /// </summary>
        private void BuildVocabulary(TextData textData)
        {
            var wordCounts = new Dictionary<string, int>();

            foreach (var tokens in textData.Tokens)
            {
                foreach (var token in tokens)
                {
                    if (!wordCounts.ContainsKey(token))
                        wordCounts[token] = 0;
                    wordCounts[token]++;
                }
            }

            // Select top words by frequency
            var topWords = wordCounts
                .OrderByDescending(kvp => kvp.Value)
                .Take(_vocabularySize)
                .Select((kvp, index) => new { Word = kvp.Key, Index = index, Count = kvp.Value })
                .ToList();

            _vocabulary.Clear();
            _idfScores.Clear();

            int totalDocs = textData.Tokens.Count;
            
            foreach (var item in topWords)
            {
                _vocabulary[item.Word] = item.Index;
                
                // Calculate IDF score
                int docsWithWord = textData.Tokens.Count(tokens => tokens.Contains(item.Word));
                _idfScores[item.Word] = _numericOps.FromDouble(Math.Log((double)totalDocs / (1 + docsWithWord)));
            }
        }

        /// <summary>
        /// Generates a random projection matrix for dimensionality reduction
        /// </summary>
        private double[,] GenerateRandomProjectionMatrix(int inputDim, int outputDim)
        {
            var matrix = new double[outputDim, inputDim];
            var random = new Random(42); // Fixed seed for consistency

            // Initialize with Gaussian random values
            double scale = Math.Sqrt(2.0 / inputDim);
            
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    // Box-Muller transform for Gaussian distribution
                    double u1 = random.NextDouble();
                    double u2 = random.NextDouble();
                    double gaussian = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
                    matrix[i, j] = gaussian * scale;
                }
            }

            return matrix;
        }

        /// <summary>
        /// Internal class for storing text data
        /// </summary>
        private class TextData
        {
            public List<string> RawTexts { get; set; } = new List<string>();
            public List<string> ProcessedTexts { get; set; } = new List<string>();
            public List<List<string>> Tokens { get; set; } = new List<List<string>>();
        }
    }
}