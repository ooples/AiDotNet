using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// FastText neural network implementation, an extension of Word2Vec that considers subword information.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// FastText is a library for learning of word representations and sentence classification. It improves 
    /// upon the original Word2Vec by representing each word as a bag of character n-grams. This approach 
    /// allows the model to compute word representations for words that did not appear in the training data 
    /// (out-of-vocabulary words).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most models see words like "playing" and "played" as completely different things. 
    /// FastText is smarter: it breaks words into pieces (like "play", "ing", and "ed"). Because it knows what 
    /// "play" means, it can guess the meaning of a new word like "player" even if it has never seen it before. 
    /// It's like a person who can understand a complex new word by looking at its root and its suffix.
    /// </para>
    /// </remarks>
    public class FastText<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        /// <summary>
        /// The size of the full-word vocabulary.
        /// </summary>
        private int _vocabSize;

        /// <summary>
        /// The number of "buckets" used to store subword (n-gram) information.
        /// </summary>
        private int _bucketSize;

        /// <summary>
        /// The dimensionality of the embedding vectors.
        /// </summary>
        private int _embeddingDimension;

        /// <summary>
        /// The tokenizer used to process text input.
        /// </summary>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// The maximum number of tokens to process per input string.
        /// </summary>
        private int _maxTokens;

        /// <summary>
        /// The loss function used during training.
        /// </summary>
        private ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimizer used to update the model's parameters.
        /// </summary>
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        /// <inheritdoc/>
        public int EmbeddingDimension => _embeddingDimension;

        /// <inheritdoc/>
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the FastText model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 10000).</param>
        /// <param name="bucketSize">The number of subword buckets (default: 2,000,000).</param>
        /// <param name="embeddingDimension">The dimension of the word vectors (default: 100).</param>
        /// <param name="maxTokens">The maximum tokens per sentence (default: 512).</param>
        /// <param name="lossFunction">Optional loss function. Defaults to Binary Cross Entropy.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        public FastText(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 10000,
            int bucketSize = 2000000,
            int embeddingDimension = 100,
            int maxTokens = 512,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), maxGradNorm)
        {
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _bucketSize = bucketSize;
            _embeddingDimension = embeddingDimension;
            _maxTokens = maxTokens;
            _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Configures the layers needed for FastText, including word and subword embedding tables.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method initializes:
        /// 1. A standard Word Embedding table (Layer 0).
        /// 2. A large N-gram Embedding table (Layer 1) using the hashing trick.
        /// 3. A projection head used for training.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This method builds the internal storage for the model. It creates 
        /// two main "dictionaries"—one for whole words and a much larger one for word fragments. 
        /// These are used together to understand sentences.
        /// </para>
        /// </remarks>
        protected override void InitializeLayers()
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                ValidateCustomLayers(Layers);
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultFastTextLayers(
                    Architecture,
                    _vocabSize,
                    _bucketSize,
                    _embeddingDimension));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Performs a forward pass to retrieve representations.
        /// </summary>
        public Tensor<T> Forward(Tensor<T> input)
        {
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            Tensor<T> output = input;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        /// <summary>
        /// Propagates error gradients backward for learning.
        /// </summary>
        public Tensor<T> Backward(Tensor<T> outputGradient)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                outputGradient = Layers[i].Backward(outputGradient);
            }

            return outputGradient;
        }

        /// <summary>
        /// Updates all trainable weights in the FastText model.
        /// </summary>
        public override void UpdateParameters(Vector<T> parameters)
        {
            int index = 0;
            foreach (var layer in Layers)
            {
                int layerParameterCount = layer.ParameterCount;
                if (layerParameterCount > 0)
                {
                    var layerParameters = parameters.Slice(index, layerParameterCount);
                    layer.UpdateParameters(layerParameters);
                    index += layerParameterCount;
                }
            }
        }

        /// <inheritdoc/>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Forward(input);
        }

        /// <summary>
        /// Trains the model on a single step of data using standard backpropagation.
        /// </summary>
        public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        {
            var prediction = Predict(input);
            LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

            var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
            var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

            var gradients = new List<Tensor<T>>();
            var currentGradient = outputGradientTensor;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                currentGradient = Layers[i].Backward(currentGradient);
                gradients.Insert(0, currentGradient);
            }

            UpdateParameters(gradients);
        }

        private void UpdateParameters(List<Tensor<T>> gradients)
        {
            ClipGradients(gradients);
            _optimizer.UpdateParameters(Layers);
        }

        /// <summary>
        /// Turns text into a robust embedding vector using both word and subword information.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the final step where the model summarizes your text. 
        /// It looks at the meaning of every word and the meaning of every word fragment (like roots 
        /// and suffixes), averages them all together, and gives you one final "meaning coordinate" 
        /// for the entire sentence.
        /// </remarks>
        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();
            var tokens = tokenResult.Tokens.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            // FastText uses word embeddings (layer 0) and n-gram embeddings (layer 1)
            var wordEmbeds = Layers[0].Forward(inputTensor);
            
            var sumVector = new Vector<T>(_embeddingDimension);
            int totalComponents = 0;

            for (int s = 0; s < tokenIds.Count; s++)
            {
                // Add word embedding
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    sumVector[d] = NumOps.Add(sumVector[d], wordEmbeds[s, d]);
                }
                totalComponents++;

                // Add n-gram embeddings
                var ngrams = GetCharacterNGrams(tokens[s], 3, 6);
                if (ngrams.Count > 0)
                {
                    var ngramIndices = ngrams.Select(ng => (double)(Math.Abs(ng.GetHashCode()) % _bucketSize)).ToArray();
                    var ngramInputTensor = Tensor<T>.FromVector(new Vector<T>(ngramIndices), [ngrams.Count]);
                    var ngramEmbeds = Layers[1].Forward(ngramInputTensor);

                    for (int n = 0; n < ngrams.Count; n++)
                    {
                        for (int d = 0; d < _embeddingDimension; d++)
                        {
                            sumVector[d] = NumOps.Add(sumVector[d], ngramEmbeds[n, d]);
                        }
                        totalComponents++;
                    }
                }
            }

            var meanVector = new Vector<T>(_embeddingDimension);
            for (int d = 0; d < _embeddingDimension; d++)
            {
                meanVector[d] = NumOps.Divide(sumVector[d], NumOps.FromDouble(totalComponents));
            }

            // Normalization ensures that "length" doesn't distort similarity comparisons
            return meanVector.SafeNormalize();
        }

        /// <inheritdoc/>
        public Task<Vector<T>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
        }

        private List<string> GetCharacterNGrams(string word, int minN, int maxN)
        {
            var ngrams = new List<string>();
            string decoratedWord = "<" + word + ">";
            
            for (int n = minN; n <= maxN; n++)
            {
                for (int i = 0; i <= decoratedWord.Length - n; i++)
                {
                    ngrams.Add(decoratedWord.Substring(i, n));
                }
            }

            return ngrams;
        }

        /// <summary>
        /// Encodes a batch of texts for high-throughput processing.
        /// </summary>
        /// <param name="texts">The texts to encode.</param>
        /// <returns>A matrix where each row is the embedding for one input text.</returns>
        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var result = new Matrix<T>(textList.Count, _embeddingDimension);

            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    result[i, j] = embedding[j];
                }
            }

            return result;
        }

        /// <inheritdoc/>
        public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new FastText<T>(
                Architecture,
                _tokenizer,
                null,
                _vocabSize,
                _bucketSize,
                _embeddingDimension,
                _maxTokens,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves detailed metadata about the FastText model configuration.
        /// </summary>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "FastText",
                ModelType = ModelType.NeuralNetwork,
                Description = "FastText embedding model with subword support",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "BucketSize", _bucketSize },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_bucketSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxTokens);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _vocabSize = reader.ReadInt32();
            _bucketSize = reader.ReadInt32();
            _embeddingDimension = reader.ReadInt32();
            _maxTokens = reader.ReadInt32();
        }

        #endregion
    }
}