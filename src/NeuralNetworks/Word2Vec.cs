using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
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
    /// Word2Vec neural network implementation supporting both Skip-Gram and CBOW architectures.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// Word2Vec is a foundational technique in Natural Language Processing (NLP) that learns to map words 
    /// to dense vectors of real numbers. These "embeddings" capture semantic and syntactic relationships 
    /// based on the contexts in which words appear.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Imagine you are learning a new language by looking at thousands of newspapers. 
    /// You notice that the word "bark" often appears near "dog," "tree," and "loud." You also notice that 
    /// "meow" appears near "cat," "kitten," and "soft." Word2Vec is an AI that does exactly this—it 
    /// builds a "map" of words where words with similar meanings or contexts are placed close together. 
    /// In this map, "dog" and "cat" might be neighbors, while "dog" and "spaceship" are on opposite ends.
    /// </para>
    /// <para>
    /// This implementation supports two main styles:
    /// <list type="bullet">
    /// <item><b>Skip-Gram:</b> Tries to guess the surrounding "context" words when given a single target word.</item>
    /// <item><b>CBOW:</b> Tries to guess a single "target" word when given a group of surrounding context words.</item>
    /// </list>
    /// </para>
    /// </remarks>
    public class Word2Vec<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        /// <summary>
        /// The number of unique words in the model's vocabulary.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the size of the "dictionary" the model knows. If the vocabulary size 
        /// is 10,000, the model has learned unique addresses for 10,000 different words.
        /// </remarks>
        private int _vocabSize;

        /// <summary>
        /// The length of the embedding vector for each word.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is how many "coordinates" we use to describe a word's location on our 
        /// mental map. More coordinates (e.g., 300 instead of 100) allow for more detail but require more 
        /// memory and processing power.
        /// </remarks>
        private int _embeddingDimension;

        /// <summary>
        /// The size of the context window used during training.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This defines how many words to the left and right of a target word the model 
        /// should pay attention to. A window of 5 means the model looks at 5 neighbors on each side.
        /// </remarks>
        private int _windowSize;

        /// <summary>
        /// The specific Word2Vec architecture type (Skip-Gram or CBOW).
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This determines the model's "learning strategy"—whether it tries to 
        /// predict context from a word (Skip-Gram) or a word from its context (CBOW).
        /// </remarks>
        private Word2VecType _type;

        /// <summary>
        /// The tokenizer used to convert text into numerical IDs.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the "translator" that turns a sentence like "I love AI" into 
        /// a list of numbers like [42, 105, 7]. The model can only understand these numbers.
        /// </remarks>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// The maximum number of tokens to process per input string.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is a safety limit to prevent the model from getting overwhelmed by 
        /// extremely long documents. It will only look at the first few hundred words of any text you give it.
        /// </remarks>
        private int _maxTokens;

        /// <summary>
        /// The loss function used to measure how well the model is learning.
        /// </summary>
        private ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimization algorithm used to update the model's parameters.
        /// </summary>
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        /// <inheritdoc/>
        /// <remarks>
        /// <b>For Beginners:</b> This tells you the size of the final list of numbers the model produces 
        /// for any given word or sentence.
        /// </remarks>
        public int EmbeddingDimension => _embeddingDimension;

        /// <inheritdoc/>
        /// <remarks>
        /// <b>For Beginners:</b> This is the maximum sentence length the model will process at one time.
        /// </remarks>
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the Word2Vec model.
        /// </summary>
        /// <param name="architecture">The architecture configuration defining the neural network's metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing. Defaults to a standard BERT-style tokenizer.</param>
        /// <param name="optimizer">Optional optimizer for training. Defaults to the Adam optimizer.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 10000).</param>
        /// <param name="embeddingDimension">The dimension of the embedding vectors (default: 100).</param>
        /// <param name="windowSize">The context window size (default: 5).</param>
        /// <param name="maxTokens">The maximum tokens to process per input (default: 512).</param>
        /// <param name="type">The Word2Vec architecture type (default: SkipGram).</param>
        /// <param name="lossFunction">Optional loss function. Defaults to Binary Cross Entropy.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        /// <remarks>
        /// <b>For Beginners:</b> This sets up the model's brain. You can decide how many words it should 
        /// know (vocabSize), how detailed its "mental map" should be (embeddingDimension), and which 
        /// learning strategy it should use (type).
        /// </remarks>
        public Word2Vec(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            int windowSize = 5,
            int maxTokens = 512,
            Word2VecType type = Word2VecType.SkipGram,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new BinaryCrossEntropyLoss<T>(), maxGradNorm)
        {
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _windowSize = windowSize;
            _maxTokens = maxTokens;
            _type = type;
            _lossFunction = lossFunction ?? new BinaryCrossEntropyLoss<T>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Initializes the layers of the Word2Vec network based on the provided architecture or standard research defaults.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This method sets up the "U" and "V" matrices described in the original Word2Vec paper. The "U" 
        /// matrix (the first layer) acts as the input lookup table, while the "V" matrix (the second layer) 
        /// acts as the context prediction head.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This method builds the actual structure of the model. It either uses 
        /// a custom setup you provide or, by default, creates a two-step process: first, look up the 
        /// numbers for a word, and second, use those numbers to try and guess other words.
        /// </para>
        /// </remarks>
        protected override void InitializeLayers()
        {
            InitializeLayersCore(true);
        }

        private void InitializeLayersCore(bool useVirtualValidation)
        {
            if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            {
                Layers.AddRange(Architecture.Layers);
                if (useVirtualValidation)
                {
                    ValidateCustomLayers(Layers);
                }
                else
                {
                    ValidateCustomLayersInternal(Layers);
                }
            }
            else
            {
                Layers.AddRange(LayerHelper<T>.CreateDefaultWord2VecLayers(
                    Architecture,
                    _vocabSize,
                    _embeddingDimension));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Performs a forward pass through the network, typically to retrieve an embedding for given token IDs.
        /// </summary>
        /// <param name="input">A tensor containing token indices.</param>
        /// <returns>A tensor containing the resulting embeddings.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the process of looking up a word's "address" on the model's 
        /// mental map. You give it the word's ID number, and it returns the list of coordinates (embedding) 
        /// for that word.
        /// </remarks>
        public Tensor<T> Forward(Tensor<T> input)
        {
            // GPU-resident optimization for high performance
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
        /// Performs a backward pass through the network to calculate gradients for training.
        /// </summary>
        /// <param name="outputGradient">The error gradient from the loss function.</param>
        /// <returns>The gradient calculated for the input.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> After the model makes a guess, we tell it how wrong it was. 
        /// This method takes that "wrongness" and works backward through the model's brain to 
        /// figure out exactly which neurons need to be adjusted to make a better guess next time.
        /// </remarks>
        public Tensor<T> Backward(Tensor<T> outputGradient)
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                outputGradient = Layers[i].Backward(outputGradient);
            }

            return outputGradient;
        }

        /// <summary>
        /// Updates the parameters of all layers in the network based on a provided update vector.
        /// </summary>
        /// <param name="parameters">A vector containing updated weights and biases.</param>
        /// <remarks>
        /// <b>For Beginners:</b> After we figure out how to improve (the backward pass), this method 
        /// actually changes the model's settings. It's like turning the knobs on a machine to fine-tune its 
        /// performance.
        /// </remarks>
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
        /// <remarks>
        /// <b>For Beginners:</b> This is identical to the Forward pass—it takes word IDs and 
        /// returns their embeddings.
        /// </remarks>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Forward(input);
        }

        /// <summary>
        /// Trains the Word2Vec model on a single batch of target and context pairs.
        /// </summary>
        /// <param name="input">The input tokens (targets or context depending on architecture).</param>
        /// <param name="expectedOutput">The expected tokens the model should predict.</param>
        /// <remarks>
        /// <b>For Beginners:</b> This is how the model learns. You show it a "puzzle"—a word and its 
        /// correct neighbor—and the model adjusts its internal map so that these two words 
        /// are placed closer together in its memory.
        /// </remarks>
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

        /// <summary>
        /// Applies gradient clipping and uses the optimizer to update layer parameters.
        /// </summary>
        private void UpdateParameters(List<Tensor<T>> gradients)
        {
            ClipGradients(gradients);
            _optimizer.UpdateParameters(Layers);
        }

        /// <summary>
        /// Encodes a single string into a normalized embedding vector by averaging its word vectors.
        /// </summary>
        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            // Extract embeddings from the first layer (The learned word vector table)
            if (Layers[0] is not EmbeddingLayer<T> embeddingLayer)
            {
                throw new InvalidOperationException("First layer must be an EmbeddingLayer<T> for word lookup.");
            }

            var tokenTensor = Tensor<T>.FromVector(
                new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray()),
                [tokenIds.Count]);
            var embeddings = embeddingLayer.Forward(tokenTensor);
            
            var sumVector = new Vector<T>(_embeddingDimension);
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    sumVector[d] = NumOps.Add(sumVector[d], embeddings[s, d]);
                }
            }

            var meanVector = new Vector<T>(_embeddingDimension);
            for (int d = 0; d < _embeddingDimension; d++)
            {
                meanVector[d] = NumOps.Divide(sumVector[d], NumOps.FromDouble(tokenIds.Count));
            }

            // Normalization ensures that "length" doesn't distort similarity comparisons
            return meanVector.SafeNormalize();
        }

        /// <inheritdoc/>
        public Task<Vector<T>> EmbedAsync(string text, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(Embed(text));
        }

        /// <summary>
        /// Encodes a collection of strings into a matrix of embedding vectors.
        /// </summary>
        /// <param name="texts">The collection of texts to encode.</param>
        /// <returns>A matrix where each row is the embedding for one input text.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is a faster way to process a whole list of sentences at once. 
        /// It gives you a "list of lists," where each sentence gets its own summary vector.
        /// </remarks>
        public Matrix<T> EmbedBatch(IEnumerable<string> texts)
        {
            var textList = texts.ToList();
            var matrix = new Matrix<T>(textList.Count, _embeddingDimension);

            for (int i = 0; i < textList.Count; i++)
            {
                var embedding = Embed(textList[i]);
                for (int j = 0; j < _embeddingDimension; j++)
                {
                    matrix[i, j] = embedding[j];
                }
            }

            return matrix;
        }

        /// <inheritdoc/>
        public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Task.FromResult(EmbedBatch(texts));
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new Word2Vec<T>(
                Architecture,
                _tokenizer,
                null, // Fresh optimizer for new instance
                _vocabSize,
                _embeddingDimension,
                _windowSize,
                _maxTokens,
                _type,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Retrieves detailed metadata about the Word2Vec model.
        /// </summary>
        /// <returns>A ModelMetadata object containing the model's configuration and complexity.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is like a "technical spec sheet" for the model. It tells you 
        /// exactly how many words it knows, how complex its map is, and what strategy it used to learn.
        /// </remarks>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "Word2Vec",
                ModelType = ModelType.NeuralNetwork,
                Description = $"Word2Vec ({_type}) embedding model",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "Type", _type.ToString() },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_windowSize);
            writer.Write(_maxTokens);
            writer.Write((int)_type);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _vocabSize = reader.ReadInt32();
            _embeddingDimension = reader.ReadInt32();
            _windowSize = reader.ReadInt32();
            _maxTokens = reader.ReadInt32();
            _type = (Word2VecType)reader.ReadInt32();
        }

        #endregion
    }
}
