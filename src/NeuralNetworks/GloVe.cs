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
    /// GloVe (Global Vectors for Word Representation) neural network implementation.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// GloVe is an unsupervised learning algorithm for obtaining vector representations for words. 
    /// Training is performed on aggregated global word-word co-occurrence statistics from a corpus, 
    /// and the resulting representations showcase interesting linear substructures of the word vector space.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If Word2Vec is like a student learning from reading newspapers one page at a time, 
    /// GloVe is like a researcher who looks at the entire library all at once. It builds a giant table 
    /// showing how often every word in the dictionary appears near every other word. It then uses 
    /// clever math to find the best "address" for each word so that the distance between addresses 
    /// matches those counts perfectly.
    /// </para>
    /// <para>
    /// The GloVe model is famous for its ability to solve word analogies, like: 
    /// "King - Man + Woman = Queen."
    /// </para>
    /// </remarks>
    public class GloVe<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        #region Fields

        /// <summary>
        /// The number of unique words the model is capable of representing.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the size of the model's vocabulary. If this is 10,000, 
        /// the model has learned a unique coordinate for 10,000 different words.
        /// </remarks>
        private readonly int _vocabSize;

        /// <summary>
        /// The number of dimensions in the learned word vector space.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the number of coordinates used to describe each word. 
        /// More dimensions (like 300) allow the model to capture more subtle nuances in meaning 
        /// but make the model larger and slower.
        /// </remarks>
        private readonly int _embeddingDimension;

        /// <summary>
        /// The tokenizer used to map text strings to numerical IDs.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the translator that turns sentences into numbers the 
        /// model can understand.
        /// </remarks>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// The maximum number of tokens to process per input.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the sentence length limit. It keeps the model efficient 
        /// by only looking at the first part of very long texts.
        /// </remarks>
        private readonly int _maxTokens;

        /// <summary>
        /// The loss function used to evaluate training progress.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the scorekeeper. It tells the model how close its guessed 
        /// word distances are to the actual counts from the library.
        /// </remarks>
        private readonly ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimization algorithm used to refine the word vectors.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the coach that tells the model exactly how to move its 
        /// word coordinates to improve its score.
        /// </remarks>
        private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

        #endregion

        #region Properties

        /// <inheritdoc/>
        /// <remarks>
        /// <b>For Beginners:</b> This returns the number of coordinates used for each word's "address."
        /// </remarks>
        public int EmbeddingDimension => _embeddingDimension;

        /// <inheritdoc/>
        /// <remarks>
        /// <b>For Beginners:</b> This is the sentence length limit for the model.
        /// </remarks>
        public int MaxTokens => _maxTokens;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the GloVe embedding model.
        /// </summary>
        /// <param name="architecture">The configuration defining the model's metadata.</param>
        /// <param name="tokenizer">Optional tokenizer for text processing.</param>
        /// <param name="optimizer">Optional optimizer for training.</param>
        /// <param name="vocabSize">The size of the vocabulary (default: 10000).</param>
        /// <param name="embeddingDimension">The dimension of the word vectors (default: 100).</param>
        /// <param name="maxTokens">The maximum tokens per input (default: 512).</param>
        /// <param name="lossFunction">Optional loss function. Defaults to Mean Squared Error.</param>
        /// <param name="maxGradNorm">Maximum gradient norm for stability (default: 1.0).</param>
        /// <remarks>
        /// <b>For Beginners:</b> This constructor builds the framework for the model. You can 
        /// decide how many words it should know and how detailed its "dictionary" should be.
        /// </remarks>
        public GloVe(
            NeuralNetworkArchitecture<T> architecture,
            ITokenizer? tokenizer = null,
            IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
            int vocabSize = 10000,
            int embeddingDimension = 100,
            int maxTokens = 512,
            ILossFunction<T>? lossFunction = null,
            double maxGradNorm = 1.0)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
        {
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxTokens = maxTokens;
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

            InitializeLayers();
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Sets up the neural network layers required for the GloVe architecture.
        /// </summary>
        /// <remarks>
        /// <para>
        /// GloVe utilizes four primary learnable components: two embedding matrices (W and W_tilde) 
        /// and two bias vectors (b and b_tilde). This method initializes them as standard layers 
        /// to leverage the library's built-in GPU and AutoDiff support.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> This method creates the internal "books" the model uses to store its 
        /// knowledge. It creates two main lists of coordinates and two lists of "popularity scores" 
        /// for words, then combines them to find the most balanced representation.
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
                Layers.AddRange(LayerHelper<T>.CreateDefaultGloVeLayers(
                    Architecture,
                    _vocabSize,
                    _embeddingDimension));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Performs a forward pass to retrieve embeddings for given token IDs.
        /// </summary>
        /// <param name="input">A tensor containing token indices.</param>
        /// <returns>A tensor containing the resulting embeddings.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is the lookup process. You give the model word ID numbers, 
        /// and it returns the coordinates for those words from its internal memory.
        /// </remarks>
        public Tensor<T> Forward(Tensor<T> input)
        {
            // GPU optimization path
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
        /// Propagates error gradients backward through the model layers.
        /// </summary>
        /// <param name="outputGradient">The error signal from the loss function.</param>
        /// <returns>The calculated gradient for the input.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This method traces mistakes back to their source. It figures 
        /// out which word coordinates need to change to better match the real-world data.
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
        /// Updates the internal weights and biases of the model.
        /// </summary>
        /// <param name="parameters">The new coordinates and scores for the model.</param>
        /// <remarks>
        /// <b>For Beginners:</b> This method actually moves the words around on the map. 
        /// It updates the "addresses" of the words based on what it learned in the backward pass.
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
        /// returns their addresses (embeddings).
        /// </remarks>
        public override Tensor<T> Predict(Tensor<T> input)
        {
            return Forward(input);
        }

        /// <summary>
        /// Trains the model on a batch of word pairs and their co-occurrence counts.
        /// </summary>
        /// <param name="input">The word pair indices.</param>
        /// <param name="expectedOutput">The actual co-occurrence counts from the dataset.</param>
        /// <remarks>
        /// <b>For Beginners:</b> This is how the model gets smarter. You show it two words and 
        /// how often they appeared together in your data. The model then adjusts its "addresses" 
        /// for those words so the distances between them reflect that frequency.
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
        /// Clips gradients and uses the optimizer to update layer parameters.
        /// </summary>
        private void UpdateParameters(List<Tensor<T>> gradients)
        {
            ClipGradients(gradients);
            _optimizer.UpdateParameters(Layers);
        }

        /// <summary>
        /// Turns a sentence into a single, summary coordinate (embedding).
        /// </summary>
        /// <param name="text">The sentence or text to encode.</param>
        /// <returns>A normalized summary vector.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> If you give this a sentence like "I love technology," it finds 
        /// the address for every word, averages them all together, and finds the "geographic center" 
        /// of that sentence's meaning.
        /// </remarks>
        public Vector<T> Embed(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Vector<T>(_embeddingDimension);

            var tokenizer = _tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
            var tokenResult = tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);
            
            // GloVe uses W (layer 0) and W_tilde (layer 1). Final vectors are often W + W_tilde.
            var w = Layers[0].Forward(inputTensor);
            var w_tilde = Layers[1].Forward(inputTensor);
            
            var sumVector = new Vector<T>(_embeddingDimension);
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    T val = NumOps.Add(w[s, d], w_tilde[s, d]);
                    sumVector[d] = NumOps.Add(sumVector[d], val);
                }
            }

            var meanVector = new Vector<T>(_embeddingDimension);
            for (int d = 0; d < _embeddingDimension; d++)
            {
                meanVector[d] = NumOps.Divide(sumVector[d], NumOps.FromDouble(tokenIds.Count));
            }

            return meanVector.Normalize();
        }

        /// <summary>
        /// Encodes a whole batch of sentences at once for speed.
        /// </summary>
        /// <param name="texts">The collection of texts to encode.</param>
        /// <returns>A matrix where each row is an embedding vector.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is a high-speed way to process many sentences at the same time.
        /// </remarks>
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
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new GloVe<T>(
                Architecture,
                _tokenizer,
                _optimizer,
                _vocabSize,
                _embeddingDimension,
                _maxTokens,
                _lossFunction,
                Convert.ToDouble(MaxGradNorm));
        }

        /// <summary>
        /// Returns technical details and configuration info about the GloVe model.
        /// </summary>
        /// <returns>A metadata object containing vocabulary and dimension details.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            return new ModelMetadata<T>
            {
                Name = "GloVe",
                ModelType = ModelType.NeuralNetwork,
                Description = "GloVe (Global Vectors) embedding model",
                Complexity = ParameterCount,
                AdditionalInfo = new Dictionary<string, object>
                {
                    { "EmbeddingDimension", _embeddingDimension },
                    { "VocabSize", _vocabSize },
                    { "MaxTokens", _maxTokens }
                }
            };
        }

        /// <inheritdoc/>
        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(_vocabSize);
            writer.Write(_embeddingDimension);
            writer.Write(_maxTokens);
        }

        /// <inheritdoc/>
        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _ = reader.ReadInt32();
            _ = reader.ReadInt32();
            _ = reader.ReadInt32();
        }

        #endregion
    }
}
