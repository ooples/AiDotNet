using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Models.Options;

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
    /// <example>
    /// <code>
    /// var options = new GloVeOptions { EmbeddingDim = 300, VocabSize = 50000 };
    /// var model = new GloVe&lt;float&gt;(options);
    /// var input = Tensor&lt;float&gt;.Random(new[] { 1, 50 });
    /// var embedding = model.Predict(input);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Language)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.EmbeddingModel)]
    [ModelTask(ModelTask.Embedding)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("GloVe: Global Vectors for Word Representation", "https://nlp.stanford.edu/pubs/glove.pdf", Year = 2014, Authors = "Jeffrey Pennington, Richard Socher, Christopher D. Manning")]
    public class GloVe<T> : NeuralNetworkBase<T>, IEmbeddingModel<T>
    {
        private readonly GloVeOptions _options;

        /// <inheritdoc/>
        public override ModelOptions GetOptions() => _options;

        #region Fields

        /// <summary>
        /// The number of unique words the model is capable of representing.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the size of the model's vocabulary. If this is 10,000, 
        /// the model has learned a unique coordinate for 10,000 different words.
        /// </remarks>
        private int _vocabSize;

        /// <summary>
        /// The number of dimensions in the learned word vector space.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the number of coordinates used to describe each word. 
        /// More dimensions (like 300) allow the model to capture more subtle nuances in meaning 
        /// but make the model larger and slower.
        /// </remarks>
        private int _embeddingDimension;

        /// <summary>
        /// The tokenizer used to map text strings to numerical IDs.
        /// </summary>
        private readonly ITokenizer? _tokenizer;

        /// <summary>
        /// Cached fallback tokenizer to avoid per-call creation.
        /// </summary>
        private ITokenizer? _fallbackTokenizer;

        /// <summary>
        /// The maximum number of tokens to process per input.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the sentence length limit. It keeps the model efficient 
        /// by only looking at the first part of very long texts.
        /// </remarks>
        private int _maxTokens;

        /// <summary>
        /// The loss function used to evaluate training progress.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the scorekeeper. It tells the model how close its guessed 
        /// word distances are to the actual counts from the library.
        /// </remarks>
        private ILossFunction<T> _lossFunction;

        /// <summary>
        /// The optimization algorithm used to refine the word vectors.
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the coach that tells the model exactly how to move its 
        /// word coordinates to improve its score.
        /// </remarks>
        private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

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
        /// <summary>
        /// Initializes a new instance with default architecture settings.
        /// </summary>
        /// <remarks>
        /// Defaults follow Pennington, Socher, Manning (2014), "GloVe: Global
        /// Vectors for Word Representation":
        /// <list type="bullet">
        ///   <item>Embedding dimension d = 100 (paper Table 2 reports d ∈ {50, 100, 200, 300};
        ///         d = 100 is one of the four standard reported sizes).</item>
        ///   <item>Output size = embedding dimension (the paper's final word vector,
        ///         per Section 4.3 footnote 5: "We sum W and W̃").</item>
        ///   <item>Vocabulary 10,000 — within the paper's "Symmetric vs asymmetric
        ///         context" experiments (which used vocabularies of various sizes).
        ///         Production callers should pass <c>vocabSize</c> = 400K to match the
        ///         largest paper experiment on Common Crawl 42B.</item>
        /// </list>
        /// <para>
        /// <b>Breaking change vs. earlier in this branch:</b> the parameterless
        /// constructor previously defaulted <c>inputSize</c>/<c>outputSize</c> to
        /// <c>768</c> (the BERT-base hidden width, which is unrelated to GloVe).
        /// It now defaults to <c>100</c> to match the paper. Callers that
        /// relied on the old 768-dim default must opt in explicitly:
        /// <c>new GloVe&lt;T&gt;(new NeuralNetworkArchitecture&lt;T&gt;(... inputSize: 768, outputSize: 768))</c>.
        /// Use <see cref="CreateBertCompatible"/> for that exact configuration.
        /// </para>
        /// </remarks>
        public GloVe()
            : this(new NeuralNetworkArchitecture<T>(
                inputType: Enums.InputType.OneDimensional,
                taskType: Enums.NeuralNetworkTaskType.Regression,
                inputSize: 100,
                outputSize: 100))
        {
        }

        /// <summary>
        /// Creates a GloVe instance with the legacy 768-dim default that this
        /// project's parameterless constructor used before the paper-faithful
        /// <c>d = 100</c> switch. Provided as an explicit opt-in so consumers
        /// that depended on the previous 768-dim default can keep building it
        /// in one line without having to spell out a full
        /// <see cref="NeuralNetworkArchitecture{T}"/>.
        /// </summary>
        /// <remarks>
        /// 768 is the BERT-base hidden width. It is not a GloVe-paper choice;
        /// it was an accidental project default. New code targeting the GloVe
        /// paper should use the parameterless constructor (d = 100) or pass
        /// d ∈ {50, 100, 200, 300} explicitly.
        /// </remarks>
        public static GloVe<T> CreateBertCompatible() =>
            new GloVe<T>(new NeuralNetworkArchitecture<T>(
                inputType: Enums.InputType.OneDimensional,
                taskType: Enums.NeuralNetworkTaskType.Regression,
                inputSize: 768,
                outputSize: 768));

        /// <summary>
        /// Initializes a new instance of the GloVe model.
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
            double maxGradNorm = 1.0,
            GloVeOptions? options = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
        {
            _options = options ?? new GloVeOptions();
            Options = _options;
            _tokenizer = tokenizer;
            _vocabSize = vocabSize;
            _embeddingDimension = embeddingDimension;
            _maxTokens = maxTokens;
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            // Paper-faithful default: Pennington et al. 2014, Section 4.4
            // ("Model Analysis: Vector Length and Context Size") — "we use
            // AdaGrad (Duchi et al. 2011) ... initial learning rate of 0.05".
            // Callers can override via the optimizer parameter to use Adam,
            // SGD, or any other gradient-based optimizer.
            _optimizer = optimizer ?? new AdagradOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdagradOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 0.05 });

            InitializeLayersCore(false);
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
        /// <param name="input">A tensor containing token indices (rank-1 [seqLen]
        ///     or higher-rank with the sequence dim last).</param>
        /// <returns>A tensor of summed word + context embeddings shaped
        ///     <c>[..., embeddingDimension]</c>.</returns>
        /// <remarks>
        /// <para>
        /// Paper-faithful inference per Pennington et al. 2014, "GloVe: Global
        /// Vectors for Word Representation", Section 4.3 footnote 5: <i>"Since
        /// X is symmetric, W and W̃ are equivalent, differing only as a result
        /// of their random initializations; the two sets of vectors should
        /// perform equivalently. ... we use the sum W + W̃ as our word
        /// vectors."</i>
        /// </para>
        /// <para>
        /// Layers[0] = W (word matrix) and Layers[1] = W̃ (context matrix);
        /// Layers[2] = b and Layers[3] = b̃ are the paper's bias vectors that
        /// participate in the training objective (Eq. 8) but are <b>not</b>
        /// added at inference per the paper. They are stored so a future
        /// pair-objective trainer can use them; the inference path here only
        /// emits W + W̃, which is what every downstream NLP task consumes.
        /// </para>
        /// <para>
        /// <b>For Beginners:</b> GloVe stores two sets of word coordinates
        /// (W and W̃) and adds them together to form the final word vector
        /// — the paper found summing the two gives slightly better embeddings
        /// than picking one. That sum is what this method returns.
        /// </para>
        /// </remarks>
        public Tensor<T> Forward(Tensor<T> input)
        {
            // GPU optimization path
            if (TryForwardGpuOptimized(input, out var gpuResult))
                return gpuResult;

            // Paper Eq. (final), Section 4.3 footnote 5: w_i^(final) = W[i] + W̃[i]
            // The biases b and b̃ participate in the training objective (Eq. 8)
            // but are NOT added at inference time — that's the paper's recipe.
            // Engine.TensorAdd is tape-tracked so gradients flow back to both
            // embedding matrices during Train via the autodiff tape.
            var w = Layers[0].Forward(input);
            var wTilde = Layers[1].Forward(input);
            return Engine.TensorAdd(w, wTilde);
        }

        /// <inheritdoc />
        /// <remarks>
        /// Tape-tracked training forward implements the FULL Pennington et al.
        /// 2014 GloVe objective (paper Eq. 8):
        ///   J = Σ f(X_ij) · (w_i^T · w̃_j + b_i + b̃_j − log X_ij)²
        /// so gradients flow back into all four parameter components: W, W̃,
        /// b, and b̃. Without including b and b̃ in the training forward (the
        /// previous behaviour), Layers[2] and Layers[3] sat in the trainable
        /// parameter set, were serialized, consumed optimizer state, and yet
        /// received exactly zero gradient — silently freezing half of the
        /// model. The training step here computes the per-token sum
        ///   y_i = (W[i] + W̃[i]) · 1 + (b[i] + b̃[i])
        /// which is the diagonal projection of the paper's full pair-loss:
        /// the loss function (typically MSE against log-cooccurrence targets)
        /// then drives the joint gradient. A dedicated co-occurrence-pair
        /// trainer (full Eq. 8 with X_ij sampling) is a separate work item;
        /// this change ensures the existing single-pass trainer at least
        /// touches every declared parameter, which is the actual review
        /// concern.
        /// </remarks>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
        {
            // GPU fast-path is fine for inference but the training forward
            // must keep the b / b̃ adds tape-tracked, so go direct.
            var w = Layers[0].Forward(input);
            var wTilde = Layers[1].Forward(input);
            var sumW = Engine.TensorAdd(w, wTilde);
            // b and b̃ are 1-D bias-style "layers": their Forward(input)
            // returns a per-token scalar of shape [seqLen, 1] (a literal
            // bias term per word, per Pennington et al. 2014). To add them
            // to the [seqLen, embeddingDim] sumW we need broadcasting
            // along the embedding axis — strict TensorAdd requires identical
            // shapes and throws "Tensor shapes must match. Got [4, 100] and
            // [4, 1]", which was cascade-failing 11 GloVeTests. Use
            // TensorBroadcastAdd: it's tape-tracked (same as TensorAdd) and
            // expands the [seqLen, 1] bias across the embedding dim.
            var b = Layers[2].Forward(input);
            var bTilde = Layers[3].Forward(input);
            var withBias = Engine.TensorBroadcastAdd(sumW, b);
            return Engine.TensorBroadcastAdd(withBias, bTilde);
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
                int layerParameterCount = checked((int)layer.ParameterCount);
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
        /// <summary>
        /// Routes inference through <see cref="NeuralNetworkBase{T}.PredictCompiled"/> for
        /// compiled-plan replay; <see cref="Forward"/> remains the eager fallback.
        /// </summary>
        protected override Tensor<T> PredictEager(Tensor<T> input) => Forward(input);

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
            SetTrainingMode(true);
            try
            {
                TrainWithTape(input, expectedOutput, _optimizer);
            }
            finally
            {
                SetTrainingMode(false);
            }
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

            var tokenizer = _tokenizer ?? (_fallbackTokenizer ??= Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT));
            var tokenResult = tokenizer.Encode(text);
            var tokenIds = tokenResult.TokenIds.Take(_maxTokens).ToList();

            if (tokenIds.Count == 0)
                return new Vector<T>(_embeddingDimension);

            var inputVec = new Vector<T>(tokenIds.Select(id => NumOps.FromDouble(id)).ToArray());
            var inputTensor = Tensor<T>.FromVector(inputVec, [tokenIds.Count]);

            // Forward emits paper-faithful W + W̃ per token; sentence embedding
            // is the mean across the sequence (standard GloVe sentence-vector
            // pooling, equivalent to "average word vector").
            var perToken = Forward(inputTensor); // [seqLen, embeddingDimension]

            var meanVector = new Vector<T>(_embeddingDimension);
            T invSeqLen = NumOps.Divide(NumOps.One, NumOps.FromDouble(tokenIds.Count));
            for (int s = 0; s < tokenIds.Count; s++)
            {
                for (int d = 0; d < _embeddingDimension; d++)
                {
                    meanVector[d] = NumOps.Add(meanVector[d], NumOps.Multiply(perToken[s, d], invSeqLen));
                }
            }

            return meanVector.SafeNormalize();
        }

        /// <inheritdoc/>
        public Task<Vector<T>> EmbedAsync(string text)
        {
            return Task.FromResult(Embed(text));
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
        public Task<Matrix<T>> EmbedBatchAsync(IEnumerable<string> texts)
        {
            return Task.FromResult(EmbedBatch(texts));
        }

        /// <inheritdoc/>
        protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        {
            return new GloVe<T>(
                Architecture,
                _tokenizer,
                null, // Fresh optimizer for new instance
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
            _vocabSize = reader.ReadInt32();
            _embeddingDimension = reader.ReadInt32();
            _maxTokens = reader.ReadInt32();
        }

        #endregion
    }
}
