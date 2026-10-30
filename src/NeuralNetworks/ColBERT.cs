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
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.NeuralNetworks
{
    /// <summary>
    /// ColBERT (Contextualized Late Interaction over BERT) neural network implementation.
    /// Uses token-level representations for high-precision document retrieval.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
    /// <remarks>
    /// <para>
    /// ColBERT is a highly efficient and accurate retrieval model that keeps a separate vector for every 
    /// token in a sentence. It calculates the similarity between a query and a document using a 
    /// "Late Interaction" MaxSim operator, allowing it to capture fine-grained semantic matches.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Most AI search models are like people who read a whole book and then 
    /// try to summarize it in just one word. ColBERT is like a person who keeps detailed notes 
    /// on every single word. When you ask a question, ColBERT compares every word in your question 
    /// to every word in the document notes. This is much more accurate because no information 
    /// is "lost" during summarization.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
    /// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 32 });
    /// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
    /// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new ColBERT&lt;float&gt;(architecture))
    ///     .Build(trainX, trainY);
    /// var embedding = result.Predict(input);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Language)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.Transformer)]
    [ModelCategory(ModelCategory.EmbeddingModel)]
    [ModelTask(ModelTask.Embedding)]
    [ModelTask(ModelTask.Ranking)]
    [ModelComplexity(ModelComplexity.High)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", "https://arxiv.org/abs/2004.12832", Year = 2020, Authors = "Omar Khattab, Matei Zaharia")]
    public partial class ColBERT<T> : TransformerEmbeddingNetwork<T>
    {
        /// <inheritdoc/>
        /// <remarks>Its layers are sized from its options, not its architecture (#2147).</remarks>
        protected override bool SizesLayersFromOptions => true;

        private readonly ColBERTOptions _options;

        #region Fields

        /// <summary>
        /// The dimension of each token-level embedding vector (typically 128).
        /// </summary>
        /// <remarks>
        /// <b>For Beginners:</b> This is the size of the "fingerprint" the model creates for 
        /// every single word. Because we keep many fingerprints per sentence, we make them 
        /// smaller (like 128 instead of 768) to save space while still being very precise.
        /// </remarks>
        private int _outputDim;

        private int _vocabSize;
        private int _numLayers;
        private int _numHeads;
        private int _feedForwardDim;
        private readonly ITokenizer _tokenizer;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance with default architecture settings.
        /// </summary>
        public ColBERT()
            : this(new NeuralNetworkArchitecture<T>(
                inputType: Enums.InputType.OneDimensional,
                taskType: Enums.NeuralNetworkTaskType.Regression,
                inputSize: 768,
                // The output is each token's OutputDimension-wide (128) late-interaction vector (#2147).
                outputSize: new ColBERTOptions().OutputDimension))
        {
        }

        /// <summary>
        /// Initializes a new instance of the ColBERT model.
        /// </summary>
        public ColBERT(
        NeuralNetworkArchitecture<T> architecture,
        ColBERTOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
            : base(architecture, options, tokenizer, optimizer, lossFunction)
        {
        _options = options ?? new ColBERTOptions();
        _options.Validate();
            Options = _options;

            _outputDim = _options.OutputDimension;
            _vocabSize = _options.VocabSize;
            _numLayers = _options.NumLayers;
            _numHeads = _options.NumHeads;
            _feedForwardDim = _options.FeedForwardDim;
            // ColBERT is BERT-based, and its query augmentation pads with BERT's [MASK]; the WordPiece tokenizer
            // carries that token, where the GPT-style default has none.
            _tokenizer = tokenizer ?? Tokenization.LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.RoBERTa);

            InitializeLayersCore(false);
        }

        #endregion

        #region Initialization

        /// <summary>
        /// Sets up the transformer layers and the token-level projection head for ColBERT.
        /// </summary>
        protected override void InitializeLayers()
        {
            InitializeLayersCore(true);
        }

        private void InitializeLayersCore(bool useVirtualValidation)
        {
            // Defensive ClearLayers — see SGPT/BGE for context (TE-base layer
            // doubling bug).
            ClearLayers();

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
                Layers.AddRange(LayerHelper<T>.CreateDefaultColBERTLayers(
                    _vocabSize,
                    _outputDim,
                    MaxTokens,
                    _numLayers,
                    _numHeads,
                    _feedForwardDim,
                    hiddenDimension: EmbeddingDimension));
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Encodes text into a multi-vector matrix where each row is a contextualized token embedding.
        /// </summary>
        public Matrix<T> EmbedLateInteraction(string text)
        {
            if (string.IsNullOrWhiteSpace(text))
                return new Matrix<T>(0, _outputDim);

            var tokenResult = _tokenizer.Encode(text);
            var tokens = tokenResult.TokenIds.Take(MaxTokens).ToList();
            if (tokens.Count == 0) tokens.Add(0);

            return EncodeTokens(tokens);
        }

        /// <summary>
        /// Encodes a search query the way ColBERT's query encoder does: exactly
        /// <see cref="ColBERTOptions.MaxQueryLength"/> token embeddings.
        /// </summary>
        /// <param name="query">The query text.</param>
        /// <returns>A MaxQueryLength x OutputDimension matrix of normalized token embeddings, or an empty
        /// matrix for a blank query.</returns>
        /// <remarks>
        /// <para>
        /// Khattab &amp; Zaharia 2020, section 3.2: a query longer than N<sub>q</sub> tokens is truncated, and a
        /// shorter one is padded with [MASK] tokens up to N<sub>q</sub>. The masks are encoded like any other
        /// token, so they become learned query-expansion terms that take part in MaxSim scoring. Documents
        /// are not padded; encode them with <see cref="EmbedLateInteraction(string)"/>.
        /// </para>
        /// <para><b>For Beginners:</b> Use this for the search text and <see cref="EmbedLateInteraction(string)"/>
        /// for the passages being searched, then compare them with <see cref="LateInteractionScore"/>.</para>
        /// </remarks>
        /// <exception cref="InvalidOperationException">The tokenizer has no mask token to pad with.</exception>
        public Matrix<T> EmbedQueryLateInteraction(string query)
        {
            if (string.IsNullOrWhiteSpace(query))
                return new Matrix<T>(0, _outputDim);

            int queryLength = _options.MaxQueryLength;
            var tokens = _tokenizer.Encode(query).TokenIds.Take(queryLength).ToList();
            if (tokens.Count < queryLength)
            {
                string mask = _tokenizer.SpecialTokens.MaskToken;
                if (string.IsNullOrEmpty(mask) || !_tokenizer.Vocabulary.ContainsToken(mask))
                {
                    throw new InvalidOperationException(
                        "ColBERT pads queries with the tokenizer's mask token, and this tokenizer has none. " +
                        "Use a BERT-style tokenizer (one with [MASK]).");
                }

                int maskId = _tokenizer.Vocabulary.GetTokenId(mask);
                while (tokens.Count < queryLength) tokens.Add(maskId);
            }

            return EncodeTokens(tokens);
        }

        private Matrix<T> EncodeTokens(List<int> tokens)
        {
            var inputTensor = Tensor<T>.FromVector(new Vector<T>(tokens.Select(id => NumOps.FromDouble(id)).ToArray()), [1, tokens.Count]);
            var output = Predict(inputTensor); // [1, seqLen, outputDim]

            int seqLen = output.Shape[1];
            var result = new Matrix<T>(seqLen, _outputDim);
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < _outputDim; d++)
                {
                    result[s, d] = output[0, s, d];
                }
            }

            // Normalization is critical for the MaxSim dot-product scoring
            for (int s = 0; s < seqLen; s++)
            {
                var row = result.GetRow(s);
                result.SetRow(s, row.SafeNormalize());
            }

            return result;
        }

        /// <summary>
        /// Fallback method that encodes a sentence into a single summary vector (mean-pooled).
        /// </summary>
        /// <param name="text">The text to encode.</param>
        /// <returns>A summary vector for the input text.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is a fallback option. While ColBERT works best when it keeps 
        /// all its notes (as a table), sometimes you just want one summary list of numbers. 
        /// This method averages all the word-level info into one overall summary.
        /// </remarks>
        public override Vector<T> Embed(string text)
        {
            var matrix = EmbedLateInteraction(text);
            if (matrix.Rows == 0) return new Vector<T>(_outputDim);

            var result = new Vector<T>(_outputDim);
            for (int d = 0; d < _outputDim; d++)
            {
                T sum = NumOps.Zero;
                for (int s = 0; s < matrix.Rows; s++)
                {
                    sum = NumOps.Add(sum, matrix[s, d]);
                }
                result[d] = NumOps.Divide(sum, NumOps.FromDouble(matrix.Rows));
            }

            return result.SafeNormalize();
        }

        /// <summary>
        /// Computes the similarity score between a query and document matrix using the MaxSim interaction.
        /// </summary>
        /// <param name="queryEmbeddings">The token-level embeddings for the query.</param>
        /// <param name="docEmbeddings">The token-level embeddings for the document.</param>
        /// <returns>A scalar interaction score.</returns>
        /// <remarks>
        /// <b>For Beginners:</b> This is how ColBERT compares a question to a document. It looks 
        /// at every word in your question and finds the absolute "best match" for it in the 
        /// entire document. It then combines all those best matches into one final score.
        /// </remarks>
        public T LateInteractionScore(Matrix<T> queryEmbeddings, Matrix<T> docEmbeddings)
        {
            T totalScore = NumOps.Zero;

            for (int q = 0; q < queryEmbeddings.Rows; q++)
            {
                var queryVec = queryEmbeddings.GetRow(q);
                T maxSim = NumOps.MinValue;

                for (int d = 0; d < docEmbeddings.Rows; d++)
                {
                    var docVec = docEmbeddings.GetRow(d);
                    T sim = Engine.DotProduct(queryVec, docVec);
                    if (NumOps.GreaterThan(sim, maxSim))
                    {
                        maxSim = sim;
                    }
                }

                totalScore = NumOps.Add(totalScore, maxSim);
            }

            return totalScore;
        }

        /// <summary>
        /// Retrieves metadata about the ColBERT model.
        /// </summary>
        /// <returns>Metadata containing model type and output dimensionality information.</returns>
        public override ModelMetadata<T> GetModelMetadata()
        {
            var metadata = base.GetModelMetadata();
            metadata.Name = "ColBERT";
            metadata.Description = "ColBERT late interaction multi-vector embedding model";
            metadata.AdditionalInfo["OutputDimension"] = _outputDim;
            return metadata;
        }





        #endregion
    }
}
