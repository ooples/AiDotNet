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
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
    /// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 50 });
    /// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
    /// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
    /// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
    ///     .ConfigureModel(new GloVe&lt;float&gt;(architecture))
    ///     .Build(trainX, trainY);
    /// var embedding = result.Predict(input);
    /// </code>
    /// </example>
    [ModelDomain(ModelDomain.Language)]
    [ModelCategory(ModelCategory.NeuralNetwork)]
    [ModelCategory(ModelCategory.EmbeddingModel)]
    [ModelTask(ModelTask.Embedding)]
    [ModelComplexity(ModelComplexity.Low)]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("GloVe: Global Vectors for Word Representation", "https://nlp.stanford.edu/pubs/glove.pdf", Year = 2014, Authors = "Jeffrey Pennington, Richard Socher, Christopher D. Manning")]
    public partial class GloVe<T> : TextEmbeddingModelLayoutBase<T>, IEmbeddingModel<T>, ITrainingObjectiveProvider<T>
    {
        private readonly GloVeOptions _options;

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
        GloVeOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
            : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), options?.MaxGradNorm ?? 1.0)
        {
        _options = options ?? new GloVeOptions();
        _options.Validate();
            Options = _options;
            _tokenizer = tokenizer;
            _vocabSize = _options.VocabSize;
            _embeddingDimension = _options.EmbeddingDimension;
            _maxTokens = _options.MaxSequenceLength;
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
        /// participate in the training objective (Eq. 8, see
        /// <see cref="ForwardForTraining"/>) but are <b>not</b> added at
        /// inference per the paper; the inference path here only emits W + W̃,
        /// which is what every downstream NLP task consumes.
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

            // Paper Eq. (final), Section 4.3 footnote 5: w_i^(final) = W[i] + W̃[i]
            // The biases b and b̃ participate in the training objective (Eq. 8)
            // but are NOT added at inference time — that's the paper's recipe.
            // Engine.TensorAdd is tape-tracked so gradients flow back to both
            // embedding matrices during Train via the autodiff tape.
            var w = Layers[0].Forward(input);
            var wTilde = Layers[1].Forward(input);
            return Engine.TensorAdd(w, wTilde);
        }

        /// <summary>
        /// The paper's training objective (Pennington et al. 2014, Eq. 8): the weighted least-squares fit of
        /// <c>w_i · w̃_j + b_i + b̃_j</c> to <c>log X_ij</c> over co-occurring word pairs, weighted by
        /// <c>f(X_ij) = min(1, (X_ij / x_max)^α)</c>.
        /// </summary>
        /// <param name="input">The pairs <see cref="PrepareTrainingPair"/> builds: [pairs, 3] rows of
        /// (i, j, sqrt f(X_ij)).</param>
        /// <returns>[pairs] values <c>sqrt f(X_ij) · (w_i · w̃_j + b_i + b̃_j)</c>. Against the target
        /// <c>sqrt f(X_ij) · log X_ij</c> the squared error is exactly the f-weighted Eq. 8 term, so the configured
        /// loss (mean squared error by default) is J divided by the pair count.</returns>
        public override Tensor<T> ForwardForTraining(Tensor<T> input)
        {
            if (input.Shape.Length != 2 || input.Shape[1] != 3)
            {
                throw new ArgumentException(
                    $"GloVe's training forward takes [pairs, 3] rows of (i, j, sqrt f(X_ij)) from PrepareTrainingPair; got [{string.Join(", ", input.Shape.ToArray())}].",
                    nameof(input));
            }

            int pairs = input.Shape[0];
            Tensor<T> Column(int c) => Engine.Reshape(Engine.TensorSlice(input, [0, c], [pairs, 1]), [pairs]);
            var i = Column(0);
            var j = Column(1);
            var dot = Engine.ReduceSum(Engine.TensorMultiply(Layers[0].Forward(i), Layers[1].Forward(j)), [1], keepDims: false);
            var b = Engine.Reshape(Layers[2].Forward(i), [pairs]);
            var bTilde = Engine.Reshape(Layers[3].Forward(j), [pairs]);
            return Engine.TensorMultiply(Engine.TensorAdd(Engine.TensorAdd(dot, b), bTilde), Column(2));
        }

        /// <summary>
        /// Turns the caller's (input, target) into Eq. 8's weighted co-occurrence pairs.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Two inputs are accepted. Explicit pairs: an [N, 2] input of (i, j) word indices with a target of N values
        /// <c>log X_ij</c> trains on those co-occurrences as given. Otherwise the input is token sequences (the
        /// sequence axis last, as <see cref="Forward"/> takes) and the target is not used: X is counted from the
        /// sequences with a symmetric window of <see cref="GloVeOptions.WindowSize"/> words, a pair d words apart
        /// adding 1/d (Section 4.2), and every co-occurring pair becomes one row.
        /// </para>
        /// <para>
        /// The weighting <c>f</c> is applied here, outside the parameter graph: each row is (i, j, sqrt f(X_ij)) and
        /// its target <c>sqrt f(X_ij) · log X_ij</c>.
        /// </para>
        /// </remarks>
        protected override (Tensor<T> Input, Tensor<T>? Target) PrepareTrainingPair(Tensor<T> input, Tensor<T>? target, long draw)
        {
            var counts = new List<(int I, int J, double LogX)>();
            if (IsExplicitPairs(input, target))
            {
                for (int r = 0; r < input.Shape[0]; r++)
                    counts.Add((TokenId(input[r, 0]), TokenId(input[r, 1]), NumOps.ToDouble(target!.GetFlat(r))));
            }
            else
            {
                var x = new Dictionary<(int, int), double>();
                int seq = input.Shape.Length == 0 ? 1 : input.Shape[input.Shape.Length - 1];
                int sequences = seq == 0 ? 0 : input.Length / seq;
                int window = Math.Max(1, _options.WindowSize);
                for (int s = 0; s < sequences; s++)
                {
                    for (int a = 0; a < seq; a++)
                    {
                        int wi = TokenId(input.GetFlat(s * seq + a));
                        for (int d = 1; d <= window && a + d < seq; d++)
                        {
                            int wj = TokenId(input.GetFlat(s * seq + a + d));
                            // X is symmetric (Section 3): both orders of the pair co-occur.
                            x[(wi, wj)] = x.TryGetValue((wi, wj), out var v1) ? v1 + 1.0 / d : 1.0 / d;
                            x[(wj, wi)] = x.TryGetValue((wj, wi), out var v2) ? v2 + 1.0 / d : 1.0 / d;
                        }
                    }
                }
                foreach (var kv in x.OrderBy(kv => kv.Key.Item1).ThenBy(kv => kv.Key.Item2))
                    counts.Add((kv.Key.Item1, kv.Key.Item2, Math.Log(kv.Value)));
            }

            // A sequence of one word co-occurs with nothing: one zero-weight row keeps the objective defined (and zero).
            if (counts.Count == 0)
            {
                int w = input.Length > 0 ? TokenId(input.GetFlat(0)) : 0;
                var emptyInput = new Tensor<T>([1, 3]);
                emptyInput[0, 0] = NumOps.FromDouble(w);
                emptyInput[0, 1] = NumOps.FromDouble(w);
                return (emptyInput, new Tensor<T>([1]));
            }

            var rows = new Tensor<T>([counts.Count, 3]);
            var weightedLogX = new Tensor<T>([counts.Count]);
            for (int r = 0; r < counts.Count; r++)
            {
                var (i, j, logX) = counts[r];
                double sqrtF = Math.Sqrt(Weighting(Math.Exp(logX)));
                rows[r, 0] = NumOps.FromDouble(i);
                rows[r, 1] = NumOps.FromDouble(j);
                rows[r, 2] = NumOps.FromDouble(sqrtF);
                weightedLogX[r] = NumOps.FromDouble(sqrtF * logX);
            }
            return (rows, weightedLogX);
        }

        /// <inheritdoc />
        TrainingObjectiveKind ITrainingObjectiveProvider<T>.TrainingObjectiveKind => TrainingObjectiveKind.CooccurrenceFactorization;

        /// <inheritdoc />
        /// <remarks>Explicit pairs ([N, 2] with N log-count targets) keep their targets. Token sequences train on the
        /// co-occurrence counted from themselves, so the proposed target plays no part and the input stands in for
        /// it.</remarks>
        Tensor<T> ITrainingObjectiveProvider<T>.ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget)
            => IsExplicitPairs(input, proposedTarget) ? proposedTarget : input;

        /// <inheritdoc />
        T ITrainingObjectiveProvider<T>.EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target)
            => EvaluateTrainingObjective(input, target);

        private static bool IsExplicitPairs(Tensor<T> input, Tensor<T>? target)
            => input.Shape.Length == 2 && input.Shape[1] == 2 && target is not null && target.Length == input.Shape[0];

        /// <summary>The pairs are already the training graph's input; no batch axis is added.</summary>
        protected override Tensor<T> PrepareInputForTraining(Tensor<T> input) => input;

        /// <summary>The weighting function of Eq. 9: <c>(x / x_max)^α</c> below <c>x_max</c>, 1 at and above it.</summary>
        private double Weighting(double x)
            => x >= _options.XMax ? 1.0 : Math.Pow(Math.Max(x, 0.0) / _options.XMax, _options.Alpha);

        private int TokenId(T value)
        {
            int id = (int)Math.Round(NumOps.ToDouble(value));
            if (id < 0 || id >= _vocabSize)
                throw new ArgumentOutOfRangeException(nameof(value), $"Word index {id} is outside the vocabulary [0, {_vocabSize}).");
            return id;
        }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
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





        #endregion
    }
}
