using System.Globalization;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NER.Options;
using AiDotNet.NER.Preprocessing;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NER.SequenceLabeling;

/// <summary>
/// Paper-faithful word + character BiLSTM-CRF for Named Entity Recognition
/// (Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition").
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Unlike <see cref="BiLSTMCRF{T}"/>, which consumes caller-supplied token embeddings, this model
/// <b>owns its embedding layers and consumes token/character indices</b> — the standard NER design
/// (AllenNLP <c>CrfTagger</c>, flair <c>SequenceTagger</c>, the canonical PyTorch BiLSTM-CRF). The full
/// architecture is:
/// <list type="number">
/// <item><b>Word + character embedding front-end</b> (<see cref="WordCharEmbeddingLayer{T}"/>): a
/// learnable word lookup (initializable from pretrained GloVe via <see cref="LoadGloVeEmbeddings"/>)
/// concatenated with a character-level BiLSTM representation per word.</item>
/// <item><b>Word-level BiLSTM</b> over the per-token representations for bidirectional context.</item>
/// <item><b>Dropout</b> regularization (paper default 0.5).</item>
/// <item><b>Linear projection</b> to per-label emission scores.</item>
/// <item><b>CRF</b> for globally-consistent BIO decoding (Viterbi) and negative-log-likelihood training.</item>
/// </list>
/// </para>
/// <para>
/// <b>Why a separate model:</b> owning the embeddings fixes the failure modes of feeding a BiLSTM-CRF
/// ad-hoc vectors: word identity gives a shared, deterministic, generalizing embedding space (no
/// process-randomized hashing), and the character BiLSTM lets the model recognize unseen words by their
/// spelling instead of hallucinating. The CRF guarantees structurally valid label sequences (no orphan
/// I- tags).
/// </para>
/// <para>
/// <b>For Beginners:</b> give this model sentences (as words) with their entity labels and it learns to
/// tag new sentences. It reads both whole words and their spelling, considers the whole sentence in both
/// directions, and makes sure the labels form valid entity spans. Use <see cref="Create"/> to build one
/// from your training sentences.
/// </para>
/// <para>
/// <b>References:</b> Lample et al., NAACL 2016; Huang, Xu, Yu, 2015; Ma and Hovy, ACL 2016.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Neural Architectures for Named Entity Recognition",
    "https://arxiv.org/abs/1603.01360",
    Year = 2016,
    Authors = "Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer")]
public class WordCharBiLSTMCRF<T> : SequenceLabelingNERBase<T>
{
    private readonly BiLSTMCRFOptions _options;
    private readonly NerTextEncoder _encoder;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private WordCharEmbeddingLayer<T>? _embeddingFrontEnd;

    /// <summary>Gets the text encoder (word/character vocabularies + tokenizer) this model was built with.</summary>
    public NerTextEncoder Encoder => _encoder;

    /// <summary>Gets the word+character embedding front-end layer (null until layers are initialized).</summary>
    public WordCharEmbeddingLayer<T>? EmbeddingFrontEnd => _embeddingFrontEnd;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Constructs a word+char BiLSTM-CRF over a prebuilt <see cref="NerTextEncoder"/>.
    /// Prefer <see cref="Create"/>, which builds the encoder from training data for you.
    /// </summary>
    /// <param name="architecture">Network architecture (input/output sizing metadata).</param>
    /// <param name="encoder">The vocabulary/tokenizer encoder.</param>
    /// <param name="options">Model options; defaults to the paper configuration when null.</param>
    /// <param name="optimizer">Gradient optimizer; defaults to AdamW with the options' learning rate.</param>
    public WordCharBiLSTMCRF(
        NeuralNetworkArchitecture<T> architecture,
        NerTextEncoder encoder,
        BiLSTMCRFOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _options = options ?? new BiLSTMCRFOptions();
        if (_options.LabelNames.Length != _options.NumLabels)
            throw new ArgumentException(
                $"LabelNames length ({_options.LabelNames.Length}) must match NumLabels ({_options.NumLabels}).");

        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate
            });

        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;

        InitializeLayers();
        WarmUp();
    }

    /// <summary>
    /// Runs one inference pass on a zero input to resolve every lazy layer's shape and materialize
    /// its parameter tensors. Without this, the first training step would collect an incomplete
    /// parameter set, and <see cref="LoadGloVeEmbeddings"/> would have no word table to write into.
    /// </summary>
    private void WarmUp()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            Tensor<T> x = new([_options.MaxSequenceLength, 1 + _encoder.MaxWordLength]);
            foreach (var layer in Layers)
                x = layer.Forward(x);
        }
        finally
        {
            if (wasTraining) SetTrainingMode(true);
        }
    }

    /// <summary>
    /// Builds a ready-to-train model from tokenized, labeled training sentences. The word/character
    /// vocabularies are derived from the sentences; pass GloVe later via <see cref="LoadGloVeEmbeddings"/>.
    /// </summary>
    /// <param name="tokenizedSentences">Training sentences, each already split into tokens (see
    /// <see cref="NerTextEncoder.Tokenize"/>).</param>
    /// <param name="options">Model options; defaults to the paper configuration when null.</param>
    /// <param name="maxWordLength">Maximum characters encoded per word.</param>
    /// <param name="optimizer">Gradient optimizer; defaults to AdamW.</param>
    /// <returns>A constructed <see cref="WordCharBiLSTMCRF{T}"/>.</returns>
    public static WordCharBiLSTMCRF<T> Create(
        IEnumerable<string[]> tokenizedSentences,
        BiLSTMCRFOptions? options = null,
        int maxWordLength = 20,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
    {
        var encoder = NerTextEncoder.Build(tokenizedSentences, maxWordLength);
        var opts = options ?? new BiLSTMCRFOptions();
        var arch = new NeuralNetworkArchitecture<T>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.SequenceClassification,
            inputSize: opts.EmbeddingDimension,
            outputSize: opts.NumLabels);
        return new WordCharBiLSTMCRF<T>(arch, encoder, opts, optimizer);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Builds <c>[WordCharEmbeddingLayer, word-BiLSTM, dropout, dense projection, CRF]</c>. The
    /// word-BiLSTM's input dimension is the fused word+char embedding size and is inferred lazily.
    /// </remarks>
    protected override void InitializeLayers()
    {
        Layers.Clear();

        var tanh = new TanhActivation<T>() as IActivationFunction<T>;
        var sigmoid = new SigmoidActivation<T>() as IActivationFunction<T>;
        var identity = new IdentityActivation<T>() as IActivationFunction<T>;

        _embeddingFrontEnd = new WordCharEmbeddingLayer<T>(
            wordVocabSize: _encoder.WordVocabSize,
            wordEmbeddingDim: _options.EmbeddingDimension,
            charVocabSize: _encoder.CharVocabSize,
            charEmbeddingDim: _options.CharEmbeddingDimension,
            charHiddenDim: _options.CharHiddenDimension,
            sequenceLength: _options.MaxSequenceLength,
            maxWordLength: _encoder.MaxWordLength);
        Layers.Add(_embeddingFrontEnd);

        for (int layer = 0; layer < _options.NumLSTMLayers; layer++)
        {
            var lstm = new LSTMLayer<T>(_options.HiddenDimension, tanh, sigmoid);
            Layers.Add(new BidirectionalLayer<T>(lstm, mergeMode: true, identity));
            if (_options.DropoutRate > 0 && layer < _options.NumLSTMLayers - 1)
                Layers.Add(new DropoutLayer<T>(_options.DropoutRate));
        }

        if (_options.DropoutRate > 0)
            Layers.Add(new DropoutLayer<T>(_options.DropoutRate));

        Layers.Add(new DenseLayer<T>(_options.NumLabels, identity));

        if (_options.UseCRF)
            Layers.Add(new ConditionalRandomFieldLayer<T>(
                numClasses: _options.NumLabels,
                sequenceLength: _options.MaxSequenceLength,
                scalarActivation: identity));
    }

    #region Input encoding (preprocessing — facade input prep, not a parallel train/predict path)

    /// <summary>
    /// Encodes a tokenized sentence into the packed-index input tensor this model consumes.
    /// </summary>
    /// <remarks>
    /// This is input <b>preprocessing</b>, not an alternate run path: it only builds the
    /// <see cref="Tensor{T}"/> you feed to the facade. Train through
    /// <c>AiModelBuilder.ConfigureModel(model).BuildAsync(...)</c> and run inference through
    /// <c>AiModelResult.Predict(...)</c> (or <see cref="PredictLabels"/>) on the encoded tensor,
    /// then map indices to names with <see cref="SequenceLabelingNERBase{T}.DecodeLabels"/> — exactly
    /// like every other NER model, which consume caller-prepared tensors.
    /// </remarks>
    /// <param name="tokens">The sentence tokens.</param>
    /// <returns>A <c>[sequenceLength, 1 + maxWordLength]</c> index tensor.</returns>
    public Tensor<T> EncodeSentence(string[] tokens)
    {
        if (tokens is null) throw new ArgumentNullException(nameof(tokens));
        double[] packed = _encoder.EncodePacked(tokens, _options.MaxSequenceLength);
        var data = new T[packed.Length];
        for (int i = 0; i < packed.Length; i++) data[i] = NumOps.FromDouble(packed[i]);
        return new Tensor<T>(new Vector<T>(data),
            [_options.MaxSequenceLength, 1 + _encoder.MaxWordLength]);
    }

    /// <summary>
    /// Initializes the word-embedding table from a pretrained GloVe text file. Each line is
    /// <c>word v1 v2 ... vd</c>; rows for words present in the vocabulary are overwritten, others
    /// keep their random initialization. The vector dimension must match
    /// <see cref="BiLSTMCRFOptions.EmbeddingDimension"/>.
    /// </summary>
    /// <param name="gloveFilePath">Path to a GloVe-format embeddings file.</param>
    /// <returns>The number of vocabulary words matched and initialized from the file.</returns>
    public int LoadGloVeEmbeddings(string gloveFilePath)
    {
        if (_embeddingFrontEnd is null)
            throw new InvalidOperationException("Embedding front-end is not initialized.");
        if (string.IsNullOrWhiteSpace(gloveFilePath) || !File.Exists(gloveFilePath))
            throw new FileNotFoundException("GloVe file not found.", gloveFilePath);

        int dim = _options.EmbeddingDimension;
        var wordEmbedding = _embeddingFrontEnd.WordEmbedding;
        var paramsVec = wordEmbedding.GetParameters();
        var tokenToId = _encoder.WordVocabulary.TokenToId;
        int matched = 0;

        foreach (var line in File.ReadLines(gloveFilePath))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            // char[] overload (not the char one) — string.Split(char, StringSplitOptions) does not
            // exist on net471, which this project also targets.
            var parts = line.Split([' '], StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != dim + 1) continue;

            string word = parts[0].ToLowerInvariant();
            if (!tokenToId.TryGetValue(word, out int id)) continue;

            int rowBase = id * dim;
            if (rowBase + dim > paramsVec.Length) continue;
            for (int d = 0; d < dim; d++)
                paramsVec[rowBase + d] = NumOps.FromDouble(
                    double.Parse(parts[d + 1], CultureInfo.InvariantCulture));
            matched++;
        }

        wordEmbedding.SetParameters(paramsVec);
        return matched;
    }

    #endregion

    #region Sequence labeling

    /// <inheritdoc/>
    public override Tensor<T> PredictLabels(Tensor<T> tokenIndices)
    {
        var preprocessed = PreprocessTokens(tokenIndices);
        var output = Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    protected override Tensor<T> ComputeEmissionScores(Tensor<T> tokenIndices)
    {
        Tensor<T> output = PreprocessTokens(tokenIndices);
        foreach (var layer in Layers)
        {
            if (layer is ConditionalRandomFieldLayer<T>) break;
            output = layer.Forward(output);
        }
        return output;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (_optimizer is null)
            throw new InvalidOperationException("Optimizer is not initialized. Cannot train without an optimizer.");
        RunCrfAwareTrainStep(input, expected, _options.UseCRF, _optimizer);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Pads or truncates the packed index input along the token axis to <c>MaxSequenceLength</c>, the
    /// fixed length the CRF layer requires. Padding rows are zero (the [PAD] word/character index).
    /// </remarks>
    protected override Tensor<T> PreprocessTokens(Tensor<T> rawIndices)
    {
        if (rawIndices.Rank != 2)
            throw new ArgumentException(
                $"WordCharBiLSTMCRF expects rank-2 packed index input [seqLen, 1 + maxWordLength]; got rank {rawIndices.Rank}.",
                nameof(rawIndices));

        int maxLen = _options.MaxSequenceLength;
        int width = rawIndices.Shape[1];
        int seqLen = rawIndices.Shape[0];
        if (seqLen == maxLen) return rawIndices;

        var padded = new Tensor<T>([maxLen, width]);
        int copyLen = Math.Min(seqLen, maxLen);
        for (int s = 0; s < copyLen; s++)
            for (int w = 0; w < width; w++)
                padded[s, w] = rawIndices[s, w];
        return padded;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        if (modelOutput.Rank >= 2 && modelOutput.Shape[^1] == _options.NumLabels)
            return ArgmaxDecode(modelOutput);
        return modelOutput;
    }

    #endregion

    #region NeuralNetworkBase plumbing

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = checked((int)layer.ParameterCount);
            if (count == 0) continue;
            layer.SetParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = "WordChar-BiLSTM-CRF",
            Description = $"Word+Char BiLSTM-CRF NER (Lample et al., NAACL 2016); " +
                          $"wordVocab={_encoder.WordVocabSize}, charVocab={_encoder.CharVocabSize}",
            Complexity = _options.NumLSTMLayers
        };
        m.AdditionalInfo["EmbeddingDimension"] = _options.EmbeddingDimension.ToString(CultureInfo.InvariantCulture);
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString(CultureInfo.InvariantCulture);
        m.AdditionalInfo["CharHiddenDimension"] = _options.CharHiddenDimension.ToString(CultureInfo.InvariantCulture);
        m.AdditionalInfo["NumLabels"] = _options.NumLabels.ToString(CultureInfo.InvariantCulture);
        m.AdditionalInfo["UseCRF"] = _options.UseCRF.ToString();
        return m;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_options.EmbeddingDimension);
        w.Write(_options.HiddenDimension);
        w.Write(_options.NumLSTMLayers);
        w.Write(_options.NumLabels);
        w.Write(_options.MaxSequenceLength);
        w.Write(_options.CharEmbeddingDimension);
        w.Write(_options.CharHiddenDimension);
        w.Write(_options.UseCRF);
        w.Write(_options.DropoutRate);
        w.Write(_options.LearningRate);
        w.Write(_options.LabelNames.Length);
        foreach (var label in _options.LabelNames) w.Write(label);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _options.EmbeddingDimension = r.ReadInt32();
        _options.HiddenDimension = r.ReadInt32();
        _options.NumLSTMLayers = r.ReadInt32();
        _options.NumLabels = r.ReadInt32();
        _options.MaxSequenceLength = r.ReadInt32();
        _options.CharEmbeddingDimension = r.ReadInt32();
        _options.CharHiddenDimension = r.ReadInt32();
        _options.UseCRF = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        _options.LearningRate = r.ReadDouble();
        int labelCount = r.ReadInt32();
        var labels = new string[labelCount];
        for (int i = 0; i < labelCount; i++) labels[i] = r.ReadString();
        _options.LabelNames = labels;

        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new WordCharBiLSTMCRF<T>(Architecture, _encoder, new BiLSTMCRFOptions(_options));
    }

    #endregion
}
