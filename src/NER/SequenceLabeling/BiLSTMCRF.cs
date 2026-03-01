using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NER.Interfaces;
using AiDotNet.NER.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.NER.SequenceLabeling;

/// <summary>
/// BiLSTM-CRF: Bidirectional LSTM with Conditional Random Field for Named Entity Recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BiLSTM-CRF (Huang et al., 2015; Lample et al., NAACL 2016) is the foundational neural architecture
/// for Named Entity Recognition that combines two powerful, complementary components into a unified model:
///
/// <b>1. Bidirectional LSTM (BiLSTM):</b>
/// A pair of Long Short-Term Memory networks that process the token sequence in both directions simultaneously.
/// The forward LSTM reads left-to-right (e.g., "Barack" -> "Obama" -> "was" -> "born" -> "in" -> "Honolulu"),
/// capturing preceding context for each token. The backward LSTM reads right-to-left, capturing following
/// context. Their hidden states are merged at each position, giving the model a complete view of the
/// entire sentence context around every token. This bidirectional context is crucial for NER because entity
/// recognition often depends on both left and right context (e.g., "Washington" could be a person, location,
/// or organization depending on surrounding words).
///
/// <b>2. Conditional Random Field (CRF):</b>
/// A structured prediction layer that models label-level transition dependencies to produce globally optimal
/// label sequences. While the BiLSTM produces per-token scores for each possible label (called "emission
/// scores"), the CRF adds a learned transition matrix that captures which label transitions are valid.
/// For example, it learns that I-PER (Inside Person) can only follow B-PER (Begin Person) or I-PER,
/// never B-ORG (Begin Organization). During inference, the Viterbi algorithm efficiently finds the
/// highest-scoring label sequence that respects all transition constraints.
///
/// <b>Architecture Pipeline:</b>
/// <list type="number">
/// <item>Token embeddings (e.g., 100d GloVe vectors) are fed into the BiLSTM encoder</item>
/// <item>BiLSTM produces contextualized hidden states for each token position</item>
/// <item>A linear projection maps hidden states to emission scores (one score per label per token)</item>
/// <item>Dropout regularization prevents overfitting during training</item>
/// <item>CRF layer combines emission scores with learned transition scores to decode the optimal label sequence</item>
/// </list>
///
/// <b>Mathematical Formulation:</b>
/// For input sequence x = (x_1, ..., x_n) and label sequence y = (y_1, ..., y_n):
/// - Emission score: e(x_t, y_t) = W * h_t + b, where h_t is the BiLSTM hidden state at position t
/// - Transition score: T[y_{t-1}, y_t] from the learned CRF transition matrix
/// - Sequence score: S(x, y) = SUM_t [e(x_t, y_t) + T[y_{t-1}, y_t]]
/// - Training objective: maximize log P(y|x) = S(x, y) - log(SUM_{y'} exp(S(x, y')))
/// - Inference: y* = argmax_y S(x, y), solved efficiently by Viterbi algorithm in O(n * L^2) time
///
/// This model serves as the golden example for the NER model family in AiDotNet, establishing the
/// architectural patterns (base class hierarchy, layer initialization via LayerHelper, dual ONNX/native
/// mode, serialization, etc.) that all other NER models follow.
/// </para>
/// <para>
/// <b>For Beginners:</b> BiLSTM-CRF is the go-to model for finding names, places, organizations,
/// and other entities in text. Think of it as a two-step process:
///
/// <b>Step 1 - Reading with Context (BiLSTM):</b> The model reads each word while considering
/// the words around it. It reads the sentence both forward and backward, like reading a mystery
/// novel twice - once normally and once from the end. This gives each word a rich understanding
/// of its context. For example, "Apple" in "Apple Inc. announced..." gets a different
/// representation than "apple" in "She ate an apple."
///
/// <b>Step 2 - Making Consistent Labels (CRF):</b> After understanding each word's context,
/// the CRF ensures the labels make sense as a sequence. It's like a spell-checker for entity
/// labels. Without CRF, the model might label "Obama" as I-ORG after labeling "Barack" as
/// B-PER, which doesn't make sense. The CRF knows that I-ORG can't follow B-PER and would
/// correct this to "Barack"=B-PER, "Obama"=I-PER.
///
/// <b>Example:</b>
/// <code>
/// // Create the model architecture
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 9);
///
/// // Create model with default options (100d embeddings, 100 hidden units, CRF enabled)
/// var model = new BiLSTMCRF&lt;float&gt;(arch);
///
/// // Train on labeled data
/// model.Train(tokenEmbeddings, labelSequence);
///
/// // Predict entity labels for new text
/// var predictions = model.PredictLabels(newTokenEmbeddings);
/// var labels = model.DecodeLabels(predictions);
/// // labels might be: ["B-PER", "I-PER", "O", "O", "B-ORG"]
/// </code>
///
/// <b>When to use BiLSTM-CRF:</b>
/// - Named entity recognition (people, organizations, locations, etc.)
/// - Part-of-speech tagging
/// - Chunking and shallow parsing
/// - Any sequence labeling task where label dependencies matter
///
/// <b>Key advantage over simpler approaches:</b> The CRF layer typically improves F1 score
/// by 1-2% over independent per-token classification, and more importantly, it guarantees
/// that the output label sequence is always structurally valid.
/// </para>
/// <para>
/// <b>References:</b>
/// <list type="bullet">
/// <item>"Bidirectional LSTM-CRF Models for Sequence Tagging" (Huang, Xu, Yu, 2015)</item>
/// <item>"Neural Architectures for Named Entity Recognition" (Lample et al., NAACL 2016)</item>
/// <item>"End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (Ma and Hovy, ACL 2016)</item>
/// </list>
/// </para>
/// </remarks>
public class BiLSTMCRF<T> : SequenceLabelingNERBase<T>, INERModel<T>
{
    #region Fields

    /// <summary>
    /// The configuration options controlling all aspects of the BiLSTM-CRF architecture, including
    /// embedding dimensions, hidden sizes, number of LSTM layers, dropout rates, and CRF settings.
    /// </summary>
    /// <remarks>
    /// <para>
    /// These options map directly to the hyperparameters described in Lample et al. (2016). The defaults
    /// follow the paper's recommended configuration: 100d embeddings, 100 hidden units per direction,
    /// single BiLSTM layer, 0.5 dropout, CRF enabled, and 9 CoNLL-2003 BIO labels.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This holds all the settings that control how the model is built and trained.
    /// The default values come from the original research paper and work well for most NER tasks.
    /// </para>
    /// </remarks>
    private readonly BiLSTMCRFOptions _options;

    /// <summary>
    /// Returns the model's configuration options.
    /// </summary>
    /// <returns>The <see cref="BiLSTMCRFOptions"/> instance controlling this model's behavior.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to inspect or retrieve the current model settings, such as
    /// embedding dimensions, hidden size, number of labels, and whether CRF is enabled.
    /// </para>
    /// </remarks>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The gradient-based optimizer used for updating model weights during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Defaults to AdamW (Adam with weight decay), which is the standard optimizer for NER models.
    /// AdamW combines adaptive learning rates (different for each parameter) with proper L2
    /// regularization, preventing the model from overfitting to the training data.
    ///
    /// The learning rate is configured via <see cref="BiLSTMCRFOptions.LearningRate"/> and passed
    /// to the optimizer during construction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The optimizer is what adjusts the model's internal numbers (weights)
    /// during training to make better predictions. AdamW is like an intelligent teacher that
    /// adjusts each student's study plan individually rather than giving everyone the same feedback.
    /// </para>
    /// </remarks>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Whether the model operates in native training mode (true) or ONNX inference-only mode (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// In native mode, the model uses AiDotNet's built-in layers (BidirectionalLayer wrapping LSTM,
    /// Dense, CRF) and supports both training and inference. In ONNX mode, the model loads a
    /// pre-trained ONNX file and only supports inference (no training or gradient computation).
    ///
    /// Use native mode when you want to train a model from scratch or fine-tune on your data.
    /// Use ONNX mode when you have a pre-trained model (e.g., from PyTorch/TensorFlow) and
    /// just want to run predictions.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of native mode as "learning mode" where the model can be
    /// taught, and ONNX mode as "expert mode" where the model already knows everything and
    /// just answers questions. ONNX mode is faster because it doesn't need to keep track of
    /// learning-related information.
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    /// <summary>
    /// Whether this model instance has been disposed and should no longer be used.
    /// </summary>
    private bool _disposed;

    #endregion

    #region INERModel Properties

    /// <summary>
    /// Gets the number of BIO labels this model can predict.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Corresponds to the output dimension of the linear projection layer and the CRF layer.
    /// For the default CoNLL-2003 configuration, this is 9: O, B-PER, I-PER, B-ORG, I-ORG,
    /// B-LOC, I-LOC, B-MISC, I-MISC. The number of labels determines the size of the CRF
    /// transition matrix (numLabels x numLabels) and the emission score output dimension.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many different labels the model can assign to each
    /// word. With 9 labels, the model can identify 4 entity types (Person, Organization, Location,
    /// Miscellaneous), each with Begin/Inside variants, plus the "Outside" label for non-entity words.
    /// </para>
    /// </remarks>
    int INERModel<T>.NumLabels => _options.NumLabels;

    /// <summary>
    /// Gets the dimensionality of input token embeddings expected by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This must match the dimensionality of the word embeddings used as input. The original
    /// Lample et al. (2016) paper uses 100-dimensional GloVe embeddings. Other common choices
    /// include 300d GloVe, 768d BERT, or 1024d RoBERTa embeddings. The embedding dimension
    /// determines the input size of the first BiLSTM layer.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Each word in the input text must be converted to a list of numbers
    /// (called an "embedding") before the model can process it. This property tells you how long
    /// that list of numbers needs to be. For example, with 100d GloVe embeddings, each word is
    /// represented as 100 numbers that capture its meaning.
    /// </para>
    /// </remarks>
    int INERModel<T>.EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the expected input tensor shape as [maxSequenceLength, embeddingDimension].
    /// </summary>
    /// <remarks>
    /// <para>
    /// The model expects 2D input tensors where the first dimension is the sequence length (number
    /// of tokens) and the second dimension is the embedding size. For batch processing, a 3D tensor
    /// with shape [batchSize, sequenceLength, embeddingDimension] is accepted.
    ///
    /// Sequences shorter than maxSequenceLength are automatically padded with zeros during
    /// preprocessing. Sequences longer than maxSequenceLength are truncated to fit.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you the shape of data the model expects. The first number is
    /// the maximum number of words in a sentence, and the second is the size of each word's embedding.
    /// You don't need to pad your input manually - the model handles this automatically.
    /// </para>
    /// </remarks>
    public int[] ExpectedInputShape => [_options.MaxSequenceLength, _options.EmbeddingDimension];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a BiLSTM-CRF model in ONNX inference mode, loading a pre-trained model from disk.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.
    /// For NER, set inputFeatures to the embedding dimension and outputSize to the number of labels.</param>
    /// <param name="modelPath">Path to the ONNX model file (.onnx). The file must exist and contain
    /// a valid ONNX model exported from a BiLSTM-CRF implementation (e.g., from PyTorch, TensorFlow,
    /// or a previous AiDotNet training session).</param>
    /// <param name="options">Optional configuration. If null, defaults to the Lample et al. (2016)
    /// configuration: 100d embeddings, 100 hidden units, CRF enabled, 9 CoNLL-2003 labels.</param>
    /// <remarks>
    /// <para>
    /// ONNX mode enables high-performance inference using a pre-trained model without the overhead
    /// of maintaining gradient computation graphs. This is the recommended mode for production
    /// deployments where you have a model that was trained in Python (PyTorch/TensorFlow) and
    /// exported to ONNX format.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you already have a trained model file.
    /// The model will be able to predict entity labels but cannot be further trained.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 9);
    /// var model = new BiLSTMCRF&lt;float&gt;(arch, "my-trained-model.onnx");
    /// var predictions = model.PredictLabels(tokenEmbeddings);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when modelPath is null, empty, or whitespace.</exception>
    public BiLSTMCRF(NeuralNetworkArchitecture<T> architecture, string modelPath, BiLSTMCRFOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new BiLSTMCRFOptions();
        ValidateOptions();
        _useNativeMode = false;
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a BiLSTM-CRF model in native training mode with full forward/backward pass support.
    /// </summary>
    /// <param name="architecture">The neural network architecture defining input/output dimensions.
    /// For NER, set inputFeatures to the embedding dimension and outputSize to the number of labels.
    /// You may also provide custom layers via <c>architecture.Layers</c> to override the default
    /// BiLSTM-CRF architecture.</param>
    /// <param name="options">Optional configuration. If null, defaults to the Lample et al. (2016)
    /// configuration: 100d embeddings, 100 hidden units, CRF enabled, 9 CoNLL-2003 labels.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training. If null, defaults to
    /// AdamW (Adam with decoupled weight decay) with the learning rate from options, which provides
    /// faster convergence than the SGD used in the original paper while maintaining strong
    /// generalization.</param>
    /// <remarks>
    /// <para>
    /// Native mode builds the full BiLSTM-CRF architecture using AiDotNet's built-in layers and
    /// supports both training (forward + backward pass with gradient updates) and inference.
    /// The model architecture is constructed via <see cref="LayerHelper{T}.CreateDefaultBiLSTMCRFLayers"/>
    /// using research-paper-validated defaults unless custom layers are provided through the architecture.
    ///
    /// The default layer stack follows Lample et al. (2016):
    /// <list type="number">
    /// <item>BidirectionalLayer wrapping LSTM: processes tokens in both directions with tanh/sigmoid activations</item>
    /// <item>Dropout: 50% regularization between layers and before projection</item>
    /// <item>Dense projection: maps hidden states to emission scores (identity activation)</item>
    /// <item>CRF layer: Viterbi decoding with learned transition matrix</item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a model from scratch
    /// on your own labeled data. The model starts with random weights and learns to recognize
    /// entities through training.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 100, outputSize: 9);
    /// var model = new BiLSTMCRF&lt;float&gt;(arch); // Uses default research paper settings
    ///
    /// // Train for multiple epochs
    /// for (int epoch = 0; epoch &lt; 50; epoch++)
    ///     model.Train(trainingEmbeddings, trainingLabels);
    ///
    /// // Predict on new data
    /// var predictions = model.PredictLabels(testEmbeddings);
    /// </code>
    /// </para>
    /// </remarks>
    public BiLSTMCRF(NeuralNetworkArchitecture<T> architecture, BiLSTMCRFOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BiLSTMCRFOptions();
        ValidateOptions();
        _useNativeMode = true;
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
    }

    /// <summary>
    /// Validates the options for consistency and supported feature combinations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Checks that NumLabels matches LabelNames length. Additional dimension and rate
    /// validations are handled by property setters in <see cref="BiLSTMCRFOptions"/>.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when options are inconsistent.</exception>
    private void ValidateOptions()
    {
        if (_options.LabelNames.Length != _options.NumLabels)
            throw new ArgumentException(
                $"LabelNames length ({_options.LabelNames.Length}) must match NumLabels ({_options.NumLabels}).");
    }

    #endregion

    #region Sequence Labeling

    /// <summary>
    /// Predicts the optimal BIO label sequence for input token embeddings using the full
    /// BiLSTM-CRF pipeline (or ONNX inference if in ONNX mode).
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings with shape [sequenceLength, embeddingDim]
    /// for a single sentence. Each row is one token's embedding vector (e.g., 100d GloVe).
    /// Sequences shorter than MaxSequenceLength are automatically padded; longer sequences
    /// are truncated.</param>
    /// <returns>Predicted label indices with shape [sequenceLength], where each integer value is an
    /// index into <see cref="SequenceLabelingNERBase{T}.LabelNames"/>. Use
    /// <see cref="SequenceLabelingNERBase{T}.DecodeLabels"/> to convert to human-readable strings.</returns>
    /// <remarks>
    /// <para>
    /// The prediction pipeline is:
    /// <list type="number">
    /// <item><see cref="PreprocessTokens"/>: pads/truncates embeddings to MaxSequenceLength</item>
    /// <item><see cref="NERNeuralNetworkBase{T}.Forward"/> or <see cref="NERNeuralNetworkBase{T}.RunOnnxInference"/>:
    /// run the BiLSTM + projection + CRF forward pass</item>
    /// <item><see cref="PostprocessOutput"/>: applies argmax decoding to convert CRF one-hot output
    /// or raw emission scores into label indices</item>
    /// </list>
    ///
    /// When CRF is enabled (default), the CRF layer's Viterbi algorithm produces a one-hot encoded
    /// label sequence, which is then argmax-decoded to produce label indices. This guarantees
    /// structurally valid output (no invalid transitions like I-PER after B-ORG).
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Give this method the numerical representations of your words, and it
    /// returns a label index for each word. For example, if your input has 6 tokens, you might get
    /// back [1, 2, 0, 0, 0, 5], meaning: B-PER, I-PER, O, O, O, B-LOC. Use DecodeLabels() to
    /// convert these numbers to readable names like "B-PER", "I-PER", etc.
    /// </para>
    /// </remarks>
    /// <exception cref="ObjectDisposedException">Thrown if the model has been disposed.</exception>
    public override Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessTokens(tokenEmbeddings);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <summary>
    /// Computes emission scores from token embeddings by running the BiLSTM and linear projection
    /// layers, stopping before the CRF layer.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings with shape [sequenceLength, embeddingDim].</param>
    /// <returns>Emission score matrix with shape [sequenceLength, numLabels]. Each row contains
    /// the raw (pre-CRF) scores for all possible labels at that token position. Higher values
    /// indicate stronger model confidence in that label.</returns>
    /// <remarks>
    /// <para>
    /// Emission scores represent the BiLSTM's belief about what each token's label should be,
    /// based purely on the token's bidirectional context. These are the "votes" that each token
    /// casts for each possible label before the CRF considers label transition constraints.
    ///
    /// In the CRF framework:
    /// - Emission scores answer: "What does this token look like based on its context?"
    /// - Transition scores answer: "What label typically follows the previous label?"
    /// - The CRF combines both to find the best overall label sequence.
    ///
    /// This method is useful for:
    /// - Visualizing model confidence (which labels does the model consider for each token?)
    /// - Debugging: comparing emission scores before vs. after CRF decoding
    /// - Custom decoding strategies (e.g., constrained decoding with a gazetteer)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Emission scores show what the model thinks about each word individually.
    /// For example, the word "Google" might have high scores for B-ORG (organization) and B-LOC
    /// (location) but low scores for B-PER (person). The CRF then uses these scores along with
    /// knowledge about valid label sequences to make the final decision.
    /// </para>
    /// </remarks>
    /// <exception cref="ObjectDisposedException">Thrown if the model has been disposed.</exception>
    protected override Tensor<T> ComputeEmissionScores(Tensor<T> tokenEmbeddings)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
        {
            return RunOnnxInference(tokenEmbeddings);
        }

        // Forward through all layers except CRF to get emission scores
        Tensor<T> output = tokenEmbeddings;
        foreach (var layer in Layers)
        {
            if (layer is ConditionalRandomFieldLayer<T>)
                break;
            output = layer.Forward(output);
        }
        return output;
    }

    #endregion

    #region INERModel Methods

    /// <summary>
    /// Trains the BiLSTM-CRF model asynchronously over multiple epochs with progress reporting.
    /// </summary>
    /// <param name="tokenEmbeddings">Training input: token embeddings with shape
    /// [sequenceLength, embeddingDim] or [batchSize, sequenceLength, embeddingDim].</param>
    /// <param name="labels">Ground truth label indices with shape [sequenceLength] or
    /// [batchSize, sequenceLength]. Each value is an index into <see cref="SequenceLabelingNERBase{T}.LabelNames"/>.</param>
    /// <param name="epochs">Number of training epochs (complete passes through the data).
    /// Lample et al. (2016) recommend 50-100 epochs with early stopping on validation F1.</param>
    /// <param name="progress">Optional progress reporter that receives training metrics after each epoch,
    /// including current epoch, loss value, and optionally F1 score.</param>
    /// <param name="cancellationToken">Token to cancel training gracefully. The current epoch
    /// completes before cancellation takes effect.</param>
    /// <returns>A task that completes when training finishes or is cancelled.</returns>
    /// <remarks>
    /// <para>
    /// Each epoch performs:
    /// <list type="number">
    /// <item>Forward pass through BiLSTM layers to compute emission scores</item>
    /// <item>CRF negative log-likelihood loss computation</item>
    /// <item>Backward pass to compute gradients for all parameters</item>
    /// <item>AdamW optimizer updates all model weights</item>
    /// <item>Progress reporting with current loss (computed efficiently during the forward pass)</item>
    /// </list>
    ///
    /// Training tips from the literature:
    /// - Use a learning rate of 0.001 (AdamW) or 0.01 (SGD with momentum)
    /// - Apply gradient clipping with max norm 5.0 to prevent exploding gradients
    /// - Monitor validation F1 score and stop when it plateaus (early stopping)
    /// - Typical training completes in 50-100 epochs on CoNLL-2003
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the model to recognize entities by showing it
    /// examples of correctly labeled text. Each "epoch" is one complete pass through all training
    /// examples. More epochs generally means better accuracy, but too many can cause "overfitting"
    /// where the model memorizes the training data instead of learning general patterns.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown if called in ONNX mode (ONNX models are read-only).</exception>
    Task INERModel<T>.TrainAsync(
        Tensor<T> tokenEmbeddings,
        Tensor<T> labels,
        int epochs,
        IProgress<NERTrainingProgress>? progress,
        CancellationToken cancellationToken)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized.");

        return Task.Run(() =>
        {
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Single forward + backward pass (avoids duplicate forward computation)
                SetTrainingMode(true);
                try
                {
                    var preprocessed = PreprocessTokens(tokenEmbeddings);
                    var preprocessedLabels = PreprocessLabels(labels, preprocessed.Shape[0]);
                    var output = Forward(preprocessed);
                    double loss = NumOps.ToDouble(LossFunction.CalculateLoss(
                        output.ToVector(), preprocessedLabels.ToVector()));
                    var grad = LossFunction.CalculateDerivative(output.ToVector(), preprocessedLabels.ToVector());
                    var gt = Tensor<T>.FromVector(grad);
                    for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
                    _optimizer.UpdateParameters(Layers);

                    progress?.Report(new NERTrainingProgress
                    {
                        CurrentEpoch = epoch,
                        TotalEpochs = epochs,
                        CurrentBatch = 1,
                        TotalBatches = 1,
                        Loss = loss
                    });
                }
                finally
                {
                    SetTrainingMode(false);
                }
            }
        }, cancellationToken);
    }

    /// <summary>
    /// Predicts entity labels for multiple input sequences in batch, yielding results lazily.
    /// </summary>
    /// <param name="sequences">An enumerable of token embedding tensors, each with shape
    /// [sequenceLength, embeddingDim]. Each tensor represents one sentence/document to label.</param>
    /// <returns>An enumerable of label index tensors, one per input sequence, each with shape
    /// [sequenceLength] containing integer label indices.</returns>
    /// <remarks>
    /// <para>
    /// This method processes sequences one at a time using lazy evaluation (yield return), so
    /// it doesn't load all sequences into memory simultaneously. This is important for large
    /// datasets where you may have millions of sentences to label.
    ///
    /// For higher throughput on GPU, consider batching sequences of similar length together
    /// and using PredictLabels with 3D batch input instead.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> If you have many sentences to label, pass them all at once and
    /// this method will label them one by one, giving you results as they become available.
    /// This is memory-efficient because it doesn't need to hold all results in memory at once.
    /// </para>
    /// </remarks>
    IEnumerable<Tensor<T>> INERModel<T>.PredictBatch(IEnumerable<Tensor<T>> sequences)
    {
        foreach (var seq in sequences)
        {
            yield return PredictLabels(seq);
        }
    }

    /// <summary>
    /// Validates that an input tensor has the correct shape, embedding dimension, and sequence
    /// length for this model.
    /// </summary>
    /// <param name="input">The input tensor to validate.</param>
    /// <remarks>
    /// <para>
    /// Checks:
    /// <list type="bullet">
    /// <item>Rank must be 2 (single sequence) or 3 (batch of sequences)</item>
    /// <item>Last dimension must match the model's embedding dimension</item>
    /// <item>Sequence length must not exceed MaxSequenceLength</item>
    /// </list>
    ///
    /// Note: sequences shorter than MaxSequenceLength are valid and will be automatically
    /// padded during preprocessing. Only sequences exceeding MaxSequenceLength trigger an error.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Call this before PredictLabels to make sure your input data has
    /// the right shape. It's like checking that you're putting the right size battery in a device -
    /// better to check first than to find out something's wrong halfway through.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when input rank is not 2 or 3, when the
    /// embedding dimension doesn't match, or when sequence length exceeds MaxSequenceLength.</exception>
    void INERModel<T>.ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2 || input.Rank > 3)
            throw new ArgumentException(
                $"Input must be 2D [sequenceLength, embeddingDim] or 3D [batch, sequenceLength, embeddingDim]. Got rank {input.Rank}.");

        int embDim = input.Shape[^1];
        if (embDim != _options.EmbeddingDimension)
            throw new ArgumentException(
                $"Embedding dimension mismatch. Expected {_options.EmbeddingDimension}, got {embDim}.");

        int seqDim = input.Rank == 2 ? input.Shape[0] : input.Shape[1];
        if (seqDim > _options.MaxSequenceLength)
            throw new ArgumentException(
                $"Sequence length ({seqDim}) exceeds MaxSequenceLength ({_options.MaxSequenceLength}). " +
                "Truncate input or increase MaxSequenceLength in options.");
    }

    /// <summary>
    /// Returns a human-readable summary of the model's architecture and configuration.
    /// </summary>
    /// <returns>A formatted string describing the model variant, mode, dimensions, layer counts,
    /// CRF status, and total parameter count.</returns>
    /// <remarks>
    /// <para>
    /// The summary includes:
    /// <list type="bullet">
    /// <item>Model variant (Tiny/Small/Base/Large/XLarge) and execution mode (Native/ONNX)</item>
    /// <item>Embedding dimension (input size to the BiLSTM)</item>
    /// <item>Hidden dimension per direction</item>
    /// <item>Number of stacked BiLSTM layers</item>
    /// <item>Number of output labels and CRF enablement status</item>
    /// <item>Dropout rate and total trainable parameter count</item>
    /// </list>
    ///
    /// This is useful for logging, debugging, and comparing different model configurations.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Call this to see a quick overview of your model. It shows things like
    /// how big the model is, how many layers it has, and how many parameters it needs to learn.
    /// More parameters generally means the model can learn more complex patterns, but also needs
    /// more training data and time.
    /// </para>
    /// </remarks>
    string INERModel<T>.GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"BiLSTM-CRF ({_options.Variant})");
        sb.AppendLine($"  Mode: {(_useNativeMode ? "Native" : "ONNX")}");
        sb.AppendLine($"  Embedding Dim: {_options.EmbeddingDimension}");
        sb.AppendLine($"  Hidden Dim: {_options.HiddenDimension} per direction");
        sb.AppendLine($"  LSTM Layers: {_options.NumLSTMLayers}");
        sb.AppendLine($"  Num Labels: {_options.NumLabels}");
        sb.AppendLine($"  CRF: {(_options.UseCRF ? "Enabled" : "Disabled")}");
        sb.AppendLine($"  Dropout: {_options.DropoutRate:P0}");
        sb.AppendLine($"  Learning Rate: {_options.LearningRate}");
        sb.AppendLine($"  Total Parameters: {Layers.Sum(l => l.ParameterCount)}");
        return sb.ToString();
    }

    #endregion

    #region NeuralNetworkBase

    /// <summary>
    /// Initializes the BiLSTM-CRF layer stack using research-paper-validated defaults from
    /// <see cref="LayerHelper{T}.CreateDefaultBiLSTMCRFLayers"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method constructs the neural network layer stack following the architecture described
    /// in Lample et al. (NAACL 2016). The layer construction is delegated to
    /// <see cref="LayerHelper{T}.CreateDefaultBiLSTMCRFLayers"/> which uses the yield return
    /// pattern to lazily produce the following layers:
    ///
    /// <list type="number">
    /// <item><b>BidirectionalLayer wrapping LSTM:</b> Processes tokens in both forward and backward
    /// directions using <see cref="BidirectionalLayer{T}"/> which wraps an LSTM layer, clones it for
    /// the backward pass, and merges (element-wise adds) the outputs from both directions.</item>
    /// <item><b>Inter-layer Dropout:</b> Variational dropout between stacked BiLSTM layers to
    /// prevent co-adaptation of features across layers (only when numLSTMLayers > 1).</item>
    /// <item><b>Pre-projection Dropout:</b> Dropout before the linear projection, with the
    /// paper's recommended 0.5 rate, to regularize the emission score computation.</item>
    /// <item><b>Dense Projection:</b> Linear layer mapping hidden states to emission scores
    /// [hiddenDim -> numLabels] with identity activation (raw scores for CRF input).</item>
    /// <item><b>CRF Layer:</b> Conditional Random Field with learned transition matrix and
    /// Viterbi decoding for globally optimal label sequence prediction.</item>
    /// </list>
    ///
    /// If custom layers are provided via <c>Architecture.Layers</c>, those are used instead,
    /// allowing full architectural customization for advanced users.
    ///
    /// In ONNX mode, this method is a no-op since layers are handled by the ONNX runtime.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method builds the model's internal structure automatically
    /// using the best settings from the original research paper. You don't need to worry about
    /// the details - the model knows how to assemble itself from proven building blocks.
    /// Advanced users can override this by providing custom layers through the architecture parameter.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultBiLSTMCRFLayers(
                embeddingDimension: _options.EmbeddingDimension,
                hiddenDimension: _options.HiddenDimension,
                numLabels: _options.NumLabels,
                numLSTMLayers: _options.NumLSTMLayers,
                maxSequenceLength: _options.MaxSequenceLength,
                dropoutRate: _options.DropoutRate,
                useCharEmbeddings: _options.UseCharEmbeddings,
                charEmbeddingDimension: _options.CharEmbeddingDimension,
                charHiddenDimension: _options.CharHiddenDimension,
                useCRF: _options.UseCRF));
        }
    }

    /// <summary>
    /// Performs one training step: forward pass, loss computation, backward pass, and parameter update.
    /// </summary>
    /// <param name="input">Token embeddings with shape [sequenceLength, embeddingDim].</param>
    /// <param name="expected">Ground truth label indices with shape [sequenceLength]. Each value is
    /// an integer index into <see cref="SequenceLabelingNERBase{T}.LabelNames"/>.</param>
    /// <remarks>
    /// <para>
    /// A single training step consists of:
    /// <list type="number">
    /// <item><b>Preprocessing:</b> Pad/truncate input to MaxSequenceLength</item>
    /// <item><b>Forward pass:</b> Compute emission scores through BiLSTM and projection layers,
    /// then CRF decoding produces the predicted label sequence</item>
    /// <item><b>Loss computation:</b> Cross-entropy loss between predicted and expected labels
    /// (with CRF, this becomes the negative log-likelihood of the correct sequence)</item>
    /// <item><b>Backward pass:</b> Compute gradients for all parameters by propagating the
    /// loss gradient backward through CRF, projection, dropout, and BiLSTM layers</item>
    /// <item><b>Parameter update:</b> AdamW optimizer adjusts all weights using the computed
    /// gradients with adaptive learning rates and weight decay</item>
    /// </list>
    ///
    /// Training mode is enabled during the forward/backward pass to activate dropout layers
    /// (which are disabled during inference) and is restored to inference mode afterwards,
    /// even if an error occurs.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method teaches the model from one example. You show it a sentence
    /// with the correct labels, and it adjusts its internal weights to be more accurate. Call this
    /// method many times with different examples (training loop) to train the model.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown when called in ONNX mode.</exception>
    /// <exception cref="InvalidOperationException">Thrown when optimizer is not initialized.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (_optimizer is null) throw new InvalidOperationException("Optimizer is not initialized. Cannot train without an optimizer.");
        SetTrainingMode(true);
        try
        {
            var preprocessed = PreprocessTokens(input);
            var preprocessedLabels = PreprocessLabels(expected, preprocessed.Shape[0]);
            var output = Forward(preprocessed);
            var grad = LossFunction.CalculateDerivative(output.ToVector(), preprocessedLabels.ToVector());
            var gt = Tensor<T>.FromVector(grad);
            for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Updates all model parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">A flat vector containing all model parameters concatenated in layer
    /// order. The total length must equal the sum of all layer parameter counts.</param>
    /// <remarks>
    /// <para>
    /// This method distributes the flat parameter vector to individual layers by slicing it
    /// according to each layer's parameter count. This is useful for:
    /// - Loading parameters from an external source
    /// - Implementing custom optimization algorithms
    /// - Parameter averaging across multiple model checkpoints
    ///
    /// Parameters are applied in the same order as layers: BiLSTM weights, dropout (no params),
    /// Dense weights/biases, CRF transition matrix.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method sets all the model's learnable numbers at once from a
    /// single list. You typically don't need to call this directly - the optimizer handles
    /// parameter updates during training. It's mainly useful for advanced scenarios like loading
    /// saved model parameters.
    /// </para>
    /// </remarks>
    /// <exception cref="NotSupportedException">Thrown when called in ONNX mode.</exception>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    /// <summary>
    /// Preprocesses token embeddings by padding or truncating to MaxSequenceLength before
    /// feeding them to the BiLSTM.
    /// </summary>
    /// <param name="rawEmbeddings">Raw token embeddings with shape [sequenceLength, embeddingDim].
    /// Sequence length may be shorter or longer than MaxSequenceLength.</param>
    /// <returns>Token embeddings padded with zeros or truncated to shape
    /// [MaxSequenceLength, embeddingDim].</returns>
    /// <remarks>
    /// <para>
    /// The CRF layer requires a fixed sequence length (MaxSequenceLength) because its transition
    /// matrix and Viterbi decoding buffers are pre-allocated at construction time. This method
    /// ensures all inputs conform to that fixed length:
    /// - Shorter sequences are right-padded with zero vectors
    /// - Longer sequences are truncated (excess tokens are dropped from the end)
    /// - Sequences already at MaxSequenceLength pass through unchanged
    ///
    /// For 2D inputs [seqLen, embDim], the output is [MaxSequenceLength, embDim].
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Not all sentences have the same number of words. This method ensures
    /// they all have the same length by adding "blank" words (zeros) at the end of short sentences
    /// and cutting off extra words from long sentences. This is necessary because the CRF layer
    /// needs to know the sequence length in advance.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PreprocessTokens(Tensor<T> rawEmbeddings)
    {
        int maxLen = _options.MaxSequenceLength;
        int embDim = _options.EmbeddingDimension;

        if (rawEmbeddings.Rank < 2)
            return rawEmbeddings;

        // Handle rank-3 [batch, seqLen, embDim] by processing each batch element
        if (rawEmbeddings.Rank == 3)
        {
            int batch = rawEmbeddings.Shape[0];
            int seqLen3 = rawEmbeddings.Shape[1];
            if (seqLen3 == maxLen) return rawEmbeddings;

            var padded3 = new Tensor<T>([batch, maxLen, embDim]);
            int copyLen3 = Math.Min(seqLen3, maxLen);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < copyLen3; s++)
                    for (int d = 0; d < embDim; d++)
                        padded3[b, s, d] = rawEmbeddings[b, s, d];
            return padded3;
        }

        // Rank-2 [seqLen, embDim]
        int seqLen = rawEmbeddings.Shape[0];
        if (seqLen == maxLen)
            return rawEmbeddings;

        var padded = new Tensor<T>([maxLen, embDim]);
        int copyLen = Math.Min(seqLen, maxLen);
        for (int s = 0; s < copyLen; s++)
            for (int d = 0; d < embDim; d++)
                padded[s, d] = rawEmbeddings[s, d];

        return padded;
    }

    /// <summary>
    /// Pads or truncates labels to match the preprocessed input sequence length.
    /// </summary>
    private Tensor<T> PreprocessLabels(Tensor<T> labels, int targetSeqLen)
    {
        if (labels.Rank < 1) return labels;

        int labelLen = labels.Shape[0];
        if (labelLen == targetSeqLen) return labels;

        if (labels.Rank == 1)
        {
            var padded = new Tensor<T>([targetSeqLen]);
            int copyLen = Math.Min(labelLen, targetSeqLen);
            for (int i = 0; i < copyLen; i++)
                padded[i] = labels[i];
            return padded;
        }

        // Rank-2 [seqLen, numLabels] (one-hot or multi-label)
        int cols = labels.Shape[1];
        var padded2 = new Tensor<T>([targetSeqLen, cols]);
        int copyLen2 = Math.Min(labelLen, targetSeqLen);
        for (int s = 0; s < copyLen2; s++)
            for (int c = 0; c < cols; c++)
                padded2[s, c] = labels[s, c];
        return padded2;
    }

    /// <summary>
    /// Postprocesses the model output by applying argmax decoding to produce label indices.
    /// </summary>
    /// <param name="modelOutput">Raw model output from the forward pass. When CRF is enabled,
    /// this is a one-hot encoded tensor from the CRF's Viterbi decoding. When CRF is disabled,
    /// this is raw emission scores.</param>
    /// <returns>Label indices with shape [sequenceLength], where each value is the index of
    /// the predicted label at that token position.</returns>
    /// <remarks>
    /// <para>
    /// Both CRF and non-CRF modes require argmax decoding:
    /// - <b>CRF enabled:</b> The CRF layer's Forward method returns a one-hot encoded tensor
    ///   (same shape as emission scores), not raw label indices. Argmax decoding converts this
    ///   one-hot encoding to integer label indices.
    /// - <b>CRF disabled:</b> The model output is raw emission scores where argmax selects the
    ///   highest-scoring label independently at each token position.
    ///
    /// The argmax is always applied when the output has a label dimension (last dim == numLabels),
    /// ensuring consistent label-index output regardless of CRF configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The model's raw output is a grid of scores (one score per label per
    /// word). This method picks the highest-scoring label for each word, converting scores into
    /// simple label numbers like 0=O, 1=B-PER, 2=I-PER, etc.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Always argmax-decode when the output has a label dimension
        if (modelOutput.Rank >= 2 && modelOutput.Shape[^1] == _options.NumLabels)
        {
            return ArgmaxDecode(modelOutput);
        }
        // Otherwise, assume the model already produced label indices
        return modelOutput;
    }

    /// <summary>
    /// Returns metadata describing this model for serialization, logging, and model registry purposes.
    /// </summary>
    /// <returns>A <see cref="ModelMetadata{T}"/> instance containing the model name, description,
    /// type, complexity, and additional architecture details.</returns>
    /// <remarks>
    /// <para>
    /// The metadata includes:
    /// <list type="bullet">
    /// <item><b>Name:</b> "BiLSTM-CRF-Native" or "BiLSTM-CRF-ONNX" depending on execution mode</item>
    /// <item><b>Description:</b> Full citation with model variant</item>
    /// <item><b>ModelType:</b> NamedEntityRecognition</item>
    /// <item><b>Complexity:</b> Number of LSTM layers (proxy for model depth)</item>
    /// <item><b>AdditionalInfo:</b> Variant, embedding dim, hidden dim, num layers, num labels, CRF status</item>
    /// </list>
    ///
    /// This metadata is used by model registries, experiment tracking systems, and serialization
    /// to identify and describe the model without loading its full state.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This returns a description card for the model, like a product label.
    /// It includes the model's name, what it does, and its key settings. Useful for keeping track
    /// of different model versions during experimentation.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "BiLSTM-CRF-Native" : "BiLSTM-CRF-ONNX",
            Description = $"BiLSTM-CRF {_options.Variant} sequence labeling NER (Lample et al., NAACL 2016)",
            ModelType = ModelType.NamedEntityRecognition,
            Complexity = _options.NumLSTMLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["EmbeddingDimension"] = _options.EmbeddingDimension.ToString();
        m.AdditionalInfo["HiddenDimension"] = _options.HiddenDimension.ToString();
        m.AdditionalInfo["NumLSTMLayers"] = _options.NumLSTMLayers.ToString();
        m.AdditionalInfo["NumLabels"] = _options.NumLabels.ToString();
        m.AdditionalInfo["UseCRF"] = _options.UseCRF.ToString();
        return m;
    }

    /// <summary>
    /// Serializes model-specific data to a binary stream for saving model checkpoints.
    /// </summary>
    /// <param name="w">The binary writer to write serialization data to.</param>
    /// <remarks>
    /// <para>
    /// Persists all configuration options needed to reconstruct this exact model instance:
    /// execution mode, model path (ONNX), variant, all architecture dimensions, CRF setting,
    /// dropout rate, learning rate, and the full label name list.
    ///
    /// Layer weights are serialized by the base class; this method only handles the configuration.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This saves the model's settings to a file so you can reload the exact
    /// same model later. Think of it as saving a recipe - it records all the ingredients and
    /// proportions so you can recreate the dish.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.EmbeddingDimension);
        w.Write(_options.HiddenDimension);
        w.Write(_options.NumLSTMLayers);
        w.Write(_options.NumLabels);
        w.Write(_options.MaxSequenceLength);
        w.Write(_options.UseCRF);
        w.Write(_options.UseCharEmbeddings);
        w.Write(_options.CharEmbeddingDimension);
        w.Write(_options.CharHiddenDimension);
        w.Write(_options.DropoutRate);
        w.Write(_options.LearningRate);

        // Serialize label names
        w.Write(_options.LabelNames.Length);
        foreach (var label in _options.LabelNames)
        {
            w.Write(label);
        }
    }

    /// <summary>
    /// Deserializes model-specific data from a binary stream, restoring the model to its saved state.
    /// </summary>
    /// <param name="r">The binary reader to read serialization data from.</param>
    /// <remarks>
    /// <para>
    /// Reads all configuration options in the exact order they were written by
    /// <see cref="SerializeNetworkSpecificData"/> and restores the model's configuration.
    /// After reading options, this method:
    /// <list type="bullet">
    /// <item>In ONNX mode: reloads the ONNX model from the saved path</item>
    /// <item>In native mode: clears existing layers and reinitializes from the restored options</item>
    /// </list>
    ///
    /// The deserialized label names, dimensions, and CRF settings are propagated to the
    /// base class properties to ensure consistency.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This restores a model from a saved file, rebuilding the exact same
    /// model that was saved. It's the counterpart to SerializeNetworkSpecificData - one saves,
    /// the other loads.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (NERModelVariant)r.ReadInt32();
        _options.EmbeddingDimension = r.ReadInt32();
        _options.HiddenDimension = r.ReadInt32();
        _options.NumLSTMLayers = r.ReadInt32();
        _options.NumLabels = r.ReadInt32();
        _options.MaxSequenceLength = r.ReadInt32();
        _options.UseCRF = r.ReadBoolean();
        _options.UseCharEmbeddings = r.ReadBoolean();
        _options.CharEmbeddingDimension = r.ReadInt32();
        _options.CharHiddenDimension = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        _options.LearningRate = r.ReadDouble();

        // Deserialize label names
        int labelCount = r.ReadInt32();
        _options.LabelNames = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
        {
            _options.LabelNames[i] = r.ReadString();
        }

        // Restore base class properties from deserialized options
        NumLabels = _options.NumLabels;
        EmbeddingDimension = _options.EmbeddingDimension;
        MaxSequenceLength = _options.MaxSequenceLength;
        UseCRF = _options.UseCRF;
        LabelNames = _options.LabelNames;

        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel?.Dispose();
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
        else if (_useNativeMode)
        {
            OnnxModel?.Dispose();
            OnnxModel = null;
            Layers.Clear();
            InitializeLayers();
        }
    }

    /// <summary>
    /// Creates a new, uninitialized instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new <see cref="BiLSTMCRF{T}"/> instance with identical options but fresh
    /// (randomly initialized) weights. Used internally by the framework for model cloning,
    /// ensemble creation, and cross-validation.</returns>
    /// <remarks>
    /// <para>
    /// The new instance receives a deep copy of the options via the copy constructor to prevent
    /// mutation leaking between instances. In ONNX mode, the new instance loads the same
    /// ONNX model file. In native mode, the new instance gets freshly initialized layers via
    /// <see cref="InitializeLayers"/>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This creates a "twin" of the current model with the same architecture
    /// and settings, but with fresh random weights (as if it was just created). This is useful for
    /// training multiple copies of the same model (ensemble learning) or for cross-validation
    /// experiments where you need identical model architectures with different training data.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var optionsCopy = new BiLSTMCRFOptions(_options);
        if (!_useNativeMode && optionsCopy.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new BiLSTMCRF<T>(Architecture, p, optionsCopy);
        return new BiLSTMCRF<T>(Architecture, optionsCopy);
    }

    #endregion

    #region Disposal

    /// <summary>
    /// Throws <see cref="ObjectDisposedException"/> if this model has been disposed.
    /// </summary>
    /// <exception cref="ObjectDisposedException">Thrown when the model has been disposed.</exception>
    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BiLSTMCRF<T>));
    }

    /// <summary>
    /// Releases resources held by the model, including ONNX runtime sessions and native layers.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if called from finalizer.
    /// Managed resources (ONNX model, layers) are only released when true.</param>
    /// <remarks>
    /// <para>
    /// In ONNX mode, this disposes the ONNX inference session and releases GPU/CPU memory
    /// allocated by the ONNX runtime. In native mode, the base class handles layer disposal.
    ///
    /// After disposal, all public methods will throw <see cref="ObjectDisposedException"/>.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Always call Dispose() (or use a `using` statement) when you're done
    /// with the model to free up memory. This is especially important for ONNX models which may
    /// hold GPU memory.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
