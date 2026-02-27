using AiDotNet.Interfaces;

namespace AiDotNet.NER.Interfaces;

/// <summary>
/// Base interface for all Named Entity Recognition (NER) AI models in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Named Entity Recognition (NER) is the task of identifying and classifying named entities
/// in text into predefined categories such as person names, organizations, locations, dates,
/// and more. This interface extends IFullModel to provide the core contract that all NER models
/// must implement, inheriting standard methods for training, inference, model persistence,
/// and gradient computation.
///
/// NER is a fundamental building block for many NLP applications:
/// - <b>Information extraction:</b> Pulling structured data from unstructured text
/// - <b>Question answering:</b> Identifying entities mentioned in questions and documents
/// - <b>Search engines:</b> Understanding queries that mention specific people, places, or organizations
/// - <b>Knowledge graph construction:</b> Populating knowledge bases from text corpora
/// - <b>Medical NLP:</b> Extracting drug names, diseases, and symptoms from clinical notes
/// </para>
/// <para>
/// <b>For Beginners:</b> A NER model reads text and highlights important named things in it,
/// like a highlighter that uses different colors for different types of entities.
///
/// For example, given the sentence "Albert Einstein worked at Princeton University in New Jersey":
/// - "Albert Einstein" is highlighted as a <b>PERSON</b> (someone's name)
/// - "Princeton University" is highlighted as an <b>ORGANIZATION</b> (an institution)
/// - "New Jersey" is highlighted as a <b>LOCATION</b> (a place)
///
/// The model works by processing each word (token) in the sentence and assigning it a label
/// from the BIO scheme:
/// - <b>B-</b> prefix means "Beginning of an entity" (e.g., B-PER for the first word of a person name)
/// - <b>I-</b> prefix means "Inside an entity" (e.g., I-PER for subsequent words of a person name)
/// - <b>O</b> means "Outside any entity" (regular words like "worked", "at", "in")
///
/// Key technical concepts:
/// - Input tensors represent token embeddings with shape [batch, sequenceLength, embeddingDim]
/// - Output tensors represent label predictions with shape [batch, sequenceLength] or label scores
/// - Models can run in Native mode (pure C# with full training) or ONNX mode (optimized inference)
/// - All models inherit full serialization, checkpointing, and gradient computation from IFullModel
///
/// Example usage:
/// <code>
/// // Create a BiLSTM-CRF NER model
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 100, outputSize: 9);
/// var model = new BiLSTMCRF&lt;double&gt;(architecture);
///
/// // Train on labeled data (token embeddings + BIO label sequences)
/// model.Train(tokenEmbeddings, labelSequence);
///
/// // Predict entities in new text
/// var predictions = model.PredictLabels(newTokenEmbeddings);
/// // predictions might be: [0, 1, 2, 0, 0, 3, 4] meaning O, B-PER, I-PER, O, O, B-ORG, I-ORG
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("NERModel")]
public interface INERModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    #region NER-Specific Properties

    /// <summary>
    /// Gets the number of entity label classes this model predicts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The number of labels depends on the annotation scheme and entity types being recognized.
    /// Common NER label sets include:
    /// - <b>CoNLL-2003:</b> 9 labels using BIO scheme for 4 entity types:
    ///   O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    /// - <b>OntoNotes 5.0:</b> 37 labels using BIO scheme for 18 entity types including
    ///   PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE,
    ///   DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL
    /// - <b>WNUT-17:</b> 13 labels for emerging/rare entities in social media text
    ///
    /// The BIO scheme uses 2 labels per entity type (B- and I-) plus one O label:
    /// numLabels = 2 * numEntityTypes + 1
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many different entity categories the model can
    /// distinguish. For example, with 9 labels the model can identify 4 types of entities
    /// (person, organization, location, miscellaneous) plus mark words that aren't part of
    /// any entity. More labels means the model recognizes more entity types.
    /// </para>
    /// </remarks>
    int NumLabels { get; }

    /// <summary>
    /// Gets the embedding dimension for input token representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Token embeddings are dense vector representations of words or subword tokens.
    /// Common embedding dimensions include:
    /// - <b>100:</b> GloVe-100d (compact, fast, good baseline for BiLSTM-CRF)
    /// - <b>300:</b> GloVe-300d or Word2Vec (richer representations, standard for research)
    /// - <b>768:</b> BERT-base hidden states (contextual embeddings from a transformer)
    /// - <b>1024:</b> BERT-large or RoBERTa-large hidden states
    ///
    /// The embedding dimension must match the pre-trained embeddings you feed into the model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Words need to be converted to numbers before a neural network
    /// can process them. An "embedding" is a list of numbers (a vector) that represents a word.
    /// The embedding dimension is how many numbers are in each word's vector. A dimension of
    /// 100 means each word is represented by 100 numbers that capture its meaning.
    /// </para>
    /// </remarks>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the expected input tensor shape for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// NER models accept token embedding tensors in these formats:
    /// - <b>2D tensor:</b> [sequenceLength, embeddingDim] for processing a single sentence
    /// - <b>3D tensor:</b> [batch, sequenceLength, embeddingDim] for processing multiple sentences at once
    ///
    /// The sequence length dimension represents the number of tokens in the sentence.
    /// Each token is represented by an embedding vector of size embeddingDim.
    /// If a sentence has fewer tokens than the expected sequence length, it should be padded.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what shape your input data needs to be.
    /// Think of it as: [how many words, how many numbers per word].
    /// If your sentence doesn't match this shape, you'll need to pad short sentences
    /// with zeros or truncate long sentences.
    /// </para>
    /// </remarks>
    int[] ExpectedInputShape { get; }

    #endregion

    #region NER-Specific Methods

    /// <summary>
    /// Performs NER prediction on input token embeddings and returns the optimal label sequence.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings tensor. Accepts either:
    /// - 2D: [sequenceLength, embeddingDim] for a single sentence
    /// - 3D: [batch, sequenceLength, embeddingDim] for multiple sentences</param>
    /// <returns>Predicted label indices as a tensor:
    /// - 1D: [sequenceLength] for a single sentence
    /// - 2D: [batch, sequenceLength] for multiple sentences
    /// Each value is an integer index into the label set (e.g., 0=O, 1=B-PER, 2=I-PER, etc.).</returns>
    /// <remarks>
    /// <para>
    /// This is the main inference method for NER. For models with CRF decoding (like BiLSTM-CRF),
    /// this method uses the Viterbi algorithm to find the globally optimal label sequence, which
    /// enforces structural constraints like valid BIO transitions. For models without CRF, it
    /// performs independent per-token classification using argmax.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main method to identify entities in text.
    /// You give it word embeddings (numerical representations of your text) and it returns
    /// a label for each word telling you what type of entity it is (or O if it's not an entity).
    ///
    /// Example: Given embeddings for ["John", "Smith", "works", "at", "Google"],
    /// the output might be [1, 2, 0, 0, 3] meaning:
    /// - 1 = B-PER (John is the start of a person name)
    /// - 2 = I-PER (Smith continues the person name)
    /// - 0 = O (works is not an entity)
    /// - 0 = O (at is not an entity)
    /// - 3 = B-ORG (Google is the start of an organization name)
    /// </para>
    /// </remarks>
    Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Trains the model on labeled NER data asynchronously with progress reporting.
    /// </summary>
    /// <param name="tokenEmbeddings">Input token embeddings tensor [batch, sequenceLength, embeddingDim].
    /// Each row is a sentence represented as a sequence of word embedding vectors.</param>
    /// <param name="labels">Target label sequences tensor [batch, sequenceLength].
    /// Each value is the ground-truth BIO label index for the corresponding token.</param>
    /// <param name="epochs">Number of complete passes over the training data. More epochs generally
    /// improve accuracy but risk overfitting. Typical values: 50-150 for BiLSTM-CRF.</param>
    /// <param name="progress">Optional progress reporter that receives updates after each epoch,
    /// including the current loss and F1 score.</param>
    /// <param name="cancellationToken">Optional token to cancel training early.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to recognize entities by showing it
    /// many examples of text with correctly labeled entities. Each "epoch" is one complete
    /// pass through all the training examples. The model gradually improves its accuracy
    /// over multiple epochs. You can monitor progress through the progress reporter.
    /// </para>
    /// </remarks>
    Task TrainAsync(
        Tensor<T> tokenEmbeddings,
        Tensor<T> labels,
        int epochs = 100,
        IProgress<NERTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Processes multiple sentences in a batch, predicting entity labels for each one.
    /// </summary>
    /// <param name="sequences">Collection of token embedding tensors, one per sentence.
    /// Each tensor has shape [sequenceLength, embeddingDim].</param>
    /// <returns>Collection of label prediction tensors, one per sentence.
    /// Each tensor has shape [sequenceLength] with integer label indices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you have multiple sentences to process, this method handles
    /// them all at once. It's more efficient than calling PredictLabels one sentence at a time
    /// because the model can process them in parallel.
    /// </para>
    /// </remarks>
    IEnumerable<Tensor<T>> PredictBatch(IEnumerable<Tensor<T>> sequences);

    /// <summary>
    /// Validates that an input tensor has the correct shape for this model.
    /// </summary>
    /// <param name="input">The input tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor shape doesn't match the expected
    /// input format (wrong rank, wrong embedding dimension, etc.).</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Call this before making predictions to check that your input
    /// data is in the right format. It will give you a clear error message if something is wrong,
    /// rather than letting the model fail with a confusing error deeper in the pipeline.
    /// </para>
    /// </remarks>
    void ValidateInputShape(Tensor<T> input);

    /// <summary>
    /// Gets a human-readable summary of the model architecture.
    /// </summary>
    /// <returns>A multi-line string describing the model's layers, dimensions, and configuration.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives you a text description of the model's structure,
    /// including how many layers it has, what dimensions they use, and how many parameters
    /// (learnable values) the model contains. Useful for debugging and understanding the model.
    /// </para>
    /// </remarks>
    string GetModelSummary();

    #endregion
}

/// <summary>
/// Reports training progress for NER models, including loss, F1 score, and epoch information.
/// </summary>
/// <remarks>
/// <para>
/// This class is used by <see cref="INERModel{T}.TrainAsync"/> to report progress during training.
/// The primary metrics for NER are:
/// - <b>Loss:</b> How wrong the model's predictions are (lower is better). For BiLSTM-CRF models,
///   this is the negative log-likelihood of the correct label sequence.
/// - <b>F1 Score:</b> The harmonic mean of precision and recall at the entity level (higher is better).
///   An F1 of 0.91 on CoNLL-2003 is considered state-of-the-art for BiLSTM-CRF models.
/// </para>
/// <para>
/// <b>For Beginners:</b> When training a NER model, you want to see the loss going down
/// and the F1 score going up over time. If the loss stops decreasing, training may have
/// converged. If the F1 score starts decreasing while loss keeps going down, the model
/// might be overfitting (memorizing training data instead of learning general patterns).
/// </para>
/// </remarks>
public class NERTrainingProgress
{
    /// <summary>
    /// Gets or sets the current epoch number (1-based).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An epoch is one complete pass through all training data.
    /// If CurrentEpoch is 5 and TotalEpochs is 100, the model has completed 5% of training.
    /// </para>
    /// </remarks>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Gets or sets the total number of epochs planned for training.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets the current mini-batch number within the current epoch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training data is often split into smaller chunks called "batches"
    /// to fit in memory. This tracks which batch within the current epoch is being processed.
    /// </para>
    /// </remarks>
    public int CurrentBatch { get; set; }

    /// <summary>
    /// Gets or sets the total number of mini-batches per epoch.
    /// </summary>
    public int TotalBatches { get; set; }

    /// <summary>
    /// Gets or sets the current training loss value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For BiLSTM-CRF models, this is the negative log-likelihood of the correct label sequence.
    /// For softmax-based models, this is the cross-entropy loss.
    /// Lower values indicate better training performance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The loss measures how wrong the model's predictions are.
    /// A loss of 0 would mean perfect predictions. You want to see this number decrease
    /// steadily during training. Typical starting values are 2-5, decreasing to 0.1-0.5.
    /// </para>
    /// </remarks>
    public double Loss { get; set; }

    /// <summary>
    /// Gets or sets the entity-level F1 score on validation data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// F1 score is the standard evaluation metric for NER. It is computed at the entity level
    /// (not token level), meaning an entity is only counted as correct if both its boundaries
    /// and type are predicted exactly right. Values range from 0.0 (worst) to 1.0 (perfect).
    /// State-of-the-art BiLSTM-CRF achieves ~0.91 F1 on CoNLL-2003.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> F1 score measures how well the model finds entities. A score of
    /// 1.0 means it found every entity perfectly. A score of 0.5 means it's getting about
    /// half right. For NER, we measure at the entity level - "John Smith" counts as one entity,
    /// and the model must get both words right to score a point.
    /// </para>
    /// </remarks>
    public double F1Score { get; set; }

    /// <summary>
    /// Gets or sets any additional metrics being tracked during training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common additional metrics include:
    /// - "Precision": fraction of predicted entities that are correct
    /// - "Recall": fraction of true entities that were found
    /// - "TokenAccuracy": per-token classification accuracy
    /// </para>
    /// </remarks>
    public Dictionary<string, double>? Metrics { get; set; }

    /// <summary>
    /// Gets the overall progress as a percentage (0-100).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This gives you a simple 0-100% progress value that accounts
    /// for both which epoch you're on and which batch within that epoch is being processed.
    /// </para>
    /// </remarks>
    public double ProgressPercentage => TotalEpochs > 0
        ? (CurrentEpoch - 1 + (TotalBatches > 0 ? (double)CurrentBatch / TotalBatches : 0)) / TotalEpochs * 100
        : 0;
}
