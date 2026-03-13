using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Validation;

namespace AiDotNet.RetrievalAugmentedGeneration.Generators;

/// <summary>
/// Neural network-based text generator for RAG systems using LSTM architecture.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations and relevance scoring.</typeparam>
/// <remarks>
/// <para>
/// This generator uses an LSTM neural network for token-by-token text generation in
/// retrieval-augmented generation tasks. It processes context through the trained LSTM
/// and generates responses using temperature-based sampling from the network's probability
/// distributions.
/// </para>
/// <para><b>For Beginners:</b> This generator uses a trained neural network to create answers.
///
/// Think of it like a smart writer:
/// - Takes your question and retrieved documents
/// - Converts text into numbers (tokens) the network understands
/// - Uses an LSTM neural network to predict next words
/// - Samples from probability distributions with temperature control
/// - Converts numbers back to readable text
///
/// How it works:
/// 1. Tokenizes input text into numerical IDs
/// 2. Feeds tokens through LSTM network layer-by-layer
/// 3. Network outputs probability distribution over vocabulary
/// 4. Samples next token using temperature (higher = more creative)
/// 5. Repeats until generating enough tokens or reaching end token
/// 6. Detokenizes back to human-readable text
///
/// Production considerations:
/// - Requires pre-trained LSTM network (not included)
/// - Actual LSTM forward pass for each token (computationally intensive)
/// - Temperature-based sampling for controlled randomness
/// - Configurable context and generation limits
/// - Proper error handling and edge cases
/// - Memory-efficient sequential processing
///
/// Note: This generator requires a pre-trained LSTM network with vocabulary matching
/// the configured vocabulary size. Without proper training, output will be meaningless.
/// For production use, you must:
/// 1. Train the LSTM on your domain data, OR
/// 2. Use transfer learning from a pre-trained language model
///
/// For production RAG systems, consider using LLM-based generators with pre-trained
/// models instead of training your own LSTM.
/// </para>
/// </remarks>
public class NeuralGenerator<T> : IGenerator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly LSTMNeuralNetwork<T> _network;
    private readonly int _maxContextTokens;
    private readonly int _maxGenerationTokens;
    private readonly double _temperature;
    private readonly int _vocabularySize;
    private readonly int _embeddingDimension;
    private readonly Dictionary<string, int> _wordToToken;
    private readonly Dictionary<int, string> _tokenToWord;
    private readonly T[,] _embeddingMatrix;
    private readonly object _vocabularyLock = new object();

    /// <summary>
    /// Gets the maximum number of tokens this generator can process in a single request.
    /// </summary>
    public int MaxContextTokens => _maxContextTokens;

    /// <summary>
    /// Gets the maximum number of tokens this generator can generate in a response.
    /// </summary>
    public int MaxGenerationTokens => _maxGenerationTokens;

    /// <summary>
    /// Initializes a new instance of the NeuralGenerator class.
    /// </summary>
    /// <param name="network">The LSTM neural network for text generation.</param>
    /// <param name="vocabularySize">Size of the token vocabulary (default: 50000).</param>
    /// <param name="embeddingDimension">Dimension of token embeddings (default: 256).</param>
    /// <param name="maxContextTokens">The maximum context tokens (default: 4096).</param>
    /// <param name="maxGenerationTokens">The maximum generation tokens (default: 1024).</param>
    /// <param name="temperature">Sampling temperature for generation (default: 0.7). Higher = more creative.</param>
    /// <param name="prebuiltVocabulary">Optional pre-built vocabulary for frozen token mapping. If provided, vocabulary is frozen and unknown words map to [UNK].</param>
    /// <remarks>
    /// The vocabulary is frozen after initialization when using a pre-built vocabulary. Unknown words encountered
    /// during tokenization are mapped to the [UNK] token to ensure compatibility with pre-trained networks.
    /// </remarks>
    public NeuralGenerator(
        LSTMNeuralNetwork<T> network,
        int vocabularySize = 50000,
        int embeddingDimension = 256,
        int maxContextTokens = 4096,
        int maxGenerationTokens = 1024,
        double temperature = 0.7,
        IDictionary<string, int>? prebuiltVocabulary = null)
    {
        Guard.NotNull(network);
        _network = network;

        if (vocabularySize <= 0)
            throw new ArgumentException("Vocabulary size must be positive", nameof(vocabularySize));
        if (embeddingDimension <= 0)
            throw new ArgumentException("Embedding dimension must be positive", nameof(embeddingDimension));
        if (maxContextTokens <= 0)
            throw new ArgumentException("MaxContextTokens must be greater than zero", nameof(maxContextTokens));
        if (maxGenerationTokens <= 0)
            throw new ArgumentException("MaxGenerationTokens must be greater than zero", nameof(maxGenerationTokens));
        if (temperature <= 0)
            throw new ArgumentException("Temperature must be positive", nameof(temperature));

        _vocabularySize = vocabularySize;
        _embeddingDimension = embeddingDimension;
        _maxContextTokens = maxContextTokens;
        _maxGenerationTokens = maxGenerationTokens;
        _temperature = temperature;

        // Validate network input dimension matches embedding dimension
        var networkInputShape = network.Architecture.GetInputShape();
        if (networkInputShape.Length > 0 && networkInputShape[networkInputShape.Length - 1] != embeddingDimension)
        {
            throw new ArgumentException(
                $"Network input dimension ({networkInputShape[networkInputShape.Length - 1]}) must match embedding dimension ({embeddingDimension})",
                nameof(network));
        }

        // Validate network output dimension matches vocabulary requirements
        var networkOutputShape = network.Architecture.GetOutputShape();
        if (networkOutputShape.Length > 0)
        {
            int networkOutputDim = networkOutputShape[networkOutputShape.Length - 1];
            if (networkOutputDim < vocabularySize)
                throw new ArgumentException(
                    $"Network output dimension ({networkOutputDim}) must be >= vocabulary size ({vocabularySize})",
                    nameof(network));
        }

        // Initialize embedding matrix with Xavier/Glorot initialization
        _embeddingMatrix = new T[vocabularySize, embeddingDimension];
        var random = RandomHelper.CreateSeededRandom(42);
        double initScale = Math.Sqrt(2.0 / (vocabularySize + embeddingDimension));

        for (int i = 0; i < vocabularySize; i++)
        {
            for (int j = 0; j < embeddingDimension; j++)
            {
                double value = (random.NextDouble() * 2.0 - 1.0) * initScale;
                _embeddingMatrix[i, j] = NumOps.FromDouble(value);
            }
        }

        // Initialize bidirectional vocabulary mapping
        if (prebuiltVocabulary != null)
        {
            // Use frozen vocabulary from pre-trained model
            _wordToToken = new Dictionary<string, int>(prebuiltVocabulary);
            _tokenToWord = new Dictionary<int, string>();
            foreach (var kvp in prebuiltVocabulary)
            {
                _tokenToWord[kvp.Value] = kvp.Key;
            }
        }
        else
        {
            // Initialize empty vocabulary with special tokens only
            _wordToToken = new Dictionary<string, int>();
            _tokenToWord = new Dictionary<int, string>();

            // Reserve special tokens
            _tokenToWord[0] = "[PAD]";   // Padding
            _tokenToWord[1] = "[UNK]";   // Unknown
            _tokenToWord[2] = "[BOS]";   // Beginning of sequence
            _tokenToWord[3] = "[EOS]";   // End of sequence

            _wordToToken["[PAD]"] = 0;
            _wordToToken["[UNK]"] = 1;
            _wordToToken["[BOS]"] = 2;
            _wordToToken["[EOS]"] = 3;
        }
    }

    /// <summary>
    /// Generates a text response based on a prompt.
    /// </summary>
    /// <param name="prompt">The input prompt or question.</param>
    /// <returns>The generated text response.</returns>
    public string Generate(string prompt)
    {
        if (string.IsNullOrWhiteSpace(prompt))
            throw new ArgumentException("Prompt cannot be null or empty", nameof(prompt));

        var tokens = TokenizeText(prompt);
        if (tokens.Count == 0)
            return "Unable to process empty input.";

        // Limit to context window
        if (tokens.Count > _maxContextTokens)
            tokens = tokens.Take(_maxContextTokens).ToList();

        // Generate response using neural network
        var generated = GenerateTokens(tokens, _maxGenerationTokens);
        return DetokenizeText(generated);
    }

    /// <summary>
    /// Generates a grounded answer using provided context documents.
    /// </summary>
    /// <param name="query">The user's original query or question.</param>
    /// <param name="context">The retrieved documents providing context for the answer.</param>
    /// <returns>A grounded answer with the generated text, source documents, and extracted citations.</returns>
    public GroundedAnswer<T> GenerateGrounded(string query, IEnumerable<Document<T>> context)
    {
        if (string.IsNullOrWhiteSpace(query))
            throw new ArgumentException("Query cannot be null or empty", nameof(query));

        var contextList = context?.ToList() ?? new List<Document<T>>();

        if (contextList.Count == 0)
        {
            return new GroundedAnswer<T>
            {
                Query = query,
                Answer = "I don't have enough information to answer this question based on the provided context.",
                SourceDocuments = new List<Document<T>>(),
                Citations = new List<string>(),
                ConfidenceScore = 0.0
            };
        }

        // Build prompt with context and query
        var promptBuilder = new StringBuilder();
        promptBuilder.AppendLine("Context documents:");
        promptBuilder.AppendLine();

        var citations = new List<string>();
        for (int i = 0; i < contextList.Count; i++)
        {
            var doc = contextList[i];
            var citationNum = i + 1;

            // Handle null content gracefully
            var content = doc.Content ?? "(no content)";
            promptBuilder.AppendLine($"[{citationNum}] {TruncateText(content, 500)}");
            promptBuilder.AppendLine();

            citations.Add($"[{citationNum}] Document ID: {doc.Id}");
        }

        promptBuilder.AppendLine($"Question: {query}");
        promptBuilder.AppendLine();
        promptBuilder.AppendLine("Answer based on the context:");

        // Generate answer using neural network
        var fullPrompt = promptBuilder.ToString();
        var generatedAnswer = Generate(fullPrompt);

        // Calculate confidence based on retrieval scores
        // IMPORTANT: This confidence reflects retrieval quality (how well documents matched the query),
        // NOT generation quality (how accurate or coherent the generated answer is).
        //
        // Limitations:
        // - Does not measure generation confidence or hallucination risk
        // - Does not track token probabilities during generation
        // - Assumes retrieval scores correlate with answer quality
        //
        // For production RAG systems, consider implementing:
        // 1. Token probability tracking during generation (average log-prob of generated tokens)
        // 2. Consistency checking (generate multiple answers and measure agreement)
        // 3. Entailment scoring (verify answer is entailed by retrieved documents)
        // 4. Combine all metrics: confidence = α*retrieval_score + β*generation_score + γ*entailment_score
        //
        // Current implementation: Simple average of document relevance scores as proxy for answer quality
        var avgScore = contextList
            .Where(d => d.HasRelevanceScore)
            .Select(d => Convert.ToDouble(d.RelevanceScore))
            .DefaultIfEmpty(0.6)
            .Average();

        var confidenceScore = Math.Min(1.0, Math.Max(0.0, avgScore));

        return new GroundedAnswer<T>
        {
            Query = query,
            Answer = generatedAnswer,
            SourceDocuments = contextList,
            Citations = citations,
            ConfidenceScore = confidenceScore
        };
    }

    private List<int> TokenizeText(string text)
    {
        // Production-ready word-based tokenization with thread-safe vocabulary tracking
        var words = text.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':' },
            StringSplitOptions.RemoveEmptyEntries);

        var tokens = new List<int>();

        foreach (var word in words)
        {
            var normalizedWord = word.ToLowerInvariant();

            int tokenId;
            lock (_vocabularyLock)
            {
                // Check if word exists in vocabulary
                if (!_wordToToken.TryGetValue(normalizedWord, out tokenId))
                {
                    // Map unknown words to [UNK] token (frozen vocabulary for pre-trained networks)
                    tokenId = 1;
                }
            }

            tokens.Add(tokenId);
        }

        return tokens;
    }

    private string DetokenizeText(List<int> tokens)
    {
        // Production-ready detokenization using vocabulary mapping
        var words = new List<string>();

        foreach (var tokenId in tokens)
        {
            if (_tokenToWord.TryGetValue(tokenId, out var word))
            {
                // Skip special tokens in output
                if (word != "[PAD]" && word != "[BOS]" && word != "[EOS]")
                {
                    words.Add(word == "[UNK]" ? "<unknown>" : word);
                }
            }
            else
            {
                words.Add("<unknown>");
            }
        }

        return string.Join(" ", words);
    }

    private List<int> GenerateTokens(List<int> inputTokens, int maxTokens)
    {
        var generated = new List<int>();
        var random = RandomHelper.CreateSeededRandom(42); // For temperature sampling

        // Start with input context
        var currentSequence = new List<int>(inputTokens);

        for (int i = 0; i < maxTokens; i++)
        {
            // Take last N tokens as context window (prevent excessive memory use)
            var contextWindow = currentSequence.Skip(Math.Max(0, currentSequence.Count - 128)).ToList();

            // Get next token using LSTM network
            var nextToken = PredictNextToken(contextWindow, random);

            // Stop if we generate end-of-sequence token ([EOS] = 3)
            if (nextToken == 3)
                break;

            generated.Add(nextToken);
            currentSequence.Add(nextToken);
        }

        return generated;
    }

    private int PredictNextToken(List<int> context, Random random)
    {
        if (context.Count == 0)
            return random.Next(1, _vocabularySize); // Fallback for empty context

        // Create input tensor from context tokens using embedding lookup
        var sequenceLength = context.Count;
        var embeddingData = new T[sequenceLength * _embeddingDimension];

        // Look up embeddings for each token
        for (int i = 0; i < sequenceLength; i++)
        {
            int tokenId = context[i];
            if (tokenId < 0 || tokenId >= _vocabularySize)
                tokenId = 1; // Use [UNK] token for out-of-vocabulary

            for (int j = 0; j < _embeddingDimension; j++)
            {
                embeddingData[i * _embeddingDimension + j] = _embeddingMatrix[tokenId, j];
            }
        }

        var inputVector = new Vector<T>(embeddingData);
        var inputTensor = new Tensor<T>(new[] { 1, sequenceLength, _embeddingDimension }, inputVector);

        // Forward pass through LSTM network
        var outputTensor = _network.Predict(inputTensor);

        // Extract logits for vocabulary
        // Output shape: [batch_size, sequence_length, output_dim]
        // We want the last time step's output
        var outputVector = outputTensor.ToVector();
        var outputDim = outputTensor.Shape[outputTensor.Shape.Length - 1];

        // Get the last time step's output
        var lastStepStart = outputVector.Length - outputDim;
        var logits = new T[Math.Min(outputDim, _vocabularySize)];

        for (int i = 0; i < logits.Length; i++)
        {
            logits[i] = outputVector[lastStepStart + i];
        }

        // Apply temperature and convert to probabilities
        var probabilities = ApplyTemperatureAndSoftmax(logits, _temperature);

        // Sample from probability distribution
        return SampleFromDistribution(probabilities, random);
    }

    private double[] ApplyTemperatureAndSoftmax(T[] logits, double temperature)
    {
        // Apply temperature scaling: logits / temperature
        var scaledLogits = new double[logits.Length];
        var maxLogit = double.NegativeInfinity;

        for (int i = 0; i < logits.Length; i++)
        {
            scaledLogits[i] = Convert.ToDouble(logits[i]) / temperature;
            if (scaledLogits[i] > maxLogit)
                maxLogit = scaledLogits[i];
        }

        // Subtract max for numerical stability
        var expSum = 0.0;
        for (int i = 0; i < scaledLogits.Length; i++)
        {
            scaledLogits[i] = Math.Exp(scaledLogits[i] - maxLogit);
            expSum += scaledLogits[i];
        }

        // Normalize to probabilities
        var probabilities = new double[scaledLogits.Length];
        for (int i = 0; i < scaledLogits.Length; i++)
        {
            probabilities[i] = scaledLogits[i] / expSum;
        }

        return probabilities;
    }

    private int SampleFromDistribution(double[] probabilities, Random random)
    {
        var sample = random.NextDouble();
        var cumulative = 0.0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            cumulative += probabilities[i];
            if (sample <= cumulative)
                return i;
        }

        // Fallback to last token (should rarely happen due to floating point precision)
        return probabilities.Length - 1;
    }

    private string TruncateText(string text, int maxLength)
    {
        if (string.IsNullOrEmpty(text) || text.Length <= maxLength)
            return text;

        return text.Substring(0, maxLength) + "...";
    }
}
