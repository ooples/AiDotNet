using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// Implements a static word embedding model (e.g., GloVe, Word2Vec, FastText).
/// </summary>
/// <typeparam name="T">The numeric type for vector operations.</typeparam>
/// <remarks>
/// <para>
/// This model loads pre-trained word vectors from a file (standard text format where each line is "word val1 val2 ...")
/// and computes sentence embeddings by averaging the vectors of the words in the sentence.
/// </para>
/// <para><b>For Beginners:</b> Before Transformers (like BERT), we used "static" embeddings.
/// - "Static" means a word always has the same vector, regardless of context.
/// - "bank" in "river bank" and "bank account" gets the same vector.
/// - Transformer models generate "contextual" embeddings where "bank" would differ based on the sentence.
/// 
/// Despite being older, these models are:
/// - Very fast (simple lookup + average)
/// - Low memory (if vocabulary is pruned)
/// - Good baselines for simple tasks
/// </para>
/// </remarks>
public class StaticWordEmbeddingModel<T> : EmbeddingModelBase<T>
{
    private readonly Dictionary<string, Vector<T>> _wordVectors;
    private readonly int _dimension;
    private readonly Vector<T> _unknownVector;
    private readonly bool _ignoreUnknown;

    public override int EmbeddingDimension => _dimension;
    
    // Static models don't really have a token limit like transformers, but we set a reasonable default
    public override int MaxTokens => 10000; 

    /// <summary>
    /// Initializes a new instance of the <see cref="StaticWordEmbeddingModel{T}"/> class.
    /// </summary>
    /// <param name="wordVectors">A dictionary mapping words to their vector representations.</param>
    /// <param name="dimension">The dimension of the embedding vectors.</param>
    /// <param name="ignoreUnknown">If true, unknown words are skipped. If false, an unknown token vector is used.</param>
    public StaticWordEmbeddingModel(
        Dictionary<string, Vector<T>> wordVectors, 
        int dimension, 
        bool ignoreUnknown = true)
    {
        if (wordVectors == null || wordVectors.Count == 0)
            throw new ArgumentException("Word vector dictionary cannot be empty", nameof(wordVectors));

        _wordVectors = wordVectors;
        _dimension = dimension;
        _ignoreUnknown = ignoreUnknown;
        _unknownVector = new Vector<T>(dimension); // Zero vector
    }

    /// <summary>
    /// Loads embeddings from a standard text format file (GloVe/FastText format).
    /// </summary>
    /// <param name="filePath">Path to the embedding file.</param>
    /// <param name="vocabLimit">Optional limit on vocabulary size to save memory.</param>
    /// <returns>A new instance of StaticWordEmbeddingModel.</returns>
    public static StaticWordEmbeddingModel<T> LoadFromTextFile(string filePath, int? vocabLimit = null)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Embedding file not found: {filePath}");

        var wordVectors = new Dictionary<string, Vector<T>>();
        int dimension = 0;
        var numOps = MathHelper.GetNumericOperations<T>();

        using (var reader = new StreamReader(filePath))
        {
            string? line;
            int count = 0;
            while ((line = reader.ReadLine()) != null)
            {
                if (vocabLimit.HasValue && count >= vocabLimit.Value) break;

                var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                
                // Skip header lines if present (FastText sometimes has one)
                if (parts.Length <= 2) continue;

                var word = parts[0];
                
                // Determine dimension from first valid line
                if (dimension == 0)
                {
                    dimension = parts.Length - 1;
                }
                else if (parts.Length - 1 != dimension)
                {
                    // Skip malformed lines
                    continue;
                }

                var values = new T[dimension];
                for (int i = 0; i < dimension; i++)
                {
                    if (double.TryParse(parts[i + 1], out double val))
                    {
                        values[i] = numOps.FromDouble(val);
                    }
                }

                wordVectors[word] = new Vector<T>(values);
                count++;
            }
        }

        return new StaticWordEmbeddingModel<T>(wordVectors, dimension);
    }

    protected override Vector<T> EmbedCore(string text)
    {
        var words = Tokenize(text);
        if (words.Count == 0) return new Vector<T>(_dimension); 

        var sumVector = new Vector<T>(_dimension);
        int validWords = 0;

        foreach (var word in words)
        {
            if (_wordVectors.TryGetValue(word, out var vector))
            {
                sumVector = sumVector.Add(vector);
                validWords++;
            }
            else if (_wordVectors.TryGetValue(word.ToLowerInvariant(), out var lowerVector))
            {
                sumVector = sumVector.Add(lowerVector);
                validWords++;
            }
            else if (!_ignoreUnknown)
            {
                sumVector = sumVector.Add(_unknownVector);
                validWords++;
            }
        }

        if (validWords == 0) return new Vector<T>(_dimension);

        // Compute mean
        var meanVector = sumVector.Divide(NumOps.FromDouble(validWords));
        
        // Normalize
        return meanVector.Normalize();
    }

    private List<string> Tokenize(string text)
    {
        // Simple whitespace tokenizer for static embeddings
        return text.Split(new[] { ' ', '\t', '\n', '\r', '.', ',', '!', '?', ';', ':', '(', ')', '[', ']', '"', '\'' }, 
            StringSplitOptions.RemoveEmptyEntries).ToList();
    }
}
