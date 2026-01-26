using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to sentence embeddings using Smooth Inverse Frequency (SIF).
/// </summary>
/// <remarks>
/// <para>
/// SIF is a simple but effective method for creating sentence embeddings from word vectors.
/// It computes a weighted average of word vectors using smooth inverse frequency weights,
/// then removes the first principal component to create better sentence representations.
/// </para>
/// <para>
/// The algorithm:
/// 1. Compute weighted average: v_s = (1/|s|) * Σ (a / (a + p(w))) * v_w
/// 2. Remove first principal component: v_s = v_s - u * u^T * v_s
/// </para>
/// <para><b>For Beginners:</b> SIF creates sentence embeddings that are surprisingly good:
/// - Simple: just weighted averaging of word vectors
/// - Effective: often competitive with complex deep learning methods
/// - Fast: no neural network inference required
/// - Requires pre-trained word vectors (Word2Vec, GloVe, etc.)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SIFVectorizer<T> : TextVectorizerBase<T>
{
    private readonly double _alpha;
    private readonly bool _removePrincipalComponent;
    private readonly int _nPrincipalComponents;

    private Dictionary<string, double[]>? _wordVectors;
    private Dictionary<string, double>? _wordFrequencies;
    private double[][]? _principalComponents;
    private int _vectorSize;

    /// <summary>
    /// Gets the vector dimensionality.
    /// </summary>
    public int VectorSize => _vectorSize;

    /// <summary>
    /// Creates a new instance of <see cref="SIFVectorizer{T}"/>.
    /// </summary>
    /// <param name="wordVectors">Pre-trained word vectors (e.g., from Word2Vec, GloVe).</param>
    /// <param name="alpha">SIF weighting parameter. Defaults to 1e-3.</param>
    /// <param name="removePrincipalComponent">Whether to remove principal components. Defaults to true.</param>
    /// <param name="nPrincipalComponents">Number of principal components to remove. Defaults to 1.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public SIFVectorizer(
        Dictionary<string, double[]> wordVectors,
        double alpha = 1e-3,
        bool removePrincipalComponent = true,
        int nPrincipalComponents = 1,
        bool lowercase = true,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(1, 1.0, null, (1, 1), lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        if (wordVectors is null || wordVectors.Count == 0)
            throw new ArgumentException("Word vectors cannot be null or empty.", nameof(wordVectors));

        _wordVectors = wordVectors;
        _vectorSize = wordVectors.Values.First().Length;
        _alpha = alpha;
        _removePrincipalComponent = removePrincipalComponent;
        _nPrincipalComponents = nPrincipalComponents;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _wordFrequencies is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _vectorSize;

    /// <summary>
    /// Fits the SIF vectorizer by computing word frequencies and principal components.
    /// </summary>
    /// <param name="documents">The text documents to fit on.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // Compute word frequencies
        _wordFrequencies = new Dictionary<string, double>();
        int totalWords = 0;

        foreach (string doc in docList)
        {
            var tokens = Tokenize(doc);
            foreach (string token in tokens)
            {
                _wordFrequencies.TryGetValue(token, out double count);
                _wordFrequencies[token] = count + 1;
                totalWords++;
            }
        }

        // Normalize to probabilities
        foreach (var key in _wordFrequencies.Keys.ToList())
        {
            _wordFrequencies[key] /= totalWords;
        }

        _vocabulary = _wordVectors!.Keys
            .Where(k => _wordFrequencies.ContainsKey(k))
            .Select((k, i) => (k, i))
            .ToDictionary(x => x.k, x => x.i);
        _featureNames = Enumerable.Range(0, _vectorSize).Select(i => $"sif_dim_{i}").ToArray();

        if (_removePrincipalComponent && docList.Count > _nPrincipalComponents)
        {
            // Compute sentence embeddings first (without PC removal)
            var embeddings = ComputeEmbeddings(docList, removePC: false);

            // Compute principal components
            _principalComponents = ComputePrincipalComponents(embeddings, _nPrincipalComponents);
        }
    }

    private double[,] ComputeEmbeddings(List<string> documents, bool removePC)
    {
        int nDocs = documents.Count;
        var embeddings = new double[nDocs, _vectorSize];

        for (int d = 0; d < nDocs; d++)
        {
            var tokens = Tokenize(documents[d]).ToList();
            if (tokens.Count == 0) continue;

            // Compute weighted average
            int count = 0;
            foreach (string token in tokens)
            {
                if (!_wordVectors!.TryGetValue(token, out var wordVec)) continue;
                if (!_wordFrequencies!.TryGetValue(token, out double freq)) continue;

                double weight = _alpha / (_alpha + freq);

                for (int i = 0; i < _vectorSize; i++)
                {
                    embeddings[d, i] += weight * wordVec[i];
                }
                count++;
            }

            if (count > 0)
            {
                for (int i = 0; i < _vectorSize; i++)
                {
                    embeddings[d, i] /= count;
                }
            }
        }

        // Remove principal components
        if (removePC && _principalComponents is not null)
        {
            for (int d = 0; d < nDocs; d++)
            {
                foreach (var pc in _principalComponents)
                {
                    // Project onto PC
                    double projection = 0;
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        projection += embeddings[d, i] * pc[i];
                    }

                    // Subtract projection
                    for (int i = 0; i < _vectorSize; i++)
                    {
                        embeddings[d, i] -= projection * pc[i];
                    }
                }
            }
        }

        return embeddings;
    }

    private double[][] ComputePrincipalComponents(double[,] embeddings, int nComponents)
    {
        int nDocs = embeddings.GetLength(0);
        int dim = embeddings.GetLength(1);

        // Center the data
        var mean = new double[dim];
        for (int d = 0; d < nDocs; d++)
        {
            for (int i = 0; i < dim; i++)
            {
                mean[i] += embeddings[d, i];
            }
        }
        for (int i = 0; i < dim; i++)
        {
            mean[i] /= nDocs;
        }

        var centered = new double[nDocs, dim];
        for (int d = 0; d < nDocs; d++)
        {
            for (int i = 0; i < dim; i++)
            {
                centered[d, i] = embeddings[d, i] - mean[i];
            }
        }

        // Power iteration to find principal components
        var pcs = new List<double[]>();
        var random = new Random(42);

        for (int comp = 0; comp < nComponents; comp++)
        {
            // Initialize random vector
            var v = new double[dim];
            double norm = 0;
            for (int i = 0; i < dim; i++)
            {
                v[i] = random.NextDouble() - 0.5;
                norm += v[i] * v[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < dim; i++)
            {
                v[i] /= norm;
            }

            // Power iteration
            for (int iter = 0; iter < 20; iter++)
            {
                // v = X^T * X * v
                var Xv = new double[nDocs];
                for (int d = 0; d < nDocs; d++)
                {
                    for (int i = 0; i < dim; i++)
                    {
                        Xv[d] += centered[d, i] * v[i];
                    }
                }

                var newV = new double[dim];
                for (int i = 0; i < dim; i++)
                {
                    for (int d = 0; d < nDocs; d++)
                    {
                        newV[i] += centered[d, i] * Xv[d];
                    }
                }

                // Remove projections onto previous PCs
                foreach (var prevPc in pcs)
                {
                    double dot = 0;
                    for (int i = 0; i < dim; i++)
                    {
                        dot += newV[i] * prevPc[i];
                    }
                    for (int i = 0; i < dim; i++)
                    {
                        newV[i] -= dot * prevPc[i];
                    }
                }

                // Normalize
                norm = 0;
                for (int i = 0; i < dim; i++)
                {
                    norm += newV[i] * newV[i];
                }
                norm = Math.Sqrt(norm);
                if (norm > 1e-10)
                {
                    for (int i = 0; i < dim; i++)
                    {
                        v[i] = newV[i] / norm;
                    }
                }
            }

            pcs.Add(v);
        }

        return pcs.ToArray();
    }

    /// <summary>
    /// Transforms documents to SIF sentence embeddings.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's SIF embedding.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_wordFrequencies is null || _wordVectors is null)
        {
            throw new InvalidOperationException("SIFVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        var embeddings = ComputeEmbeddings(docList, removePC: _removePrincipalComponent);

        // Convert to output type
        int nDocs = docList.Count;
        var output = new T[nDocs, _vectorSize];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < _vectorSize; j++)
            {
                output[i, j] = NumOps.FromDouble(embeddings[i, j]);
            }
        }

        return new Matrix<T>(output);
    }
}
