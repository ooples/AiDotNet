using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to latent semantic vectors using Latent Semantic Analysis (LSA/LSI).
/// </summary>
/// <remarks>
/// <para>
/// LSA (Latent Semantic Analysis), also known as LSI (Latent Semantic Indexing), uses
/// Singular Value Decomposition (SVD) to reduce the dimensionality of the term-document
/// matrix while capturing latent semantic relationships between terms and documents.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Creating a TF-IDF weighted term-document matrix
/// 2. Applying truncated SVD to get a low-rank approximation
/// 3. Projecting documents into the reduced semantic space
/// </para>
/// <para><b>For Beginners:</b> LSA discovers hidden topics in your text:
/// - "car" and "automobile" become similar because they appear in similar contexts
/// - Reduces thousands of word features to ~100-500 semantic concepts
/// - Great for document similarity, clustering, and information retrieval
/// - Can handle synonymy (different words, same meaning) and polysemy (same word, different meanings)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LSAVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _nComponents;
    private readonly int _nIterations;
    private readonly double _tolerance;
    private readonly int? _randomState;

    private TfidfVectorizer<T>? _tfidfVectorizer;
    private double[,]? _components; // V^T from SVD: (n_components, n_features)
    private double[]? _singularValues;
    private double[]? _explainedVarianceRatio;

    /// <summary>
    /// Gets the number of components (latent dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the singular values from the SVD decomposition.
    /// </summary>
    public double[]? SingularValues => _singularValues;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public double[]? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets the components (topic-term matrix) from LSA.
    /// </summary>
    /// <remarks>
    /// Each row represents a latent topic, and each column represents a term's
    /// contribution to that topic.
    /// </remarks>
    public double[,]? Components => _components;

    /// <summary>
    /// Creates a new instance of <see cref="LSAVectorizer{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of latent components/topics. Defaults to 100.</param>
    /// <param name="nIterations">Number of iterations for SVD algorithm. Defaults to 5.</param>
    /// <param name="tolerance">Convergence tolerance for iterative SVD. Defaults to 1e-7.</param>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public LSAVectorizer(
        int nComponents = 100,
        int nIterations = 5,
        double tolerance = 1e-7,
        int minDf = 1,
        double maxDf = 1.0,
        int? maxFeatures = null,
        (int Min, int Max)? nGramRange = null,
        bool lowercase = true,
        int? randomState = null,
        Func<string, IEnumerable<string>>? tokenizer = null,
        HashSet<string>? stopWords = null,
        ITokenizer? advancedTokenizer = null)
        : base(minDf, maxDf, maxFeatures, nGramRange, lowercase, tokenizer, stopWords, advancedTokenizer)
    {
        if (nComponents < 1)
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));

        _nComponents = nComponents;
        _nIterations = nIterations;
        _tolerance = tolerance;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _components is not null && _tfidfVectorizer is not null;

    /// <inheritdoc/>
    public override int FeatureCount => _nComponents;

    /// <summary>
    /// Fits the LSA vectorizer to the corpus.
    /// </summary>
    /// <param name="documents">The text documents to learn latent topics from.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // First, create TF-IDF representation
        _tfidfVectorizer = new TfidfVectorizer<T>(
            minDf: _minDf,
            maxDf: _maxDf,
            maxFeatures: _maxFeatures,
            nGramRange: _nGramRange,
            lowercase: _lowercase,
            tokenizer: _tokenizer,
            stopWords: _stopWords);

        var tfidfMatrix = _tfidfVectorizer.FitTransform(docList);
        _vocabulary = _tfidfVectorizer.Vocabulary;
        _featureNames = Enumerable.Range(0, _nComponents).Select(i => $"lsa_component_{i}").ToArray();

        // Convert to double array for SVD
        int nDocs = tfidfMatrix.Rows;
        int nFeatures = tfidfMatrix.Columns;
        int actualComponents = Math.Min(_nComponents, Math.Min(nDocs, nFeatures));

        var matrix = new double[nDocs, nFeatures];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                matrix[i, j] = NumOps.ToDouble(tfidfMatrix[i, j]);
            }
        }

        // Perform truncated SVD using power iteration method
        PerformTruncatedSVD(matrix, actualComponents);
    }

    /// <summary>
    /// Performs truncated SVD using randomized algorithm.
    /// </summary>
    private void PerformTruncatedSVD(double[,] matrix, int nComponents)
    {
        int m = matrix.GetLength(0); // documents
        int n = matrix.GetLength(1); // features

        var random = _randomState.HasValue ? new Random(_randomState.Value) : new Random();

        // Initialize random matrix Q (n x nComponents)
        var Q = new double[n, nComponents];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < nComponents; j++)
            {
                Q[i, j] = random.NextDouble() - 0.5;
            }
        }

        // Power iteration to approximate right singular vectors
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Y = A * Q
            var Y = MultiplyMatrices(matrix, Q, m, n, nComponents);

            // Orthonormalize Y using modified Gram-Schmidt
            Y = OrthonormalizeColumns(Y, m, nComponents);

            // Q = A^T * Y
            Q = MultiplyTransposeMatrices(matrix, Y, n, m, nComponents);

            // Orthonormalize Q
            Q = OrthonormalizeColumns(Q, n, nComponents);
        }

        // Final computation: U = A * Q, then compute sigma
        var U = MultiplyMatrices(matrix, Q, m, n, nComponents);

        // Compute singular values and normalize U
        _singularValues = new double[nComponents];
        for (int j = 0; j < nComponents; j++)
        {
            double sigma = 0;
            for (int i = 0; i < m; i++)
            {
                sigma += U[i, j] * U[i, j];
            }
            sigma = Math.Sqrt(sigma);
            _singularValues[j] = sigma;

            if (sigma > _tolerance)
            {
                for (int i = 0; i < m; i++)
                {
                    U[i, j] /= sigma;
                }
            }
        }

        // Store V^T (components)
        _components = new double[nComponents, n];
        for (int i = 0; i < nComponents; i++)
        {
            for (int j = 0; j < n; j++)
            {
                _components[i, j] = Q[j, i];
            }
        }

        // Compute explained variance ratio
        double totalVariance = 0;
        for (int i = 0; i < nComponents; i++)
        {
            totalVariance += _singularValues[i] * _singularValues[i];
        }

        _explainedVarianceRatio = new double[nComponents];
        for (int i = 0; i < nComponents; i++)
        {
            _explainedVarianceRatio[i] = (_singularValues[i] * _singularValues[i]) / totalVariance;
        }
    }

    private double[,] MultiplyMatrices(double[,] A, double[,] B, int m, int k, int n)
    {
        var result = new double[m, n];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += A[i, l] * B[l, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    private double[,] MultiplyTransposeMatrices(double[,] A, double[,] B, int n, int m, int k)
    {
        // Computes A^T * B where A is (m x n), result is (n x k)
        var result = new double[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int l = 0; l < m; l++)
                {
                    sum += A[l, i] * B[l, j];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    private double[,] OrthonormalizeColumns(double[,] matrix, int m, int n)
    {
        var result = new double[m, n];
        Array.Copy(matrix, result, matrix.Length);

        for (int j = 0; j < n; j++)
        {
            // Subtract projections onto previous columns
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++)
                {
                    dot += result[i, j] * result[i, k];
                }
                for (int i = 0; i < m; i++)
                {
                    result[i, j] -= dot * result[i, k];
                }
            }

            // Normalize
            double norm = 0;
            for (int i = 0; i < m; i++)
            {
                norm += result[i, j] * result[i, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > _tolerance)
            {
                for (int i = 0; i < m; i++)
                {
                    result[i, j] /= norm;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Transforms documents to LSA vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's LSA vector in latent semantic space.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_components is null || _tfidfVectorizer is null)
        {
            throw new InvalidOperationException("LSAVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        // Transform to TF-IDF first
        var docList = documents.ToList();
        var tfidfMatrix = _tfidfVectorizer.Transform(docList);

        int nDocs = tfidfMatrix.Rows;
        int nFeatures = tfidfMatrix.Columns;
        int nComponents = _components.GetLength(0);

        // Project into latent space: X_lsa = X_tfidf * V
        var result = new T[nDocs, nComponents];

        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < nComponents; j++)
            {
                double sum = 0;
                for (int k = 0; k < nFeatures; k++)
                {
                    sum += NumOps.ToDouble(tfidfMatrix[i, k]) * _components[j, k];
                }
                result[i, j] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }
}
