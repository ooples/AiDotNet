using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.Preprocessing.TextVectorizers;

/// <summary>
/// Converts text documents to topic vectors using Non-negative Matrix Factorization (NMF).
/// </summary>
/// <remarks>
/// <para>
/// NMF factorizes the term-document matrix V into two non-negative matrices W and H such that V ≈ W × H.
/// - W: document-topic matrix (n_docs × n_topics)
/// - H: topic-term matrix (n_topics × n_terms)
/// </para>
/// <para>
/// Unlike LSA which can have negative values, NMF produces purely additive, parts-based representations
/// that are often more interpretable. Topics are combinations of words with positive weights.
/// </para>
/// <para><b>For Beginners:</b> NMF is like LSA but with only positive values:
/// - Topics are "built from" words (additive combinations)
/// - Results are often more interpretable than LSA
/// - Works well for topic modeling when you want to "see" what makes up each topic
/// - Each document is a combination of topics with positive weights
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class NMFVectorizer<T> : TextVectorizerBase<T>
{
    private readonly int _nComponents;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly int? _randomState;
    private readonly NMFInitialization _init;

    private int _effectiveComponents;
    private TfidfVectorizer<T>? _tfidfVectorizer;
    private double[,]? _components; // H: (n_components, n_features)

    /// <summary>
    /// Gets the number of components (topics).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the topic-term matrix (H) from NMF.
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Creates a new instance of <see cref="NMFVectorizer{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components/topics. Defaults to 10.</param>
    /// <param name="maxIterations">Maximum number of iterations. Defaults to 200.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="init">Initialization method. Defaults to Random.</param>
    /// <param name="minDf">Minimum document frequency (absolute count). Defaults to 1.</param>
    /// <param name="maxDf">Maximum document frequency (proportion 0-1). Defaults to 1.0.</param>
    /// <param name="maxFeatures">Maximum vocabulary size. Null for unlimited.</param>
    /// <param name="nGramRange">N-gram range (min, max). Defaults to (1, 1) for unigrams.</param>
    /// <param name="lowercase">Convert all text to lowercase. Defaults to true.</param>
    /// <param name="randomState">Random seed for reproducibility. Null for random.</param>
    /// <param name="tokenizer">Custom tokenizer function. Null for default.</param>
    /// <param name="stopWords">Words to exclude. Null for no filtering.</param>
    /// <param name="advancedTokenizer">Optional ITokenizer for subword tokenization.</param>
    public NMFVectorizer(
        int nComponents = 10,
        int maxIterations = 200,
        double tolerance = 1e-4,
        NMFInitialization init = NMFInitialization.Random,
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
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _init = init;
        _randomState = randomState;
    }

    /// <inheritdoc/>
    public override bool IsFitted => _components is not null && _tfidfVectorizer is not null;

    /// <inheritdoc/>
    /// <remarks>
    /// Returns the effective number of components, which may be less than the requested
    /// nComponents if the corpus is smaller than the requested dimensions.
    /// </remarks>
    public override int FeatureCount => _effectiveComponents > 0 ? _effectiveComponents : _nComponents;

    /// <summary>
    /// Fits the NMF model to the corpus using multiplicative update rules.
    /// </summary>
    /// <param name="documents">The text documents to learn topics from.</param>
    public override void Fit(IEnumerable<string> documents)
    {
        var docList = documents.ToList();
        _nDocs = docList.Count;

        // Create TF-IDF representation
        _tfidfVectorizer = new TfidfVectorizer<T>(
            minDf: _minDf,
            maxDf: _maxDf,
            maxFeatures: _maxFeatures,
            nGramRange: _nGramRange,
            lowercase: _lowercase,
            tokenizer: _tokenizer,
            stopWords: _stopWords,
            advancedTokenizer: _advancedTokenizer);

        var tfidfMatrix = _tfidfVectorizer.FitTransform(docList);
        _vocabulary = _tfidfVectorizer.Vocabulary;

        int nDocs = tfidfMatrix.Rows;
        int nFeatures = tfidfMatrix.Columns;

        // Calculate effective components - may be less than requested if corpus is small
        _effectiveComponents = Math.Min(_nComponents, Math.Min(nDocs, nFeatures));
        if (_effectiveComponents < 1)
            throw new InvalidOperationException("NMF requires at least one component; check corpus size and vocabulary.");

        // Set feature names based on effective components
        _featureNames = Enumerable.Range(0, _effectiveComponents).Select(i => $"nmf_topic_{i}").ToArray();

        // Convert to double array
        var V = new double[nDocs, nFeatures];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                V[i, j] = Math.Max(0, NumOps.ToDouble(tfidfMatrix[i, j])); // Ensure non-negative
            }
        }

        // Initialize W and H
        var random = _randomState.HasValue ? RandomHelper.CreateSeededRandom(_randomState.Value) : RandomHelper.CreateSecureRandom();
        int k = _effectiveComponents;

        var W = new double[nDocs, k];
        var H = new double[k, nFeatures];

        if (_init == NMFInitialization.NNDSVD)
        {
            InitializeNNDSVD(V, W, H, nDocs, nFeatures, k);
        }
        else
        {
            // Random initialization
            for (int i = 0; i < nDocs; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    W[i, j] = random.NextDouble() * 0.1 + 0.01;
                }
            }
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    H[i, j] = random.NextDouble() * 0.1 + 0.01;
                }
            }
        }

        // Multiplicative update rules (Lee & Seung)
        double prevError = double.MaxValue;
        const double eps = 1e-10;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update W: W = W * (V * H^T) / (W * H * H^T)
            var VHt = MultiplyMatrixTranspose(V, H, nDocs, nFeatures, k);
            var WHHt = MultiplyMatrices(MultiplyMatrices(W, H, nDocs, k, nFeatures), TransposeMatrix(H, k, nFeatures), nDocs, nFeatures, k);

            for (int i = 0; i < nDocs; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    W[i, j] = W[i, j] * VHt[i, j] / (WHHt[i, j] + eps);
                }
            }

            // Update H: H = H * (W^T * V) / (W^T * W * H)
            var WtV = MultiplyTransposeMatrix(W, V, k, nDocs, nFeatures);
            var WtWH = MultiplyMatrices(MultiplyTransposeMatrix(W, W, k, nDocs, k), H, k, k, nFeatures);

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < nFeatures; j++)
                {
                    H[i, j] = H[i, j] * WtV[i, j] / (WtWH[i, j] + eps);
                }
            }

            // Check convergence
            if (iter % 10 == 0)
            {
                double error = ComputeFrobeniusError(V, W, H, nDocs, nFeatures, k);
                if (Math.Abs(prevError - error) < _tolerance)
                    break;
                prevError = error;
            }
        }

        _components = H;
    }

    /// <summary>
    /// NNDSVD-inspired initialization using scaled random values.
    /// </summary>
    /// <remarks>
    /// This is a simplified approximation that uses the mean of the input matrix
    /// to scale random initialization values. It does not implement the full NNDSVD
    /// algorithm (which requires SVD decomposition), but provides better starting
    /// values than pure random initialization for faster convergence.
    /// </remarks>
    private void InitializeNNDSVD(double[,] V, double[,] W, double[,] H, int m, int n, int k)
    {
        var random = _randomState.HasValue ? RandomHelper.CreateSeededRandom(_randomState.Value) : RandomHelper.CreateSecureRandom();

        // Initialize with small positive random values as fallback
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                W[i, j] = random.NextDouble() * 0.1 + 0.01;
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                H[i, j] = random.NextDouble() * 0.1 + 0.01;
            }
        }

        // Use mean of V for scaling
        double mean = 0;
        int count = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (V[i, j] > 0)
                {
                    mean += V[i, j];
                    count++;
                }
            }
        }
        mean = count > 0 ? mean / count : 1;

        double scale = Math.Sqrt(mean / k);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                W[i, j] *= scale;
            }
        }
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                H[i, j] *= scale;
            }
        }
    }

    private double ComputeFrobeniusError(double[,] V, double[,] W, double[,] H, int m, int n, int k)
    {
        double error = 0;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double approx = 0;
                for (int l = 0; l < k; l++)
                {
                    approx += W[i, l] * H[l, j];
                }
                double diff = V[i, j] - approx;
                error += diff * diff;
            }
        }
        return Math.Sqrt(error);
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

    private double[,] MultiplyMatrixTranspose(double[,] A, double[,] B, int m, int n, int k)
    {
        // Computes A * B^T where B is (k x n), result is (m x k)
        var result = new double[m, k];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int l = 0; l < n; l++)
                {
                    sum += A[i, l] * B[j, l];
                }
                result[i, j] = sum;
            }
        }
        return result;
    }

    private double[,] MultiplyTransposeMatrix(double[,] A, double[,] B, int k, int m, int n)
    {
        // Computes A^T * B where A is (m x k), result is (k x n)
        var result = new double[k, n];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
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

    private double[,] TransposeMatrix(double[,] A, int m, int n)
    {
        var result = new double[n, m];
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[j, i] = A[i, j];
            }
        }
        return result;
    }

    /// <summary>
    /// Transforms documents to NMF topic vectors.
    /// </summary>
    /// <param name="documents">The documents to transform.</param>
    /// <returns>Matrix where each row is a document's topic representation.</returns>
    public override Matrix<T> Transform(IEnumerable<string> documents)
    {
        if (_components is null || _tfidfVectorizer is null)
        {
            throw new InvalidOperationException("NMFVectorizer has not been fitted. Call Fit() or FitTransform() first.");
        }

        var docList = documents.ToList();
        var tfidfMatrix = _tfidfVectorizer.Transform(docList);

        int nDocs = tfidfMatrix.Rows;
        int nFeatures = tfidfMatrix.Columns;
        int k = _components.GetLength(0);

        // Convert to double array
        var V = new double[nDocs, nFeatures];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < nFeatures; j++)
            {
                V[i, j] = Math.Max(0, NumOps.ToDouble(tfidfMatrix[i, j]));
            }
        }

        // Solve for W using NNLS-like update
        // Use a fixed seed derived from _randomState for reproducibility
        var random = _randomState.HasValue ? RandomHelper.CreateSeededRandom(_randomState.Value) : RandomHelper.CreateSeededRandom(42);
        var W = new double[nDocs, k];

        // Initialize W
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < k; j++)
            {
                W[i, j] = random.NextDouble() * 0.1 + 0.01;
            }
        }

        // Update W with fixed H
        const double eps = 1e-10;
        for (int iter = 0; iter < 50; iter++)
        {
            var VHt = MultiplyMatrixTranspose(V, _components, nDocs, nFeatures, k);
            var WHHt = MultiplyMatrices(MultiplyMatrices(W, _components, nDocs, k, nFeatures), TransposeMatrix(_components, k, nFeatures), nDocs, nFeatures, k);

            for (int i = 0; i < nDocs; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    W[i, j] = W[i, j] * VHt[i, j] / (WHHt[i, j] + eps);
                }
            }
        }

        // Convert to output type
        var output = new T[nDocs, k];
        for (int i = 0; i < nDocs; i++)
        {
            for (int j = 0; j < k; j++)
            {
                output[i, j] = NumOps.FromDouble(W[i, j]);
            }
        }

        return new Matrix<T>(output);
    }

    /// <summary>
    /// Gets the top words for each topic.
    /// </summary>
    /// <param name="nWords">Number of top words to return per topic.</param>
    /// <returns>Array of arrays, where each inner array contains the top words for a topic.</returns>
    public string[][] GetTopWordsPerTopic(int nWords = 10)
    {
        if (_components is null || _tfidfVectorizer?.FeatureNames is null)
        {
            throw new InvalidOperationException("NMFVectorizer has not been fitted.");
        }

        var vocabArray = _tfidfVectorizer.FeatureNames;
        int nTopics = _components.GetLength(0);
        var result = new string[nTopics][];

        for (int k = 0; k < nTopics; k++)
        {
            var wordWeights = new List<(int Index, double Weight)>();
            for (int w = 0; w < vocabArray.Length; w++)
            {
                wordWeights.Add((w, _components[k, w]));
            }

            result[k] = wordWeights
                .OrderByDescending(ww => ww.Weight)
                .Take(nWords)
                .Select(ww => vocabArray[ww.Index])
                .ToArray();
        }

        return result;
    }
}
