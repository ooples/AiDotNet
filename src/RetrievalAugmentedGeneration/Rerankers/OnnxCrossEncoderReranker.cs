using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.Tokenization.HuggingFace;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;

namespace AiDotNet.RetrievalAugmentedGeneration.Rerankers;

/// <summary>
/// A production-ready cross-encoder reranker that scores (query, document) pairs with a real
/// transformer model loaded via ONNX Runtime (e.g. BAAI/bge-reranker or ms-marco-MiniLM).
/// </summary>
/// <typeparam name="T">The numeric data type used for scoring.</typeparam>
/// <remarks>
/// <para>
/// Unlike <see cref="CrossEncoderReranker{T}"/> (which requires the caller to supply a
/// <c>Func&lt;string, string, T&gt;</c> scorer), this reranker works out of the box: point it at a
/// cross-encoder ONNX model and its tokenizer, and it will tokenize each (query, document) pair
/// jointly (<c>[CLS] query [SEP] document [SEP]</c>), run the model to produce a relevance logit,
/// and reorder documents by that logit (highest first).
/// </para>
/// <para><b>For Beginners:</b> This is the "batteries-included" cross-encoder reranker.
///
/// A cross-encoder reads the query and a document together and outputs a single number telling you
/// how relevant the document is to the query. Higher means more relevant. This class handles all of
/// the plumbing:
/// - Loading the model (ONNX file) and its tokenizer.
/// - Building the joint input the model expects for each pair.
/// - Running the model (in batches for speed).
/// - Sorting your documents from most to least relevant.
///
/// Typical models to use:
/// - BAAI/bge-reranker-base / bge-reranker-large (single relevance logit).
/// - cross-encoder/ms-marco-MiniLM-L-6-v2 (single relevance logit).
///
/// Model loading is lazy: nothing is opened until the first rerank call. If the model file is
/// missing, a clear <see cref="FileNotFoundException"/> is thrown — there is no silent lexical
/// fallback.
/// </para>
/// </remarks>
[Attributes.ComponentType(Enums.ComponentType.Reranker)]
[Attributes.PipelineStage(Enums.PipelineStage.PostRetrieval)]
public class OnnxCrossEncoderReranker<T> : RerankerBase<T>, IDisposable
{
    private readonly string _modelPath;
    private readonly string _tokenizerPath;
    private readonly int _maxLength;
    private readonly int _maxPairsToScore;
    private readonly int _batchSize;
    private readonly int _scoreLabelIndex;

    private volatile InferenceSession? _session;
    private volatile ITokenizer? _tokenizer;
    private readonly object _initLock = new();
    private bool _disposed;

    /// <summary>
    /// Gets a value indicating whether this reranker modifies relevance scores.
    /// </summary>
    public override bool ModifiesScores => true;

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxCrossEncoderReranker{T}"/> class.
    /// </summary>
    /// <param name="modelPath">Path to the cross-encoder ONNX model file (e.g. <c>model.onnx</c>).</param>
    /// <param name="tokenizerPath">
    /// Path to the tokenizer. May be a directory containing the tokenizer files
    /// (<c>tokenizer.json</c> / <c>vocab.txt</c> + <c>tokenizer_config.json</c>), or a path to one of
    /// those files (the containing directory is used).
    /// </param>
    /// <param name="maxLength">Maximum joint sequence length (query + document); default 512.</param>
    /// <param name="maxPairsToScore">Maximum number of query-document pairs to score; default 100.</param>
    /// <param name="batchSize">Number of pairs to run through the model per inference call; default 16.</param>
    /// <param name="scoreLabelIndex">
    /// The output label index to read as the relevance score. Use -1 (default) to read the last label,
    /// which is correct for single-logit rerankers (index 0) and for binary classifiers whose positive
    /// class is last. Set explicitly if your model orders labels differently.
    /// </param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The two paths you must supply are the model file and its tokenizer.
    /// Everything else has sensible defaults. Loading is deferred until the first rerank, so
    /// constructing this object is cheap and never throws just because a file is missing.
    /// </para>
    /// </remarks>
    public OnnxCrossEncoderReranker(
        string modelPath,
        string tokenizerPath,
        int maxLength = 512,
        int maxPairsToScore = 100,
        int batchSize = 16,
        int scoreLabelIndex = -1)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be empty", nameof(modelPath));
        if (string.IsNullOrWhiteSpace(tokenizerPath))
            throw new ArgumentException("Tokenizer path cannot be empty", nameof(tokenizerPath));
        if (maxLength <= 0)
            throw new ArgumentException("maxLength must be greater than zero", nameof(maxLength));
        if (maxPairsToScore <= 0)
            throw new ArgumentException("maxPairsToScore must be greater than zero", nameof(maxPairsToScore));
        if (batchSize <= 0)
            throw new ArgumentException("batchSize must be greater than zero", nameof(batchSize));

        _modelPath = modelPath;
        _tokenizerPath = tokenizerPath;
        _maxLength = maxLength;
        _maxPairsToScore = maxPairsToScore;
        _batchSize = batchSize;
        _scoreLabelIndex = scoreLabelIndex;
    }

    /// <summary>
    /// Reranks documents by running the cross-encoder model on each (query, document) pair.
    /// </summary>
    /// <param name="query">The validated search query.</param>
    /// <param name="documents">The validated, materialized documents to rerank.</param>
    /// <returns>Documents reordered by descending cross-encoder relevance score.</returns>
    protected override IEnumerable<Document<T>> RerankCore(string query, IList<Document<T>> documents)
    {
        var docList = documents.Take(_maxPairsToScore).ToList();
        if (docList.Count == 0)
            return Enumerable.Empty<Document<T>>();

        var contents = new List<string>(docList.Count);
        for (int i = 0; i < docList.Count; i++)
        {
            contents.Add(docList[i].Content ?? string.Empty);
        }

        var scores = ScorePairs(query, contents);
        if (scores == null || scores.Length != docList.Count)
        {
            throw new InvalidOperationException(
                "ScorePairs must return exactly one score per document.");
        }

        var scored = new List<(Document<T> Doc, double Score)>(docList.Count);
        for (int i = 0; i < docList.Count; i++)
        {
            scored.Add((docList[i], scores[i]));
        }

        var reranked = scored
            .OrderByDescending(x => x.Score)
            .Select(x =>
            {
                var doc = x.Doc;
                doc.RelevanceScore = NumOps.FromDouble(x.Score);
                doc.HasRelevanceScore = true;
                return doc;
            })
            .ToList();

        return reranked;
    }

    /// <summary>
    /// Scores every (query, document) pair with the cross-encoder model and returns one relevance
    /// score per document, in input order.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="documentContents">The document contents to score against the query.</param>
    /// <returns>One relevance score per document, in the same order as <paramref name="documentContents"/>.</returns>
    /// <remarks>
    /// <para>
    /// This is the inference seam. The default implementation loads the ONNX model + tokenizer,
    /// tokenizes each pair jointly, runs the model in batches, and reads the relevance logit. Tests
    /// override this method to inject deterministic scores and exercise the ranking logic without a
    /// real model file.
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file cannot be found.</exception>
    protected virtual double[] ScorePairs(string query, IList<string> documentContents)
    {
        if (documentContents == null)
            throw new ArgumentNullException(nameof(documentContents));

        if (!EnsureModelLoaded())
            throw new FileNotFoundException($"ONNX cross-encoder model file not found: {_modelPath}", _modelPath);

        var session = Session;
        var tokenizer = Tokenizer;
        var declaredInputs = new HashSet<string>(session.InputMetadata.Keys);
        long padId = GetSpecialTokenId(tokenizer, tokenizer.SpecialTokens.PadToken, 0L);

        var results = new double[documentContents.Count];

        for (int start = 0; start < documentContents.Count; start += _batchSize)
        {
            int count = Math.Min(_batchSize, documentContents.Count - start);

            // Encode each pair and track the longest sequence in this batch.
            var encoded = new List<(long[] Ids, long[] Mask, long[] Types)>(count);
            int maxLen = 0;
            for (int j = 0; j < count; j++)
            {
                var pair = EncodePair(tokenizer, query, documentContents[start + j]);
                encoded.Add(pair);
                if (pair.Ids.Length > maxLen)
                    maxLen = pair.Ids.Length;
            }
            if (maxLen == 0)
                maxLen = 1;

            // Pad every sequence in the batch to maxLen so they form a rectangular tensor.
            var ids = new long[count * maxLen];
            var mask = new long[count * maxLen];
            var types = new long[count * maxLen];
            for (int j = 0; j < count; j++)
            {
                var pair = encoded[j];
                for (int k = 0; k < maxLen; k++)
                {
                    int idx = (j * maxLen) + k;
                    if (k < pair.Ids.Length)
                    {
                        ids[idx] = pair.Ids[k];
                        mask[idx] = pair.Mask[k];
                        types[idx] = pair.Types[k];
                    }
                    else
                    {
                        ids[idx] = padId;
                        mask[idx] = 0L;
                        types[idx] = 0L;
                    }
                }
            }

            var shape = new[] { count, maxLen };
            var inputs = BuildPairInputs(declaredInputs, ids, mask, types, shape);

            using var outputs = session.Run(inputs);
            var logits = outputs.First().AsTensor<float>();
            var dims = logits.Dimensions.ToArray();
            int numLabels = dims.Length >= 2 ? dims[dims.Length - 1] : 1;
            var flat = System.Linq.Enumerable.ToArray(logits);

            var batchScores = ExtractScoresFromLogits(flat, count, numLabels, _scoreLabelIndex);
            for (int j = 0; j < count; j++)
            {
                results[start + j] = batchScores[j];
            }
        }

        return results;
    }

    /// <summary>
    /// Scores a single (query, document) pair and returns the relevance score as <typeparamref name="T"/>.
    /// Convenient for wiring an ONNX cross-encoder into <see cref="CrossEncoderReranker{T}"/> or ad-hoc use.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <param name="document">The document content.</param>
    /// <returns>The relevance score.</returns>
    public T ScorePair(string query, string document)
    {
        var scores = ScorePairs(query, new List<string> { document ?? string.Empty });
        return NumOps.FromDouble(scores.Length > 0 ? scores[0] : 0.0);
    }

    /// <summary>
    /// Tokenizes a (query, document) pair into the joint BERT-style sequence
    /// <c>[CLS] query [SEP] document [SEP]</c> with matching attention mask and token-type ids.
    /// </summary>
    private (long[] Ids, long[] Mask, long[] Types) EncodePair(ITokenizer tokenizer, string query, string document)
    {
        var special = tokenizer.SpecialTokens;
        bool hasCls = !string.IsNullOrEmpty(special.ClsToken);
        bool hasSep = !string.IsNullOrEmpty(special.SepToken);

        var queryTokens = tokenizer.Tokenize(query ?? string.Empty);
        var docTokens = tokenizer.Tokenize(document ?? string.Empty);

        // Reserve room for the special tokens: [CLS] ... [SEP] ... [SEP]
        int specialCount = (hasCls ? 1 : 0) + (hasSep ? 2 : 1);
        int available = _maxLength - specialCount;
        if (available < 0)
            available = 0;

        TruncateLongestFirst(queryTokens, docTokens, available);

        long clsId = hasCls ? GetSpecialTokenId(tokenizer, special.ClsToken, 0L) : 0L;
        long sepId = hasSep ? GetSpecialTokenId(tokenizer, special.SepToken, 0L) : 0L;

        var queryIds = tokenizer.ConvertTokensToIds(queryTokens);
        var docIds = tokenizer.ConvertTokensToIds(docTokens);

        var ids = new List<long>(specialCount + queryIds.Count + docIds.Count);
        var types = new List<long>(ids.Capacity);

        // Segment 0: [CLS] + query + [SEP]
        if (hasCls)
        {
            ids.Add(clsId);
            types.Add(0L);
        }
        for (int i = 0; i < queryIds.Count; i++)
        {
            ids.Add(queryIds[i]);
            types.Add(0L);
        }
        if (hasSep)
        {
            ids.Add(sepId);
            types.Add(0L);
        }

        // Segment 1: document + [SEP]
        for (int i = 0; i < docIds.Count; i++)
        {
            ids.Add(docIds[i]);
            types.Add(1L);
        }
        if (hasSep)
        {
            ids.Add(sepId);
            types.Add(1L);
        }

        var idArray = ids.ToArray();
        var typeArray = types.ToArray();
        var maskArray = new long[idArray.Length];
        for (int i = 0; i < maskArray.Length; i++)
        {
            maskArray[i] = 1L;
        }

        return (idArray, maskArray, typeArray);
    }

    /// <summary>
    /// Truncates the two token lists in place using HuggingFace's "longest_first" strategy so their
    /// combined length does not exceed <paramref name="available"/>.
    /// </summary>
    internal static void TruncateLongestFirst(List<string> queryTokens, List<string> docTokens, int available)
    {
        if (available < 0)
            available = 0;

        while (queryTokens.Count + docTokens.Count > available)
        {
            if (queryTokens.Count == 0 && docTokens.Count == 0)
                break;

            if (queryTokens.Count > docTokens.Count)
            {
                queryTokens.RemoveAt(queryTokens.Count - 1);
            }
            else
            {
                docTokens.RemoveAt(docTokens.Count - 1);
            }
        }
    }

    /// <summary>
    /// Builds the ONNX input list for a batch of tokenized (query, document) pairs. <c>input_ids</c> is
    /// always supplied; <c>attention_mask</c> and <c>token_type_ids</c> are added only when the loaded
    /// session declares them, so models exported without them keep working.
    /// </summary>
    /// <param name="declaredInputs">The names of the inputs the loaded session declares.</param>
    /// <param name="inputIds">Int64 token ids, flattened row-major with shape [batch, seqLen].</param>
    /// <param name="attentionMask">Int64 attention mask, same layout as <paramref name="inputIds"/>.</param>
    /// <param name="tokenTypeIds">Int64 segment ids, same layout as <paramref name="inputIds"/>.</param>
    /// <param name="shape">The [batch, seqLen] shape shared by every input tensor.</param>
    internal static List<NamedOnnxValue> BuildPairInputs(
        ICollection<string> declaredInputs,
        long[] inputIds,
        long[] attentionMask,
        long[] tokenTypeIds,
        int[] shape)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(inputIds, shape))
        };

        if (declaredInputs.Contains("attention_mask"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("attention_mask", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(attentionMask, shape)));
        }

        if (declaredInputs.Contains("token_type_ids"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(tokenTypeIds, shape)));
        }

        return inputs;
    }

    /// <summary>
    /// Extracts one relevance score per batch item from a flattened [batch, numLabels] logits array.
    /// </summary>
    /// <param name="flatLogits">Row-major logits, length batch * numLabels.</param>
    /// <param name="batch">Number of items in the batch.</param>
    /// <param name="numLabels">Number of output labels per item.</param>
    /// <param name="labelIndex">
    /// The label index to read. Values &lt; 0 or out of range select the last label, which is correct
    /// for single-logit rerankers and last-is-positive binary classifiers.
    /// </param>
    internal static double[] ExtractScoresFromLogits(float[] flatLogits, int batch, int numLabels, int labelIndex)
    {
        if (numLabels < 1)
            numLabels = 1;

        int li = labelIndex;
        if (li < 0 || li >= numLabels)
            li = numLabels - 1;

        var scores = new double[batch];
        for (int i = 0; i < batch; i++)
        {
            int idx = (i * numLabels) + li;
            scores[i] = (idx >= 0 && idx < flatLogits.Length) ? flatLogits[idx] : 0.0;
        }

        return scores;
    }

    private static long GetSpecialTokenId(ITokenizer tokenizer, string token, long fallback)
    {
        if (string.IsNullOrEmpty(token))
            return fallback;

        var ids = tokenizer.ConvertTokensToIds(new List<string> { token });
        if (ids == null || ids.Count == 0)
            return fallback;

        return ids[0];
    }

    /// <summary>
    /// Ensures the ONNX model and tokenizer are loaded, loading lazily on first use.
    /// Returns false only if the model file is unavailable.
    /// </summary>
    private bool EnsureModelLoaded()
    {
        if (_session != null && _tokenizer != null)
            return true;

        lock (_initLock)
        {
            if (_session != null && _tokenizer != null)
                return true;

            if (!File.Exists(_modelPath))
                return false;

            var tokenizerDir = ResolveTokenizerDirectory(_tokenizerPath);
            var tokenizer = AutoTokenizer.FromPretrained(tokenizerDir);
            var session = new InferenceSession(_modelPath);

            // Assign both atomically so concurrent readers never see a partial state.
            _tokenizer = tokenizer;
            _session = session;
            return true;
        }
    }

    private static string ResolveTokenizerDirectory(string tokenizerPath)
    {
        if (Directory.Exists(tokenizerPath))
            return tokenizerPath;

        if (File.Exists(tokenizerPath))
        {
            var dir = Path.GetDirectoryName(tokenizerPath);
            if (!string.IsNullOrEmpty(dir))
                return dir!;
        }

        // Fall back to the given path; AutoTokenizer will surface a clear error if it is invalid.
        return tokenizerPath;
    }

    /// <summary>
    /// Gets the inference session, ensuring it is loaded.
    /// </summary>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file is not found.</exception>
    private InferenceSession Session
    {
        get
        {
            if (!EnsureModelLoaded())
                throw new FileNotFoundException($"ONNX cross-encoder model file not found: {_modelPath}", _modelPath);

            return _session ?? throw new InvalidOperationException("Failed to load ONNX session.");
        }
    }

    /// <summary>
    /// Gets the tokenizer, ensuring it is loaded.
    /// </summary>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX model file is not found.</exception>
    private ITokenizer Tokenizer
    {
        get
        {
            if (!EnsureModelLoaded())
                throw new FileNotFoundException($"ONNX cross-encoder model file not found: {_modelPath}", _modelPath);

            return _tokenizer ?? throw new InvalidOperationException("Failed to load tokenizer.");
        }
    }

    /// <summary>
    /// Releases the ONNX session and tokenizer resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Releases resources held by the reranker.
    /// </summary>
    /// <param name="disposing">True when called from <see cref="Dispose()"/>.</param>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed)
            return;

        if (disposing)
        {
            _session?.Dispose();
            if (_tokenizer is IDisposable disposableTokenizer)
            {
                disposableTokenizer.Dispose();
            }
        }

        _disposed = true;
    }
}
