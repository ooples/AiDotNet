using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Qdrant-based document store built for performance and scalability with advanced filtering.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Qdrant provides high-performance vector similarity search with powerful filtering capabilities
/// and horizontal scalability for production workloads.
/// </remarks>
public class QdrantDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly string _endpoint;
    private readonly string _collectionName;
    private readonly string _apiKey;
    private readonly int _vectorDimension;
    private readonly HttpClient _httpClient;
    private int _documentCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="QdrantDocumentStore{T}"/> class.
    /// </summary>
    /// <param name="endpoint">The Qdrant endpoint URL.</param>
    /// <param name="collectionName">The name of the collection to use.</param>
    /// <param name="apiKey">The API key for authentication.</param>
    /// <param name="vectorDimension">The dimensionality of document vectors.</param>
    /// <param name="httpClient">Optional HTTP client for testing. If null, creates a new one.</param>
    public QdrantDocumentStore(
        string endpoint,
        string collectionName,
        string apiKey,
        int vectorDimension,
        HttpClient? httpClient = null)
    {
        _endpoint = endpoint ?? throw new ArgumentNullException(nameof(endpoint));
        _collectionName = collectionName ?? throw new ArgumentNullException(nameof(collectionName));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        _vectorDimension = vectorDimension;
        _httpClient = httpClient ?? new HttpClient();
        
        ConfigureHttpClient();
        EnsureCollectionExists();
        InitializeDocumentCount();
    }

    private void ConfigureHttpClient()
    {
        _httpClient.DefaultRequestHeaders.Clear();
        _httpClient.DefaultRequestHeaders.Add("api-key", _apiKey);
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    private void EnsureCollectionExists()
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}";
            var response = _httpClient.GetAsync(url).Result;
            
            if (!response.IsSuccessStatusCode)
            {
                CreateCollection();
            }
        }
        catch
        {
            CreateCollection();
        }
    }

    private void CreateCollection()
    {
        var url = $"{_endpoint}/collections/{_collectionName}";
        var payload = new
        {
            vectors = new
            {
                size = _vectorDimension,
                distance = "Cosine"
            }
        };
        
        var content = new StringContent(
            JsonSerializer.Serialize(payload),
            Encoding.UTF8,
            "application/json");
        
        _httpClient.PutAsync(url, content).Wait();
    }

    private void InitializeDocumentCount()
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}";
            var response = _httpClient.GetAsync(url).Result;
            
            if (response.IsSuccessStatusCode)
            {
                var json = response.Content.ReadAsStringAsync().Result;
                var doc = JsonSerializer.Deserialize<JsonElement>(json);
                
                if (doc.TryGetProperty("result", out var result) &&
                    result.TryGetProperty("points_count", out var count))
                {
                    _documentCount = count.GetInt32();
                }
            }
        }
        catch
        {
            _documentCount = 0;
        }
    }

    /// <inheritdoc />
    public override int DocumentCount => _documentCount;

    /// <inheritdoc />
    public override int VectorDimension => _vectorDimension;

    /// <inheritdoc />
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}/points?wait=true";
            
            var vectorArray = vectorDocument.Vector.ToArray();
            var vectorDoubles = new double[vectorArray.Length];
            for (int i = 0; i < vectorArray.Length; i++)
            {
                vectorDoubles[i] = NumOps.ToDouble(vectorArray[i]);
            }
            
            var payload = new
            {
                points = new[]
                {
                    new
                    {
                        id = Guid.NewGuid().ToString(),
                        vector = vectorDoubles,
                        payload = new
                        {
                            documentId = vectorDocument.Id,
                            content = vectorDocument.Content,
                            metadata = vectorDocument.Metadata
                        }
                    }
                }
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PutAsync(url, content).Result;
            response.EnsureSuccessStatusCode();
            
            _documentCount++;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to add document to Qdrant: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    protected override void AddBatchCore(IList<VectorDocument<T>> vectorDocuments)
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}/points?wait=true";
            
            var points = vectorDocuments.Select(doc =>
            {
                var vectorArray = doc.Vector.ToArray();
                var vectorDoubles = new double[vectorArray.Length];
                for (int i = 0; i < vectorArray.Length; i++)
                {
                    vectorDoubles[i] = NumOps.ToDouble(vectorArray[i]);
                }
                
                return new
                {
                    id = Guid.NewGuid().ToString(),
                    vector = vectorDoubles,
                    payload = new
                    {
                        documentId = doc.Id,
                        content = doc.Content,
                        metadata = doc.Metadata
                    }
                };
            }).ToArray();
            
            var payload = new { points };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PutAsync(url, content).Result;
            response.EnsureSuccessStatusCode();
            
            _documentCount += vectorDocuments.Count;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to add batch to Qdrant: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}/points/search";
            
            var vectorArray = queryVector.ToArray();
            var vectorDoubles = new double[vectorArray.Length];
            for (int i = 0; i < vectorArray.Length; i++)
            {
                vectorDoubles[i] = NumOps.ToDouble(vectorArray[i]);
            }
            
            var payload = new
            {
                vector = vectorDoubles,
                limit = topK,
                with_payload = true
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PostAsync(url, content).Result;
            response.EnsureSuccessStatusCode();
            
            var json = response.Content.ReadAsStringAsync().Result;
            var result = JsonSerializer.Deserialize<JsonElement>(json);
            
            var documents = new List<Document<T>>();
            
            if (result.TryGetProperty("result", out var results))
            {
                foreach (var point in results.EnumerateArray())
                {
                    if (point.TryGetProperty("payload", out var payload))
                    {
                        var id = payload.GetProperty("documentId").GetString() ?? string.Empty;
                        var contentStr = payload.GetProperty("content").GetString() ?? string.Empty;
                        
                        var metadata = new Dictionary<string, object>();
                        if (payload.TryGetProperty("metadata", out var metaElement))
                        {
                            foreach (var prop in metaElement.EnumerateObject())
                            {
                                metadata[prop.Name] = prop.Value.ToString() ?? string.Empty;
                            }
                        }
                        
                        var document = new Document<T>(id, contentStr, metadata);
                        
                        if (point.TryGetProperty("score", out var scoreElement))
                        {
                            var score = NumOps.FromDouble(scoreElement.GetDouble());
                            document.RelevanceScore = score;
                            document.HasRelevanceScore = true;
                        }
                        
                        documents.Add(document);
                    }
                }
            }
            
            return documents;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to search Qdrant: {ex.Message}", ex);
        }
    }

    /// <inheritdoc />
    protected override Document<T>? GetByIdCore(string documentId)
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}/points/scroll";
            var payload = new
            {
                filter = new
                {
                    must = new[]
                    {
                        new
                        {
                            key = "documentId",
                            match = new { value = documentId }
                        }
                    }
                },
                limit = 1,
                with_payload = true,
                with_vector = false
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PostAsync(url, content).Result;
            
            if (!response.IsSuccessStatusCode)
                return null;
            
            var json = response.Content.ReadAsStringAsync().Result;
            var result = JsonSerializer.Deserialize<JsonElement>(json);
            
            if (result.TryGetProperty("result", out var resultObj) &&
                resultObj.TryGetProperty("points", out var points))
            {
                var pointsArray = points.EnumerateArray().ToArray();
                if (pointsArray.Length > 0)
                {
                    var point = pointsArray[0];
                    if (point.TryGetProperty("payload", out var payload))
                    {
                        var contentStr = payload.GetProperty("content").GetString() ?? string.Empty;
                        
                        var metadata = new Dictionary<string, object>();
                        if (payload.TryGetProperty("metadata", out var metaElement))
                        {
                            foreach (var prop in metaElement.EnumerateObject())
                            {
                                metadata[prop.Name] = prop.Value.ToString() ?? string.Empty;
                            }
                        }
                        
                        return new Document<T>(documentId, contentStr, metadata);
                    }
                }
            }
            
            return null;
        }
        catch
        {
            return null;
        }
    }

    /// <inheritdoc />
    protected override bool RemoveCore(string documentId)
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}/points/delete?wait=true";
            var payload = new
            {
                filter = new
                {
                    must = new[]
                    {
                        new
                        {
                            key = "documentId",
                            match = new { value = documentId }
                        }
                    }
                }
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PostAsync(url, content).Result;
            
            if (response.IsSuccessStatusCode && _documentCount > 0)
            {
                _documentCount--;
                return true;
            }
            
            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <inheritdoc />
    public override void Clear()
    {
        try
        {
            var url = $"{_endpoint}/collections/{_collectionName}";
            var response = _httpClient.DeleteAsync(url).Result;
            response.EnsureSuccessStatusCode();
            
            CreateCollection();
            _documentCount = 0;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to clear Qdrant collection: {ex.Message}", ex);
        }
    }
}

