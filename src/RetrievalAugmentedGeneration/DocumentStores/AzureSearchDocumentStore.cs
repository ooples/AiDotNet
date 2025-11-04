using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// Azure Cognitive Search document store providing fully managed search capabilities.
/// </summary>
/// <typeparam name="T">The numeric data type used for vector operations.</typeparam>
/// <remarks>
/// Azure Cognitive Search combines full-text search, semantic search, and vector search
/// in a fully managed cloud service with enterprise-grade security and compliance.
/// </remarks>
public class AzureSearchDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly string _serviceName;
    private readonly string _indexName;
    private readonly string _apiKey;
    private readonly int _vectorDimension;
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private int _documentCount;

    /// <summary>
    /// Initializes a new instance of the <see cref="AzureSearchDocumentStore{T}"/> class.
    /// </summary>
    /// <param name="serviceName">The Azure Search service name.</param>
    /// <param name="indexName">The name of the index to use.</param>
    /// <param name="apiKey">The admin API key for authentication.</param>
    /// <param name="vectorDimension">The dimensionality of document vectors.</param>
    /// <param name="httpClient">Optional HTTP client for testing. If null, creates a new one.</param>
    public AzureSearchDocumentStore(
        string serviceName,
        string indexName,
        string apiKey,
        int vectorDimension,
        HttpClient? httpClient = null)
    {
        _serviceName = serviceName ?? throw new ArgumentNullException(nameof(serviceName));
        _indexName = indexName ?? throw new ArgumentNullException(nameof(indexName));
        _apiKey = apiKey ?? throw new ArgumentNullException(nameof(apiKey));
        
        if (vectorDimension <= 0)
            throw new ArgumentException("Vector dimension must be positive", nameof(vectorDimension));
        
        _vectorDimension = vectorDimension;
        _httpClient = httpClient ?? new HttpClient();
        _baseUrl = $"https://{_serviceName}.search.windows.net";
        
        ConfigureHttpClient();
        InitializeDocumentCount();
    }

    private void ConfigureHttpClient()
    {
        _httpClient.DefaultRequestHeaders.Clear();
        _httpClient.DefaultRequestHeaders.Add("api-key", _apiKey);
        _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
    }

    private void InitializeDocumentCount()
    {
        try
        {
            var url = $"{_baseUrl}/indexes/{_indexName}/docs/$count?api-version=2023-11-01";
            var response = _httpClient.GetAsync(url).Result;
            
            if (response.IsSuccessStatusCode)
            {
                var countStr = response.Content.ReadAsStringAsync().Result;
                _documentCount = int.Parse(countStr);
            }
        }
        catch
        {
            _documentCount = 0;
        }
    }

    /// <summary>
    /// Gets the total number of documents in the index.
    /// </summary>
    public override int DocumentCount => _documentCount;

    /// <summary>
    /// Gets the dimensionality of vectors in this store.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Removes all documents from the store.
    /// </summary>
    public override void Clear()
    {
        try
        {
            // Delete all documents using Azure Search batch delete
            var url = $"{_baseUrl}/indexes/{_indexName}/docs/index?api-version=2023-11-01";
            
            // Get all document IDs first
            var searchUrl = $"{_baseUrl}/indexes/{_indexName}/docs/search?api-version=2023-11-01";
            var searchPayload = new
            {
                search = "*",
                select = "id",
                top = 1000
            };
            
            var searchContent = new StringContent(
                JsonSerializer.Serialize(searchPayload),
                Encoding.UTF8,
                "application/json");
            
            var searchResponse = _httpClient.PostAsync(searchUrl, searchContent).Result;
            
            if (searchResponse.IsSuccessStatusCode)
            {
                var resultJson = searchResponse.Content.ReadAsStringAsync().Result;
                var result = JsonSerializer.Deserialize<JsonElement>(resultJson);
                
                if (result.TryGetProperty("value", out var docs))
                {
                    var deleteActions = new List<object>();
                    foreach (var doc in docs.EnumerateArray())
                    {
                        if (doc.TryGetProperty("id", out var id))
                        {
                            deleteActions.Add(new
                            {
                                __delete = new { id = id.GetString() }
                            });
                        }
                    }
                    
                    if (deleteActions.Count > 0)
                    {
                        var deletePayload = new { value = deleteActions };
                        var deleteContent = new StringContent(
                            JsonSerializer.Serialize(deletePayload),
                            Encoding.UTF8,
                            "application/json");
                        
                        _httpClient.PostAsync(url, deleteContent).Wait();
                    }
                }
            }
            
            _documentCount = 0;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException("Failed to clear Azure Search index", ex);
        }
    }

    /// <summary>
    /// Core logic for adding a single vector document.
    /// </summary>
    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        try
        {
            var url = $"{_baseUrl}/indexes/{_indexName}/docs/index?api-version=2023-11-01";
            
            var vectorArray = vectorDocument.Vector.ToArray();
            var vectorDoubles = new double[vectorArray.Length];
            for (int i = 0; i < vectorArray.Length; i++)
            {
                vectorDoubles[i] = NumOps.ToDouble(vectorArray[i]);
            }
            
            var document = new
            {
                id = vectorDocument.Id,
                content = vectorDocument.Content,
                vector = vectorDoubles,
                metadata = vectorDocument.Metadata
            };
            
            var payload = new
            {
                value = new[] { new { __upload = document } }
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(payload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PostAsync(url, content).Result;
            response.EnsureSuccessStatusCode();
            
            _documentCount++;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to add document to Azure Search: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Core logic for similarity search with optional filtering.
    /// </summary>
    protected override IEnumerable<Document<T>> GetSimilarCore(Vector<T> queryVector, int topK, Dictionary<string, object> metadataFilters)
    {
        try
        {
            var url = $"{_baseUrl}/indexes/{_indexName}/docs/search?api-version=2023-11-01";
            
            var vectorArray = queryVector.ToArray();
            var vectorDoubles = new double[vectorArray.Length];
            for (int i = 0; i < vectorArray.Length; i++)
            {
                vectorDoubles[i] = NumOps.ToDouble(vectorArray[i]);
            }
            
            var searchPayload = new
            {
                vectorQueries = new[]
                {
                    new
                    {
                        kind = "vector",
                        vector = vectorDoubles,
                        fields = "vector",
                        k = topK
                    }
                },
                select = "id,content,metadata",
                top = topK
            };
            
            var content = new StringContent(
                JsonSerializer.Serialize(searchPayload),
                Encoding.UTF8,
                "application/json");
            
            var response = _httpClient.PostAsync(url, content).Result;
            response.EnsureSuccessStatusCode();
            
            var resultJson = response.Content.ReadAsStringAsync().Result;
            var result = JsonSerializer.Deserialize<JsonElement>(resultJson);
            
            var documents = new List<Document<T>>();
            
            if (result.TryGetProperty("value", out var docs))
            {
                foreach (var doc in docs.EnumerateArray())
                {
                    var id = doc.GetProperty("id").GetString() ?? string.Empty;
                    var content = doc.GetProperty("content").GetString() ?? string.Empty;
                    
                    var metadata = new Dictionary<string, object>();
                    if (doc.TryGetProperty("metadata", out var metaElement))
                    {
                        foreach (var prop in metaElement.EnumerateObject())
                        {
                            metadata[prop.Name] = prop.Value.ToString() ?? string.Empty;
                        }
                    }
                    
                    var document = new Document<T>(id, content, metadata);
                    
                    if (doc.TryGetProperty("@search.score", out var scoreElement))
                    {
                        var score = NumOps.FromDouble(scoreElement.GetDouble());
                        document.RelevanceScore = score;
                        document.HasRelevanceScore = true;
                    }
                    
                    documents.Add(document);
                }
            }
            
            return documents;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to search Azure Search index: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Core logic for retrieving a document by ID.
    /// </summary>
    protected override Document<T>? GetByIdCore(string documentId)
    {
        try
        {
            var url = $"{_baseUrl}/indexes/{_indexName}/docs('{documentId}')?api-version=2023-11-01";
            var response = _httpClient.GetAsync(url).Result;
            
            if (!response.IsSuccessStatusCode)
                return null;
            
            var resultJson = response.Content.ReadAsStringAsync().Result;
            var doc = JsonSerializer.Deserialize<JsonElement>(resultJson);
            
            var content = doc.GetProperty("content").GetString() ?? string.Empty;
            
            var metadata = new Dictionary<string, object>();
            if (doc.TryGetProperty("metadata", out var metaElement))
            {
                foreach (var prop in metaElement.EnumerateObject())
                {
                    metadata[prop.Name] = prop.Value.ToString() ?? string.Empty;
                }
            }
            
            return new Document<T>(documentId, content, metadata);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Core logic for removing a document by ID.
    /// </summary>
    protected override bool RemoveCore(string documentId)
    {
        try
        {
            var url = $"{_baseUrl}/indexes/{_indexName}/docs/index?api-version=2023-11-01";
            
            var payload = new
            {
                value = new[] { new { __delete = new { id = documentId } } }
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
}
