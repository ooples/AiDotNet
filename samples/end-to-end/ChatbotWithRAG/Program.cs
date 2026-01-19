// =============================================================================
// AiDotNet Sample: ChatbotWithRAG
// =============================================================================
// A complete chatbot application with Retrieval-Augmented Generation (RAG).
// Features:
// - Document ingestion with chunking
// - Vector store with semantic search
// - RAG pipeline for grounded responses
// - Web UI for interactive chat
// - REST API for integration
//
// Run with: dotnet run
// Then open: http://localhost:5000
// =============================================================================

using AiDotNet.RetrievalAugmentedGeneration;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.VectorStores;
using AiDotNet.RetrievalAugmentedGeneration.Chunking;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddSingleton<RAGChatbot>();
builder.Services.AddEndpointsApiExplorer();

var app = builder.Build();

// Initialize the chatbot with sample documents
var chatbot = app.Services.GetRequiredService<RAGChatbot>();
await chatbot.InitializeAsync();

// Serve static HTML UI
app.MapGet("/", () => Results.Content(GetHtmlUI(), "text/html"));

// REST API endpoints
app.MapPost("/api/chat", async (ChatRequest request, RAGChatbot bot) =>
{
    var response = await bot.ChatAsync(request.Message);
    return Results.Json(new ChatResponse
    {
        Answer = response.Answer,
        Sources = response.Sources.ToList(),
        Confidence = response.Confidence
    });
});

app.MapPost("/api/documents", async (DocumentRequest request, RAGChatbot bot) =>
{
    await bot.AddDocumentAsync(request.Title, request.Content);
    return Results.Ok(new { message = "Document added successfully" });
});

app.MapGet("/api/documents", (RAGChatbot bot) =>
{
    return Results.Json(bot.GetDocuments());
});

app.MapDelete("/api/documents/{id}", async (string id, RAGChatbot bot) =>
{
    await bot.RemoveDocumentAsync(id);
    return Results.Ok(new { message = "Document removed" });
});

app.MapGet("/api/stats", (RAGChatbot bot) =>
{
    return Results.Json(bot.GetStats());
});

Console.WriteLine("===========================================");
Console.WriteLine("  AiDotNet ChatbotWithRAG Sample");
Console.WriteLine("===========================================");
Console.WriteLine();
Console.WriteLine("Server running at: http://localhost:5000");
Console.WriteLine();
Console.WriteLine("Features:");
Console.WriteLine("  - RAG-powered responses with source citations");
Console.WriteLine("  - Document ingestion via API");
Console.WriteLine("  - Semantic search over your documents");
Console.WriteLine("  - Interactive web UI");
Console.WriteLine();
Console.WriteLine("API Endpoints:");
Console.WriteLine("  POST /api/chat          - Send a message");
Console.WriteLine("  POST /api/documents     - Add a document");
Console.WriteLine("  GET  /api/documents     - List documents");
Console.WriteLine("  DELETE /api/documents/  - Remove a document");
Console.WriteLine("  GET  /api/stats         - Get system stats");
Console.WriteLine();
Console.WriteLine("Press Ctrl+C to stop the server.");
Console.WriteLine("===========================================");

app.Run("http://localhost:5000");

// =============================================================================
// RAG Chatbot Service
// =============================================================================

public class RAGChatbot
{
    private RAGPipeline<float>? _pipeline;
    private readonly Dictionary<string, DocumentInfo> _documents = new();
    private int _queryCount;
    private int _documentCount;

    public async Task InitializeAsync()
    {
        Console.WriteLine("Initializing RAG pipeline...");

        // Build the RAG pipeline with AiDotNet components
        _pipeline = new RAGPipeline<float>()
            .WithEmbeddings(new SentenceTransformerEmbeddings<float>("all-MiniLM-L6-v2"))
            .WithVectorStore(new InMemoryVectorStore<float>(dimension: 384))
            .WithChunker(new RecursiveChunker(chunkSize: 512, chunkOverlap: 50))
            .WithRetriever(new DenseRetriever<float>(topK: 5))
            .Build();

        // Add sample documents about AiDotNet
        var sampleDocs = GetSampleDocuments();
        foreach (var doc in sampleDocs)
        {
            await AddDocumentAsync(doc.Title, doc.Content);
        }

        Console.WriteLine($"Initialized with {_documentCount} documents.");
    }

    public async Task<RAGResponse> ChatAsync(string message)
    {
        if (_pipeline == null)
            throw new InvalidOperationException("Pipeline not initialized");

        _queryCount++;

        // Query the RAG pipeline
        var result = await _pipeline.QueryAsync(message);

        return new RAGResponse
        {
            Answer = result.Answer,
            Sources = result.SourceDocuments.Select(d => new SourceInfo
            {
                Title = d.Metadata.GetValueOrDefault("title", "Unknown")?.ToString() ?? "Unknown",
                Snippet = d.Text.Length > 200 ? d.Text[..200] + "..." : d.Text,
                Score = d.Score
            }).ToArray(),
            Confidence = result.SourceDocuments.Any() ? result.SourceDocuments.Average(d => d.Score) : 0
        };
    }

    public async Task AddDocumentAsync(string title, string content)
    {
        if (_pipeline == null)
            throw new InvalidOperationException("Pipeline not initialized");

        var id = Guid.NewGuid().ToString();

        var document = new Document
        {
            Id = id,
            Text = content,
            Metadata = new Dictionary<string, object>
            {
                ["title"] = title,
                ["addedAt"] = DateTime.UtcNow.ToString("o")
            }
        };

        await _pipeline.IndexDocumentsAsync(new[] { document });

        _documents[id] = new DocumentInfo
        {
            Id = id,
            Title = title,
            Length = content.Length,
            AddedAt = DateTime.UtcNow
        };

        _documentCount++;
    }

    public async Task RemoveDocumentAsync(string id)
    {
        if (_documents.ContainsKey(id))
        {
            _documents.Remove(id);
            _documentCount--;
        }

        await Task.CompletedTask;
    }

    public List<DocumentInfo> GetDocuments()
    {
        return _documents.Values.ToList();
    }

    public ChatbotStats GetStats()
    {
        return new ChatbotStats
        {
            TotalDocuments = _documentCount,
            TotalQueries = _queryCount,
            EmbeddingModel = "all-MiniLM-L6-v2",
            EmbeddingDimension = 384,
            ChunkSize = 512,
            TopK = 5
        };
    }

    private static List<(string Title, string Content)> GetSampleDocuments()
    {
        return new List<(string, string)>
        {
            ("AiDotNet Overview", @"
AiDotNet is the most comprehensive AI/ML framework for .NET. It provides over 4,300 implementations
across 60+ feature categories, including 100+ neural network architectures, 106+ classical ML algorithms,
50+ computer vision models, 90+ audio processing models, and 80+ reinforcement learning agents.

Key features include:
- Native .NET implementation with SIMD optimizations
- GPU acceleration via CUDA and OpenCL
- HuggingFace model integration
- Distributed training with DDP, FSDP, and ZeRO
- LoRA fine-tuning for LLMs
- Production model serving with AiDotNet.Serving

Unlike TorchSharp or TensorFlow.NET, AiDotNet is a pure .NET implementation without external runtime
dependencies, providing instant startup and native .NET type support including Memory<T> and Span<T>.
"),
            ("AiModelBuilder Guide", @"
AiModelBuilder is the main entry point for building and training models in AiDotNet.
It follows a fluent API pattern that makes model configuration simple and readable.

Basic usage:
```csharp
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetwork<double>(inputSize: 4, hiddenSize: 16, outputSize: 3))
    .ConfigureOptimizer(new AdamOptimizer<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);
```

The builder supports 73+ configuration methods including:
- ConfigureModel() - Set the model architecture
- ConfigureOptimizer() - Choose optimizer (Adam, SGD, AdamW, etc.)
- ConfigurePreprocessing() - Data normalization and augmentation
- ConfigureLearningRate() - Learning rate schedules
- ConfigureGpuAcceleration() - Enable GPU training
- ConfigureDistributedTraining() - Multi-GPU training
- ConfigureAutoML() - Automatic model selection
"),
            ("Neural Network Architectures", @"
AiDotNet supports 100+ neural network architectures:

Convolutional Networks (CNN):
- Standard CNN, ResNet, VGG, Inception, EfficientNet
- MobileNet, ShuffleNet, DenseNet, SENet

Recurrent Networks (RNN):
- LSTM, GRU, Bidirectional LSTM/GRU
- Attention-based RNNs

Transformers:
- BERT, GPT, T5, ViT (Vision Transformer)
- CLIP, BLIP, LLaMA, Mistral

Generative Models:
- GAN, DCGAN, StyleGAN, CycleGAN
- VAE, CVAE, VQ-VAE
- Diffusion Models (Stable Diffusion, DALL-E)

Graph Neural Networks:
- GCN, GAT, GraphSAGE, GIN
- Message Passing Neural Networks

Specialized Architectures:
- Capsule Networks
- Neural Radiance Fields (NeRF)
- Physics-Informed Neural Networks (PINNs)
"),
            ("RAG Components", @"
AiDotNet provides 50+ RAG (Retrieval-Augmented Generation) components:

Embeddings:
- SentenceTransformerEmbeddings - all-MiniLM-L6-v2, all-mpnet-base-v2
- OpenAIEmbeddings, CohereEmbeddings
- Custom embedding models

Vector Stores:
- InMemoryVectorStore - Simple, fast, in-memory storage
- FAISSVectorStore - Billion-scale similarity search
- ChromaVectorStore, PineconeVectorStore

Chunking Strategies:
- FixedSizeChunker - Split by character count
- SentenceChunker - Split by sentences
- RecursiveChunker - Respects document structure
- SemanticChunker - Groups similar content

Retrievers:
- DenseRetriever - Semantic similarity search
- SparseRetriever - BM25 keyword matching
- HybridRetriever - Combines dense and sparse

Rerankers:
- CrossEncoderReranker - High-quality reranking
- ColBERTReranker - Efficient late interaction
"),
            ("Distributed Training", @"
AiDotNet supports 10+ distributed training strategies for scaling to multiple GPUs:

DDP (Distributed Data Parallel):
- Replicates model on each GPU
- Splits data across GPUs
- Synchronizes gradients
- Best for models that fit on one GPU

FSDP (Fully Sharded Data Parallel):
- Shards model parameters across GPUs
- Reduces memory per GPU
- Enables training larger models
- Supports mixed precision

ZeRO Optimization:
- Stage 1: Partition optimizer states
- Stage 2: Partition gradients
- Stage 3: Partition parameters
- Memory reduction up to 8x

Pipeline Parallelism:
- Splits model layers across GPUs
- Uses micro-batching for efficiency
- Good for very deep models

Configuration example:
```csharp
.ConfigureDistributedTraining(new DistributedConfig
{
    Strategy = DistributedStrategy.FSDP,
    WorldSize = 8,
    ShardingStrategy = ShardingStrategy.FullShard
})
```
")
        };
    }
}

// =============================================================================
// Request/Response Models
// =============================================================================

public record ChatRequest(string Message);
public record DocumentRequest(string Title, string Content);

public class ChatResponse
{
    public string Answer { get; set; } = "";
    public List<SourceInfo> Sources { get; set; } = new();
    public double Confidence { get; set; }
}

public class RAGResponse
{
    public string Answer { get; set; } = "";
    public SourceInfo[] Sources { get; set; } = Array.Empty<SourceInfo>();
    public double Confidence { get; set; }
}

public class SourceInfo
{
    public string Title { get; set; } = "";
    public string Snippet { get; set; } = "";
    public double Score { get; set; }
}

public class DocumentInfo
{
    public string Id { get; set; } = "";
    public string Title { get; set; } = "";
    public int Length { get; set; }
    public DateTime AddedAt { get; set; }
}

public class ChatbotStats
{
    public int TotalDocuments { get; set; }
    public int TotalQueries { get; set; }
    public string EmbeddingModel { get; set; } = "";
    public int EmbeddingDimension { get; set; }
    public int ChunkSize { get; set; }
    public int TopK { get; set; }
}

public class Document
{
    public string Id { get; set; } = "";
    public string Text { get; set; } = "";
    public Dictionary<string, object> Metadata { get; set; } = new();
    public double Score { get; set; }
}

// =============================================================================
// HTML UI
// =============================================================================

static string GetHtmlUI() => """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiDotNet RAG Chatbot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        h1 { font-size: 2.5rem; margin-bottom: 10px; }
        h1 span { color: #4ecdc4; }
        .subtitle { color: rgba(255,255,255,0.7); font-size: 1.1rem; }
        .main-grid { display: grid; grid-template-columns: 1fr 350px; gap: 30px; }
        .chat-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { text-align: right; }
        .message-content {
            display: inline-block;
            padding: 12px 18px;
            border-radius: 18px;
            max-width: 80%;
        }
        .user .message-content { background: #4ecdc4; color: #000; }
        .assistant .message-content { background: rgba(255,255,255,0.1); }
        .sources {
            margin-top: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            font-size: 0.85rem;
        }
        .sources-title { color: #4ecdc4; margin-bottom: 8px; }
        .source-item {
            padding: 8px;
            margin: 5px 0;
            background: rgba(255,255,255,0.05);
            border-radius: 4px;
        }
        .source-score { color: #4ecdc4; font-size: 0.8rem; }
        .chat-input-area {
            display: flex;
            gap: 10px;
        }
        .chat-input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 25px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            font-size: 1rem;
        }
        .chat-input:focus { outline: none; background: rgba(255,255,255,0.15); }
        .chat-input::placeholder { color: rgba(255,255,255,0.5); }
        .send-btn {
            padding: 15px 30px;
            border: none;
            border-radius: 25px;
            background: #4ecdc4;
            color: #000;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .send-btn:hover { transform: scale(1.05); }
        .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .panel {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
        }
        .panel h3 { margin-bottom: 15px; color: #4ecdc4; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat {
            padding: 10px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.5rem; font-weight: 600; color: #4ecdc4; }
        .stat-label { font-size: 0.8rem; color: rgba(255,255,255,0.6); }
        .doc-list { max-height: 200px; overflow-y: auto; }
        .doc-item {
            padding: 10px;
            margin: 5px 0;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .doc-title { font-weight: 500; }
        .doc-size { font-size: 0.8rem; color: rgba(255,255,255,0.5); }
        .add-doc-btn {
            width: 100%;
            padding: 10px;
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 8px;
            background: transparent;
            color: rgba(255,255,255,0.6);
            cursor: pointer;
            transition: all 0.2s;
        }
        .add-doc-btn:hover { border-color: #4ecdc4; color: #4ecdc4; }
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal.active { display: flex; }
        .modal-content {
            background: #1a1a2e;
            padding: 30px;
            border-radius: 16px;
            width: 500px;
            max-width: 90%;
        }
        .modal-content h3 { margin-bottom: 20px; }
        .modal-input {
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
        }
        .modal-textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            margin-bottom: 15px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: #fff;
            resize: vertical;
        }
        .modal-buttons { display: flex; gap: 10px; justify-content: flex-end; }
        .modal-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .modal-btn.primary { background: #4ecdc4; color: #000; }
        .modal-btn.secondary { background: rgba(255,255,255,0.1); color: #fff; }
        .loading { display: inline-block; }
        .loading::after {
            content: '';
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: #4ecdc4;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
            .chat-container { height: 500px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><span>AiDotNet</span> RAG Chatbot</h1>
            <p class="subtitle">Ask questions about your documents with AI-powered retrieval</p>
        </header>

        <div class="main-grid">
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message assistant">
                        <div class="message-content">
                            Hello! I'm an AI assistant powered by AiDotNet's RAG pipeline.
                            I can answer questions about the documents in my knowledge base.
                            Try asking about AiDotNet features, neural networks, or RAG components!
                        </div>
                    </div>
                </div>
                <div class="chat-input-area">
                    <input type="text" class="chat-input" id="chatInput"
                           placeholder="Ask a question about your documents..."
                           onkeypress="if(event.key==='Enter')sendMessage()">
                    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel">
                    <h3>System Stats</h3>
                    <div class="stat-grid">
                        <div class="stat">
                            <div class="stat-value" id="docCount">-</div>
                            <div class="stat-label">Documents</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value" id="queryCount">-</div>
                            <div class="stat-label">Queries</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">384</div>
                            <div class="stat-label">Embedding Dim</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">5</div>
                            <div class="stat-label">Top-K</div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h3>Knowledge Base</h3>
                    <div class="doc-list" id="docList"></div>
                    <button class="add-doc-btn" onclick="showAddDocModal()">+ Add Document</button>
                </div>
            </div>
        </div>
    </div>

    <div class="modal" id="addDocModal">
        <div class="modal-content">
            <h3>Add Document</h3>
            <input type="text" class="modal-input" id="docTitle" placeholder="Document title">
            <textarea class="modal-textarea" id="docContent" placeholder="Document content..."></textarea>
            <div class="modal-buttons">
                <button class="modal-btn secondary" onclick="hideAddDocModal()">Cancel</button>
                <button class="modal-btn primary" onclick="addDocument()">Add Document</button>
            </div>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message) return;

            const messagesDiv = document.getElementById('chatMessages');
            const sendBtn = document.getElementById('sendBtn');

            // Add user message
            messagesDiv.innerHTML += `
                <div class="message user">
                    <div class="message-content">${escapeHtml(message)}</div>
                </div>
            `;

            // Add loading message
            const loadingId = 'loading-' + Date.now();
            messagesDiv.innerHTML += `
                <div class="message assistant" id="${loadingId}">
                    <div class="message-content">
                        <span class="loading">Thinking</span>
                    </div>
                </div>
            `;

            input.value = '';
            sendBtn.disabled = true;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = `
                        <div class="sources">
                            <div class="sources-title">Sources:</div>
                            ${data.sources.map(s => `
                                <div class="source-item">
                                    <strong>${escapeHtml(s.title)}</strong>
                                    <span class="source-score">(${(s.score * 100).toFixed(1)}% match)</span>
                                    <p style="margin-top:5px;color:rgba(255,255,255,0.7);font-size:0.85rem;">
                                        ${escapeHtml(s.snippet)}
                                    </p>
                                </div>
                            `).join('')}
                        </div>
                    `;
                }

                document.getElementById(loadingId).innerHTML = `
                    <div class="message-content">${escapeHtml(data.answer)}</div>
                    ${sourcesHtml}
                `;

                loadStats();
            } catch (error) {
                document.getElementById(loadingId).innerHTML = `
                    <div class="message-content" style="color:#ff6b6b;">
                        Error: ${error.message}
                    </div>
                `;
            }

            sendBtn.disabled = false;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const stats = await response.json();
                document.getElementById('docCount').textContent = stats.totalDocuments;
                document.getElementById('queryCount').textContent = stats.totalQueries;
            } catch (e) { console.error(e); }
        }

        async function loadDocuments() {
            try {
                const response = await fetch('/api/documents');
                const docs = await response.json();
                const docList = document.getElementById('docList');
                docList.innerHTML = docs.map(d => `
                    <div class="doc-item">
                        <div>
                            <div class="doc-title">${escapeHtml(d.title)}</div>
                            <div class="doc-size">${d.length} chars</div>
                        </div>
                    </div>
                `).join('');
            } catch (e) { console.error(e); }
        }

        function showAddDocModal() {
            document.getElementById('addDocModal').classList.add('active');
        }

        function hideAddDocModal() {
            document.getElementById('addDocModal').classList.remove('active');
            document.getElementById('docTitle').value = '';
            document.getElementById('docContent').value = '';
        }

        async function addDocument() {
            const title = document.getElementById('docTitle').value.trim();
            const content = document.getElementById('docContent').value.trim();
            if (!title || !content) return alert('Please fill in both fields');

            try {
                await fetch('/api/documents', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title, content })
                });
                hideAddDocModal();
                loadDocuments();
                loadStats();
            } catch (e) {
                alert('Error adding document: ' + e.message);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Initial load
        loadStats();
        loadDocuments();
    </script>
</body>
</html>
""";
