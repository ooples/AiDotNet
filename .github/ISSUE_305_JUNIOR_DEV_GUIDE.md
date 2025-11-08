# Junior Developer Implementation Guide: Issue #305
## File-Based Document Store for RAG

**Issue:** [#305 - Implement In-House, File-Based Document Store for RAG](https://github.com/ooples/AiDotNet/issues/305)

**Target Audience:** Junior developers new to RAG, vector databases, and file-based storage systems

**Estimated Time:** 3-5 days for a junior developer

---

## Table of Contents

1. [Understanding the Problem](#understanding-the-problem)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Phase-by-Phase Implementation Guide](#phase-by-phase-implementation-guide)
5. [Testing Strategy](#testing-strategy)
6. [Common Pitfalls](#common-pitfalls)
7. [Resources](#resources)

---

## Understanding the Problem

### What Are We Building?

We're creating a **file-based document store** that saves documents and their vector embeddings to disk, allowing them to persist between application restarts. Think of it as creating your own mini-database system using files.

### Why Do We Need This?

Currently, `InMemoryDocumentStore` keeps everything in RAM. When your application closes, all data is lost. This is like writing notes on a whiteboard - great for quick work, but they disappear when you erase the board.

A file-based store is like writing in a notebook - your notes stay there even when you close the notebook and come back later.

### Real-World Analogy

Imagine you're building a search system for a company's documentation:

- **In-Memory Store**: Like keeping all documents on sticky notes in your office. Fast to access, but if you leave and come back tomorrow, they're gone.
- **File-Based Store**: Like filing documents in a filing cabinet. A bit slower to access, but they're always there when you need them.

---

## Core Concepts

### 1. Vector Embeddings Explained

**What is a vector embedding?**

A vector embedding is a list of numbers that represents the "meaning" of text. Think of it like GPS coordinates for ideas:

```csharp
// Example: Converting text to a vector
string text = "The quick brown fox jumps over the lazy dog";
Vector<float> embedding = embeddingModel.Embed(text);
// Result: [0.23, -0.45, 0.67, ..., 0.12] (e.g., 384 numbers)
```

**Why vectors?**

Vectors let computers compare meanings mathematically:

```csharp
// Documents with similar meanings have vectors that are "close" to each other
var doc1 = "I love cats"; // Vector: [0.8, 0.6, 0.1, ...]
var doc2 = "I adore kittens"; // Vector: [0.82, 0.58, 0.09, ...] - CLOSE!
var doc3 = "Democracy is important"; // Vector: [-0.3, 0.1, 0.9, ...] - FAR!

// Similarity calculation (simplified)
var similarity = CosineSimilarity(doc1.Embedding, doc2.Embedding); // ~0.95 (very similar!)
var similarity2 = CosineSimilarity(doc1.Embedding, doc3.Embedding); // ~0.15 (not similar)
```

### 2. Document Chunking

**What is chunking?**

Chunking means breaking large documents into smaller pieces. Why?

- Embedding models have size limits (e.g., 512 words max)
- Smaller chunks = more precise retrieval
- Better context for answers

**Example:**

```
Original Document (5000 words):
"Introduction to Machine Learning... [lots of text] ...Conclusion."

After Chunking (10 chunks of ~500 words each):
Chunk 1: "Introduction to Machine Learning..."
Chunk 2: "What is supervised learning..."
Chunk 3: "Neural networks explained..."
...
Chunk 10: "...Conclusion."

Each chunk becomes a separate Document<T> with its own embedding.
```

### 3. Similarity Search

**How does similarity search work?**

Given a query, find the most similar documents:

```csharp
// User query
string query = "How do I train a neural network?";
Vector<float> queryEmbedding = embeddingModel.Embed(query);

// Search process
1. Compare queryEmbedding to EVERY document's embedding
2. Calculate similarity score for each (cosine similarity)
3. Sort by score (highest first)
4. Return top K results (e.g., top 5)

// Result
Document 1: "Neural network training basics..." (score: 0.92)
Document 2: "Backpropagation explained..." (score: 0.87)
Document 3: "Optimizers for training..." (score: 0.81)
Document 4: "Loss functions..." (score: 0.76)
Document 5: "Hyperparameter tuning..." (score: 0.71)
```

### 4. Metadata Filtering

**What is metadata filtering?**

Filtering documents by their properties BEFORE similarity search:

```csharp
// Metadata examples
var doc = new Document<float>("doc1", "Content about Python...")
{
    Metadata = new Dictionary<string, object>
    {
        ["language"] = "Python",
        ["year"] = 2024,
        ["category"] = "tutorial",
        ["author"] = "Jane Smith"
    }
};

// Search with filters
var filters = new Dictionary<string, object>
{
    ["year"] = 2024,  // Only 2024 documents
    ["language"] = "Python"  // Only Python documents
};

var results = store.GetSimilarWithFilters(queryVector, topK: 5, filters);
// Returns: Only 2024 Python documents, ranked by similarity
```

### 5. Inverted Index

**What is an inverted index?**

An inverted index is like the index at the back of a book. Instead of reading every page to find where "neural networks" is mentioned, you look in the index:

```
Traditional approach (slow):
Check document 1: Does it have category="science"? No
Check document 2: Does it have category="science"? No
Check document 3: Does it have category="science"? Yes! ✓
... (check ALL documents)

Inverted index approach (fast):
InvertedIndex["category"]["science"] = ["doc3", "doc7", "doc15", "doc42"]
// Instant lookup! Only need to check 4 documents instead of thousands.
```

**Structure:**

```json
{
  "category": {
    "science": ["doc3", "doc7", "doc15"],
    "history": ["doc1", "doc9"],
    "tutorial": ["doc2", "doc4", "doc8"]
  },
  "year": {
    "2024": ["doc1", "doc2", "doc3"],
    "2023": ["doc4", "doc5"]
  },
  "language": {
    "Python": ["doc1", "doc3", "doc8"],
    "JavaScript": ["doc2", "doc4"]
  }
}
```

### 6. B-Tree Index

**What is a B-Tree?**

A B-Tree is a data structure that lets you find things quickly, like a phone book:

```
Finding "Smith, John" in a phone book:
1. Open to middle (M section)
2. "Smith" > "M", go to right half
3. Open to middle of right half (S section)
4. Found "Smith"!

Instead of checking all 1000 pages, you only checked 3 sections!
```

**For our use case:**

We use B-Trees to map document IDs to their file locations:

```
B-Tree Index:
documentId -> byte offset in file

"doc1" -> 0 bytes
"doc2" -> 1024 bytes
"doc3" -> 2048 bytes
...

To get "doc2":
1. Look up "doc2" in B-Tree index: offset = 1024
2. Seek to byte 1024 in documents.data file
3. Read and deserialize the document
```

---

## Architecture Overview

### File Structure

The `FileDocumentStore` manages these files:

```
/data/document-store/
├── documents.data          # Serialized documents (binary)
├── doc_index.db           # B-Tree: documentId -> file offset
├── deletions.log          # List of deleted document IDs
├── metadata_index.json    # Inverted index for metadata
├── vectors.bin            # Binary vector embeddings
└── vector_map.log         # Map: line number -> documentId
```

### How Files Work Together

**Adding a document:**

```
1. Serialize document -> append to documents.data at offset 0
2. Add to B-Tree: doc_index["doc1"] = 0
3. Update inverted index: metadata_index["category"]["science"].add("doc1")
4. Append embedding to vectors.bin
5. Append "doc1" to vector_map.log
```

**Retrieving a document:**

```
1. Look up in B-Tree: offset = doc_index["doc1"] = 0
2. Check if in deletions.log (if yes, return null)
3. Seek to offset 0 in documents.data
4. Read and deserialize document
5. Return document
```

**Searching by similarity:**

```
1. Apply metadata filters using inverted index (get candidate doc IDs)
2. For each candidate:
   a. Find vector index from vector_map.log
   b. Read vector from vectors.bin
   c. Calculate similarity
3. Sort by similarity, return top K
```

### Class Relationships

```
IDocumentStore<T>
    ↑
    |
DocumentStoreBase<T>
    ↑
    |
FileDocumentStore<T>
    |
    +-- uses --> BTreeIndex (for doc_index.db)
    |
    +-- uses --> InvertedIndex (for metadata_index.json)
```

---

## Phase-by-Phase Implementation Guide

### Phase 1: Core Document Persistence

#### AC 1.1: FileDocumentStore Scaffolding

**Goal:** Create the basic class structure and file management.

**Step 1: Create the file**

Location: `src/RetrievalAugmentedGeneration/DocumentStores/FileDocumentStore.cs`

**Step 2: Basic class structure**

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.LinearAlgebra;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

/// <summary>
/// File-based document store that persists documents and vectors to disk.
/// </summary>
/// <remarks>
/// <para>
/// FileDocumentStore provides persistent storage without requiring external databases.
/// All data is stored in a local directory using efficient file formats and indexing.
/// </para>
/// <para><b>For Beginners:</b> This is like creating your own mini-database using files.
///
/// Think of it as a filing cabinet system:
/// - documents.data: The file folders containing actual documents
/// - doc_index.db: The catalog card system (find documents fast)
/// - vectors.bin: A separate drawer for numeric "fingerprints"
/// - metadata_index.json: Tags system for quick filtering
///
/// When your program stops and restarts, all your data is still there!
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type for vector operations (typically float or double).</typeparam>
public class FileDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly string _directoryPath;
    private readonly string _documentsFilePath;
    private readonly string _indexFilePath;
    private readonly string _deletionsFilePath;

    private readonly BTreeIndex _docIndex;
    private readonly HashSet<string> _deletedIds;

    private int _documentCount;
    private int _vectorDimension;

    /// <summary>
    /// Gets the number of documents currently stored (excluding deleted).
    /// </summary>
    public override int DocumentCount => _documentCount;

    /// <summary>
    /// Gets the dimensionality of vectors stored in this document store.
    /// </summary>
    public override int VectorDimension => _vectorDimension;

    /// <summary>
    /// Initializes a new instance of the FileDocumentStore class.
    /// </summary>
    /// <param name="directoryPath">The directory where all data files will be stored.</param>
    /// <exception cref="ArgumentException">Thrown when directoryPath is null or empty.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up your file-based database.
    ///
    /// Example:
    /// <code>
    /// // Create a store in a specific directory
    /// var store = new FileDocumentStore&lt;float&gt;("./data/my-documents");
    ///
    /// // The directory will contain:
    /// // - documents.data (your documents)
    /// // - doc_index.db (fast lookup index)
    /// // - deletions.log (deleted document IDs)
    /// // - metadata_index.json (filter index)
    /// // - vectors.bin (embeddings)
    /// // - vector_map.log (vector ID mapping)
    /// </code>
    ///
    /// If the directory doesn't exist, it will be created automatically.
    /// If it already exists, existing data will be loaded.
    /// </para>
    /// </remarks>
    public FileDocumentStore(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            throw new ArgumentException("Directory path cannot be null or empty", nameof(directoryPath));

        _directoryPath = directoryPath;

        // Ensure directory exists
        Directory.CreateDirectory(_directoryPath);

        // Define file paths
        _documentsFilePath = Path.Combine(_directoryPath, "documents.data");
        _indexFilePath = Path.Combine(_directoryPath, "doc_index.db");
        _deletionsFilePath = Path.Combine(_directoryPath, "deletions.log");

        // Initialize B-Tree index
        _docIndex = new BTreeIndex(_indexFilePath);

        // Load deletions
        _deletedIds = LoadDeletions();

        // Initialize counts
        _documentCount = _docIndex.Count - _deletedIds.Count;
        _vectorDimension = 0; // Will be set when first document is added
    }

    /// <summary>
    /// Loads the set of deleted document IDs from the deletions log.
    /// </summary>
    private HashSet<string> LoadDeletions()
    {
        var deletedIds = new HashSet<string>();

        if (File.Exists(_deletionsFilePath))
        {
            foreach (var line in File.ReadLines(_deletionsFilePath))
            {
                if (!string.IsNullOrWhiteSpace(line))
                    deletedIds.Add(line.Trim());
            }
        }

        return deletedIds;
    }
}
```

**Understanding the Constructor:**

1. **Validate input:** Always check parameters aren't null/empty
2. **Create directory:** `Directory.CreateDirectory()` creates the directory if it doesn't exist, or does nothing if it already exists (safe to call always)
3. **Define file paths:** Use `Path.Combine()` to build cross-platform paths
4. **Initialize components:** Create the B-Tree index and load deletion log
5. **Calculate counts:** How many documents are actually available (total - deleted)

#### AC 1.2: Document Indexing and CRUD

**Goal:** Implement Add, Get, and Remove operations.

**Understanding CRUD:**
- **C**reate: Add documents
- **R**ead: Get documents
- **U**pdate: Not needed (documents are immutable in this design)
- **D**elete: Remove documents

**Step 1: Serialization Helper**

Documents need to be converted to binary format for storage:

```csharp
/// <summary>
/// Serializes a document to binary format for disk storage.
/// </summary>
private byte[] SerializeDocument(Document<T> document)
{
    using var memoryStream = new MemoryStream();
    using var writer = new BinaryWriter(memoryStream);

    // Write document ID (length-prefixed string)
    writer.Write(document.Id);

    // Write content (length-prefixed string)
    writer.Write(document.Content);

    // Write metadata count
    writer.Write(document.Metadata.Count);

    // Write each metadata entry
    foreach (var kvp in document.Metadata)
    {
        writer.Write(kvp.Key);

        // Write type marker and value
        if (kvp.Value is string strValue)
        {
            writer.Write((byte)1); // String type
            writer.Write(strValue);
        }
        else if (kvp.Value is int intValue)
        {
            writer.Write((byte)2); // Int type
            writer.Write(intValue);
        }
        else if (kvp.Value is double doubleValue)
        {
            writer.Write((byte)3); // Double type
            writer.Write(doubleValue);
        }
        else if (kvp.Value is bool boolValue)
        {
            writer.Write((byte)4); // Bool type
            writer.Write(boolValue);
        }
        // Add more types as needed
    }

    return memoryStream.ToArray();
}

/// <summary>
/// Deserializes a document from binary format.
/// </summary>
private Document<T> DeserializeDocument(byte[] data)
{
    using var memoryStream = new MemoryStream(data);
    using var reader = new BinaryReader(memoryStream);

    // Read document ID
    var id = reader.ReadString();

    // Read content
    var content = reader.ReadString();

    // Read metadata count
    var metadataCount = reader.ReadInt32();

    var metadata = new Dictionary<string, object>();
    for (int i = 0; i < metadataCount; i++)
    {
        var key = reader.ReadString();
        var typeMarker = reader.ReadByte();

        object value = typeMarker switch
        {
            1 => reader.ReadString(),
            2 => reader.ReadInt32(),
            3 => reader.ReadDouble(),
            4 => reader.ReadBoolean(),
            _ => throw new InvalidOperationException($"Unknown type marker: {typeMarker}")
        };

        metadata[key] = value;
    }

    return new Document<T>(id, content, metadata);
}
```

**Why serialize this way?**

- **Length-prefixed strings:** Write the length first, then the string data. This lets us know how many bytes to read when deserializing.
- **Type markers:** Since metadata values can be different types, we store a byte indicating the type before the value.
- **Efficient:** Binary format is compact and fast to read/write.

**Step 2: Implement AddCore**

```csharp
/// <summary>
/// Core logic for adding a single vector document to the file store.
/// </summary>
protected override void AddCore(VectorDocument<T> vectorDocument)
{
    // Set vector dimension from first document
    if (_documentCount == 0)
        _vectorDimension = vectorDocument.Embedding.Length;

    var document = vectorDocument.Document;

    // Serialize document to binary
    var documentBytes = SerializeDocument(document);

    // Append to documents.data file and get the byte offset
    long offset;
    using (var fileStream = new FileStream(_documentsFilePath, FileMode.Append, FileAccess.Write))
    {
        offset = fileStream.Position;
        fileStream.Write(documentBytes, 0, documentBytes.Length);
    }

    // Add to B-Tree index: documentId -> file offset
    _docIndex.Add(document.Id, offset);

    // Remove from deletions if it was previously deleted (re-adding)
    _deletedIds.Remove(document.Id);

    // Increment count
    _documentCount++;
}
```

**Understanding AddCore:**

1. **Set vector dimension:** On the first document, record what dimension we're using (all subsequent documents must match)
2. **Serialize:** Convert document to bytes
3. **Append to file:** Open in Append mode, write bytes, record the starting position (offset)
4. **Index:** Store mapping from document ID to file offset in B-Tree
5. **Update state:** Remove from deletions if re-adding, increment count

**Why Append mode?**

`FileMode.Append` always writes at the end of the file. This is perfect for our use case:
- New documents are always appended
- We never modify existing documents (immutable)
- Simple and efficient

**Step 3: Implement GetByIdCore**

```csharp
/// <summary>
/// Core logic for retrieving a document by its unique identifier.
/// </summary>
protected override Document<T>? GetByIdCore(string documentId)
{
    // Check if document was deleted
    if (_deletedIds.Contains(documentId))
        return null;

    // Look up offset in B-Tree index
    var offset = _docIndex.Get(documentId);
    if (!offset.HasValue)
        return null; // Document doesn't exist

    // Read document from file at the offset
    using var fileStream = new FileStream(_documentsFilePath, FileMode.Open, FileAccess.Read);

    // Seek to the document's position
    fileStream.Seek(offset.Value, SeekOrigin.Begin);

    // Read length prefix (first 4 bytes in our serialization format)
    // Note: We need to know the document size. Options:
    // 1. Store size in index (better)
    // 2. Write size as first field in serialization
    // 3. Read until next document (inefficient)

    // For simplicity, let's assume we stored size in the index
    // We'll need to modify AddCore and BTreeIndex to support this

    // Alternative: Read the entire file into memory (inefficient for large files)
    // Better approach: Modify serialization to include document size

    // Let's use a simpler approach: store (offset, size) in index
    // This requires modifying BTreeIndex to store tuples

    // For now, read a large chunk and find the document boundary
    var buffer = new byte[1024 * 1024]; // 1 MB buffer
    var bytesRead = fileStream.Read(buffer, 0, buffer.Length);

    // Deserialize from buffer
    var document = DeserializeDocument(buffer.Take(bytesRead).ToArray());

    return document;
}
```

**Challenge: Variable-Length Documents**

Documents have different sizes. How do we know how many bytes to read?

**Solutions:**

1. **Store size in index:** Modify B-Tree to store `(offset, size)` tuples
2. **Size prefix in serialization:** Write document size as first field
3. **Fixed-size records:** Pad documents to fixed size (wasteful)

**Best approach:** Size prefix in serialization

Let's modify our serialization:

```csharp
private byte[] SerializeDocument(Document<T> document)
{
    using var memoryStream = new MemoryStream();
    using var writer = new BinaryWriter(memoryStream);

    // Reserve space for size (will write at end)
    writer.Write(0); // Placeholder for size

    // Write document data
    writer.Write(document.Id);
    writer.Write(document.Content);

    // ... rest of metadata writing ...

    // Get total size
    var totalSize = (int)memoryStream.Length;

    // Go back and write size at the beginning
    memoryStream.Seek(0, SeekOrigin.Begin);
    writer.Write(totalSize);

    return memoryStream.ToArray();
}

private Document<T> DeserializeDocument(byte[] data)
{
    using var memoryStream = new MemoryStream(data);
    using var reader = new BinaryReader(memoryStream);

    // Read size (we already know it from reading, but skip it)
    var size = reader.ReadInt32();

    // Read rest of document
    var id = reader.ReadString();
    var content = reader.ReadString();

    // ... rest of deserialization ...

    return new Document<T>(id, content, metadata);
}
```

**Updated GetByIdCore:**

```csharp
protected override Document<T>? GetByIdCore(string documentId)
{
    if (_deletedIds.Contains(documentId))
        return null;

    var offset = _docIndex.Get(documentId);
    if (!offset.HasValue)
        return null;

    using var fileStream = new FileStream(_documentsFilePath, FileMode.Open, FileAccess.Read);
    fileStream.Seek(offset.Value, SeekOrigin.Begin);

    // Read size prefix
    var sizeBytes = new byte[4];
    fileStream.Read(sizeBytes, 0, 4);
    var size = BitConverter.ToInt32(sizeBytes, 0);

    // Read entire document
    var documentBytes = new byte[size];
    fileStream.Seek(offset.Value, SeekOrigin.Begin); // Go back to start
    fileStream.Read(documentBytes, 0, size);

    return DeserializeDocument(documentBytes);
}
```

**Step 4: Implement RemoveCore**

```csharp
/// <summary>
/// Core logic for removing a document from the file store.
/// </summary>
protected override bool RemoveCore(string documentId)
{
    // Check if document exists
    var offset = _docIndex.Get(documentId);
    if (!offset.HasValue)
        return false; // Doesn't exist

    // Check if already deleted
    if (_deletedIds.Contains(documentId))
        return false; // Already deleted

    // Add to deletions log
    File.AppendAllText(_deletionsFilePath, documentId + Environment.NewLine);
    _deletedIds.Add(documentId);

    // Decrement count
    _documentCount--;

    return true;
}
```

**Why not actually delete from file?**

Deleting from the middle of a file is expensive (requires rewriting the entire file). Instead:

1. Mark as deleted in deletions.log
2. When retrieving, check deletions first
3. Later, we can implement compaction (rebuild file without deleted docs)

This is called **lazy deletion** or **tombstoning**.

### Phase 2: Search and Retrieval

#### AC 2.1: Metadata Indexing

**Goal:** Build an inverted index for fast metadata filtering.

**Step 1: Create InvertedIndex class**

Create a new file: `src/RetrievalAugmentedGeneration/DocumentStores/Helpers/InvertedIndex.cs`

```csharp
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace AiDotNet.RetrievalAugmentedGeneration.DocumentStores.Helpers;

/// <summary>
/// Inverted index for fast metadata-based document filtering.
/// </summary>
/// <remarks>
/// <para>
/// An inverted index maps metadata field values to document IDs, enabling O(1) lookups
/// instead of scanning all documents. Structure: FieldName -> FieldValue -> [DocumentIds]
/// </para>
/// <para><b>For Beginners:</b> This is like the index at the back of a book.
///
/// Instead of reading every page to find where "Python" is mentioned,
/// you look in the index: "Python: pages 15, 42, 67, 103"
///
/// Here, instead of checking every document for category="Python",
/// we look up: index["category"]["Python"] = ["doc1", "doc5", "doc12"]
///
/// This is MUCH faster for large document collections!
/// </para>
/// </remarks>
public class InvertedIndex
{
    private readonly string _filePath;
    private Dictionary<string, Dictionary<string, HashSet<string>>> _index;

    /// <summary>
    /// Initializes a new instance of the InvertedIndex class.
    /// </summary>
    /// <param name="filePath">Path to the JSON file storing the index.</param>
    public InvertedIndex(string filePath)
    {
        _filePath = filePath;
        _index = new Dictionary<string, Dictionary<string, HashSet<string>>>();

        // Load existing index if file exists
        if (File.Exists(_filePath))
        {
            Load();
        }
    }

    /// <summary>
    /// Adds a document to the index based on its metadata.
    /// </summary>
    /// <param name="documentId">The document ID.</param>
    /// <param name="metadata">The document's metadata.</param>
    public void AddDocument(string documentId, Dictionary<string, object> metadata)
    {
        foreach (var kvp in metadata)
        {
            var fieldName = kvp.Key;
            var fieldValue = kvp.Value.ToString() ?? string.Empty;

            // Ensure field exists in index
            if (!_index.ContainsKey(fieldName))
                _index[fieldName] = new Dictionary<string, HashSet<string>>();

            // Ensure value exists in field
            if (!_index[fieldName].ContainsKey(fieldValue))
                _index[fieldName][fieldValue] = new HashSet<string>();

            // Add document ID to value's set
            _index[fieldName][fieldValue].Add(documentId);
        }
    }

    /// <summary>
    /// Removes a document from the index.
    /// </summary>
    public void RemoveDocument(string documentId, Dictionary<string, object> metadata)
    {
        foreach (var kvp in metadata)
        {
            var fieldName = kvp.Key;
            var fieldValue = kvp.Value.ToString() ?? string.Empty;

            if (_index.ContainsKey(fieldName) &&
                _index[fieldName].ContainsKey(fieldValue))
            {
                _index[fieldName][fieldValue].Remove(documentId);

                // Clean up empty sets
                if (_index[fieldName][fieldValue].Count == 0)
                    _index[fieldName].Remove(fieldValue);

                if (_index[fieldName].Count == 0)
                    _index.Remove(fieldName);
            }
        }
    }

    /// <summary>
    /// Gets document IDs matching the specified filters.
    /// </summary>
    /// <param name="filters">Metadata filters to apply.</param>
    /// <returns>Set of document IDs matching ALL filters (intersection).</returns>
    public HashSet<string> GetMatchingDocuments(Dictionary<string, object> filters)
    {
        if (filters.Count == 0)
            return new HashSet<string>(); // No filters = no filtering (handled by caller)

        HashSet<string>? result = null;

        foreach (var filter in filters)
        {
            var fieldName = filter.Key;
            var fieldValue = filter.Value.ToString() ?? string.Empty;

            // Get documents for this filter
            HashSet<string> matchingDocs;
            if (_index.ContainsKey(fieldName) &&
                _index[fieldName].ContainsKey(fieldValue))
            {
                matchingDocs = _index[fieldName][fieldValue];
            }
            else
            {
                // No documents match this filter
                return new HashSet<string>();
            }

            // Intersect with previous results
            if (result == null)
                result = new HashSet<string>(matchingDocs);
            else
                result.IntersectWith(matchingDocs);

            // Early exit if no matches
            if (result.Count == 0)
                return result;
        }

        return result ?? new HashSet<string>();
    }

    /// <summary>
    /// Saves the index to disk as JSON.
    /// </summary>
    public void Save()
    {
        // Convert HashSet to List for JSON serialization
        var serializableIndex = new Dictionary<string, Dictionary<string, List<string>>>();

        foreach (var field in _index)
        {
            serializableIndex[field.Key] = new Dictionary<string, List<string>>();
            foreach (var value in field.Value)
            {
                serializableIndex[field.Key][value.Key] = new List<string>(value.Value);
            }
        }

        var json = JsonSerializer.Serialize(serializableIndex, new JsonSerializerOptions
        {
            WriteIndented = true
        });

        File.WriteAllText(_filePath, json);
    }

    /// <summary>
    /// Loads the index from disk.
    /// </summary>
    private void Load()
    {
        var json = File.ReadAllText(_filePath);
        var serializableIndex = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, List<string>>>>(json);

        if (serializableIndex == null)
            return;

        // Convert List back to HashSet
        _index = new Dictionary<string, Dictionary<string, HashSet<string>>>();
        foreach (var field in serializableIndex)
        {
            _index[field.Key] = new Dictionary<string, HashSet<string>>();
            foreach (var value in field.Value)
            {
                _index[field.Key][value.Key] = new HashSet<string>(value.Value);
            }
        }
    }

    /// <summary>
    /// Clears the entire index.
    /// </summary>
    public void Clear()
    {
        _index.Clear();
        if (File.Exists(_filePath))
            File.Delete(_filePath);
    }
}
```

**Understanding the Index Structure:**

```
_index = {
    "category": {
        "science": ["doc1", "doc3", "doc7"],
        "history": ["doc2", "doc5"],
        "tutorial": ["doc4", "doc6", "doc8"]
    },
    "year": {
        "2024": ["doc1", "doc2", "doc3"],
        "2023": ["doc4", "doc5", "doc6"]
    },
    "language": {
        "Python": ["doc1", "doc3", "doc5"],
        "JavaScript": ["doc2", "doc4"]
    }
}
```

**Step 2: Integrate InvertedIndex into FileDocumentStore**

```csharp
public class FileDocumentStore<T> : DocumentStoreBase<T>
{
    // ... existing fields ...
    private readonly InvertedIndex _metadataIndex;

    public FileDocumentStore(string directoryPath)
    {
        // ... existing initialization ...

        // Initialize metadata index
        var metadataIndexPath = Path.Combine(_directoryPath, "metadata_index.json");
        _metadataIndex = new InvertedIndex(metadataIndexPath);
    }

    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        // ... existing code ...

        // Update metadata index
        _metadataIndex.AddDocument(document.Id, document.Metadata);
        _metadataIndex.Save(); // Persist to disk

        // ... rest of method ...
    }

    protected override bool RemoveCore(string documentId)
    {
        // ... existing code ...

        // Get document to access its metadata
        var document = GetByIdCore(documentId);
        if (document != null)
        {
            _metadataIndex.RemoveDocument(documentId, document.Metadata);
            _metadataIndex.Save();
        }

        // ... rest of method ...
    }
}
```

#### AC 2.2: Vector Storage

**Goal:** Store and retrieve vector embeddings efficiently.

**Step 1: Add vector file paths**

```csharp
public class FileDocumentStore<T> : DocumentStoreBase<T>
{
    // ... existing fields ...
    private readonly string _vectorsFilePath;
    private readonly string _vectorMapFilePath;

    public FileDocumentStore(string directoryPath)
    {
        // ... existing initialization ...

        _vectorsFilePath = Path.Combine(_directoryPath, "vectors.bin");
        _vectorMapFilePath = Path.Combine(_directoryPath, "vector_map.log");
    }
}
```

**Step 2: Vector serialization helpers**

```csharp
/// <summary>
/// Writes a vector to the vectors.bin file.
/// </summary>
private void WriteVector(Vector<T> vector)
{
    using var fileStream = new FileStream(_vectorsFilePath, FileMode.Append, FileAccess.Write);
    using var writer = new BinaryWriter(fileStream);

    // Write vector dimension
    writer.Write(vector.Length);

    // Write each element
    for (int i = 0; i < vector.Length; i++)
    {
        // Convert T to double for serialization
        var value = Convert.ToDouble(vector[i]);
        writer.Write(value);
    }
}

/// <summary>
/// Reads a vector at a specific index from vectors.bin.
/// </summary>
private Vector<T> ReadVector(int vectorIndex)
{
    using var fileStream = new FileStream(_vectorsFilePath, FileMode.Open, FileAccess.Read);
    using var reader = new BinaryReader(fileStream);

    // Calculate byte offset
    // Each vector: 4 bytes (dimension) + (dimension * 8 bytes per double)
    // But dimension varies! We need to store dimension or calculate offset differently.

    // Better approach: All vectors have same dimension (VectorDimension property)
    int bytesPerVector = 4 + (_vectorDimension * 8); // 4 for dimension int, 8 per double
    long offset = vectorIndex * bytesPerVector;

    fileStream.Seek(offset, SeekOrigin.Begin);

    // Read dimension (should match VectorDimension)
    var dimension = reader.ReadInt32();
    if (dimension != _vectorDimension)
        throw new InvalidOperationException($"Vector dimension mismatch: expected {_vectorDimension}, got {dimension}");

    // Read elements
    var elements = new T[dimension];
    for (int i = 0; i < dimension; i++)
    {
        var value = reader.ReadDouble();
        elements[i] = NumOps.FromDouble(value);
    }

    return new Vector<T>(elements);
}
```

**Step 3: Update AddCore to store vectors**

```csharp
protected override void AddCore(VectorDocument<T> vectorDocument)
{
    // ... existing document storage code ...

    // Store vector
    WriteVector(vectorDocument.Embedding);

    // Store mapping: line number in vector_map.log = vector index
    File.AppendAllText(_vectorMapFilePath, vectorDocument.Document.Id + Environment.NewLine);

    // ... rest of method ...
}
```

**Understanding the vector storage:**

```
vectors.bin:
[dim][v1][v2][v3]...[vN] [dim][v1][v2][v3]...[vN] [dim][v1][v2][v3]...[vN]
|-- Vector 0 ---------|   |-- Vector 1 ---------|   |-- Vector 2 ---------|

vector_map.log:
doc1
doc2
doc3

Line 0 (doc1) corresponds to Vector 0
Line 1 (doc2) corresponds to Vector 1
Line 2 (doc3) corresponds to Vector 2
```

#### AC 2.3: Similarity Search Implementation

**Goal:** Implement full similarity search with metadata filtering.

```csharp
/// <summary>
/// Core logic for similarity search with optional metadata filtering.
/// </summary>
protected override IEnumerable<Document<T>> GetSimilarCore(
    Vector<T> queryVector,
    int topK,
    Dictionary<string, object> metadataFilters)
{
    // Step 1: Get candidate document IDs from metadata filters
    HashSet<string>? candidateIds = null;
    if (metadataFilters.Count > 0)
    {
        candidateIds = _metadataIndex.GetMatchingDocuments(metadataFilters);

        // If no documents match filters, return empty
        if (candidateIds.Count == 0)
            return Enumerable.Empty<Document<T>>();
    }

    // Step 2: Scan vectors and calculate similarities
    var scoredDocuments = new List<(string documentId, T similarity)>();

    // Read vector map to get document IDs
    var lines = File.ReadAllLines(_vectorMapFilePath);

    for (int vectorIndex = 0; vectorIndex < lines.Length; vectorIndex++)
    {
        var documentId = lines[vectorIndex].Trim();

        // Skip if deleted
        if (_deletedIds.Contains(documentId))
            continue;

        // Skip if doesn't match metadata filters
        if (candidateIds != null && !candidateIds.Contains(documentId))
            continue;

        // Read vector from file
        var vector = ReadVector(vectorIndex);

        // Calculate similarity
        var similarity = StatisticsHelper<T>.CosineSimilarity(queryVector, vector);

        scoredDocuments.Add((documentId, similarity));
    }

    // Step 3: Sort by similarity and take top K
    var topDocuments = scoredDocuments
        .OrderByDescending(x => Convert.ToDouble(x.similarity))
        .Take(topK)
        .ToList();

    // Step 4: Retrieve full documents and set relevance scores
    var results = new List<Document<T>>();
    foreach (var (documentId, similarity) in topDocuments)
    {
        var document = GetByIdCore(documentId);
        if (document != null)
        {
            document.RelevanceScore = similarity;
            document.HasRelevanceScore = true;
            results.Add(document);
        }
    }

    return results;
}
```

**Understanding the search flow:**

```
Query: "How to train neural networks?"
Filters: { "category": "tutorial", "year": 2024 }
TopK: 5

Step 1: Metadata Filtering
    InvertedIndex["category"]["tutorial"] = [doc1, doc3, doc5, doc7, doc8]
    InvertedIndex["year"]["2024"] = [doc1, doc2, doc3, doc4]
    Intersection = [doc1, doc3] (only 2 docs match BOTH filters)

Step 2: Vector Scanning
    For doc1:
        Read vector at index 0: [0.23, -0.45, ...]
        Similarity = CosineSimilarity(queryVector, doc1Vector) = 0.92
    For doc3:
        Read vector at index 2: [0.18, -0.41, ...]
        Similarity = CosineSimilarity(queryVector, doc3Vector) = 0.87

Step 3: Sort and Return
    Results (already sorted):
        doc1 (score: 0.92)
        doc3 (score: 0.87)

    TopK=5 but only 2 results (that's OK, fewer matches than requested)
```

---

## Testing Strategy

### Unit Tests

**Test File:** `tests/AiDotNet.Tests/RetrievalAugmentedGeneration/FileDocumentStoreTests.cs`

#### Test 1: Basic CRUD Operations

```csharp
[Fact]
public void AddAndRetrieve_SingleDocument_Success()
{
    // Arrange
    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    var store = new FileDocumentStore<float>(tempDir);

    var document = new Document<float>("doc1", "Test content");
    var embedding = new Vector<float>(new float[] { 0.1f, 0.2f, 0.3f });
    var vectorDoc = new VectorDocument<float>(document, embedding);

    // Act
    store.Add(vectorDoc);
    var retrieved = store.GetById("doc1");

    // Assert
    Assert.NotNull(retrieved);
    Assert.Equal("doc1", retrieved.Id);
    Assert.Equal("Test content", retrieved.Content);

    // Cleanup
    Directory.Delete(tempDir, true);
}

[Fact]
public void Remove_ExistingDocument_ReturnsTrue()
{
    // Arrange
    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    var store = new FileDocumentStore<float>(tempDir);

    var document = new Document<float>("doc1", "Test content");
    var embedding = new Vector<float>(new float[] { 0.1f, 0.2f, 0.3f });
    var vectorDoc = new VectorDocument<float>(document, embedding);

    store.Add(vectorDoc);

    // Act
    var removed = store.Remove("doc1");
    var retrieved = store.GetById("doc1");

    // Assert
    Assert.True(removed);
    Assert.Null(retrieved); // Document should not be retrievable after deletion

    // Cleanup
    Directory.Delete(tempDir, true);
}
```

#### Test 2: Persistence Across Restarts

```csharp
[Fact]
public void Persistence_DataSurvivesRestart()
{
    // Arrange
    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

    // First instance: Add documents
    var store1 = new FileDocumentStore<float>(tempDir);
    var document = new Document<float>("doc1", "Test content");
    var embedding = new Vector<float>(new float[] { 0.1f, 0.2f, 0.3f });
    var vectorDoc = new VectorDocument<float>(document, embedding);
    store1.Add(vectorDoc);

    // Simulate restart: Create new instance with same directory
    var store2 = new FileDocumentStore<float>(tempDir);

    // Act
    var retrieved = store2.GetById("doc1");

    // Assert
    Assert.NotNull(retrieved);
    Assert.Equal("doc1", retrieved.Id);
    Assert.Equal("Test content", retrieved.Content);
    Assert.Equal(1, store2.DocumentCount);

    // Cleanup
    Directory.Delete(tempDir, true);
}
```

#### Test 3: Similarity Search

```csharp
[Fact]
public void GetSimilar_ReturnsClosestDocuments()
{
    // Arrange
    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    var store = new FileDocumentStore<float>(tempDir);

    // Add documents with different embeddings
    var doc1 = new VectorDocument<float>(
        new Document<float>("doc1", "About cats"),
        new Vector<float>(new float[] { 1.0f, 0.0f, 0.0f })
    );
    var doc2 = new VectorDocument<float>(
        new Document<float>("doc2", "About dogs"),
        new Vector<float>(new float[] { 0.9f, 0.1f, 0.0f }) // Close to doc1
    );
    var doc3 = new VectorDocument<float>(
        new Document<float>("doc3", "About cars"),
        new Vector<float>(new float[] { 0.0f, 0.0f, 1.0f }) // Far from doc1
    );

    store.Add(doc1);
    store.Add(doc2);
    store.Add(doc3);

    // Act
    var queryVector = new Vector<float>(new float[] { 1.0f, 0.0f, 0.0f }); // Same as doc1
    var results = store.GetSimilar(queryVector, topK: 2).ToList();

    // Assert
    Assert.Equal(2, results.Count);
    Assert.Equal("doc1", results[0].Id); // Most similar
    Assert.Equal("doc2", results[1].Id); // Second most similar
    Assert.True(Convert.ToDouble(results[0].RelevanceScore) > Convert.ToDouble(results[1].RelevanceScore));

    // Cleanup
    Directory.Delete(tempDir, true);
}
```

#### Test 4: Metadata Filtering

```csharp
[Fact]
public void GetSimilarWithFilters_OnlyReturnsMatchingDocuments()
{
    // Arrange
    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
    var store = new FileDocumentStore<float>(tempDir);

    var embedding = new Vector<float>(new float[] { 1.0f, 0.0f, 0.0f });

    var doc1 = new VectorDocument<float>(
        new Document<float>("doc1", "Python tutorial 2024", new Dictionary<string, object>
        {
            ["language"] = "Python",
            ["year"] = 2024
        }),
        embedding
    );

    var doc2 = new VectorDocument<float>(
        new Document<float>("doc2", "Python tutorial 2023", new Dictionary<string, object>
        {
            ["language"] = "Python",
            ["year"] = 2023
        }),
        embedding
    );

    var doc3 = new VectorDocument<float>(
        new Document<float>("doc3", "JavaScript tutorial 2024", new Dictionary<string, object>
        {
            ["language"] = "JavaScript",
            ["year"] = 2024
        }),
        embedding
    );

    store.Add(doc1);
    store.Add(doc2);
    store.Add(doc3);

    // Act
    var filters = new Dictionary<string, object>
    {
        ["language"] = "Python",
        ["year"] = 2024
    };
    var results = store.GetSimilarWithFilters(embedding, topK: 10, filters).ToList();

    // Assert
    Assert.Single(results); // Only doc1 matches both filters
    Assert.Equal("doc1", results[0].Id);

    // Cleanup
    Directory.Delete(tempDir, true);
}
```

### Integration Tests

**Test File:** `tests/AiDotNet.Integration.Tests/FileDocumentStoreIntegrationTests.cs`

```csharp
[Fact]
public void RealWorldScenario_AddSearchRestart()
{
    // This test simulates a real-world RAG pipeline:
    // 1. Add documents with metadata
    // 2. Search for similar documents
    // 3. Restart (new instance)
    // 4. Search again and verify results persist

    var tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());

    // Step 1: Add documents
    var store1 = new FileDocumentStore<float>(tempDir);

    for (int i = 0; i < 100; i++)
    {
        var embedding = GenerateRandomEmbedding(384); // Typical embedding size
        var doc = new Document<float>($"doc{i}", $"Content {i}", new Dictionary<string, object>
        {
            ["category"] = i % 3 == 0 ? "science" : i % 3 == 1 ? "history" : "art",
            ["year"] = 2020 + (i % 5)
        });
        store1.Add(new VectorDocument<float>(doc, embedding));
    }

    // Step 2: Search
    var queryVector = GenerateRandomEmbedding(384);
    var results1 = store1.GetSimilar(queryVector, 10).ToList();

    Assert.Equal(10, results1.Count);

    // Step 3: Restart
    var store2 = new FileDocumentStore<float>(tempDir);

    // Step 4: Search again
    var results2 = store2.GetSimilar(queryVector, 10).ToList();

    // Assert
    Assert.Equal(100, store2.DocumentCount);
    Assert.Equal(10, results2.Count);

    // Results should be identical (same query, same data)
    for (int i = 0; i < 10; i++)
    {
        Assert.Equal(results1[i].Id, results2[i].Id);
        Assert.Equal(results1[i].RelevanceScore, results2[i].RelevanceScore);
    }

    // Cleanup
    Directory.Delete(tempDir, true);
}

private Vector<float> GenerateRandomEmbedding(int dimension)
{
    var random = new Random();
    var values = new float[dimension];
    for (int i = 0; i < dimension; i++)
        values[i] = (float)random.NextDouble();
    return new Vector<float>(values);
}
```

---

## Common Pitfalls

### 1. File Locking Issues

**Problem:** Multiple processes accessing the same files simultaneously.

**Solution:**

```csharp
// Always use FileShare.Read when opening for reading
using var fileStream = new FileStream(
    _documentsFilePath,
    FileMode.Open,
    FileAccess.Read,
    FileShare.Read  // Allow other readers
);

// For writing, use FileMode.Append or exclusive locks when needed
using var fileStream = new FileStream(
    _documentsFilePath,
    FileMode.Append,
    FileAccess.Write,
    FileShare.None  // Exclusive write access
);
```

### 2. Incorrect Byte Offsets

**Problem:** Reading from wrong file positions due to offset calculation errors.

**Solution:**
- Always serialize document size as the first field
- Test with documents of varying sizes
- Use debug logging to verify offsets

```csharp
// Good: Log offsets during development
var offset = fileStream.Position;
Console.WriteLine($"Writing document {documentId} at offset {offset}");
```

### 3. Type Conversion Errors with Generics

**Problem:** Converting between generic type `T` and concrete types like `double`.

**Solution:**

```csharp
// Use NumOps for conversions
var doubleValue = NumOps.ToDouble(genericValue);  // T -> double
var genericValue = NumOps.FromDouble(doubleValue); // double -> T

// Or use Convert class
var doubleValue = Convert.ToDouble(genericValue);
var genericValue = (T)Convert.ChangeType(doubleValue, typeof(T));
```

### 4. Memory Leaks with File Streams

**Problem:** Not disposing file streams properly.

**Solution:**

```csharp
// Always use 'using' statements
using (var stream = new FileStream(...))
{
    // Work with stream
} // Automatically disposed here, even if exception thrown

// Or use 'using' declaration (C# 8+)
using var stream = new FileStream(...);
// Work with stream
// Disposed at end of scope
```

### 5. Race Conditions in Concurrent Access

**Problem:** Multiple threads modifying index simultaneously.

**Solution:**

```csharp
public class FileDocumentStore<T> : DocumentStoreBase<T>
{
    private readonly object _writeLock = new object();

    protected override void AddCore(VectorDocument<T> vectorDocument)
    {
        lock (_writeLock)
        {
            // Critical section - only one thread at a time
            // Serialize, write to file, update indexes
        }
    }
}
```

### 6. Inverted Index Not Persisted

**Problem:** Forgetting to save index to disk after modifications.

**Solution:**

```csharp
protected override void AddCore(VectorDocument<T> vectorDocument)
{
    // ... add document ...

    _metadataIndex.AddDocument(document.Id, document.Metadata);
    _metadataIndex.Save(); // DON'T FORGET THIS!
}
```

### 7. Vector Dimension Mismatch

**Problem:** Adding vectors with different dimensions.

**Solution:**

```csharp
protected override void AddCore(VectorDocument<T> vectorDocument)
{
    // Validate dimension
    if (_documentCount > 0 && vectorDocument.Embedding.Length != _vectorDimension)
    {
        throw new ArgumentException(
            $"Vector dimension mismatch. Expected {_vectorDimension}, got {vectorDocument.Embedding.Length}");
    }

    // Set dimension from first document
    if (_documentCount == 0)
        _vectorDimension = vectorDocument.Embedding.Length;

    // ... rest of method ...
}
```

---

## Resources

### Understanding Vector Embeddings

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### B-Tree Data Structures

- [B-Tree Visualization](https://www.cs.usfca.edu/~galles/visualization/BTree.html)
- [B-Tree Tutorial](https://www.geeksforgeeks.org/introduction-of-b-tree-2/)

### File I/O Best Practices

- [C# BinaryReader/BinaryWriter](https://docs.microsoft.com/en-us/dotnet/api/system.io.binaryreader)
- [FileStream Class](https://docs.microsoft.com/en-us/dotnet/api/system.io.filestream)

### RAG Systems

- [Retrieval-Augmented Generation Paper](https://arxiv.org/abs/2005.11401)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Database Comparison](https://github.com/erikbern/ann-benchmarks)

### Testing

- [xUnit Documentation](https://xunit.net/)
- [Moq Mocking Framework](https://github.com/moq/moq4)
- [FluentAssertions](https://fluentassertions.com/)

---

## Summary Checklist

Before submitting your PR, ensure:

- [ ] `FileDocumentStore<T>` class created and inherits from `DocumentStoreBase<T>`
- [ ] Constructor creates directory and initializes all file paths
- [ ] `AddCore` serializes documents and appends to file
- [ ] `GetByIdCore` reads from correct file offset
- [ ] `RemoveCore` appends to deletions.log
- [ ] `BTreeIndex` class implemented for fast ID lookups
- [ ] `InvertedIndex` class implemented for metadata filtering
- [ ] Vectors stored in `vectors.bin` with proper serialization
- [ ] `vector_map.log` maps line numbers to document IDs
- [ ] `GetSimilarCore` implements full similarity search with filtering
- [ ] Unit tests cover all CRUD operations
- [ ] Integration tests verify persistence across restarts
- [ ] Code coverage >= 90%
- [ ] All files properly disposed (no memory leaks)
- [ ] Thread-safe for concurrent reads
- [ ] Documentation includes beginner-friendly explanations

Good luck with your implementation! Remember: Start simple, test often, and iterate. Don't try to implement everything perfectly the first time - get it working, then optimize.
