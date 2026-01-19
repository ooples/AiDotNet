using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Graph;
using Vector = AiDotNet.Tensors.LinearAlgebra.Vector<float>;

Console.WriteLine("=== AiDotNet Graph RAG ===");
Console.WriteLine("Knowledge Graph-Enhanced Retrieval-Augmented Generation\n");

// Create embedding model for vector representations
var embeddingModel = new StubEmbeddingModel<float>(embeddingDimension: 384);

// Create knowledge graph with in-memory storage
var graphStore = new MemoryGraphStore<float>();
var knowledgeGraph = new KnowledgeGraph<float>(graphStore);

Console.WriteLine("Building Knowledge Graph...\n");

// Build a knowledge graph about AI/ML concepts
BuildAIKnowledgeGraph(knowledgeGraph, embeddingModel);

Console.WriteLine($"Knowledge Graph Statistics:");
Console.WriteLine($"  - Nodes: {knowledgeGraph.NodeCount}");
Console.WriteLine($"  - Edges: {knowledgeGraph.EdgeCount}");
Console.WriteLine();

// Display the graph structure
Console.WriteLine("Graph Structure:");
Console.WriteLine(new string('-', 60));

var nodesByLabel = knowledgeGraph.GetAllNodes()
    .GroupBy(n => n.Label)
    .OrderBy(g => g.Key);

foreach (var group in nodesByLabel)
{
    Console.WriteLine($"\n{group.Key}:");
    foreach (var node in group)
    {
        var name = node.GetProperty<string>("name") ?? node.Id;
        Console.WriteLine($"  - {name}");

        // Show relationships
        var edges = knowledgeGraph.GetOutgoingEdges(node.Id).ToList();
        foreach (var edge in edges.Take(3))
        {
            var targetNode = knowledgeGraph.GetNode(edge.TargetId);
            var targetName = targetNode?.GetProperty<string>("name") ?? edge.TargetId;
            Console.WriteLine($"      --[{edge.RelationType}]--> {targetName}");
        }
        if (edges.Count > 3)
        {
            Console.WriteLine($"      ... and {edges.Count - 3} more relationships");
        }
    }
}

// Entity and Relation Extraction Demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Entity and Relation Extraction");
Console.WriteLine(new string('=', 60));

var sampleTexts = new[]
{
    "Geoffrey Hinton, often called the godfather of AI, developed the backpropagation algorithm at the University of Toronto.",
    "OpenAI released GPT-4, a large language model that powers ChatGPT for conversational AI applications.",
    "Google DeepMind created AlphaFold which solved the protein folding problem using deep learning."
};

Console.WriteLine("\nExtracting entities and relations from text:\n");

foreach (var text in sampleTexts)
{
    Console.WriteLine($"Text: \"{text}\"");
    Console.WriteLine("\nExtracted:");

    var (entities, relations) = ExtractEntitiesAndRelations(text);

    Console.WriteLine("  Entities:");
    foreach (var entity in entities)
    {
        Console.WriteLine($"    - {entity.Type}: {entity.Name}");
    }

    Console.WriteLine("  Relations:");
    foreach (var relation in relations)
    {
        Console.WriteLine($"    - {relation.Subject} --[{relation.Predicate}]--> {relation.Object}");
    }
    Console.WriteLine();
}

// Graph-Based Retrieval Demonstration
Console.WriteLine(new string('=', 60));
Console.WriteLine("Graph-Based Retrieval");
Console.WriteLine(new string('=', 60));

var queries = new[]
{
    "What techniques are used in deep learning?",
    "How is natural language processing related to machine learning?",
    "What are the applications of neural networks?"
};

foreach (var query in queries)
{
    Console.WriteLine($"\nQuery: \"{query}\"");
    Console.WriteLine(new string('-', 50));

    // Step 1: Find relevant entities in the query
    var queryEntities = FindQueryEntities(query, knowledgeGraph);
    Console.WriteLine($"\nIdentified entities in query:");
    foreach (var entity in queryEntities)
    {
        Console.WriteLine($"  - {entity.Label}: {entity.GetProperty<string>("name")}");
    }

    // Step 2: Expand context using graph traversal
    Console.WriteLine($"\nGraph traversal (1-hop neighborhood):");
    var context = new HashSet<GraphNode<float>>();

    foreach (var entity in queryEntities)
    {
        // Get direct neighbors
        var neighbors = knowledgeGraph.GetNeighbors(entity.Id).ToList();
        foreach (var neighbor in neighbors)
        {
            context.Add(neighbor);
        }

        // Get incoming connections
        var incoming = knowledgeGraph.GetIncomingEdges(entity.Id);
        foreach (var edge in incoming)
        {
            var sourceNode = knowledgeGraph.GetNode(edge.SourceId);
            if (sourceNode != null)
            {
                context.Add(sourceNode);
            }
        }
    }

    Console.WriteLine($"  Retrieved {context.Count} related entities:");
    foreach (var node in context.Take(5))
    {
        var name = node.GetProperty<string>("name") ?? node.Id;
        Console.WriteLine($"    - [{node.Label}] {name}");
    }
    if (context.Count > 5)
    {
        Console.WriteLine($"    ... and {context.Count - 5} more");
    }

    // Step 3: Generate contextual answer
    Console.WriteLine($"\nGenerated context for LLM:");
    var contextText = GenerateContextFromGraph(queryEntities, context, knowledgeGraph);
    Console.WriteLine($"  {contextText}");
}

// Path Finding Demonstration
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Path Finding in Knowledge Graph");
Console.WriteLine(new string('=', 60));

var pathQueries = new[]
{
    ("deep_learning", "natural_language_processing", "How is deep learning connected to NLP?"),
    ("neural_networks", "computer_vision", "What's the relationship between neural networks and computer vision?"),
    ("machine_learning", "transformers", "How do transformers relate to machine learning?")
};

foreach (var (start, end, question) in pathQueries)
{
    Console.WriteLine($"\nQuestion: {question}");

    var path = knowledgeGraph.FindShortestPath(start, end);

    if (path.Count > 0)
    {
        Console.WriteLine($"Path found ({path.Count} hops):");
        for (int i = 0; i < path.Count; i++)
        {
            var node = knowledgeGraph.GetNode(path[i]);
            var name = node?.GetProperty<string>("name") ?? path[i];

            if (i < path.Count - 1)
            {
                var edges = knowledgeGraph.GetOutgoingEdges(path[i])
                    .Where(e => e.TargetId == path[i + 1])
                    .ToList();
                var relation = edges.FirstOrDefault()?.RelationType ?? "RELATED_TO";
                Console.WriteLine($"  {name} --[{relation}]-->");
            }
            else
            {
                Console.WriteLine($"  {name}");
            }
        }
    }
    else
    {
        Console.WriteLine("  No path found");
    }
}

// Hybrid Retrieval (Vector + Graph)
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("Hybrid Retrieval: Vector Similarity + Graph Context");
Console.WriteLine(new string('=', 60));

var hybridQuery = "What are modern techniques for understanding language?";
Console.WriteLine($"\nQuery: \"{hybridQuery}\"\n");

// Step 1: Vector similarity search
var queryEmbedding = embeddingModel.Embed(hybridQuery);
var vectorResults = new List<(GraphNode<float> node, double similarity)>();

foreach (var node in knowledgeGraph.GetAllNodes())
{
    if (node.Embedding != null)
    {
        double sim = CosineSimilarity(queryEmbedding, node.Embedding);
        vectorResults.Add((node, sim));
    }
}

var topVectorResults = vectorResults.OrderByDescending(r => r.similarity).Take(3).ToList();

Console.WriteLine("1. Vector Similarity Results:");
foreach (var (node, sim) in topVectorResults)
{
    var name = node.GetProperty<string>("name") ?? node.Id;
    Console.WriteLine($"   [{sim:F4}] {name}");
}

// Step 2: Expand with graph context
Console.WriteLine("\n2. Graph-Expanded Context:");
var expandedContext = new HashSet<string>();

foreach (var (node, _) in topVectorResults)
{
    expandedContext.Add(node.Id);

    // Add 1-hop neighbors
    foreach (var neighbor in knowledgeGraph.GetNeighbors(node.Id))
    {
        expandedContext.Add(neighbor.Id);
    }
}

Console.WriteLine($"   Expanded from {topVectorResults.Count} to {expandedContext.Count} relevant entities");

// Step 3: Rank combined results
Console.WriteLine("\n3. Final Ranked Results (Hybrid):");
var finalResults = expandedContext
    .Select(id => knowledgeGraph.GetNode(id))
    .Where(n => n != null)
    .Select(n =>
    {
        double vectorScore = n!.Embedding != null
            ? CosineSimilarity(queryEmbedding, n.Embedding)
            : 0;
        double graphScore = topVectorResults.Any(r => r.node.Id == n.Id) ? 0.3 : 0.1;
        return (node: n, score: vectorScore + graphScore);
    })
    .OrderByDescending(r => r.score)
    .Take(5);

foreach (var (node, score) in finalResults)
{
    var name = node.GetProperty<string>("name") ?? node.Id;
    var desc = node.GetProperty<string>("description") ?? "";
    Console.WriteLine($"   [{score:F4}] {name}");
    if (!string.IsNullOrEmpty(desc))
    {
        Console.WriteLine($"            {desc[..Math.Min(50, desc.Length)]}...");
    }
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions

static void BuildAIKnowledgeGraph(KnowledgeGraph<float> graph, StubEmbeddingModel<float> embeddingModel)
{
    // Create concept nodes
    var concepts = new[]
    {
        ("machine_learning", "CONCEPT", "Machine Learning", "A subset of AI that enables systems to learn from data"),
        ("deep_learning", "CONCEPT", "Deep Learning", "Machine learning using neural networks with multiple layers"),
        ("neural_networks", "CONCEPT", "Neural Networks", "Computing systems inspired by biological neural networks"),
        ("natural_language_processing", "CONCEPT", "Natural Language Processing", "AI for understanding human language"),
        ("computer_vision", "CONCEPT", "Computer Vision", "AI for understanding visual information"),
        ("reinforcement_learning", "CONCEPT", "Reinforcement Learning", "Learning through trial and reward"),
        ("supervised_learning", "CONCEPT", "Supervised Learning", "Learning from labeled training data"),
        ("unsupervised_learning", "CONCEPT", "Unsupervised Learning", "Learning patterns without labels"),
        ("transformers", "ARCHITECTURE", "Transformers", "Neural architecture using self-attention mechanisms"),
        ("attention", "TECHNIQUE", "Attention Mechanism", "Technique to focus on relevant parts of input"),
        ("cnn", "ARCHITECTURE", "Convolutional Neural Networks", "Neural networks for grid-like data"),
        ("rnn", "ARCHITECTURE", "Recurrent Neural Networks", "Neural networks for sequential data"),
        ("lstm", "ARCHITECTURE", "LSTM", "Long Short-Term Memory networks for sequences"),
        ("gpt", "MODEL", "GPT", "Generative Pre-trained Transformer models"),
        ("bert", "MODEL", "BERT", "Bidirectional Encoder Representations from Transformers"),
        ("backpropagation", "TECHNIQUE", "Backpropagation", "Algorithm for training neural networks"),
        ("gradient_descent", "TECHNIQUE", "Gradient Descent", "Optimization algorithm for minimizing loss")
    };

    foreach (var (id, label, name, description) in concepts)
    {
        var node = new GraphNode<float>(id, label);
        node.SetProperty("name", name);
        node.SetProperty("description", description);
        node.Embedding = embeddingModel.Embed($"{name}. {description}");
        graph.AddNode(node);
    }

    // Create relationships
    var relationships = new[]
    {
        ("deep_learning", "SUBSET_OF", "machine_learning"),
        ("neural_networks", "USED_IN", "deep_learning"),
        ("natural_language_processing", "USES", "deep_learning"),
        ("computer_vision", "USES", "deep_learning"),
        ("reinforcement_learning", "SUBSET_OF", "machine_learning"),
        ("supervised_learning", "SUBSET_OF", "machine_learning"),
        ("unsupervised_learning", "SUBSET_OF", "machine_learning"),
        ("transformers", "USED_IN", "natural_language_processing"),
        ("attention", "COMPONENT_OF", "transformers"),
        ("cnn", "USED_IN", "computer_vision"),
        ("rnn", "USED_IN", "natural_language_processing"),
        ("lstm", "TYPE_OF", "rnn"),
        ("gpt", "BASED_ON", "transformers"),
        ("bert", "BASED_ON", "transformers"),
        ("backpropagation", "TRAINS", "neural_networks"),
        ("gradient_descent", "USED_IN", "backpropagation"),
        ("natural_language_processing", "SUBSET_OF", "machine_learning"),
        ("computer_vision", "SUBSET_OF", "machine_learning"),
        ("neural_networks", "TYPE_OF", "machine_learning"),
        ("transformers", "TYPE_OF", "neural_networks"),
        ("cnn", "TYPE_OF", "neural_networks"),
        ("rnn", "TYPE_OF", "neural_networks")
    };

    foreach (var (source, relation, target) in relationships)
    {
        var edge = new GraphEdge<float>(source, target, relation);
        graph.AddEdge(edge);
    }

    Console.WriteLine($"  Added {concepts.Length} concept nodes");
    Console.WriteLine($"  Added {relationships.Length} relationships");
}

static (List<Entity> entities, List<Relation> relations) ExtractEntitiesAndRelations(string text)
{
    var entities = new List<Entity>();
    var relations = new List<Relation>();

    var personPatterns = new[] { "Geoffrey Hinton", "Elon Musk", "Sam Altman" };
    var orgPatterns = new[] { "OpenAI", "Google", "DeepMind", "University of Toronto", "Google DeepMind" };
    var techPatterns = new[] { "GPT-4", "ChatGPT", "AlphaFold", "backpropagation", "deep learning" };

    foreach (var pattern in personPatterns)
    {
        if (text.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            entities.Add(new Entity(pattern, "PERSON"));
    }

    foreach (var pattern in orgPatterns)
    {
        if (text.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            entities.Add(new Entity(pattern, "ORGANIZATION"));
    }

    foreach (var pattern in techPatterns)
    {
        if (text.Contains(pattern, StringComparison.OrdinalIgnoreCase))
            entities.Add(new Entity(pattern, "TECHNOLOGY"));
    }

    if (text.Contains("developed", StringComparison.OrdinalIgnoreCase))
    {
        var person = entities.FirstOrDefault(e => e.Type == "PERSON");
        var tech = entities.FirstOrDefault(e => e.Type == "TECHNOLOGY");
        if (person != null && tech != null)
            relations.Add(new Relation(person.Name, "DEVELOPED", tech.Name));
    }

    if (text.Contains("released", StringComparison.OrdinalIgnoreCase))
    {
        var org = entities.FirstOrDefault(e => e.Type == "ORGANIZATION");
        var tech = entities.FirstOrDefault(e => e.Type == "TECHNOLOGY");
        if (org != null && tech != null)
            relations.Add(new Relation(org.Name, "RELEASED", tech.Name));
    }

    if (text.Contains("created", StringComparison.OrdinalIgnoreCase))
    {
        var org = entities.FirstOrDefault(e => e.Type == "ORGANIZATION");
        var tech = entities.FirstOrDefault(e => e.Type == "TECHNOLOGY");
        if (org != null && tech != null)
            relations.Add(new Relation(org.Name, "CREATED", tech.Name));
    }

    if (text.Contains("at the", StringComparison.OrdinalIgnoreCase) || text.Contains("at ", StringComparison.OrdinalIgnoreCase))
    {
        var person = entities.FirstOrDefault(e => e.Type == "PERSON");
        var org = entities.FirstOrDefault(e => e.Type == "ORGANIZATION");
        if (person != null && org != null)
            relations.Add(new Relation(person.Name, "AFFILIATED_WITH", org.Name));
    }

    return (entities, relations);
}

static List<GraphNode<float>> FindQueryEntities(string query, KnowledgeGraph<float> graph)
{
    var results = new List<GraphNode<float>>();
    var queryLower = query.ToLowerInvariant();

    foreach (var node in graph.GetAllNodes())
    {
        var name = node.GetProperty<string>("name")?.ToLowerInvariant() ?? "";
        var nameWords = name.Split(' ');
        foreach (var word in nameWords)
        {
            if (word.Length > 3 && queryLower.Contains(word))
            {
                results.Add(node);
                break;
            }
        }
    }

    return results.Distinct().ToList();
}

static string GenerateContextFromGraph(
    List<GraphNode<float>> queryEntities,
    HashSet<GraphNode<float>> context,
    KnowledgeGraph<float> graph)
{
    var contextParts = new List<string>();

    foreach (var entity in queryEntities)
    {
        var name = entity.GetProperty<string>("name");
        var desc = entity.GetProperty<string>("description");

        if (!string.IsNullOrEmpty(name) && !string.IsNullOrEmpty(desc))
            contextParts.Add($"{name}: {desc}");

        var edges = graph.GetOutgoingEdges(entity.Id).Take(3);
        foreach (var edge in edges)
        {
            var target = graph.GetNode(edge.TargetId);
            var targetName = target?.GetProperty<string>("name") ?? edge.TargetId;
            contextParts.Add($"{name} {edge.RelationType.ToLowerInvariant().Replace("_", " ")} {targetName}");
        }
    }

    return string.Join(". ", contextParts.Take(5));
}

static double CosineSimilarity(Vector a, Vector b)
{
    if (a.Length != b.Length) return 0;

    double dot = 0, normA = 0, normB = 0;
    for (int i = 0; i < a.Length; i++)
    {
        double va = Convert.ToDouble(a[i]);
        double vb = Convert.ToDouble(b[i]);
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }

    double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
    return denom > 0 ? dot / denom : 0;
}

record Entity(string Name, string Type);
record Relation(string Subject, string Predicate, string Object);
