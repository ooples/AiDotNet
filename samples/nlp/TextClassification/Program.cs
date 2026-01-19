using AiDotNet;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Models.Options;

Console.WriteLine("=== AiDotNet Text Classification ===");
Console.WriteLine("News Article Categorization with Multi-Class Classification\n");

// Sample news articles with categories
// Categories: 0=Technology, 1=Sports, 2=Politics, 3=Business
var (articles, labels, categoryNames) = LoadNewsDataset();

Console.WriteLine($"Loaded {articles.Length} news articles across {categoryNames.Length} categories\n");
Console.WriteLine("Categories:");
foreach (var (name, index) in categoryNames.Select((n, i) => (n, i)))
{
    var count = labels.Count(l => (int)l == index);
    Console.WriteLine($"  {index}. {name}: {count} articles");
}
Console.WriteLine();

// Split into train/test sets (80/20)
var random = new Random(42);
var indices = Enumerable.Range(0, articles.Length).OrderBy(_ => random.Next()).ToArray();
var splitIndex = (int)(articles.Length * 0.8);

var trainIndices = indices.Take(splitIndex).ToArray();
var testIndices = indices.Skip(splitIndex).ToArray();

var trainArticles = trainIndices.Select(i => articles[i]).ToArray();
var trainLabels = trainIndices.Select(i => labels[i]).ToArray();
var testArticles = testIndices.Select(i => articles[i]).ToArray();
var testLabels = testIndices.Select(i => labels[i]).ToArray();

Console.WriteLine($"Training set: {trainArticles.Length} articles");
Console.WriteLine($"Test set: {testArticles.Length} articles\n");

// Convert text to bag-of-words features
Console.WriteLine("Preprocessing: Converting text to TF-IDF features...");
var (trainFeatures, testFeatures, vocabulary) = TextToFeatures(trainArticles, testArticles);
Console.WriteLine($"  Vocabulary size: {vocabulary.Count} terms");
Console.WriteLine($"  Feature vector size: {trainFeatures[0].Length}\n");

// Build and train the classifier using Multinomial Naive Bayes
Console.WriteLine("Building Multinomial Naive Bayes classifier...");
Console.WriteLine("  - Smoothing: Laplace (alpha=1.0)");
Console.WriteLine("  - Suitable for: Multi-class text classification\n");

try
{
    // Create classifier with options
    var options = new NaiveBayesOptions<double>
    {
        Alpha = 1.0  // Laplace smoothing
    };
    var classifier = new MultinomialNaiveBayes<double>(options);

    Console.WriteLine("Training classifier...\n");

    // Convert to matrix format for training
    var trainMatrix = ArrayToMatrix(trainFeatures);
    var trainLabelVector = ArrayToVector(trainLabels);

    // Train the model
    classifier.Train(trainMatrix, trainLabelVector);

    // Evaluate on test set
    Console.WriteLine("Evaluation Results:");
    Console.WriteLine(new string('-', 60));

    // Per-category metrics
    var predictions = new double[testLabels.Length];
    var categoryCorrect = new int[categoryNames.Length];
    var categoryTotal = new int[categoryNames.Length];
    var categoryPredicted = new int[categoryNames.Length];

    for (int i = 0; i < testFeatures.Length; i++)
    {
        var testMatrix = ArrayToMatrix(new[] { testFeatures[i] });
        var prediction = classifier.Predict(testMatrix);
        predictions[i] = prediction[0];
        int actual = (int)testLabels[i];
        int predicted = (int)Math.Round(predictions[i]);

        categoryTotal[actual]++;
        if (predicted >= 0 && predicted < categoryNames.Length)
            categoryPredicted[predicted]++;

        if (predicted == actual)
        {
            categoryCorrect[actual]++;
        }
    }

    // Display per-category precision, recall, and F1
    Console.WriteLine("\nPer-Category Metrics:");
    Console.WriteLine($"{"Category",-15} {"Precision",10} {"Recall",10} {"F1-Score",10} {"Support",10}");
    Console.WriteLine(new string('-', 60));

    double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
    int totalSupport = 0;

    for (int i = 0; i < categoryNames.Length; i++)
    {
        double precision = categoryPredicted[i] > 0
            ? (double)categoryCorrect[i] / categoryPredicted[i]
            : 0;
        double recall = categoryTotal[i] > 0
            ? (double)categoryCorrect[i] / categoryTotal[i]
            : 0;
        double f1 = (precision + recall) > 0
            ? 2 * precision * recall / (precision + recall)
            : 0;

        Console.WriteLine($"{categoryNames[i],-15} {precision,10:P1} {recall,10:P1} {f1,10:P1} {categoryTotal[i],10}");

        totalPrecision += precision * categoryTotal[i];
        totalRecall += recall * categoryTotal[i];
        totalF1 += f1 * categoryTotal[i];
        totalSupport += categoryTotal[i];
    }

    Console.WriteLine(new string('-', 60));
    Console.WriteLine($"{"Weighted Avg",-15} {totalPrecision / totalSupport,10:P1} {totalRecall / totalSupport,10:P1} {totalF1 / totalSupport,10:P1} {totalSupport,10}");

    // Overall accuracy
    int totalCorrect = categoryCorrect.Sum();
    double accuracy = (double)totalCorrect / testLabels.Length;
    Console.WriteLine($"\nOverall Accuracy: {accuracy:P2}");

    // Sample predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine(new string('-', 60));

    for (int i = 0; i < Math.Min(5, testArticles.Length); i++)
    {
        int predicted = (int)Math.Round(predictions[i]);
        int actual = (int)testLabels[i];
        string status = predicted == actual ? "[Correct]" : "[Wrong]";
        string predName = predicted >= 0 && predicted < categoryNames.Length ? categoryNames[predicted] : "Unknown";
        string actualName = actual >= 0 && actual < categoryNames.Length ? categoryNames[actual] : "Unknown";

        Console.WriteLine($"\nArticle {i + 1}: \"{testArticles[i][..Math.Min(50, testArticles[i].Length)]}...\"");
        Console.WriteLine($"  Predicted: {predName}");
        Console.WriteLine($"  Actual:    {actualName} {status}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for text classification.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper function to create sample news dataset
static (string[] articles, double[] labels, string[] categories) LoadNewsDataset()
{
    var articles = new List<string>();
    var labels = new List<double>();
    var categories = new[] { "Technology", "Sports", "Politics", "Business" };

    // Technology articles (label 0)
    var techArticles = new[]
    {
        "Apple announces new iPhone with revolutionary AI features and enhanced camera system",
        "Google releases major update to Android operating system with improved security",
        "Microsoft unveils new cloud computing services for enterprise customers",
        "Tesla introduces self-driving software update for electric vehicles",
        "Amazon Web Services expands data centers across Asia Pacific region",
        "Meta develops new virtual reality headset for metaverse applications",
        "NVIDIA launches powerful graphics processing units for AI training",
        "OpenAI releases GPT-5 with enhanced reasoning capabilities",
        "SpaceX successfully deploys new satellite internet constellation",
        "Intel announces breakthrough in quantum computing technology",
        "Samsung unveils foldable smartphone with improved durability",
        "Cybersecurity firm discovers major vulnerability in cloud software",
        "Startup develops revolutionary battery technology for electric cars",
        "Tech giants invest billions in artificial intelligence research",
        "New programming language gains popularity among developers"
    };
    foreach (var article in techArticles)
    {
        articles.Add(article);
        labels.Add(0);
    }

    // Sports articles (label 1)
    var sportsArticles = new[]
    {
        "Lakers defeat Celtics in thrilling NBA finals game seven overtime",
        "Manchester United signs star striker for record transfer fee",
        "Serena Williams announces comeback to professional tennis tour",
        "Olympics committee selects host city for upcoming summer games",
        "NFL quarterback breaks all-time passing yards record this season",
        "World Cup final draws record television audience worldwide",
        "Golf champion wins fourth major tournament of the season",
        "Formula One team unveils new car design for next racing season",
        "Baseball playoffs feature unexpected underdog team championship run",
        "Swimming star sets new world record at national championships",
        "Boxing match generates millions in pay-per-view revenue",
        "Soccer club announces new stadium construction project downtown",
        "Hockey team trades star player to division rival",
        "Tennis tournament introduces new electronic line calling system",
        "Marathon runner completes race in record breaking time"
    };
    foreach (var article in sportsArticles)
    {
        articles.Add(article);
        labels.Add(1);
    }

    // Politics articles (label 2)
    var politicsArticles = new[]
    {
        "President signs landmark climate change legislation into law",
        "Congress debates new infrastructure spending bill this week",
        "Supreme Court rules on controversial voting rights case today",
        "Senate confirms new cabinet secretary after lengthy hearings",
        "Governor announces emergency measures for state budget crisis",
        "International summit addresses global security concerns",
        "Political parties prepare for upcoming midterm election campaigns",
        "United Nations passes resolution on humanitarian aid distribution",
        "Mayor proposes new housing policy for affordable development",
        "Parliament votes on immigration reform legislation package",
        "Diplomatic tensions rise between neighboring countries over trade",
        "Election results show shift in voter demographics nationwide",
        "Government launches investigation into corruption allegations",
        "Policy makers debate healthcare reform proposals extensively",
        "Constitutional amendment proposed for campaign finance reform"
    };
    foreach (var article in politicsArticles)
    {
        articles.Add(article);
        labels.Add(2);
    }

    // Business articles (label 3)
    var businessArticles = new[]
    {
        "Stock market reaches record highs amid strong corporate earnings",
        "Federal Reserve announces interest rate decision affecting markets",
        "Major retailer reports quarterly revenue exceeding analyst expectations",
        "Merger between pharmaceutical giants creates industry leader",
        "Oil prices surge following supply disruption in Middle East",
        "Bank introduces new digital payment platform for consumers",
        "Startup raises billion dollar funding round from venture capital",
        "Manufacturing sector shows signs of recovery after slowdown",
        "Real estate market cools as mortgage rates increase steadily",
        "Cryptocurrency exchange files for regulatory approval status",
        "Airline industry reports strong passenger demand this quarter",
        "Consumer spending data indicates economic growth momentum",
        "Trade deficit narrows as exports increase substantially",
        "Hedge fund manager predicts market correction coming soon",
        "Supply chain disruptions affect global shipping operations"
    };
    foreach (var article in businessArticles)
    {
        articles.Add(article);
        labels.Add(3);
    }

    return (articles.ToArray(), labels.ToArray(), categories);
}

// Convert text to TF-IDF features
static (double[][] trainFeatures, double[][] testFeatures, Dictionary<string, int> vocabulary)
    TextToFeatures(string[] trainTexts, string[] testTexts)
{
    // Build vocabulary from training data
    var vocabulary = new Dictionary<string, int>();
    var stopWords = new HashSet<string>
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "this", "that", "these", "those",
        "it", "its", "their", "they", "them", "he", "she", "his", "her", "we", "our"
    };

    // Tokenize and build vocabulary
    foreach (var text in trainTexts)
    {
        var tokens = Tokenize(text, stopWords);
        foreach (var token in tokens)
        {
            if (!vocabulary.ContainsKey(token))
            {
                vocabulary[token] = vocabulary.Count;
            }
        }
    }

    // Compute document frequencies for IDF
    var documentFrequencies = new int[vocabulary.Count];
    foreach (var text in trainTexts)
    {
        var tokens = Tokenize(text, stopWords).Distinct();
        foreach (var token in tokens)
        {
            if (vocabulary.TryGetValue(token, out int index))
            {
                documentFrequencies[index]++;
            }
        }
    }

    // Compute IDF values
    var idf = new double[vocabulary.Count];
    for (int i = 0; i < vocabulary.Count; i++)
    {
        idf[i] = Math.Log((double)trainTexts.Length / (1 + documentFrequencies[i]));
    }

    // Convert to TF-IDF features
    double[][] trainFeatures = TextsToTfIdf(trainTexts, vocabulary, idf, stopWords);
    double[][] testFeatures = TextsToTfIdf(testTexts, vocabulary, idf, stopWords);

    return (trainFeatures, testFeatures, vocabulary);
}

static double[][] TextsToTfIdf(string[] texts, Dictionary<string, int> vocabulary,
    double[] idf, HashSet<string> stopWords)
{
    var features = new double[texts.Length][];

    for (int i = 0; i < texts.Length; i++)
    {
        var tokens = Tokenize(texts[i], stopWords);
        var termFrequencies = new Dictionary<string, int>();

        foreach (var token in tokens)
        {
            if (vocabulary.ContainsKey(token))
            {
                termFrequencies[token] = termFrequencies.GetValueOrDefault(token, 0) + 1;
            }
        }

        features[i] = new double[vocabulary.Count];
        foreach (var (term, tf) in termFrequencies)
        {
            int index = vocabulary[term];
            features[i][index] = tf * idf[index];
        }

        // L2 normalize the feature vector
        double norm = Math.Sqrt(features[i].Sum(x => x * x));
        if (norm > 0)
        {
            for (int j = 0; j < features[i].Length; j++)
            {
                features[i][j] /= norm;
            }
        }
    }

    return features;
}

static List<string> Tokenize(string text, HashSet<string> stopWords)
{
    return text.ToLowerInvariant()
        .Split(new[] { ' ', ',', '.', '!', '?', ';', ':', '"', '\'', '-', '(', ')' },
            StringSplitOptions.RemoveEmptyEntries)
        .Where(t => t.Length > 2 && !stopWords.Contains(t))
        .ToList();
}

// Convert 2D array to Matrix format
static AiDotNet.Tensors.LinearAlgebra.Matrix<double> ArrayToMatrix(double[][] array)
{
    int rows = array.Length;
    int cols = array[0].Length;
    var matrix = new AiDotNet.Tensors.LinearAlgebra.Matrix<double>(rows, cols);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i, j] = array[i][j];
        }
    }
    return matrix;
}

// Convert 1D array to Vector format
static AiDotNet.Tensors.LinearAlgebra.Vector<double> ArrayToVector(double[] array)
{
    return new AiDotNet.Tensors.LinearAlgebra.Vector<double>(array);
}
