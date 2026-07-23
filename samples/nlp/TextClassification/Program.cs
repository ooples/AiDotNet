// AiDotNet — Text Classification (News Categorization)
//
// Multi-class text classification entirely through the AiModelBuilder facade. A TF-IDF
// vectorizer turns raw article text into features (ConfigureTextVectorizer +
// DataLoaders.FromTextDocuments), a Multinomial Naive Bayes model is trained, and new
// articles are categorized straight from text via result.PredictText — no manual
// tokenizing or feature engineering. Metrics come off result.GetDataSetStats.

using AiDotNet;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Text Classification ===");
Console.WriteLine("News Article Categorization with Multi-Class Classification\n");

// Categories: 0=Technology, 1=Sports, 2=Politics, 3=Business
var (articles, labels, categoryNames) = LoadNewsDataset();

Console.WriteLine($"Loaded {articles.Length} news articles across {categoryNames.Length} categories\n");
Console.WriteLine("Categories:");
foreach (var (name, index) in categoryNames.Select((n, i) => (n, i)))
    Console.WriteLine($"  {index}. {name}: {labels.Count(l => (int)l == index)} articles");
Console.WriteLine();

// Split into train/test sets (80/20).
var random = new Random(42);
var indices = Enumerable.Range(0, articles.Length).OrderBy(_ => random.Next()).ToArray();
var splitIndex = (int)(articles.Length * 0.8);

var trainArticles = indices.Take(splitIndex).Select(i => articles[i]).ToArray();
var trainLabels = indices.Take(splitIndex).Select(i => labels[i]).ToArray();
var testArticles = indices.Skip(splitIndex).Select(i => articles[i]).ToArray();
var testLabels = indices.Skip(splitIndex).Select(i => labels[i]).ToArray();

Console.WriteLine($"Training set: {trainArticles.Length} articles");
Console.WriteLine($"Test set: {testArticles.Length} articles\n");

// Build and train through the facade. The TF-IDF vectorizer converts raw article text
// into numeric features; FromTextDocuments fits it on the training articles, and
// ConfigureTextVectorizer hands the fitted vectorizer to the result for PredictText.
Console.WriteLine("Building Multinomial Naive Bayes classifier through AiModelBuilder...\n");

var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultinomialNaiveBayes<double>(new NaiveBayesOptions<double> { Alpha = 1.0 }))
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(trainArticles, trainLabels, vectorizer))
    .BuildAsync();

Console.WriteLine("Training complete.\n");

// Evaluate on the held-out test set — vectorize the test text, then read rich metrics
// off the facade (ErrorStats auto-selects the classification measures).
var testFeatures = vectorizer.Transform(testArticles);
var testStats = result.GetDataSetStats(testFeatures, new Vector<double>(testLabels));

Console.WriteLine("Evaluation Results (held-out test set):");
Console.WriteLine(new string('-', 60));
Console.WriteLine($"  Accuracy:  {testStats.ErrorStats.Accuracy:P2}");
Console.WriteLine($"  Precision: {testStats.ErrorStats.Precision:P2}");
Console.WriteLine($"  Recall:    {testStats.ErrorStats.Recall:P2}");
Console.WriteLine($"  F1 Score:  {testStats.ErrorStats.F1Score:P2}");

// Categorize new articles straight from raw text — result.PredictText applies the same
// fitted vectorizer for you. No manual feature engineering anywhere in this sample.
Console.WriteLine("\nSample Predictions (result.PredictText):");
Console.WriteLine(new string('-', 60));
for (int i = 0; i < Math.Min(5, testArticles.Length); i++)
{
    var prediction = result.PredictText(new[] { testArticles[i] });
    int predicted = (int)Math.Round(prediction[0]);
    int actual = (int)testLabels[i];
    string status = predicted == actual ? "[Correct]" : "[Wrong]";
    string predName = predicted >= 0 && predicted < categoryNames.Length ? categoryNames[predicted] : "Unknown";
    string actualName = actual >= 0 && actual < categoryNames.Length ? categoryNames[actual] : "Unknown";

    Console.WriteLine($"\nArticle: \"{testArticles[i][..Math.Min(50, testArticles[i].Length)]}...\"");
    Console.WriteLine($"  Predicted: {predName}");
    Console.WriteLine($"  Actual:    {actualName} {status}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Sample news dataset: 15 articles per category.
static (string[] articles, double[] labels, string[] categories) LoadNewsDataset()
{
    var articles = new List<string>();
    var labels = new List<double>();
    var categories = new[] { "Technology", "Sports", "Politics", "Business" };

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
    foreach (var article in techArticles) { articles.Add(article); labels.Add(0); }

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
    foreach (var article in sportsArticles) { articles.Add(article); labels.Add(1); }

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
    foreach (var article in politicsArticles) { articles.Add(article); labels.Add(2); }

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
    foreach (var article in businessArticles) { articles.Add(article); labels.Add(3); }

    return (articles.ToArray(), labels.ToArray(), categories);
}
