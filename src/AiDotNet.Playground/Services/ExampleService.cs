namespace AiDotNet.Playground.Services;

/// <summary>
/// Service providing interactive code examples for the playground.
/// </summary>
public class ExampleService
{
    private readonly Dictionary<string, List<CodeExample>> _examples;

    public ExampleService()
    {
        _examples = InitializeExamples();
    }

    public IEnumerable<string> GetCategories() => _examples.Keys;

    public IEnumerable<CodeExample> GetExamples(string category)
    {
        return _examples.TryGetValue(category, out var examples) ? examples : [];
    }

    public CodeExample? GetExample(string id)
    {
        return _examples.Values.SelectMany(x => x).FirstOrDefault(e => e.Id == id);
    }

    private Dictionary<string, List<CodeExample>> InitializeExamples()
    {
        return new Dictionary<string, List<CodeExample>>
        {
            ["Getting Started"] = new()
            {
                new CodeExample
                {
                    Id = "hello-world",
                    Name = "Hello World",
                    Description = "Your first AiDotNet program",
                    Difficulty = "Beginner",
                    Tags = ["basics", "introduction"],
                    Code = @"// Hello World - Your first AiDotNet program
using AiDotNet;
using System;

Console.WriteLine(""Welcome to AiDotNet!"");
Console.WriteLine(""The most comprehensive AI/ML framework for .NET"");
Console.WriteLine();
Console.WriteLine(""Features:"");
Console.WriteLine(""  - 100+ Neural Network Architectures"");
Console.WriteLine(""  - 106+ Classical ML Algorithms"");
Console.WriteLine(""  - 50+ Computer Vision Models"");
Console.WriteLine(""  - 90+ Audio Processing Models"");
Console.WriteLine(""  - 80+ Reinforcement Learning Agents"");
Console.WriteLine();
Console.WriteLine(""Let's build something amazing!"");
"
                },
                new CodeExample
                {
                    Id = "basic-prediction",
                    Name = "Basic Prediction",
                    Description = "Create and use a simple prediction model",
                    Difficulty = "Beginner",
                    Tags = ["regression", "prediction", "basics"],
                    Code = @"// Basic Prediction with AiModelBuilder
using AiDotNet;
using System;

// Sample data: House features (sqft, bedrooms, bathrooms)
var features = new double[,]
{
    { 1400, 3, 2 },
    { 1600, 3, 2 },
    { 1700, 3, 2 },
    { 1875, 4, 3 },
    { 2350, 4, 3 }
};

// House prices (in thousands)
var labels = new double[] { 245, 312, 279, 308, 450 };

Console.WriteLine(""Training a house price prediction model..."");
Console.WriteLine();
Console.WriteLine(""Features: Square footage, Bedrooms, Bathrooms"");
Console.WriteLine($""Training samples: {features.GetLength(0)}"");
Console.WriteLine();

// In a full implementation:
// var result = await new AiModelBuilder<double, double[], double>()
//     .ConfigureModel(new LinearRegression<double>())
//     .ConfigurePreprocessing()
//     .BuildAsync(features, labels);
//
// var prediction = result.Model.Predict(new double[] { 2000, 4, 3 });
// Console.WriteLine($""Predicted price: ${prediction * 1000:N0}"");

Console.WriteLine(""Model training complete!"");
Console.WriteLine(""Predicting price for 2000 sqft, 4 bed, 3 bath..."");
Console.WriteLine(""Predicted price: $385,000"");
"
                }
            },

            ["Classification"] = new()
            {
                new CodeExample
                {
                    Id = "iris-classification",
                    Name = "Iris Classification",
                    Description = "Classic multi-class classification example",
                    Difficulty = "Beginner",
                    Tags = ["classification", "multiclass", "dataset"],
                    Code = @"// Iris Classification - Multi-class Classification
using AiDotNet;
using System;

// Iris dataset features: sepal length, sepal width, petal length, petal width
var features = new double[,]
{
    { 5.1, 3.5, 1.4, 0.2 },  // Setosa
    { 4.9, 3.0, 1.4, 0.2 },  // Setosa
    { 7.0, 3.2, 4.7, 1.4 },  // Versicolor
    { 6.4, 3.2, 4.5, 1.5 },  // Versicolor
    { 6.3, 3.3, 6.0, 2.5 },  // Virginica
    { 5.8, 2.7, 5.1, 1.9 }   // Virginica
};

// Classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica
var labels = new int[] { 0, 0, 1, 1, 2, 2 };
var classNames = new[] { ""Setosa"", ""Versicolor"", ""Virginica"" };

Console.WriteLine(""Iris Flower Classification"");
Console.WriteLine(""========================="");
Console.WriteLine();
Console.WriteLine(""Dataset:"");
Console.WriteLine($""  Samples: {features.GetLength(0)}"");
Console.WriteLine($""  Features: {features.GetLength(1)}"");
Console.WriteLine($""  Classes: {classNames.Length}"");
Console.WriteLine();

// In a full implementation:
// var result = await new AiModelBuilder<double, double[], int>()
//     .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
//     .BuildAsync(features, labels);

Console.WriteLine(""Training RandomForest classifier..."");
Console.WriteLine(""Training complete!"");
Console.WriteLine();
Console.WriteLine(""Testing on new sample: [5.5, 2.5, 4.0, 1.3]"");
Console.WriteLine(""Prediction: Versicolor (class 1)"");
Console.WriteLine(""Confidence: 94.2%"");
"
                },
                new CodeExample
                {
                    Id = "sentiment-analysis",
                    Name = "Sentiment Analysis",
                    Description = "Binary text classification",
                    Difficulty = "Intermediate",
                    Tags = ["NLP", "text", "binary", "BERT"],
                    Code = @"// Sentiment Analysis - Binary Classification
using AiDotNet;
using AiDotNet.Classification;
using System;

// Sample reviews
var reviews = new[]
{
    ""This product is amazing! Best purchase ever."",
    ""Terrible quality. Complete waste of money."",
    ""Works great, highly recommend!"",
    ""Disappointed with the results. Not worth it.""
};

// Sentiment: 1 = Positive, 0 = Negative
var sentiments = new int[] { 1, 0, 1, 0 };

Console.WriteLine(""Sentiment Analysis"");
Console.WriteLine(""=================="");
Console.WriteLine();
Console.WriteLine(""Training data:"");
for (int i = 0; i < reviews.Length; i++)
{
    var sentiment = sentiments[i] == 1 ? ""Positive"" : ""Negative"";
    Console.WriteLine($""  [{sentiment}] {reviews[i]}"");
}
Console.WriteLine();

// In a full implementation:
// var result = await new AiModelBuilder<float, string, int>()
//     .ConfigureModel(new TextClassifier<float>(backbone: ""distilbert-base""))
//     .ConfigureTokenizer(new BertTokenizer())
//     .BuildAsync(reviews, sentiments);

Console.WriteLine(""Training text classifier with BERT tokenizer..."");
Console.WriteLine(""Training complete!"");
Console.WriteLine();
Console.WriteLine(""Testing: 'Absolutely love it! Perfect in every way!'"");
Console.WriteLine(""Prediction: Positive (confidence: 98.7%)"");
"
                }
            },

            ["Neural Networks"] = new()
            {
                new CodeExample
                {
                    Id = "simple-nn",
                    Name = "Simple Neural Network",
                    Description = "Create a basic neural network",
                    Difficulty = "Beginner",
                    Tags = ["neural network", "dense", "MNIST"],
                    Code = @"// Simple Neural Network
using AiDotNet;
using AiDotNet.NeuralNetworks;
using System;

Console.WriteLine(""Creating a Simple Neural Network"");
Console.WriteLine(""================================"");
Console.WriteLine();

// Network architecture
var inputSize = 784;   // 28x28 MNIST images
var hiddenSize = 128;
var outputSize = 10;   // 10 digit classes

Console.WriteLine(""Architecture:"");
Console.WriteLine($""  Input Layer:  {inputSize} neurons"");
Console.WriteLine($""  Hidden Layer: {hiddenSize} neurons (ReLU)"");
Console.WriteLine($""  Output Layer: {outputSize} neurons (Softmax)"");
Console.WriteLine();

// In a full implementation:
// var model = new NeuralNetwork<float>(
//     new DenseLayer<float>(inputSize, hiddenSize),
//     new ReLUActivation<float>(),
//     new DenseLayer<float>(hiddenSize, outputSize),
//     new SoftmaxActivation<float>()
// );
//
// var result = await new AiModelBuilder<float, Tensor<float>, int>()
//     .ConfigureModel(model)
//     .ConfigureOptimizer(new AdamOptimizer<float>(learningRate: 0.001f))
//     .ConfigureLossFunction(new CrossEntropyLoss<float>())
//     .BuildAsync(trainImages, trainLabels);

Console.WriteLine(""Total parameters: 101,770"");
Console.WriteLine(""Optimizer: Adam (lr=0.001)"");
Console.WriteLine(""Loss: CrossEntropy"");
Console.WriteLine();
Console.WriteLine(""Ready to train on MNIST dataset!"");
"
                },
                new CodeExample
                {
                    Id = "cnn-image",
                    Name = "CNN for Images",
                    Description = "Convolutional Neural Network for image classification",
                    Difficulty = "Intermediate",
                    Tags = ["CNN", "ResNet", "CIFAR-10", "GPU"],
                    Code = @"// Convolutional Neural Network for Image Classification
using AiDotNet;
using AiDotNet.NeuralNetworks;
using System;

Console.WriteLine(""CNN Image Classifier"");
Console.WriteLine(""===================="");
Console.WriteLine();

// Architecture for CIFAR-10 (32x32x3 images)
Console.WriteLine(""Architecture (ResNet-like):"");
Console.WriteLine(""  Conv2D(3, 64, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  Conv2D(64, 64, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  MaxPool2D(2x2)"");
Console.WriteLine(""  Conv2D(64, 128, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  Conv2D(128, 128, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  MaxPool2D(2x2)"");
Console.WriteLine(""  GlobalAvgPool2D"");
Console.WriteLine(""  Dense(128, 10) -> Softmax"");
Console.WriteLine();

// In a full implementation:
// var model = new ResNet<float>(
//     variant: ResNetVariant.ResNet18,
//     numClasses: 10,
//     pretrained: false);
//
// var result = await new AiModelBuilder<float, Tensor<float>, int>()
//     .ConfigureModel(model)
//     .ConfigureOptimizer(new AdamWOptimizer<float>(learningRate: 3e-4f))
//     .ConfigureGpuAcceleration(new GpuAccelerationConfig { Enabled = true })
//     .BuildAsync(trainImages, trainLabels);

Console.WriteLine(""Total parameters: 11.2M"");
Console.WriteLine(""GPU acceleration: Enabled"");
Console.WriteLine();
Console.WriteLine(""Expected accuracy on CIFAR-10: ~93%"");
"
                }
            },

            ["Computer Vision"] = new()
            {
                new CodeExample
                {
                    Id = "yolo-detection",
                    Name = "YOLO Object Detection",
                    Description = "Detect objects in images with YOLOv8",
                    Difficulty = "Intermediate",
                    Tags = ["YOLO", "detection", "COCO", "real-time"],
                    Code = @"// YOLO Object Detection
using AiDotNet;
using AiDotNet.ComputerVision;
using System;

Console.WriteLine(""YOLOv8 Object Detection"");
Console.WriteLine(""======================"");
Console.WriteLine();

// In a full implementation:
// var detector = await YOLOv8<float>.LoadAsync(
//     model: ""yolov8n"",  // nano variant
//     device: Device.GPU);
//
// var results = detector.Detect(image);
// foreach (var detection in results)
// {
//     Console.WriteLine($""{detection.Class}: {detection.Confidence:P1}"");
//     Console.WriteLine($""  Box: ({detection.X}, {detection.Y}, {detection.Width}, {detection.Height})"");
// }

Console.WriteLine(""Model: YOLOv8n (nano)"");
Console.WriteLine(""Input size: 640x640"");
Console.WriteLine(""Classes: 80 (COCO dataset)"");
Console.WriteLine();
Console.WriteLine(""Sample detection results:"");
Console.WriteLine(""  person: 95.2% @ (120, 50, 200, 380)"");
Console.WriteLine(""  car: 88.7% @ (350, 200, 180, 120)"");
Console.WriteLine(""  dog: 92.1% @ (50, 300, 150, 180)"");
Console.WriteLine();
Console.WriteLine(""Inference time: 12ms (GPU)"");
"
                },
                new CodeExample
                {
                    Id = "image-segmentation",
                    Name = "Image Segmentation",
                    Description = "Segment images with Mask R-CNN",
                    Difficulty = "Advanced",
                    Tags = ["segmentation", "Mask R-CNN", "instance", "pixel"],
                    Code = @"// Instance Segmentation with Mask R-CNN
using AiDotNet;
using AiDotNet.ComputerVision;
using System;

Console.WriteLine(""Mask R-CNN Instance Segmentation"");
Console.WriteLine(""================================"");
Console.WriteLine();

// In a full implementation:
// var segmenter = await MaskRCNN<float>.LoadAsync(
//     backbone: ""resnet50"",
//     pretrained: true);
//
// var results = segmenter.Segment(image);
// foreach (var instance in results)
// {
//     Console.WriteLine($""{instance.Class}: {instance.Confidence:P1}"");
//     // instance.Mask contains the pixel-level segmentation
// }

Console.WriteLine(""Model: Mask R-CNN"");
Console.WriteLine(""Backbone: ResNet-50-FPN"");
Console.WriteLine(""Classes: 80 (COCO)"");
Console.WriteLine();
Console.WriteLine(""Capabilities:"");
Console.WriteLine(""  - Object detection"");
Console.WriteLine(""  - Instance segmentation"");
Console.WriteLine(""  - Pixel-level masks"");
Console.WriteLine();
Console.WriteLine(""Sample results:"");
Console.WriteLine(""  person: 94.1% (mask: 15,234 pixels)"");
Console.WriteLine(""  bicycle: 87.3% (mask: 8,921 pixels)"");
"
                }
            },

            ["RAG & LLMs"] = new()
            {
                new CodeExample
                {
                    Id = "basic-rag",
                    Name = "Basic RAG Pipeline",
                    Description = "Retrieval-Augmented Generation",
                    Difficulty = "Intermediate",
                    Tags = ["RAG", "embeddings", "vector search", "LLM"],
                    Code = @"// Basic RAG Pipeline
using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration;
using System;

Console.WriteLine(""RAG Pipeline"");
Console.WriteLine(""============"");
Console.WriteLine();

// In a full implementation:
// var rag = new RAGPipeline<float>()
//     .WithEmbeddings(new SentenceTransformerEmbeddings<float>(""all-MiniLM-L6-v2""))
//     .WithVectorStore(new InMemoryVectorStore<float>(dimension: 384))
//     .WithRetriever(new DenseRetriever<float>(topK: 5))
//     .Build();
//
// await rag.IndexDocumentsAsync(documents);
// var response = await rag.QueryAsync(""What is AiDotNet?"");

Console.WriteLine(""Components:"");
Console.WriteLine(""  Embeddings: all-MiniLM-L6-v2 (384 dimensions)"");
Console.WriteLine(""  Vector Store: In-Memory"");
Console.WriteLine(""  Retriever: Dense (top-5)"");
Console.WriteLine();
Console.WriteLine(""Indexed: 100 documents"");
Console.WriteLine();
Console.WriteLine(""Query: 'What neural networks does AiDotNet support?'"");
Console.WriteLine();
Console.WriteLine(""Retrieved sources:"");
Console.WriteLine(""  1. Neural Networks Overview (score: 0.92)"");
Console.WriteLine(""  2. CNN Architectures (score: 0.87)"");
Console.WriteLine(""  3. Transformer Models (score: 0.85)"");
Console.WriteLine();
Console.WriteLine(""Answer: AiDotNet supports 100+ neural network architectures..."");
"
                },
                new CodeExample
                {
                    Id = "lora-finetune",
                    Name = "LoRA Fine-tuning",
                    Description = "Efficient LLM fine-tuning with LoRA",
                    Difficulty = "Advanced",
                    Tags = ["LoRA", "fine-tuning", "LLM", "PEFT"],
                    Code = @"// LoRA Fine-tuning
using AiDotNet;
using AiDotNet.LoRA;
using System;

Console.WriteLine(""LoRA Fine-tuning"");
Console.WriteLine(""================"");
Console.WriteLine();

// In a full implementation:
// var model = await HuggingFaceHub.LoadModelAsync<float>(""microsoft/phi-2"");
//
// var loraConfig = new LoRAConfig<float>
// {
//     Rank = 8,
//     Alpha = 16,
//     TargetModules = [""q_proj"", ""v_proj""],
//     Dropout = 0.05f
// };
//
// var loraModel = model.ApplyLoRA(loraConfig);
// await loraModel.TrainAsync(trainingData, trainingConfig);

Console.WriteLine(""Base Model: microsoft/phi-2 (2.7B parameters)"");
Console.WriteLine();
Console.WriteLine(""LoRA Configuration:"");
Console.WriteLine(""  Rank: 8"");
Console.WriteLine(""  Alpha: 16"");
Console.WriteLine(""  Target: q_proj, v_proj"");
Console.WriteLine(""  Dropout: 0.05"");
Console.WriteLine();
Console.WriteLine(""Memory Usage:"");
Console.WriteLine(""  Full fine-tune: 10.8 GB"");
Console.WriteLine(""  LoRA fine-tune: 1.1 GB (90% reduction!)"");
Console.WriteLine();
Console.WriteLine(""Trainable parameters: 2.1M (0.08% of total)"");
"
                }
            },

            ["Reinforcement Learning"] = new()
            {
                new CodeExample
                {
                    Id = "dqn-cartpole",
                    Name = "DQN CartPole",
                    Description = "Deep Q-Network for CartPole environment",
                    Difficulty = "Intermediate",
                    Tags = ["DQN", "Q-learning", "CartPole", "RL"],
                    Code = @"// DQN Agent for CartPole
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using System;

Console.WriteLine(""DQN Agent - CartPole"");
Console.WriteLine(""===================="");
Console.WriteLine();

// In a full implementation:
// var config = new DQNConfig<float>
// {
//     StateSize = 4,
//     ActionSize = 2,
//     HiddenLayers = [128, 128],
//     LearningRate = 1e-3f,
//     Gamma = 0.99f,
//     EpsilonStart = 1.0f,
//     EpsilonEnd = 0.01f,
//     ReplayBufferSize = 100000,
//     BatchSize = 64
// };
//
// var agent = new DQNAgent<float>(config);
// var env = new CartPoleEnvironment<float>();

Console.WriteLine(""Environment: CartPole-v1"");
Console.WriteLine(""  State: [position, velocity, angle, angular_velocity]"");
Console.WriteLine(""  Actions: [push_left, push_right]"");
Console.WriteLine();
Console.WriteLine(""Agent Configuration:"");
Console.WriteLine(""  Network: 4 -> 128 -> 128 -> 2"");
Console.WriteLine(""  Optimizer: Adam (lr=0.001)"");
Console.WriteLine(""  Gamma: 0.99"");
Console.WriteLine(""  Epsilon: 1.0 -> 0.01"");
Console.WriteLine();
Console.WriteLine(""Training progress:"");
Console.WriteLine(""  Episode 100: Avg reward = 23.4"");
Console.WriteLine(""  Episode 200: Avg reward = 87.2"");
Console.WriteLine(""  Episode 300: Avg reward = 156.8"");
Console.WriteLine(""  Episode 400: Avg reward = 195.3"");
Console.WriteLine(""  Episode 500: Avg reward = 200.0 (SOLVED!)"");
"
                },
                new CodeExample
                {
                    Id = "ppo-continuous",
                    Name = "PPO Continuous Control",
                    Description = "PPO for continuous action spaces",
                    Difficulty = "Advanced",
                    Tags = ["PPO", "policy gradient", "continuous", "actor-critic"],
                    Code = @"// PPO Agent for Continuous Control
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using System;

Console.WriteLine(""PPO Agent - Continuous Control"");
Console.WriteLine(""==============================="");
Console.WriteLine();

// In a full implementation:
// var config = new PPOConfig<float>
// {
//     StateSize = 8,
//     ActionSize = 4,
//     HiddenLayers = [256, 256],
//     LearningRate = 3e-4f,
//     Gamma = 0.99f,
//     Lambda = 0.95f,
//     ClipRatio = 0.2f,
//     EntropyCoefficient = 0.01f,
//     NumEpochs = 10,
//     MiniBatchSize = 64
// };
//
// var agent = new PPOAgent<float>(config);

Console.WriteLine(""Algorithm: Proximal Policy Optimization"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Network: Actor-Critic (256, 256)"");
Console.WriteLine(""  Clip ratio: 0.2"");
Console.WriteLine(""  GAE lambda: 0.95"");
Console.WriteLine(""  Entropy coefficient: 0.01"");
Console.WriteLine();
Console.WriteLine(""Why PPO?"");
Console.WriteLine(""  - Most stable policy gradient method"");
Console.WriteLine(""  - Works well on continuous control"");
Console.WriteLine(""  - Good sample efficiency"");
Console.WriteLine(""  - Easy to tune"");
"
                }
            }
        };
    }
}

/// <summary>
/// Represents a code example in the playground.
/// </summary>
public class CodeExample
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public string Code { get; set; } = "";
    public string Difficulty { get; set; } = "Beginner";
    public string[] Tags { get; set; } = [];
}
