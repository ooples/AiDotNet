using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Architectures;
using AiDotNet.ComputerVision;
using AiDotNet.LinearAlgebra;
using AiDotNet.Optimizers;
using AiDotNet.LossFunctions;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json;

// =============================================
// Image Classification Web Application
// Complete end-to-end example using AiDotNet
// =============================================

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddRazorPages();
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Register classification service as singleton
builder.Services.AddSingleton<ImageClassificationService>();

var app = builder.Build();

// Initialize the classification model
var classificationService = app.Services.GetRequiredService<ImageClassificationService>();
await classificationService.InitializeAsync();

// Configure pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseStaticFiles();
app.UseRouting();

// API Endpoints
app.MapGet("/", () => Results.Content(GetHomePage(), "text/html"));

app.MapPost("/api/classify", async (HttpRequest request, ImageClassificationService service) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "Expected multipart/form-data" });

    var form = await request.ReadFormAsync();
    var file = form.Files.FirstOrDefault();

    if (file == null || file.Length == 0)
        return Results.BadRequest(new { error = "No image file provided" });

    using var stream = file.OpenReadStream();
    using var ms = new MemoryStream();
    await stream.CopyToAsync(ms);

    var result = await service.ClassifyAsync(ms.ToArray());
    return Results.Ok(result);
});

app.MapPost("/api/classify/batch", async (HttpRequest request, ImageClassificationService service) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "Expected multipart/form-data" });

    var form = await request.ReadFormAsync();
    var files = form.Files;

    if (files.Count == 0)
        return Results.BadRequest(new { error = "No image files provided" });

    var results = new List<ClassificationResult>();
    foreach (var file in files)
    {
        using var stream = file.OpenReadStream();
        using var ms = new MemoryStream();
        await stream.CopyToAsync(ms);

        var result = await service.ClassifyAsync(ms.ToArray());
        result.FileName = file.FileName;
        results.Add(result);
    }

    return Results.Ok(new { results });
});

app.MapGet("/api/model/info", (ImageClassificationService service) =>
{
    return Results.Ok(service.GetModelInfo());
});

app.MapGet("/api/model/classes", (ImageClassificationService service) =>
{
    return Results.Ok(new { classes = service.GetClassLabels() });
});

app.MapGet("/api/health", () => Results.Ok(new { status = "healthy" }));

Console.WriteLine("\n==========================================");
Console.WriteLine("  Image Classification Web App           ");
Console.WriteLine("==========================================\n");
Console.WriteLine("  Open in browser: http://localhost:5200\n");
Console.WriteLine("  API Endpoints:");
Console.WriteLine("    POST /api/classify       - Classify an image");
Console.WriteLine("    POST /api/classify/batch - Classify multiple images");
Console.WriteLine("    GET  /api/model/info     - Get model info");
Console.WriteLine("    GET  /api/model/classes  - Get class labels\n");

app.Run("http://localhost:5200");

// =============================================
// Image Classification Service
// =============================================
public class ImageClassificationService
{
    private NeuralNetwork<float>? _model;
    private bool _isInitialized;
    private readonly object _lock = new();

    public static readonly string[] ClassLabels =
    [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ];

    public async Task InitializeAsync()
    {
        Console.WriteLine("Initializing image classification model...");

        // Create a CNN model (ResNet-18 style)
        var config = new ResNetConfig<float>
        {
            InputChannels = 3,
            InputHeight = 32,
            InputWidth = 32,
            NumBlocks = [2, 2, 2, 2],
            NumFilters = [64, 128, 256, 512],
            NumClasses = 10
        };

        _model = new NeuralNetwork<float>(
            new NeuralNetworkArchitecture<float>(
                inputFeatures: 3 * 32 * 32,
                numClasses: 10,
                complexity: NetworkComplexity.Medium));

        // Train on synthetic data for demo purposes
        await TrainOnSyntheticDataAsync();

        _isInitialized = true;
        Console.WriteLine("Model initialized successfully!");
    }

    private async Task TrainOnSyntheticDataAsync()
    {
        Console.WriteLine("Training model on synthetic data...");

        var random = new Random(42);
        const int numSamples = 500;
        const int batchSize = 32;
        const int epochs = 10;

        // Generate synthetic training data
        var trainFeatures = new Tensor<float>([numSamples, 3, 32, 32]);
        var trainLabels = new Tensor<float>([numSamples, 10]);

        for (int i = 0; i < numSamples; i++)
        {
            int label = i % 10;

            // Create class-specific patterns
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < 32; h++)
                {
                    for (int w = 0; w < 32; w++)
                    {
                        // Create distinct patterns for each class
                        float value = label switch
                        {
                            0 => (h + w) / 64f, // Gradient
                            1 => h < 16 ? 0.8f : 0.2f, // Horizontal split
                            2 => w < 16 ? 0.8f : 0.2f, // Vertical split
                            3 => (h + w) % 8 < 4 ? 0.9f : 0.1f, // Checkerboard
                            4 => MathF.Sin(h * 0.3f) * 0.5f + 0.5f, // Waves
                            5 => (h * w) % 32 / 32f, // Pattern
                            6 => h % 4 < 2 ? 0.7f : 0.3f, // Stripes
                            7 => MathF.Sqrt(MathF.Pow(h - 16, 2) + MathF.Pow(w - 16, 2)) / 22f, // Circle
                            8 => MathF.Abs(h - 16) / 16f, // V pattern
                            _ => random.NextSingle() * 0.5f + 0.25f // Noise
                        };

                        value += (random.NextSingle() - 0.5f) * 0.2f;
                        trainFeatures[[i, c, h, w]] = Math.Clamp(value, 0, 1);
                    }
                }
            }

            trainLabels[[i, label]] = 1.0f;
        }

        // Train
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            _model!.Train(trainFeatures, trainLabels);

            if ((epoch + 1) % 2 == 0)
            {
                Console.WriteLine($"  Epoch {epoch + 1}/{epochs} completed");
            }
        }

        await Task.CompletedTask;
    }

    public async Task<ClassificationResult> ClassifyAsync(byte[] imageData)
    {
        if (!_isInitialized || _model == null)
            throw new InvalidOperationException("Model not initialized");

        // Preprocess image
        var tensor = PreprocessImage(imageData);

        // Run inference
        var predictions = _model.Predict(tensor);

        // Get results
        var probabilities = new Dictionary<string, float>();
        int predictedClass = 0;
        float maxProb = float.MinValue;

        for (int i = 0; i < 10; i++)
        {
            float prob = predictions[[0, i]];
            probabilities[ClassLabels[i]] = prob;

            if (prob > maxProb)
            {
                maxProb = prob;
                predictedClass = i;
            }
        }

        // Apply softmax normalization
        float sum = probabilities.Values.Sum();
        foreach (var key in probabilities.Keys.ToList())
        {
            probabilities[key] /= sum;
        }

        return await Task.FromResult(new ClassificationResult
        {
            Prediction = ClassLabels[predictedClass],
            Confidence = probabilities[ClassLabels[predictedClass]],
            Probabilities = probabilities,
            ProcessingTimeMs = 10 // Simulated
        });
    }

    private Tensor<float> PreprocessImage(byte[] imageData)
    {
        // For demo, create a synthetic tensor from image bytes
        // In production, use proper image decoding library
        var tensor = new Tensor<float>([1, 3, 32, 32]);
        var random = new Random(BitConverter.ToInt32(imageData, 0));

        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    // Use bytes from image data to create pattern
                    int byteIndex = (c * 32 * 32 + h * 32 + w) % imageData.Length;
                    tensor[[0, c, h, w]] = imageData[byteIndex] / 255f;
                }
            }
        }

        return tensor;
    }

    public ModelInfo GetModelInfo()
    {
        return new ModelInfo
        {
            Name = "CIFAR-10 Classifier",
            Architecture = "ResNet-18 style CNN",
            InputShape = "3x32x32 (RGB)",
            NumClasses = 10,
            Parameters = _model?.GetParameterCount() ?? 0,
            IsInitialized = _isInitialized
        };
    }

    public string[] GetClassLabels() => ClassLabels;
}

// =============================================
// Data Transfer Objects
// =============================================
public class ClassificationResult
{
    public string FileName { get; set; } = "";
    public string Prediction { get; set; } = "";
    public float Confidence { get; set; }
    public Dictionary<string, float> Probabilities { get; set; } = new();
    public double ProcessingTimeMs { get; set; }
}

public class ModelInfo
{
    public string Name { get; set; } = "";
    public string Architecture { get; set; } = "";
    public string InputShape { get; set; } = "";
    public int NumClasses { get; set; }
    public long Parameters { get; set; }
    public bool IsInitialized { get; set; }
}

// =============================================
// HTML Home Page
// =============================================
static string GetHomePage() => """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AiDotNet Image Classifier</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            color: white;
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }
        .subtitle {
            color: rgba(255,255,255,0.8);
            text-align: center;
            margin-bottom: 2rem;
        }
        .card {
            background: white;
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            margin-bottom: 1.5rem;
        }
        .upload-zone {
            border: 3px dashed #ddd;
            border-radius: 0.5rem;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-zone:hover, .upload-zone.dragover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-zone h3 { color: #333; margin-bottom: 0.5rem; }
        .upload-zone p { color: #666; }
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            font-size: 1rem;
            margin-top: 1rem;
        }
        .btn:hover { opacity: 0.9; }
        #fileInput { display: none; }
        #preview {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 1rem;
        }
        .preview-item {
            width: 100px;
            height: 100px;
            border-radius: 0.5rem;
            object-fit: cover;
        }
        #results {
            margin-top: 1.5rem;
        }
        .result-item {
            background: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 0.5rem;
        }
        .result-label {
            font-weight: bold;
            color: #667eea;
            font-size: 1.2rem;
        }
        .confidence {
            color: #28a745;
            font-size: 0.9rem;
        }
        .prob-bar {
            display: flex;
            align-items: center;
            margin: 0.25rem 0;
        }
        .prob-label { width: 100px; font-size: 0.8rem; }
        .prob-fill {
            height: 8px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.3s;
        }
        .prob-value { font-size: 0.8rem; margin-left: 0.5rem; color: #666; }
        .loading {
            text-align: center;
            padding: 2rem;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .classes {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        .class-tag {
            background: #e9ecef;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AiDotNet Image Classifier</h1>
        <p class="subtitle">Powered by AiDotNet Neural Networks</p>

        <div class="card">
            <div class="upload-zone" id="uploadZone">
                <h3>Drag & Drop Images</h3>
                <p>or click to browse</p>
                <button class="btn" onclick="document.getElementById('fileInput').click()">
                    Select Images
                </button>
                <input type="file" id="fileInput" accept="image/*" multiple>
            </div>
            <div id="preview"></div>
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Classifying images...</p>
            </div>
            <div id="results"></div>
        </div>

        <div class="card">
            <h3>Supported Classes</h3>
            <div class="classes" id="classes"></div>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const results = document.getElementById('results');
        const loading = document.getElementById('loading');
        const classesDiv = document.getElementById('classes');

        // Load classes
        fetch('/api/model/classes')
            .then(r => r.json())
            .then(data => {
                classesDiv.innerHTML = data.classes
                    .map(c => `<span class="class-tag">${c}</span>`)
                    .join('');
            });

        // Drag and drop
        uploadZone.addEventListener('dragover', e => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', e => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', e => handleFiles(e.target.files));

        async function handleFiles(files) {
            if (!files.length) return;

            // Show previews
            preview.innerHTML = '';
            for (const file of files) {
                const img = document.createElement('img');
                img.src = URL.createObjectURL(file);
                img.className = 'preview-item';
                preview.appendChild(img);
            }

            // Classify
            loading.style.display = 'block';
            results.innerHTML = '';

            const formData = new FormData();
            for (const file of files) {
                formData.append('images', file);
            }

            try {
                const response = await fetch('/api/classify/batch', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                results.innerHTML = data.results.map(r => `
                    <div class="result-item">
                        <div class="result-label">${r.prediction}</div>
                        <div class="confidence">${(r.confidence * 100).toFixed(1)}% confidence</div>
                        ${Object.entries(r.probabilities)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 5)
                            .map(([label, prob]) => `
                                <div class="prob-bar">
                                    <span class="prob-label">${label}</span>
                                    <div style="flex:1;background:#eee;border-radius:4px;">
                                        <div class="prob-fill" style="width:${prob*100}%"></div>
                                    </div>
                                    <span class="prob-value">${(prob*100).toFixed(1)}%</span>
                                </div>
                            `).join('')}
                    </div>
                `).join('');
            } catch (err) {
                results.innerHTML = `<div class="result-item" style="color:red">Error: ${err.message}</div>`;
            } finally {
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
""";
