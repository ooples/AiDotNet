using AiDotNet;
using AiDotNet.Classification;
using AiDotNet.Serving;
using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Services;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Options;

Console.WriteLine("==========================================");
Console.WriteLine("  AiDotNet.Serving - Model Serving Demo  ");
Console.WriteLine("==========================================\n");

// ============================================
// This sample demonstrates how to use the
// AiDotNet.Serving library to deploy trained
// models as production REST APIs.
// ============================================

Console.WriteLine("This sample shows two deployment approaches:\n");
Console.WriteLine("  1. Embedded: Add serving to an existing ASP.NET app");
Console.WriteLine("  2. Standalone: Run AiDotNet.Serving as its own server\n");

// ============================================
// Part 1: Train a model using AiDotNet
// ============================================
Console.WriteLine("Step 1: Training a classification model...\n");

// Iris-like classification data
var features = new double[][]
{
    // Setosa
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
    // Versicolor
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
    // Virginica
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1]
};
var labels = new double[] { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 };

var modelResult = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

Console.WriteLine("  Model trained successfully!");
Console.WriteLine($"  Type: Random Forest Classifier");
Console.WriteLine($"  Input features: 4");
Console.WriteLine($"  Classes: 3 (Setosa, Versicolor, Virginica)");

// Test the model locally
var testSample = new double[] { 5.9, 3.0, 5.1, 1.8 };
var prediction = modelResult.Model.Predict(testSample);
Console.WriteLine($"\n  Local prediction for [5.9, 3.0, 5.1, 1.8]: Class {(int)prediction}");

// ============================================
// Part 2: Save model for serving
// ============================================
Console.WriteLine("\n\nStep 2: Saving model for serving...\n");

var modelPath = Path.Combine(Path.GetTempPath(), "iris-classifier.aidotnet");
modelResult.SaveToFile(modelPath);
Console.WriteLine($"  Model saved to: {modelPath}");

// ============================================
// Part 3: Start AiDotNet.Serving server
// ============================================
Console.WriteLine("\n\nStep 3: Starting AiDotNet.Serving server...\n");

var builder = WebApplication.CreateBuilder(args);

// Configure serving options
builder.Services.Configure<ServingOptions>(options =>
{
    options.Port = 5100;
    options.ModelDirectory = Path.GetTempPath();
    options.BatchingWindowMs = 10;
    options.MaxBatchSize = 32;
});

// Register AiDotNet.Serving services
builder.Services.AddSingleton<IModelRepository, ModelRepository>();
builder.Services.AddSingleton<IRequestBatcher, RequestBatcher>();

// Add controllers from AiDotNet.Serving
builder.Services.AddControllers()
    .AddApplicationPart(typeof(ServingOptions).Assembly);

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// Configure Kestrel
builder.WebHost.ConfigureKestrel(options =>
{
    options.ListenLocalhost(5100);
});

var app = builder.Build();

// Configure middleware
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseAuthorization();
app.MapControllers();

// ============================================
// Part 4: Load model into serving registry
// ============================================
Console.WriteLine("Step 4: Loading model into serving registry...\n");

var repository = app.Services.GetRequiredService<IModelRepository>();

// Create a servable model wrapper
// This wraps the AiModelResult for serving
var servableModel = new ServableModelWrapper<double>(
    modelName: "iris-classifier",
    inputDimension: 4,
    outputDimension: 1,
    predictFunc: input =>
    {
        var inputArray = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
            inputArray[i] = input[i];

        var result = modelResult.Model.Predict(inputArray);
        return new Vector<double>([(double)(int)result]);
    },
    predictBatchFunc: inputs =>
    {
        var results = new Matrix<double>(inputs.Rows, 1);
        for (int i = 0; i < inputs.Rows; i++)
        {
            var inputArray = new double[inputs.Columns];
            for (int j = 0; j < inputs.Columns; j++)
                inputArray[j] = inputs[i, j];

            var result = modelResult.Model.Predict(inputArray);
            results[i, 0] = (double)(int)result;
        }
        return results;
    }
);

repository.LoadModel("iris-classifier", servableModel, modelPath);

Console.WriteLine("  Model 'iris-classifier' loaded into registry");

// ============================================
// Part 5: Display API information
// ============================================
Console.WriteLine("\n\nStep 5: Server ready!\n");
Console.WriteLine("==========================================");
Console.WriteLine("  AiDotNet Model Serving API Ready        ");
Console.WriteLine("==========================================\n");

Console.WriteLine("  Base URL: http://localhost:5100");
Console.WriteLine("  Swagger:  http://localhost:5100/swagger\n");

Console.WriteLine("  Available Endpoints:");
Console.WriteLine("  ────────────────────────────────────────");
Console.WriteLine("  GET  /api/models                    - List loaded models");
Console.WriteLine("  GET  /api/models/{name}             - Get model info");
Console.WriteLine("  POST /api/models                    - Load a model");
Console.WriteLine("  DELETE /api/models/{name}           - Unload a model");
Console.WriteLine("  POST /api/inference/predict/{name}  - Make predictions");
Console.WriteLine("  GET  /api/inference/stats           - Batching statistics\n");

Console.WriteLine("  Example API Calls:");
Console.WriteLine("  ────────────────────────────────────────\n");

Console.WriteLine("  # List models:");
Console.WriteLine("  curl http://localhost:5100/api/models\n");

Console.WriteLine("  # Make a prediction:");
Console.WriteLine("  curl -X POST http://localhost:5100/api/inference/predict/iris-classifier \\");
Console.WriteLine("    -H 'Content-Type: application/json' \\");
Console.WriteLine("    -d '{\"features\": [[5.9, 3.0, 5.1, 1.8]]}'\n");

Console.WriteLine("  # Batch prediction (multiple samples):");
Console.WriteLine("  curl -X POST http://localhost:5100/api/inference/predict/iris-classifier \\");
Console.WriteLine("    -H 'Content-Type: application/json' \\");
Console.WriteLine("    -d '{\"features\": [[5.1, 3.5, 1.4, 0.2], [7.0, 3.2, 4.7, 1.4], [6.3, 3.3, 6.0, 2.5]]}'\n");

Console.WriteLine("  Expected Response:");
Console.WriteLine("  {");
Console.WriteLine("    \"predictions\": [[0.0], [1.0], [2.0]],");
Console.WriteLine("    \"processingTimeMs\": 5,");
Console.WriteLine("    \"batchSize\": 3");
Console.WriteLine("  }\n");

Console.WriteLine("  Class Mapping:");
Console.WriteLine("    0 = Setosa");
Console.WriteLine("    1 = Versicolor");
Console.WriteLine("    2 = Virginica\n");

Console.WriteLine("  Press Ctrl+C to stop the server...\n");

// Start the server
await app.RunAsync();
