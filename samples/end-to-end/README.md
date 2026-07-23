# End-to-End Applications

This directory contains complete, production-ready applications built with AiDotNet.

## Available Applications

| Application | Description |
|-------------|-------------|
| [ImageClassificationWebApp](./ImageClassificationWebApp/) | Web app for image classification |

## ImageClassificationWebApp

A complete web application demonstrating:
- Modern web UI with drag-and-drop upload
- CNN-based image classification
- REST API endpoints
- Real-time predictions
- Confidence visualization

### Running

```bash
cd ImageClassificationWebApp
dotnet run
```

Open `http://localhost:5200` in your browser.

### Features

- **Web UI**: Drag-and-drop image upload
- **API**: REST endpoints for programmatic access
- **Batch Processing**: Classify multiple images at once
- **Visualization**: Confidence bars for all classes

## Architecture Pattern

All end-to-end apps follow this pattern:

```
App/
├── Program.cs           # Entry point, DI configuration
├── Services/            # Business logic, ML integration
├── Models/              # DTOs, domain models
├── Pages/               # Razor pages (if Blazor)
└── wwwroot/             # Static files
```

## Key Integration Points

### 1. Service Registration
```csharp
builder.Services.AddSingleton<ClassificationService>();
```

### 2. Model Initialization
```csharp
var service = app.Services.GetRequiredService<ClassificationService>();
await service.InitializeAsync();
```

### 3. API Endpoints
```csharp
app.MapPost("/api/classify", async (IFormFile file, ClassificationService svc) =>
{
    var result = await svc.ClassifyAsync(file);
    return Results.Ok(result);
});
```

## Coming Soon

- **ChatbotWithRAG**: Conversational AI with knowledge retrieval
- **SpeechAssistant**: Voice-enabled assistant
- **RecommendationEngine**: Product recommendation system

## Learn More

- [Building Web Apps Tutorial](/docs/tutorials/web-apps/)
- [API Design Guide](/docs/guides/api-design/)
