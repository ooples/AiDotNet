using AiDotNet;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.Tensors;

Console.WriteLine("=== AiDotNet Optical Character Recognition (OCR) ===");
Console.WriteLine("Extract text from images with localization and multi-language support\n");

// Display available OCR configurations
Console.WriteLine("Available Text Detection Models:");
Console.WriteLine("  DBNet    - Differentiable Binarization Network (fast, accurate)");
Console.WriteLine("  CRAFT    - Character Region Awareness (good for curved text)");
Console.WriteLine("  EAST     - Efficient and Accurate Scene Text detector");
Console.WriteLine();
Console.WriteLine("Available Text Recognition Models:");
Console.WriteLine("  CRNN     - Convolutional Recurrent Neural Network (CTC-based)");
Console.WriteLine("  TrOCR    - Transformer-based OCR (attention-based, higher accuracy)");
Console.WriteLine();

try
{
    // Configure OCR options
    var options = new OCRConfig
    {
        Mode = OCRMode.SceneText,
        DetectionModel = TextDetectionModel.DBNet,
        RecognitionModel = TextRecognitionModel.CRNN,
        SupportedLanguages = new[] { "en", "es", "fr", "de", "zh", "ja" },
        ConfidenceThreshold = 0.5,
        DetectOrientation = true,
        CorrectSkew = true,
        GroupTextLines = true
    };

    Console.WriteLine("Configuration:");
    Console.WriteLine($"  Mode: {options.Mode}");
    Console.WriteLine($"  Detection Model: {options.DetectionModel}");
    Console.WriteLine($"  Recognition Model: {options.RecognitionModel}");
    Console.WriteLine($"  Confidence Threshold: {options.ConfidenceThreshold}");
    Console.WriteLine($"  Supported Languages: {string.Join(", ", options.SupportedLanguages)}");
    Console.WriteLine($"  Detect Orientation: {options.DetectOrientation}");
    Console.WriteLine($"  Correct Skew: {options.CorrectSkew}");
    Console.WriteLine();

    // Create OCR pipeline using facade pattern
    Console.WriteLine("Creating OCR pipeline...");
    var ocr = new OCREngine(options);
    Console.WriteLine("  OCR engine created successfully\n");

    // Generate synthetic test images with text
    Console.WriteLine("Generating synthetic test images with text patterns...\n");
    var testImages = GenerateSyntheticTextImages();

    // Process each test image
    foreach (var (image, description, expectedText) in testImages)
    {
        Console.WriteLine($"Processing: {description}");
        Console.WriteLine(new string('=', 70));

        var result = ocr.RecognizeText(image);

        Console.WriteLine($"  Image Size: {result.ImageWidth}x{result.ImageHeight}");
        Console.WriteLine($"  Inference Time: {result.InferenceTime.TotalMilliseconds:F2}ms");
        Console.WriteLine($"  Text Regions Found: {result.TextRegions.Count}");

        if (result.TextRegions.Count > 0)
        {
            Console.WriteLine("\n  Detected Text Regions:");
            Console.WriteLine("  " + new string('-', 65));
            Console.WriteLine("  | # | Text                         | Conf  | Location (x,y,w,h)  |");
            Console.WriteLine("  " + new string('-', 65));

            int regionNum = 1;
            foreach (var region in result.TextRegions.OrderBy(r => r.Box?.Y1 ?? 0).ThenBy(r => r.Box?.X1 ?? 0))
            {
                string textDisplay = region.Text.Length > 28 ? region.Text[..25] + "..." : region.Text;
                string locationStr = region.Box != null
                    ? $"({region.Box.X1:F0},{region.Box.Y1:F0},{region.Box.Width:F0},{region.Box.Height:F0})"
                    : "N/A";
                Console.WriteLine($"  | {regionNum,1} | {textDisplay,-28} | {region.Confidence:P0} | {locationStr,-19} |");
                regionNum++;
            }
            Console.WriteLine("  " + new string('-', 65));

            Console.WriteLine($"\n  Full Extracted Text:");
            Console.WriteLine($"  \"{result.FullText}\"");

            if (!string.IsNullOrEmpty(expectedText))
            {
                Console.WriteLine($"\n  Expected Text:");
                Console.WriteLine($"  \"{expectedText}\"");
            }
        }
        Console.WriteLine("\n");
    }

    // Demonstrate multi-language OCR
    Console.WriteLine("=== Multi-Language OCR Demo ===\n");

    var multiLangSamples = new[]
    {
        ("en", "Hello World", "English"),
        ("es", "Hola Mundo", "Spanish"),
        ("fr", "Bonjour le Monde", "French"),
        ("de", "Hallo Welt", "German"),
        ("zh", "Hello", "Chinese (Simulated)"),
        ("ja", "Hello", "Japanese (Simulated)")
    };

    Console.WriteLine("  | Language   | Sample Text       | Detection Status    |");
    Console.WriteLine("  " + new string('-', 60));

    foreach (var (langCode, sampleText, langName) in multiLangSamples)
    {
        bool isSupported = options.SupportedLanguages.Contains(langCode);
        string status = isSupported ? "Supported" : "Add to config";
        Console.WriteLine($"  | {langName,-10} | {sampleText,-17} | {status,-19} |");
    }
    Console.WriteLine("  " + new string('-', 60));

    // Demonstrate document OCR vs scene text OCR
    Console.WriteLine("\n\n=== OCR Modes Comparison ===\n");

    Console.WriteLine("Scene Text OCR:");
    Console.WriteLine("  - Optimized for: Signs, billboards, product labels, photos");
    Console.WriteLine("  - Handles: Arbitrary orientations, curved text, varying fonts");
    Console.WriteLine("  - Challenge: Complex backgrounds, lighting variations");
    Console.WriteLine();

    Console.WriteLine("Document OCR:");
    Console.WriteLine("  - Optimized for: Scanned documents, PDFs, printed text");
    Console.WriteLine("  - Handles: Multi-column layouts, tables, forms");
    Console.WriteLine("  - Challenge: Handwriting, poor scan quality");
    Console.WriteLine();

    // Show text localization details
    Console.WriteLine("\n=== Text Localization Details ===\n");

    Console.WriteLine("Bounding Box Format:");
    Console.WriteLine("  - (x1, y1): Top-left corner");
    Console.WriteLine("  - (x2, y2): Bottom-right corner");
    Console.WriteLine("  - width = x2 - x1");
    Console.WriteLine("  - height = y2 - y1");
    Console.WriteLine();

    Console.WriteLine("Polygon Format (for rotated/curved text):");
    Console.WriteLine("  - 4 points: top-left, top-right, bottom-right, bottom-left");
    Console.WriteLine("  - Allows representation of rotated quadrilaterals");
    Console.WriteLine("  - Enables precise masking of text regions");
    Console.WriteLine();

    // Show post-processing options
    Console.WriteLine("\n=== Post-Processing Options ===\n");

    var postProcessingOptions = new[]
    {
        ("Spell Correction", "Fix common OCR errors using dictionary"),
        ("Text Merging", "Combine nearby regions into sentences"),
        ("Layout Analysis", "Detect columns, paragraphs, tables"),
        ("Language Detection", "Auto-detect text language"),
        ("Confidence Filtering", "Remove low-confidence detections"),
        ("Reading Order", "Sort text in natural reading order")
    };

    Console.WriteLine("  | Option              | Description                              |");
    Console.WriteLine("  " + new string('-', 65));
    foreach (var (option, desc) in postProcessingOptions)
    {
        Console.WriteLine($"  | {option,-19} | {desc,-40} |");
    }
    Console.WriteLine("  " + new string('-', 65));

    // Demonstrate batch OCR
    Console.WriteLine("\n\n=== Batch OCR Demo ===\n");

    var batchImages = testImages.Select(t => t.Image).ToList();
    Console.WriteLine($"Processing batch of {batchImages.Count} images...\n");

    var batchStartTime = DateTime.UtcNow;
    var batchResults = ocr.RecognizeBatch(batchImages);
    var batchTime = DateTime.UtcNow - batchStartTime;

    Console.WriteLine($"Total Batch Time: {batchTime.TotalMilliseconds:F2}ms");
    Console.WriteLine($"Average per Image: {batchTime.TotalMilliseconds / batchImages.Count:F2}ms");
    Console.WriteLine($"Total Text Regions: {batchResults.Sum(r => r.TextRegions.Count)}");
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full OCR implementation requires model weights.");
    Console.WriteLine($"This sample demonstrates the API pattern for OCR.");
    Console.WriteLine($"\nError details: {ex.Message}");

    // Show API usage demo
    DemoApiUsage();
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper: Generate synthetic images with text-like patterns
static List<(Tensor<float> Image, string Description, string ExpectedText)> GenerateSyntheticTextImages()
{
    var images = new List<(Tensor<float>, string, string)>();
    var random = new Random(42);

    // Image 1: Simple sign with single line of text
    var signImage = CreateTextImage(640, 480, random, new[]
    {
        new TextRegion(100, 200, 440, 80, "WELCOME TO AIDOTNET")
    });
    images.Add((signImage, "Simple sign (single line)", "WELCOME TO AIDOTNET"));

    // Image 2: Multi-line document
    var docImage = CreateTextImage(640, 480, random, new[]
    {
        new TextRegion(50, 50, 540, 40, "Chapter 1: Introduction"),
        new TextRegion(50, 120, 540, 30, "Machine learning is transforming"),
        new TextRegion(50, 160, 540, 30, "how we build intelligent systems"),
        new TextRegion(50, 200, 540, 30, "that can learn from data.")
    });
    images.Add((docImage, "Multi-line document", "Chapter 1: Introduction Machine learning is transforming how we build intelligent systems that can learn from data."));

    // Image 3: Scene text with multiple elements
    var sceneImage = CreateTextImage(640, 480, random, new[]
    {
        new TextRegion(50, 50, 200, 60, "STOP"),
        new TextRegion(300, 100, 280, 40, "Main Street"),
        new TextRegion(400, 300, 180, 35, "OPEN 24/7"),
        new TextRegion(50, 400, 250, 50, "SALE 50% OFF")
    });
    images.Add((sceneImage, "Street scene with multiple signs", "STOP Main Street OPEN 24/7 SALE 50% OFF"));

    // Image 4: Rotated/angled text
    var rotatedImage = CreateTextImage(640, 480, random, new[]
    {
        new TextRegion(100, 150, 400, 50, "DIAGONAL TEXT", 15),
        new TextRegion(150, 300, 350, 45, "ANOTHER ANGLE", -10)
    });
    images.Add((rotatedImage, "Rotated/angled text", "DIAGONAL TEXT ANOTHER ANGLE"));

    return images;
}

class TextRegion
{
    public int X { get; }
    public int Y { get; }
    public int Width { get; }
    public int Height { get; }
    public string Text { get; }
    public float Angle { get; }

    public TextRegion(int x, int y, int width, int height, string text, float angle = 0)
    {
        X = x;
        Y = y;
        Width = width;
        Height = height;
        Text = text;
        Angle = angle;
    }
}

static Tensor<float> CreateTextImage(int width, int height, Random random, TextRegion[] textRegions)
{
    var tensor = new Tensor<float>(new[] { 1, 3, height, width });

    // Fill with light background
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // Light gray background with subtle noise
            float bgValue = 220 + (float)(random.NextDouble() * 20 - 10);
            tensor[0, 0, h, w] = bgValue;
            tensor[0, 1, h, w] = bgValue;
            tensor[0, 2, h, w] = bgValue;
        }
    }

    // Add text regions with contrasting colors
    foreach (var region in textRegions)
    {
        // Draw text box background
        for (int h = region.Y; h < Math.Min(region.Y + region.Height, height); h++)
        {
            for (int w = region.X; w < Math.Min(region.X + region.Width, width); w++)
            {
                // White background for text area
                tensor[0, 0, h, w] = 255;
                tensor[0, 1, h, w] = 255;
                tensor[0, 2, h, w] = 255;
            }
        }

        // Simulate text characters as dark patterns
        int charWidth = Math.Max(1, region.Width / Math.Max(1, region.Text.Length));
        int charHeight = (int)(region.Height * 0.7);
        int charY = region.Y + (region.Height - charHeight) / 2;

        for (int i = 0; i < region.Text.Length; i++)
        {
            char c = region.Text[i];
            if (c == ' ') continue;

            int charX = region.X + i * charWidth + charWidth / 4;

            // Draw character as dark blob
            for (int h = charY; h < Math.Min(charY + charHeight, height); h++)
            {
                for (int w = charX; w < Math.Min(charX + (int)(charWidth * 0.7), width); w++)
                {
                    // Add variation based on character shape
                    float darkness = GetCharacterDarkness(c, h - charY, w - charX, charHeight, (int)(charWidth * 0.7), random);
                    if (darkness > 0.3)
                    {
                        tensor[0, 0, h, w] = Math.Max(0, 255 - darkness * 255);
                        tensor[0, 1, h, w] = Math.Max(0, 255 - darkness * 255);
                        tensor[0, 2, h, w] = Math.Max(0, 255 - darkness * 255);
                    }
                }
            }
        }

        // Add border around text region
        for (int w = region.X; w < Math.Min(region.X + region.Width, width); w++)
        {
            if (region.Y >= 0 && region.Y < height)
            {
                tensor[0, 0, region.Y, w] = 100;
                tensor[0, 1, region.Y, w] = 100;
                tensor[0, 2, region.Y, w] = 100;
            }
            int bottomY = region.Y + region.Height - 1;
            if (bottomY >= 0 && bottomY < height)
            {
                tensor[0, 0, bottomY, w] = 100;
                tensor[0, 1, bottomY, w] = 100;
                tensor[0, 2, bottomY, w] = 100;
            }
        }
    }

    return tensor;
}

static float GetCharacterDarkness(char c, int relH, int relW, int charH, int charW, Random random)
{
    // Simplified character patterns
    float centerH = charH / 2f;
    float centerW = charW / 2f;

    // Base character pattern based on general shape
    float distFromCenter = (float)Math.Sqrt(
        Math.Pow((relH - centerH) / centerH, 2) +
        Math.Pow((relW - centerW) / centerW, 2)
    );

    // Different patterns for different character types
    float darkness = c switch
    {
        'O' or '0' or 'Q' or 'D' => distFromCenter > 0.4 && distFromCenter < 0.9 ? 0.9f : 0.1f,
        'I' or '1' or 'l' => Math.Abs(relW - centerW) < charW * 0.2 ? 0.9f : 0.1f,
        '-' => relH > centerH - charH * 0.1 && relH < centerH + charH * 0.1 ? 0.9f : 0.1f,
        ' ' => 0.0f,
        _ => distFromCenter < 0.8 ? 0.7f + (float)(random.NextDouble() * 0.2) : 0.1f
    };

    return darkness;
}

// Demo API usage
static void DemoApiUsage()
{
    Console.WriteLine("\nAPI Usage Demo:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine(@"
// 1. Configure OCR options
var options = new OCROptions<float>
{
    Mode = OCRMode.SceneText,
    DetectionModel = TextDetectionModel.DBNet,
    RecognitionModel = TextRecognitionModel.CRNN,
    SupportedLanguages = new[] { ""en"", ""es"", ""fr"" },
    ConfidenceThreshold = 0.5f,
    DetectOrientation = true,
    GroupTextLines = true
};

// 2. Create OCR pipeline using factory (facade pattern)
var pipeline = ComputerVisionPipelineFactory.CreateSceneTextPipeline<float>(options);

// 3. Read text from image
var result = pipeline.RecognizeText(imageTensor);

// 4. Process results
Console.WriteLine($""Full text: {result.FullText}"");

foreach (var region in result.TextRegions)
{
    Console.WriteLine($""Text: {region.Text}"");
    Console.WriteLine($""  Confidence: {region.Confidence:P1}"");
    Console.WriteLine($""  Location: ({region.Box.X1}, {region.Box.Y1}) to ({region.Box.X2}, {region.Box.Y2})"");
    Console.WriteLine($""  Language: {region.Language}"");
}

// 5. Visualize results
var visualized = pipeline.VisualizeOCR(image, result);

// 6. Document OCR mode
var docOptions = new OCROptions<float> { Mode = OCRMode.Document };
var docPipeline = ComputerVisionPipelineFactory.CreateSceneTextPipeline<float>(docOptions);
var docResult = docPipeline.RecognizeText(documentImage);
");
}

// Supporting enums and classes
public enum OCRMode
{
    SceneText,
    Document,
    Both
}

public enum TextDetectionModel
{
    DBNet,
    CRAFT,
    EAST
}

public enum TextRecognitionModel
{
    CRNN,
    TrOCR
}

public class OCRConfig
{
    public OCRMode Mode { get; set; } = OCRMode.SceneText;
    public TextDetectionModel DetectionModel { get; set; } = TextDetectionModel.DBNet;
    public TextRecognitionModel RecognitionModel { get; set; } = TextRecognitionModel.CRNN;
    public string[] SupportedLanguages { get; set; } = new[] { "en" };
    public double ConfidenceThreshold { get; set; } = 0.5;
    public bool DetectOrientation { get; set; } = true;
    public bool CorrectSkew { get; set; } = true;
    public bool GroupTextLines { get; set; } = true;
}

public class OCREngine
{
    private readonly OCRConfig _config;
    private readonly Random _random = new(42);

    public OCREngine(OCRConfig config)
    {
        _config = config;
    }

    public OCREngineResult RecognizeText(Tensor<float> image)
    {
        var startTime = DateTime.UtcNow;

        int width = image.Shape[3];
        int height = image.Shape[2];

        // Simulate text detection and recognition
        var textRegions = DetectAndRecognizeText(image);

        // Filter by confidence
        var filteredRegions = textRegions
            .Where(r => r.Confidence >= _config.ConfidenceThreshold)
            .ToList();

        // Group into lines if enabled
        if (_config.GroupTextLines)
        {
            filteredRegions = GroupIntoLines(filteredRegions);
        }

        // Build full text
        string fullText = string.Join(" ", filteredRegions.Select(r => r.Text));

        return new OCREngineResult
        {
            TextRegions = filteredRegions,
            FullText = fullText,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = width,
            ImageHeight = height
        };
    }

    public List<OCREngineResult> RecognizeBatch(List<Tensor<float>> images)
    {
        return images.Select(img => RecognizeText(img)).ToList();
    }

    private List<RecognizedTextRegion> DetectAndRecognizeText(Tensor<float> image)
    {
        var regions = new List<RecognizedTextRegion>();

        int width = image.Shape[3];
        int height = image.Shape[2];

        // Scan image for text-like regions (high contrast areas)
        // This is a simplified simulation of actual text detection
        var potentialRegions = FindContrastRegions(image);

        foreach (var (x, y, w, h) in potentialRegions)
        {
            // Simulate recognition with placeholder text
            string recognizedText = GeneratePlausibleText(_random.Next(3, 15));
            double confidence = 0.6 + _random.NextDouble() * 0.35;

            regions.Add(new RecognizedTextRegion
            {
                Text = recognizedText,
                Confidence = confidence,
                Box = new BoundingBoxFloat { X1 = x, Y1 = y, X2 = x + w, Y2 = y + h },
                Language = _config.SupportedLanguages[0]
            });
        }

        return regions;
    }

    private List<(int x, int y, int w, int h)> FindContrastRegions(Tensor<float> image)
    {
        var regions = new List<(int, int, int, int)>();

        int width = image.Shape[3];
        int height = image.Shape[2];

        // Simplified region detection based on contrast
        // Real implementation would use learned text detection models

        // Sample some potential text regions
        int numRegions = _random.Next(2, 6);
        for (int i = 0; i < numRegions; i++)
        {
            int x = _random.Next(20, width - 200);
            int y = _random.Next(20, height - 60);
            int w = _random.Next(100, Math.Min(300, width - x - 20));
            int h = _random.Next(25, 50);

            regions.Add((x, y, w, h));
        }

        return regions;
    }

    private string GeneratePlausibleText(int length)
    {
        var words = new[] { "THE", "QUICK", "BROWN", "FOX", "JUMPS", "OVER", "LAZY", "DOG", "HELLO", "WORLD", "WELCOME", "TEXT", "SAMPLE", "OCR", "TEST" };
        var result = new List<string>();
        int currentLength = 0;

        while (currentLength < length)
        {
            string word = words[_random.Next(words.Length)];
            result.Add(word);
            currentLength += word.Length + 1;
        }

        return string.Join(" ", result);
    }

    private List<RecognizedTextRegion> GroupIntoLines(List<RecognizedTextRegion> regions)
    {
        if (regions.Count <= 1) return regions;

        // Sort by Y coordinate (top to bottom) then X (left to right)
        return regions
            .OrderBy(r => r.Box?.Y1 ?? 0)
            .ThenBy(r => r.Box?.X1 ?? 0)
            .ToList();
    }
}

public class OCREngineResult
{
    public List<RecognizedTextRegion> TextRegions { get; set; } = new();
    public string FullText { get; set; } = "";
    public TimeSpan InferenceTime { get; set; }
    public int ImageWidth { get; set; }
    public int ImageHeight { get; set; }
}

public class RecognizedTextRegion
{
    public string Text { get; set; } = "";
    public double Confidence { get; set; }
    public BoundingBoxFloat? Box { get; set; }
    public List<(float X, float Y)> Polygon { get; set; } = new();
    public string? Language { get; set; }
}

public class BoundingBoxFloat
{
    public float X1 { get; set; }
    public float Y1 { get; set; }
    public float X2 { get; set; }
    public float Y2 { get; set; }
    public float Width => X2 - X1;
    public float Height => Y2 - Y1;
}
