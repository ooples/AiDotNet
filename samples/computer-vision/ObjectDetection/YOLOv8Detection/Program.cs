using AiDotNet;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

Console.WriteLine("=== AiDotNet YOLOv8 Object Detection ===");
Console.WriteLine("Real-time object detection with bounding boxes\n");

// Display available model sizes
Console.WriteLine("Available YOLOv8 model sizes:");
Console.WriteLine("  Nano   - 3.2M params  - Fastest inference, mobile deployment");
Console.WriteLine("  Small  - 11.2M params - Fast with good accuracy");
Console.WriteLine("  Medium - 25.9M params - Balanced speed and accuracy (recommended)");
Console.WriteLine("  Large  - 43.7M params - High accuracy, slower inference");
Console.WriteLine("  XLarge - 68.2M params - Highest accuracy, production use");
Console.WriteLine();

try
{
    // Configure object detection options
    var options = new ObjectDetectionOptions<float>
    {
        Architecture = DetectionArchitecture.YOLOv8,
        Size = ModelSize.Medium,
        NumClasses = 80,  // COCO dataset classes
        ConfidenceThreshold = 0.25,  // Minimum confidence to keep detection
        NmsThreshold = 0.45,  // IoU threshold for Non-Maximum Suppression
        InputSize = new[] { 640, 640 },  // Standard YOLO input size
        UsePretrained = true
    };

    Console.WriteLine("Configuration:");
    Console.WriteLine($"  Architecture: {options.Architecture}");
    Console.WriteLine($"  Model Size: {options.Size}");
    Console.WriteLine($"  Input Size: {options.InputSize[0]}x{options.InputSize[1]}");
    Console.WriteLine($"  Confidence Threshold: {options.ConfidenceThreshold}");
    Console.WriteLine($"  NMS Threshold: {options.NmsThreshold}");
    Console.WriteLine($"  Number of Classes: {options.NumClasses}");
    Console.WriteLine();

    // Create the detection pipeline using the facade pattern
    Console.WriteLine("Creating YOLOv8 detection pipeline...");
    var pipeline = ComputerVisionPipelineFactory.CreateYOLOv8Pipeline<float>(options);

    Console.WriteLine("  Pipeline created successfully\n");

    // Generate synthetic test images (in production, load real images)
    Console.WriteLine("Generating synthetic test images...");
    var testImages = GenerateSyntheticImages();

    // Process each test image
    for (int i = 0; i < testImages.Count; i++)
    {
        var (image, description) = testImages[i];
        Console.WriteLine($"\nProcessing Image {i + 1}: {description}");
        Console.WriteLine(new string('-', 50));

        // Perform detection
        var result = pipeline.DetectObjects(image);

        // Display results
        Console.WriteLine($"  Image Size: {result.ImageWidth}x{result.ImageHeight}");
        Console.WriteLine($"  Inference Time: {result.InferenceTime.TotalMilliseconds:F2}ms");
        Console.WriteLine($"  Objects Detected: {result.Detections.Count}");

        if (result.Detections.Count > 0)
        {
            Console.WriteLine("\n  Detected Objects:");
            Console.WriteLine("  " + new string('-', 60));
            Console.WriteLine("  | Class               | Confidence | Bounding Box (x1,y1,x2,y2) |");
            Console.WriteLine("  " + new string('-', 60));

            foreach (var detection in result.Detections.OrderByDescending(d => d.Confidence))
            {
                var box = detection.BoundingBox;
                string className = GetClassName(detection.ClassId);
                string boxStr = $"({box.X1:F0},{box.Y1:F0},{box.X2:F0},{box.Y2:F0})";
                Console.WriteLine($"  | {className,-19} | {detection.Confidence:P1}     | {boxStr,-26} |");
            }
            Console.WriteLine("  " + new string('-', 60));
        }
    }

    // Demonstrate batch detection
    Console.WriteLine("\n\n=== Batch Detection Demo ===");
    Console.WriteLine("Processing multiple images in a single batch...\n");

    var batchTensor = CreateBatchTensor(testImages.Select(t => t.Image).ToList());
    Console.WriteLine($"Batch Size: {batchTensor.Shape[0]} images");

    if (pipeline.ObjectDetector != null)
    {
        var batchResult = pipeline.ObjectDetector.DetectBatch(batchTensor);
        Console.WriteLine($"Total Batch Inference Time: {batchResult.TotalInferenceTime.TotalMilliseconds:F2}ms");
        Console.WriteLine($"Average per Image: {batchResult.TotalInferenceTime.TotalMilliseconds / batchResult.Results.Count:F2}ms");

        int totalDetections = batchResult.Results.Sum(r => r.Detections.Count);
        Console.WriteLine($"Total Objects Detected: {totalDetections}");
    }

    // Demonstrate confidence threshold tuning
    Console.WriteLine("\n\n=== Confidence Threshold Tuning ===");
    Console.WriteLine("Effect of different confidence thresholds:\n");

    var sampleImage = testImages[0].Image;
    double[] thresholds = { 0.1, 0.25, 0.5, 0.75, 0.9 };

    Console.WriteLine("  | Threshold | Detections | Notes                          |");
    Console.WriteLine("  " + new string('-', 60));

    foreach (var threshold in thresholds)
    {
        if (pipeline.ObjectDetector != null)
        {
            var thresholdResult = pipeline.ObjectDetector.Detect(sampleImage, threshold, options.NmsThreshold);
            string notes = threshold switch
            {
                0.1 => "More detections, higher false positives",
                0.25 => "Balanced (recommended starting point)",
                0.5 => "Fewer but more confident detections",
                0.75 => "Only high-confidence detections",
                0.9 => "Very few, very confident detections",
                _ => ""
            };
            Console.WriteLine($"  | {threshold:P0}      | {thresholdResult.Detections.Count,10} | {notes,-30} |");
        }
    }
    Console.WriteLine("  " + new string('-', 60));

    // Show COCO classes
    Console.WriteLine("\n\n=== COCO Dataset Classes (80 classes) ===");
    Console.WriteLine("YOLOv8 can detect the following object categories:\n");

    var cocoClasses = GetCocoClassNames();
    for (int i = 0; i < cocoClasses.Length; i += 5)
    {
        var row = cocoClasses.Skip(i).Take(5).Select((c, idx) => $"{i + idx,2}: {c,-14}");
        Console.WriteLine("  " + string.Join(" | ", row));
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full YOLOv8 implementation requires model weights.");
    Console.WriteLine($"This sample demonstrates the API pattern for object detection.");
    Console.WriteLine($"\nError details: {ex.Message}");

    // Show API usage demo
    DemoApiUsage();
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper: Generate synthetic test images
static List<(Tensor<float> Image, string Description)> GenerateSyntheticImages()
{
    var images = new List<(Tensor<float>, string)>();
    var random = new Random(42);

    // Image 1: Simple scene with objects at different positions
    var image1 = CreateSyntheticImage(640, 640, random, new[]
    {
        (100, 100, 200, 200, "person-like"),     // Person-shaped blob
        (300, 150, 450, 350, "car-like"),        // Car-shaped rectangle
        (500, 400, 600, 550, "dog-like")         // Small animal shape
    });
    images.Add((image1, "Urban scene simulation"));

    // Image 2: Dense scene with many objects
    var image2 = CreateSyntheticImage(640, 640, random, new[]
    {
        (50, 50, 150, 250, "person-like"),
        (160, 80, 260, 280, "person-like"),
        (280, 60, 380, 260, "person-like"),
        (400, 100, 500, 300, "bicycle-like"),
        (520, 200, 620, 400, "car-like")
    });
    images.Add((image2, "Crowded street simulation"));

    // Image 3: Single object detection
    var image3 = CreateSyntheticImage(640, 640, random, new[]
    {
        (200, 150, 440, 490, "dog-like")
    });
    images.Add((image3, "Single object - pet detection"));

    return images;
}

// Helper: Create a synthetic image tensor with object-like regions
static Tensor<float> CreateSyntheticImage(int height, int width, Random random, (int x1, int y1, int x2, int y2, string type)[] objects)
{
    // Create image tensor [batch=1, channels=3, height, width]
    var tensor = new Tensor<float>(new[] { 1, 3, height, width });

    // Fill with background (varying gray levels to simulate scene)
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // Create gradient background
            float bgValue = 100 + (float)(h + w) / (height + width) * 50;
            bgValue += (float)(random.NextDouble() * 10 - 5);  // Add noise

            tensor[0, 0, h, w] = bgValue;  // R
            tensor[0, 1, h, w] = bgValue;  // G
            tensor[0, 2, h, w] = bgValue;  // B
        }
    }

    // Add object regions with distinct colors/patterns
    foreach (var (x1, y1, x2, y2, objType) in objects)
    {
        float r = 0, g = 0, b = 0;

        // Different colors for different object types
        switch (objType)
        {
            case "person-like":
                r = 200 + (float)(random.NextDouble() * 55);
                g = 150 + (float)(random.NextDouble() * 50);
                b = 150 + (float)(random.NextDouble() * 50);
                break;
            case "car-like":
                r = 100 + (float)(random.NextDouble() * 50);
                g = 100 + (float)(random.NextDouble() * 50);
                b = 200 + (float)(random.NextDouble() * 55);
                break;
            case "bicycle-like":
                r = 50 + (float)(random.NextDouble() * 30);
                g = 50 + (float)(random.NextDouble() * 30);
                b = 50 + (float)(random.NextDouble() * 30);
                break;
            case "dog-like":
                r = 180 + (float)(random.NextDouble() * 50);
                g = 140 + (float)(random.NextDouble() * 40);
                b = 100 + (float)(random.NextDouble() * 30);
                break;
            default:
                r = g = b = 128;
                break;
        }

        // Fill object region
        for (int h = y1; h < Math.Min(y2, height); h++)
        {
            for (int w = x1; w < Math.Min(x2, width); w++)
            {
                // Add some texture/variation
                float noise = (float)(random.NextDouble() * 20 - 10);
                tensor[0, 0, h, w] = Math.Clamp(r + noise, 0, 255);
                tensor[0, 1, h, w] = Math.Clamp(g + noise, 0, 255);
                tensor[0, 2, h, w] = Math.Clamp(b + noise, 0, 255);
            }
        }
    }

    return tensor;
}

// Helper: Create batch tensor from list of images
static Tensor<float> CreateBatchTensor(List<Tensor<float>> images)
{
    if (images.Count == 0)
        return new Tensor<float>(new[] { 0, 3, 640, 640 });

    int channels = images[0].Shape[1];
    int height = images[0].Shape[2];
    int width = images[0].Shape[3];

    var batch = new Tensor<float>(new[] { images.Count, channels, height, width });

    for (int b = 0; b < images.Count; b++)
    {
        var img = images[b];
        int pixelsPerImage = channels * height * width;

        for (int i = 0; i < pixelsPerImage; i++)
        {
            batch[b * pixelsPerImage + i] = img[i];
        }
    }

    return batch;
}

// Helper: Get class name from class ID
static string GetClassName(int classId)
{
    var names = GetCocoClassNames();
    return classId >= 0 && classId < names.Length ? names[classId] : $"class_{classId}";
}

// Helper: Get COCO class names
static string[] GetCocoClassNames()
{
    return new[]
    {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}

// Demo API usage when model isn't available
static void DemoApiUsage()
{
    Console.WriteLine("\nAPI Usage Demo:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine(@"
// 1. Configure detection options
var options = new ObjectDetectionOptions<float>
{
    Architecture = DetectionArchitecture.YOLOv8,
    Size = ModelSize.Medium,
    ConfidenceThreshold = 0.25,
    NmsThreshold = 0.45,
    InputSize = new[] { 640, 640 }
};

// 2. Create detection pipeline (facade pattern)
var pipeline = ComputerVisionPipelineFactory.CreateYOLOv8Pipeline<float>(options);

// 3. Load an image (as tensor)
var image = LoadImage(""photo.jpg"");  // Returns Tensor<float>

// 4. Detect objects
var result = pipeline.DetectObjects(image);

// 5. Process detections
foreach (var detection in result.Detections)
{
    Console.WriteLine($""Found: {detection.ClassName}"");
    Console.WriteLine($""  Confidence: {detection.Confidence:P1}"");
    Console.WriteLine($""  Box: ({detection.BoundingBox.X1}, {detection.BoundingBox.Y1}) "");
    Console.WriteLine($""       to ({detection.BoundingBox.X2}, {detection.BoundingBox.Y2})"");
}

// 6. Visualize detections (optional)
var visualizedImage = pipeline.VisualizeDetections(image, result);

// 7. Batch detection for multiple images
var batchResult = pipeline.ObjectDetector.DetectBatch(batchOfImages);
");
}

// Supporting classes for demonstration (simplified versions)
namespace AiDotNet.ComputerVision.Detection.ObjectDetection
{
    /// <summary>
    /// Represents a single object detection result.
    /// </summary>
    public class Detection<T>
    {
        /// <summary>Class ID of the detected object.</summary>
        public int ClassId { get; set; }

        /// <summary>Class name of the detected object.</summary>
        public string ClassName { get; set; } = "";

        /// <summary>Detection confidence score (0 to 1).</summary>
        public float Confidence { get; set; }

        /// <summary>Bounding box coordinates.</summary>
        public BoundingBox<T> BoundingBox { get; set; } = new();
    }

    /// <summary>
    /// Bounding box for detected object.
    /// </summary>
    public class BoundingBox<T>
    {
        /// <summary>Left coordinate.</summary>
        public float X1 { get; set; }

        /// <summary>Top coordinate.</summary>
        public float Y1 { get; set; }

        /// <summary>Right coordinate.</summary>
        public float X2 { get; set; }

        /// <summary>Bottom coordinate.</summary>
        public float Y2 { get; set; }

        /// <summary>Width of the box.</summary>
        public float Width => X2 - X1;

        /// <summary>Height of the box.</summary>
        public float Height => Y2 - Y1;

        /// <summary>Area of the box.</summary>
        public float Area => Width * Height;
    }

    /// <summary>
    /// Result of object detection on a single image.
    /// </summary>
    public class DetectionResult<T>
    {
        /// <summary>List of detected objects.</summary>
        public List<Detection<T>> Detections { get; set; } = new();

        /// <summary>Time taken for inference.</summary>
        public TimeSpan InferenceTime { get; set; }

        /// <summary>Width of the input image.</summary>
        public int ImageWidth { get; set; }

        /// <summary>Height of the input image.</summary>
        public int ImageHeight { get; set; }
    }

    /// <summary>
    /// Result of batch object detection.
    /// </summary>
    public class BatchDetectionResult<T>
    {
        /// <summary>Detection results for each image in the batch.</summary>
        public List<DetectionResult<T>> Results { get; set; } = new();

        /// <summary>Total time for batch inference.</summary>
        public TimeSpan TotalInferenceTime { get; set; }
    }
}
