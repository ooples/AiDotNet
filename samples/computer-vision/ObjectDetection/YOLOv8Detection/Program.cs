// AiDotNet — YOLOv8 Object Detection
//
// Object detection inference through the AiModelBuilder facade. A YOLOv8 detector
// is configured via ConfigureModel and BuildAsync returns an AiModelResult; a
// forward pass on an image flows through result.Predict(image), returning the raw
// detection tensor. (A small, untrained model is used so the sample runs anywhere
// without downloading pretrained weights.)

using AiDotNet;
using AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;   // YOLOv8<T>
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet YOLOv8 Object Detection ===\n");

// ── Configure the detector ─────────────────────────────────────────────────
var options = new ObjectDetectionOptions<float>
{
    Architecture = DetectionArchitecture.YOLOv8,
    Size = ModelSize.Nano,
    NumClasses = 80,                 // COCO
    ConfidenceThreshold = 0.25,
    NmsThreshold = 0.45,
    InputSize = new[] { 320, 320 },
    UsePretrained = false
};

Console.WriteLine("Configuration:");
Console.WriteLine($"  Architecture: {options.Architecture}");
Console.WriteLine($"  Model size:   {options.Size}");
Console.WriteLine($"  Input size:   {options.InputSize[0]}x{options.InputSize[1]}");
Console.WriteLine($"  Classes:      {options.NumClasses} (COCO)\n");

// Synthetic image batch: [batch, channels, height, width].
var rng = new Random(42);
var image = new Tensor<float>(new[] { 1, 3, options.InputSize[1], options.InputSize[0] });
for (int c = 0; c < 3; c++)
    for (int h = 0; h < options.InputSize[1]; h++)
        for (int w = 0; w < options.InputSize[0]; w++)
            image[new[] { 0, c, h, w }] = (float)rng.NextDouble();

// The facade fits the detector on a labeled image pipeline; here we attach a
// single-image pipeline so BuildAsync produces a ready detector, then run a
// forward pass through result.Predict. Output is the raw detection grid which a
// post-process step decodes into boxes using the confidence + NMS thresholds.
Console.WriteLine("Building the detector through AiModelBuilder.ConfigureModel ...");
try
{
    var detector = new YOLOv8<float>(options);
    var probe = detector.Predict(image);   // shape of the detection grid for the dummy targets

    var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
        .ConfigureModel(detector)
        .ConfigureDataLoader(AiDotNet.Data.Loaders.DataLoaders.FromTensors(image, probe))
        .BuildAsync();

    Console.WriteLine("  Detector ready.\n");
    Console.WriteLine("Running detection (forward pass) via result.Predict ...");
    var output = result.Predict(image);
    Console.WriteLine($"  Raw detection output shape: [{string.Join(", ", output.Shape)}]");
    Console.WriteLine("  (decode this grid with confidence + NMS thresholds to get boxes)\n");
}
catch (Exception ex)
{
    Console.WriteLine($"  Detection reported: {ex.Message}\n");
}

Console.WriteLine("YOLOv8 detects 80 COCO categories — person, bicycle, car, dog, cat,");
Console.WriteLine("bottle, chair, laptop, cell phone, and more.");
Console.WriteLine("\n=== Sample Complete ===");
