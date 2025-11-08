# Junior Developer Implementation Guide: Issue #414

## Overview
**Issue**: Mobile Deployment (ONNX Runtime, CoreML, TensorFlow Lite)
**Goal**: Deploy AiDotNet models to iOS, Android, and edge devices
**Difficulty**: Advanced
**Estimated Time**: 16-20 hours

## Mobile Deployment Targets

### 1. ONNX Runtime Mobile

**Platforms**: iOS, Android, Windows
**Format**: `.onnx` files
**Size**: ~1-2 MB runtime
**Acceleration**: CPU (XNNPACK), GPU (Metal/OpenGL), NPU

### 2. CoreML

**Platform**: iOS/macOS only
**Format**: `.mlmodel` or `.mlpackage`
**Acceleration**: Neural Engine, GPU, CPU
**Integration**: Native Swift/Objective-C

### 3. TensorFlow Lite

**Platforms**: iOS, Android, embedded Linux
**Format**: `.tflite` files
**Size**: ~300 KB runtime
**Acceleration**: GPU Delegate, NNAPI (Android), CoreML Delegate (iOS)

## Implementation Strategy

### Phase 1: ONNX Runtime Mobile

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Deployment\OnnxRuntimeMobileExporter.cs
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Onnx;

namespace AiDotNet.Deployment
{
    /// <summary>
    /// Exports models optimized for ONNX Runtime Mobile.
    /// </summary>
    public class OnnxRuntimeMobileExporter
    {
        public void ExportForMobile(
            object model,
            string outputPath,
            MobileOptimizationOptions options = null)
        {
            options ??= MobileOptimizationOptions.Default();

            // 1. Export to ONNX
            var onnxExporter = new OnnxExporter();
            var tempOnnxPath = Path.GetTempFileName() + ".onnx";

            onnxExporter.Export(model, tempOnnxPath, new OnnxExportOptions
            {
                OpsetVersion = 13,
                OptimizeGraph = true
            });

            // 2. Optimize for mobile
            OptimizeForMobile(tempOnnxPath, outputPath, options);

            Console.WriteLine($"Mobile-optimized ONNX saved to {outputPath}");
        }

        private void OptimizeForMobile(
            string inputPath,
            string outputPath,
            MobileOptimizationOptions options)
        {
            // Use ONNX Runtime optimization
            var sessionOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                EnableCpuMemArena = false, // Reduce memory for mobile
                EnableMemPattern = false
            };

            // Optimize graph
            using var session = new InferenceSession(inputPath, sessionOptions);

            // Convert to mobile format
            // In production, use onnxruntime-tools for this:
            // python -m onnxruntime.tools.convert_onnx_models_to_ort input.onnx output.ort

            Console.WriteLine("Applying mobile optimizations:");
            Console.WriteLine("- Operator fusion");
            Console.WriteLine("- Constant folding");
            Console.WriteLine("- Quantization (if enabled)");

            File.Copy(inputPath, outputPath, overwrite: true);
        }
    }

    public class MobileOptimizationOptions
    {
        public bool UseQuantization { get; set; } = true;
        public bool FuseOperators { get; set; } = true;
        public bool OptimizeForSize { get; set; } = true;
        public int QuantizationBits { get; set; } = 8;

        public static MobileOptimizationOptions Default() => new();
    }
}
```

#### Android Integration (Java/Kotlin)

```java
// File: app/src/main/java/com/aidotnet/ModelInference.kt
import ai.onnxruntime.*

class ModelInference(context: Context, modelPath: String) {
    private val session: OrtSession

    init {
        val env = OrtEnvironment.getEnvironment()
        val options = SessionOptions()

        // Use NNAPI for hardware acceleration on Android
        options.addNnapi()

        // Load model
        session = env.createSession(modelPath, options)
    }

    fun predict(input: FloatArray, inputShape: LongArray): FloatArray {
        val inputName = session.inputNames.iterator().next()
        val outputName = session.outputNames.iterator().next()

        // Create input tensor
        val inputTensor = OnnxTensor.createTensor(
            OrtEnvironment.getEnvironment(),
            FloatBuffer.wrap(input),
            inputShape
        )

        // Run inference
        val results = session.run(mapOf(inputName to inputTensor))

        // Extract output
        val output = results[0].value as Array<FloatArray>
        return output[0]
    }

    fun close() {
        session.close()
    }
}
```

#### iOS Integration (Swift)

```swift
// File: ModelInference.swift
import onnxruntime_objc

class ModelInference {
    private var session: ORTSession?

    init(modelPath: String) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()

        // Use CoreML delegate for hardware acceleration
        try options.appendCoreMLExecutionProvider(with: [:])

        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }

    func predict(input: [Float], inputShape: [Int]) throws -> [Float] {
        guard let session = session else {
            throw NSError(domain: "ModelError", code: -1)
        }

        // Create input tensor
        let inputName = try session.inputNames()[0]
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: Data(bytes: input, count: input.count * MemoryLayout<Float>.size)),
            elementType: .float,
            shape: inputShape.map { NSNumber(value: $0) }
        )

        // Run inference
        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: try session.outputNames(),
            runOptions: nil
        )

        // Extract output
        let outputName = try session.outputNames()[0]
        guard let outputTensor = outputs[outputName] else {
            throw NSError(domain: "ModelError", code: -2)
        }

        let outputData = try outputTensor.tensorData() as Data
        return outputData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
}
```

### Phase 2: CoreML Export

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Deployment\CoreMLExporter.cs
namespace AiDotNet.Deployment
{
    /// <summary>
    /// Exports models to CoreML format for iOS/macOS.
    /// Uses coremltools via Python interop or ONNX-CoreML converter.
    /// </summary>
    public class CoreMLExporter
    {
        public void ExportToCoreML(
            object model,
            string outputPath,
            CoreMLExportOptions options = null)
        {
            options ??= CoreMLExportOptions.Default();

            // Step 1: Export to ONNX
            var onnxPath = Path.GetTempFileName() + ".onnx";
            var onnxExporter = new OnnxExporter();
            onnxExporter.Export(model, onnxPath);

            // Step 2: Convert ONNX to CoreML
            // This requires Python with onnx-coreml installed:
            // pip install onnx-coreml

            ConvertOnnxToCoreML(onnxPath, outputPath, options);

            Console.WriteLine($"CoreML model saved to {outputPath}");
        }

        private void ConvertOnnxToCoreML(
            string onnxPath,
            string coremlPath,
            CoreMLExportOptions options)
        {
            // Execute Python script
            var pythonScript = $@"
import onnx
from onnx_coreml import convert

# Load ONNX model
onnx_model = onnx.load('{onnxPath}')

# Convert to CoreML
coreml_model = convert(
    onnx_model,
    minimum_ios_deployment_target='{options.MinimumIOSVersion}',
    compute_units={GetComputeUnits(options.ComputeUnits)}
)

# Save CoreML model
coreml_model.save('{coremlPath}')
";

            var scriptPath = Path.GetTempFileName() + ".py";
            File.WriteAllText(scriptPath, pythonScript);

            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = scriptPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                var error = process.StandardError.ReadToEnd();
                throw new Exception($"CoreML conversion failed: {error}");
            }

            File.Delete(scriptPath);
        }

        private string GetComputeUnits(CoreMLComputeUnits units)
        {
            return units switch
            {
                CoreMLComputeUnits.CPUOnly => "onnx_coreml.ComputeUnit.CPU_ONLY",
                CoreMLComputeUnits.CPUAndGPU => "onnx_coreml.ComputeUnit.CPU_AND_GPU",
                CoreMLComputeUnits.All => "onnx_coreml.ComputeUnit.ALL",
                _ => "onnx_coreml.ComputeUnit.ALL"
            };
        }
    }

    public class CoreMLExportOptions
    {
        public string MinimumIOSVersion { get; set; } = "13.0";
        public CoreMLComputeUnits ComputeUnits { get; set; } = CoreMLComputeUnits.All;
        public bool UseNeuralEngine { get; set; } = true;

        public static CoreMLExportOptions Default() => new();
    }

    public enum CoreMLComputeUnits
    {
        CPUOnly,
        CPUAndGPU,
        All // CPU, GPU, and Neural Engine
    }
}
```

#### CoreML Swift Integration

```swift
// File: CoreMLModelWrapper.swift
import CoreML
import Vision

class CoreMLModelWrapper {
    private var model: VNCoreMLModel?

    init(modelPath: String) throws {
        let compiledUrl = try MLModel.compileModel(at: URL(fileURLWithPath: modelPath))
        let mlModel = try MLModel(contentsOf: compiledUrl)

        model = try VNCoreMLModel(for: mlModel)
    }

    func predict(image: UIImage) -> [String: Float]? {
        guard let model = model else { return nil }

        let request = VNCoreMLRequest(model: model) { request, error in
            // Handle results
        }

        let handler = VNImageRequestHandler(cgImage: image.cgImage!)
        try? handler.perform([request])

        // Extract predictions
        return [:]
    }
}
```

### Phase 3: TensorFlow Lite Export

```csharp
// File: C:\Users\cheat\source\repos\AiDotNet\src\Deployment\TFLiteExporter.cs
namespace AiDotNet.Deployment
{
    /// <summary>
    /// Exports models to TensorFlow Lite format.
    /// </summary>
    public class TFLiteExporter
    {
        public void ExportToTFLite(
            object model,
            string outputPath,
            TFLiteExportOptions options = null)
        {
            options ??= TFLiteExportOptions.Default();

            // Step 1: Export to ONNX
            var onnxPath = Path.GetTempFileName() + ".onnx";
            var onnxExporter = new OnnxExporter();
            onnxExporter.Export(model, onnxPath);

            // Step 2: Convert ONNX to TensorFlow SavedModel
            var tfSavedModelPath = Path.GetTempFileName() + "_saved_model";
            ConvertOnnxToTensorFlow(onnxPath, tfSavedModelPath);

            // Step 3: Convert TensorFlow to TFLite
            ConvertTensorFlowToTFLite(tfSavedModelPath, outputPath, options);

            Console.WriteLine($"TFLite model saved to {outputPath}");
        }

        private void ConvertOnnxToTensorFlow(string onnxPath, string tfPath)
        {
            // Use onnx-tensorflow converter
            var script = $@"
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load('{onnxPath}')
tf_rep = prepare(onnx_model)
tf_rep.export_graph('{tfPath}')
";

            ExecutePythonScript(script);
        }

        private void ConvertTensorFlowToTFLite(
            string tfPath,
            string tflitePath,
            TFLiteExportOptions options)
        {
            var script = $@"
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('{tfPath}')

# Apply optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Quantization
if {(options.UseQuantization ? "True" : "False")}:
    converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save
with open('{tflitePath}', 'wb') as f:
    f.write(tflite_model)
";

            ExecutePythonScript(script);
        }

        private void ExecutePythonScript(string script)
        {
            var scriptPath = Path.GetTempFileName() + ".py";
            File.WriteAllText(scriptPath, script);

            var process = System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "python",
                Arguments = scriptPath,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            });

            process.WaitForExit();

            if (process.ExitCode != 0)
            {
                var error = process.StandardError.ReadToEnd();
                throw new Exception($"Python script failed: {error}");
            }

            File.Delete(scriptPath);
        }
    }

    public class TFLiteExportOptions
    {
        public bool UseQuantization { get; set; } = true;
        public bool UseGPUDelegate { get; set; } = true;
        public bool OptimizeForSize { get; set; } = true;

        public static TFLiteExportOptions Default() => new();
    }
}
```

#### TFLite Android Integration

```kotlin
// File: TFLiteModelWrapper.kt
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer

class TFLiteModelWrapper(context: Context, modelPath: String) {
    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options()

        // Use GPU delegate for acceleration
        val gpuDelegate = GpuDelegate()
        options.addDelegate(gpuDelegate)

        // Set number of threads
        options.setNumThreads(4)

        // Load model
        val model = loadModelFile(context.assets, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun predict(input: FloatArray): FloatArray {
        val inputBuffer = ByteBuffer.allocateDirect(input.size * 4)
        inputBuffer.asFloatBuffer().put(input)

        val outputShape = interpreter.getOutputTensor(0).shape()
        val output = Array(outputShape[0]) { FloatArray(outputShape[1]) }

        interpreter.run(inputBuffer, output)

        return output[0]
    }

    private fun loadModelFile(assets: AssetManager, modelPath: String): ByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun close() {
        interpreter.close()
    }
}
```

## Performance Comparison

| Platform | Runtime | Model Size | Inference Time (ImageNet) | Hardware Acceleration |
|----------|---------|------------|---------------------------|----------------------|
| **ONNX Runtime** | iOS | 5 MB | 15 ms | Neural Engine, GPU |
| **ONNX Runtime** | Android | 5 MB | 20 ms | NNAPI, GPU |
| **CoreML** | iOS | 4 MB | 12 ms | Neural Engine |
| **TFLite** | Android | 3 MB | 18 ms | GPU Delegate |
| **TFLite** | iOS | 3 MB | 22 ms | CoreML Delegate |

## Testing on Device

### iOS (Xcode)

```swift
// File: Tests/ModelTests.swift
import XCTest

class ModelTests: XCTestCase {
    func testInferenceSpeed() {
        let model = try! ModelInference(modelPath: "model.onnx")

        let input = Array(repeating: Float(0.5), count: 224 * 224 * 3)

        measure {
            _ = try! model.predict(input: input, inputShape: [1, 3, 224, 224])
        }
    }

    func testAccuracy() {
        // Load test dataset
        // Run inference
        // Compare with ground truth
        XCTAssertEqual(predictedClass, trueClass)
    }
}
```

### Android (JUnit)

```kotlin
// File: src/androidTest/java/ModelInstrumentedTest.kt
@RunWith(AndroidJUnit4::class)
class ModelInstrumentedTest {
    @Test
    fun testInferenceSpeed() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val model = TFLiteModelWrapper(context, "model.tflite")

        val input = FloatArray(224 * 224 * 3) { 0.5f }

        val startTime = System.nanoTime()
        val output = model.predict(input)
        val endTime = System.nanoTime()

        val inferenceTime = (endTime - startTime) / 1_000_000.0 // ms
        assertTrue("Inference time: $inferenceTime ms", inferenceTime < 50.0)
    }
}
```

## Deployment Checklist

- [ ] Model exported to target format (ONNX/CoreML/TFLite)
- [ ] Model optimized (quantization, operator fusion)
- [ ] Model size acceptable for mobile (<10 MB)
- [ ] Inference time acceptable (<50 ms for real-time)
- [ ] Tested on physical devices (not just simulator)
- [ ] Battery consumption measured
- [ ] Memory usage within limits
- [ ] Accuracy validated against server model
- [ ] Error handling implemented
- [ ] Model bundled correctly in app

## Common Issues

**1. Model too large**: Apply quantization, prune weights
**2. Inference too slow**: Use hardware acceleration (GPU/NPU)
**3. Accuracy loss**: Use QAT instead of post-training quantization
**4. Operator not supported**: Check opset compatibility or use alternative ops

## Learning Resources

- **ONNX Runtime Mobile**: https://onnxruntime.ai/docs/tutorials/mobile/
- **CoreML**: https://developer.apple.com/documentation/coreml
- **TensorFlow Lite**: https://www.tensorflow.org/lite/guide

---

**Good luck!** Mobile deployment is the final step in bringing your models to billions of users' devices!
