using AiDotNet.Metrics;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Examples.ConcreteExamples;

/// <summary>
/// Comprehensive examples for 3D AI capabilities in AiDotNet.
/// Demonstrates point cloud processing, neural radiance fields, and 3D metrics.
/// </summary>
public static class ThreeDAIExamples
{
    #region Point Cloud Classification Example

    /// <summary>
    /// Example: Point cloud classification using PointNet.
    /// Demonstrates loading, preprocessing, and classifying 3D point clouds.
    /// </summary>
    /// <remarks>
    /// This example shows:
    /// - Creating synthetic point cloud data
    /// - Building a PointNet classification model
    /// - Running inference
    /// - Evaluating with accuracy metrics
    /// </remarks>
    public static void PointCloudClassificationExample()
    {
        Console.WriteLine("=== Point Cloud Classification with PointNet ===\n");

        // 1. Create synthetic point cloud data (simulating ModelNet40-style data)
        int numPoints = 1024;
        int numClasses = 10;
        int inputDim = 3; // xyz coordinates

        // Generate random point cloud (in practice, load from OBJ/PLY)
        var random = RandomHelper.CreateSeededRandom(42);
        var pointCloudData = new float[numPoints * inputDim];
        for (int i = 0; i < pointCloudData.Length; i++)
        {
            pointCloudData[i] = (float)(random.NextDouble() * 2 - 1); // [-1, 1]
        }
        var inputTensor = new Tensor<float>(new[] { 1, numPoints, inputDim }, new Vector<float>(pointCloudData));

        Console.WriteLine($"Input shape: [{string.Join(", ", inputTensor.Shape)}]");
        Console.WriteLine($"Number of points: {numPoints}");
        Console.WriteLine($"Number of classes: {numClasses}\n");

        // 2. Create PointNet model
        // Note: In production, use PointNetClassifier from LayerHelper
        Console.WriteLine("Building PointNet model...");
        Console.WriteLine("- Input transform (spatial transformer)");
        Console.WriteLine("- Shared MLPs: 64 -> 128 -> 1024");
        Console.WriteLine("- Max pooling (global feature)");
        Console.WriteLine("- FC layers: 512 -> 256 -> numClasses\n");

        // 3. Simulate inference (in practice, would use trained weights)
        Console.WriteLine("Running inference...");
        var predictions = new float[numClasses];
        predictions[3] = 0.85f; // Simulate high confidence for class 3
        for (int i = 0; i < numClasses; i++)
        {
            if (i != 3) predictions[i] = (float)random.NextDouble() * 0.1f;
        }

        // 4. Show predictions
        Console.WriteLine("\nPredicted class probabilities:");
        for (int i = 0; i < numClasses; i++)
        {
            Console.WriteLine($"  Class {i}: {predictions[i]:P1}");
        }

        int predictedClass = Array.IndexOf(predictions, predictions.Max());
        Console.WriteLine($"\nPredicted class: {predictedClass}");

        // 5. Evaluate with accuracy metric
        var accuracy = new OverallAccuracy<float>();
        var predTensor = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { (float)predictedClass }));
        var gtTensor = new Tensor<float>(new[] { 1 }, new Vector<float>(new[] { 3f })); // Ground truth is class 3
        float acc = accuracy.Compute(predTensor, gtTensor);
        Console.WriteLine($"Accuracy: {acc:P1}");

        Console.WriteLine("\n=== Point Cloud Classification Complete ===\n");
    }

    #endregion

    #region Point Cloud Segmentation Example

    /// <summary>
    /// Example: Point cloud part segmentation using PointNet++.
    /// Demonstrates semantic segmentation of 3D point clouds.
    /// </summary>
    public static void PointCloudSegmentationExample()
    {
        Console.WriteLine("=== Point Cloud Segmentation with PointNet++ ===\n");

        // 1. Create synthetic segmentation data
        int numPoints = 2048;
        int numPartClasses = 4; // e.g., wing, body, tail, engine for airplane

        var random = RandomHelper.CreateSeededRandom(42);

        // Generate point cloud with xyz + features
        var points = new float[numPoints * 3];
        var groundTruthLabels = new float[numPoints];

        for (int i = 0; i < numPoints; i++)
        {
            // Random points in [-1, 1]^3
            points[i * 3] = (float)(random.NextDouble() * 2 - 1);
            points[i * 3 + 1] = (float)(random.NextDouble() * 2 - 1);
            points[i * 3 + 2] = (float)(random.NextDouble() * 2 - 1);

            // Assign labels based on z-coordinate (simplified)
            float z = points[i * 3 + 2];
            groundTruthLabels[i] = z < -0.5f ? 0 : z < 0 ? 1 : z < 0.5f ? 2 : 3;
        }

        Console.WriteLine($"Point cloud shape: [{numPoints}, 3]");
        Console.WriteLine($"Number of part classes: {numPartClasses}\n");

        // 2. Describe PointNet++ architecture
        Console.WriteLine("PointNet++ Architecture:");
        Console.WriteLine("- Set Abstraction Layer 1: 512 points, radius=0.2");
        Console.WriteLine("- Set Abstraction Layer 2: 128 points, radius=0.4");
        Console.WriteLine("- Set Abstraction Layer 3: 32 points, radius=0.8");
        Console.WriteLine("- Feature Propagation (upsample) layers");
        Console.WriteLine("- Per-point classification head\n");

        // 3. Simulate per-point predictions
        Console.WriteLine("Running segmentation inference...");
        var predictedLabels = new float[numPoints];
        for (int i = 0; i < numPoints; i++)
        {
            // Simulate predictions (with some noise that guarantees incorrect predictions)
            if (random.NextDouble() < 0.1)
            {
                // Add noise: pick a different class than ground truth
                int wrongClass = random.Next(numPartClasses - 1);
                predictedLabels[i] = wrongClass >= groundTruthLabels[i] ? wrongClass + 1 : wrongClass;
            }
            else
            {
                predictedLabels[i] = groundTruthLabels[i];
            }
        }

        // 4. Evaluate with mIoU
        var miou = new MeanIntersectionOverUnion<float>(numPartClasses);
        var predTensor = new Tensor<float>(new[] { numPoints }, new Vector<float>(predictedLabels));
        var gtTensor = new Tensor<float>(new[] { numPoints }, new Vector<float>(groundTruthLabels));

        float miouValue = miou.Compute(predTensor, gtTensor);
        float[] perClassIoU = miou.ComputePerClass(predTensor, gtTensor);

        Console.WriteLine($"\nmIoU: {miouValue:P1}");
        Console.WriteLine("Per-class IoU:");
        string[] partNames = { "Wing", "Body", "Tail", "Engine" };
        for (int i = 0; i < numPartClasses; i++)
        {
            Console.WriteLine($"  {partNames[i]}: {perClassIoU[i]:P1}");
        }

        var accuracy = new OverallAccuracy<float>();
        float acc = accuracy.Compute(predTensor, gtTensor);
        Console.WriteLine($"Overall Accuracy: {acc:P1}");

        Console.WriteLine("\n=== Point Cloud Segmentation Complete ===\n");
    }

    #endregion

    #region Neural Radiance Fields Example

    /// <summary>
    /// Example: Novel view synthesis with Neural Radiance Fields (NeRF).
    /// Demonstrates NeRF inference pipeline with quality metrics.
    /// </summary>
    public static void NeRFRenderingExample()
    {
        Console.WriteLine("=== Neural Radiance Fields (NeRF) Rendering ===\n");

        // 1. Setup scene parameters
        int imageWidth = 64;
        int imageHeight = 64;
        int numSamplesPerRay = 64;
        int numFrequencies = 10; // For positional encoding

        Console.WriteLine("Scene Configuration:");
        Console.WriteLine($"  Image size: {imageWidth}x{imageHeight}");
        Console.WriteLine($"  Samples per ray: {numSamplesPerRay}");
        Console.WriteLine($"  Positional encoding frequencies: {numFrequencies}\n");

        // 2. Generate camera rays (simplified)
        Console.WriteLine("Generating camera rays...");
        int numRays = imageWidth * imageHeight;
        var random = RandomHelper.CreateSeededRandom(42);

        // Ray origins and directions (simplified - in practice use camera model)
        var rayOrigins = new float[numRays * 3];
        var rayDirections = new float[numRays * 3];

        for (int i = 0; i < numRays; i++)
        {
            // Camera at origin, looking along -Z
            rayOrigins[i * 3] = 0;
            rayOrigins[i * 3 + 1] = 0;
            rayOrigins[i * 3 + 2] = 0;

            // Pixel-based ray direction
            int px = i % imageWidth;
            int py = i / imageWidth;
            float x = (px - imageWidth / 2f) / imageWidth;
            float y = (py - imageHeight / 2f) / imageHeight;
            float z = -1f; // Looking into screen

            float len = (float)Math.Sqrt(x * x + y * y + z * z);
            rayDirections[i * 3] = x / len;
            rayDirections[i * 3 + 1] = y / len;
            rayDirections[i * 3 + 2] = z / len;
        }

        Console.WriteLine($"Generated {numRays} rays\n");

        // 3. Describe NeRF pipeline
        Console.WriteLine("NeRF Pipeline:");
        Console.WriteLine("1. Sample points along each ray");
        Console.WriteLine("2. Apply positional encoding (sin/cos at multiple frequencies)");
        Console.WriteLine("3. MLP network: 8 layers, 256 units each");
        Console.WriteLine("4. Output: RGB color + density (sigma)");
        Console.WriteLine("5. Volume rendering: accumulate colors along ray\n");

        // 4. Simulate rendering (in practice, use trained NeRF model)
        Console.WriteLine("Rendering image...");
        var renderedImage = new float[imageWidth * imageHeight * 3];

        // Simulate a simple sphere at origin
        for (int i = 0; i < numRays; i++)
        {
            int px = i % imageWidth;
            int py = i / imageWidth;

            // Simple sphere shading
            float cx = (px - imageWidth / 2f) / (imageWidth / 2f);
            float cy = (py - imageHeight / 2f) / (imageHeight / 2f);
            float r2 = cx * cx + cy * cy;

            if (r2 < 0.5f)
            {
                // Inside sphere - compute normal-based shading
                float z = (float)Math.Sqrt(Math.Max(0, 0.5f - r2));
                float shade = 0.3f + 0.7f * z; // Simple Lambertian

                renderedImage[i * 3] = shade * 0.8f; // R
                renderedImage[i * 3 + 1] = shade * 0.2f; // G
                renderedImage[i * 3 + 2] = shade * 0.2f; // B
            }
            else
            {
                // Background
                renderedImage[i * 3] = 0.1f;
                renderedImage[i * 3 + 1] = 0.1f;
                renderedImage[i * 3 + 2] = 0.3f;
            }
        }

        // 5. Create ground truth (perfect sphere render)
        var groundTruth = new float[imageWidth * imageHeight * 3];
        Array.Copy(renderedImage, groundTruth, renderedImage.Length);

        // Add some noise to rendered to simulate training progress
        for (int i = 0; i < renderedImage.Length; i++)
        {
            renderedImage[i] += (float)(random.NextDouble() * 0.02 - 0.01);
            renderedImage[i] = Math.Max(0, Math.Min(1, renderedImage[i]));
        }

        // 6. Evaluate rendering quality
        Console.WriteLine("\nEvaluating rendering quality...");

        var renderedTensor = new Tensor<float>(new[] { imageHeight, imageWidth, 3 }, new Vector<float>(renderedImage));
        var gtTensor = new Tensor<float>(new[] { imageHeight, imageWidth, 3 }, new Vector<float>(groundTruth));

        var psnr = new PeakSignalToNoiseRatio<float>();
        var ssim = new StructuralSimilarity<float>();

        float psnrValue = psnr.Compute(renderedTensor, gtTensor);
        float ssimValue = ssim.Compute(renderedTensor, gtTensor);

        Console.WriteLine($"PSNR: {psnrValue:F2} dB");
        Console.WriteLine($"SSIM: {ssimValue:F4}");

        // Typical NeRF paper benchmarks:
        Console.WriteLine("\nReference benchmarks (NeRF paper):");
        Console.WriteLine("  Synthetic scenes: PSNR ~31 dB, SSIM ~0.95");
        Console.WriteLine("  Real scenes: PSNR ~26 dB, SSIM ~0.81");

        Console.WriteLine("\n=== NeRF Rendering Complete ===\n");
    }

    #endregion

    #region 3D Reconstruction Evaluation Example

    /// <summary>
    /// Example: Evaluating 3D reconstruction quality using geometry metrics.
    /// Demonstrates Chamfer Distance, EMD, and F-Score computation.
    /// </summary>
    public static void ReconstructionEvaluationExample()
    {
        Console.WriteLine("=== 3D Reconstruction Quality Evaluation ===\n");

        // 1. Create ground truth point cloud (unit sphere)
        int numPointsGT = 1000;
        var random = RandomHelper.CreateSeededRandom(42);

        var gtPoints = new float[numPointsGT * 3];
        for (int i = 0; i < numPointsGT; i++)
        {
            // Uniform points on unit sphere
            float theta = (float)(random.NextDouble() * 2 * Math.PI);
            float phi = (float)Math.Acos(2 * random.NextDouble() - 1);

            gtPoints[i * 3] = (float)(Math.Sin(phi) * Math.Cos(theta));
            gtPoints[i * 3 + 1] = (float)(Math.Sin(phi) * Math.Sin(theta));
            gtPoints[i * 3 + 2] = (float)Math.Cos(phi);
        }

        // 2. Create reconstructed point cloud (with noise and missing regions)
        int numPointsPred = 900;
        var predPoints = new float[numPointsPred * 3];

        for (int i = 0; i < numPointsPred; i++)
        {
            // Copy from GT with noise
            int srcIdx = random.Next(numPointsGT);
            float noise = 0.02f;

            predPoints[i * 3] = gtPoints[srcIdx * 3] + (float)(random.NextDouble() * noise - noise / 2);
            predPoints[i * 3 + 1] = gtPoints[srcIdx * 3 + 1] + (float)(random.NextDouble() * noise - noise / 2);
            predPoints[i * 3 + 2] = gtPoints[srcIdx * 3 + 2] + (float)(random.NextDouble() * noise - noise / 2);
        }

        var gtTensor = new Tensor<float>(new[] { numPointsGT, 3 }, new Vector<float>(gtPoints));
        var predTensor = new Tensor<float>(new[] { numPointsPred, 3 }, new Vector<float>(predPoints));

        Console.WriteLine($"Ground truth points: {numPointsGT}");
        Console.WriteLine($"Reconstructed points: {numPointsPred}\n");

        // 3. Compute Chamfer Distance
        Console.WriteLine("Computing Chamfer Distance...");
        var chamfer = new ChamferDistance<float>(squared: true);
        float cdValue = chamfer.Compute(predTensor, gtTensor);
        Console.WriteLine($"Chamfer Distance (squared): {cdValue:E4}");

        float cdOneWay = chamfer.ComputeOneWay(predTensor, gtTensor);
        Console.WriteLine($"CD (pred -> GT): {cdOneWay:E4}");

        // 4. Compute F-Score at different thresholds
        Console.WriteLine("\nComputing F-Scores at different thresholds...");
        float[] thresholds = { 0.01f, 0.02f, 0.05f };

        foreach (var threshold in thresholds)
        {
            var fScore = new FScore<float>(threshold);
            var (precision, recall) = fScore.ComputePrecisionRecall(predTensor, gtTensor);
            float f = fScore.Compute(predTensor, gtTensor);

            Console.WriteLine($"τ={threshold:F2}: F-Score={f:P1}, Precision={precision:P1}, Recall={recall:P1}");
        }

        // 5. Compute approximate EMD (expensive, use subset)
        Console.WriteLine("\nComputing Earth Mover's Distance (subset)...");
        int subsetSize = 100;
        var gtSubset = new float[subsetSize * 3];
        var predSubset = new float[subsetSize * 3];

        for (int i = 0; i < subsetSize * 3; i++)
        {
            gtSubset[i] = gtPoints[i];
            predSubset[i] = predPoints[i];
        }

        var gtSubsetTensor = new Tensor<float>(new[] { subsetSize, 3 }, new Vector<float>(gtSubset));
        var predSubsetTensor = new Tensor<float>(new[] { subsetSize, 3 }, new Vector<float>(predSubset));

        var emd = new EarthMoversDistance<float>(iterations: 50);
        float emdValue = emd.Compute(predSubsetTensor, gtSubsetTensor);
        Console.WriteLine($"EMD (Sinkhorn approx): {emdValue:F6}");

        // 6. Reference benchmarks
        Console.WriteLine("\nTypical benchmarks for 3D reconstruction:");
        Console.WriteLine("  ShapeNet completion: CD ~2e-4, F1@0.01 ~0.70");
        Console.WriteLine("  Multi-view reconstruction: CD ~5e-5, F1@0.01 ~0.85");

        Console.WriteLine("\n=== Reconstruction Evaluation Complete ===\n");
    }

    #endregion

    #region Voxel-Based Detection Example

    /// <summary>
    /// Example: 3D object detection with voxel-based methods.
    /// Demonstrates 3D IoU computation for detection evaluation.
    /// </summary>
    public static void VoxelDetectionExample()
    {
        Console.WriteLine("=== Voxel-Based 3D Detection ===\n");

        // 1. Create voxel grid
        int gridSize = 32;
        var random = RandomHelper.CreateSeededRandom(42);

        Console.WriteLine($"Voxel grid size: {gridSize}³\n");

        // 2. Create ground truth occupancy (simple cube)
        var gtVoxels = new float[gridSize * gridSize * gridSize];
        var predVoxels = new float[gridSize * gridSize * gridSize];

        for (int z = 0; z < gridSize; z++)
        {
            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    int idx = z * gridSize * gridSize + y * gridSize + x;

                    // Ground truth: cube from (8,8,8) to (24,24,24)
                    if (x >= 8 && x < 24 && y >= 8 && y < 24 && z >= 8 && z < 24)
                    {
                        gtVoxels[idx] = 1;
                    }

                    // Prediction: slightly offset and noisy
                    if (x >= 9 && x < 25 && y >= 9 && y < 25 && z >= 9 && z < 25)
                    {
                        // Add some noise
                        predVoxels[idx] = random.NextDouble() > 0.1 ? 1 : 0;
                    }
                }
            }
        }

        var gtTensor = new Tensor<float>(new[] { gridSize, gridSize, gridSize }, new Vector<float>(gtVoxels));
        var predTensor = new Tensor<float>(new[] { gridSize, gridSize, gridSize }, new Vector<float>(predVoxels));

        // 3. Compute voxel IoU
        var iou3d = new IoU3D<float>();
        float voxelIoU = iou3d.ComputeVoxelIoU(predTensor, gtTensor);
        Console.WriteLine($"Voxel IoU: {voxelIoU:P1}\n");

        // 4. Compute bounding box IoU
        Console.WriteLine("Computing bounding box IoU...");

        // GT box: [8, 8, 8, 24, 24, 24]
        var gtBox = new float[] { 8, 8, 8, 24, 24, 24 };
        // Pred box: [9, 9, 9, 25, 25, 25]
        var predBox = new float[] { 9, 9, 9, 25, 25, 25 };

        float boxIoU = iou3d.ComputeBoxIoU(predBox, gtBox);
        Console.WriteLine($"Bounding box 3D IoU: {boxIoU:P1}");

        // 5. Detection evaluation thresholds
        Console.WriteLine("\n3D Detection AP thresholds:");
        Console.WriteLine("  Easy: IoU > 0.70");
        Console.WriteLine("  Moderate: IoU > 0.50");
        Console.WriteLine("  Hard: IoU > 0.25");

        string difficulty = boxIoU >= 0.7f ? "Easy" : boxIoU >= 0.5f ? "Moderate" : boxIoU >= 0.25f ? "Hard" : "Not detected";
        Console.WriteLine($"\nDetection result: {difficulty}");

        Console.WriteLine("\n=== Voxel Detection Complete ===\n");
    }

    #endregion

    #region Mesh Processing Example

    /// <summary>
    /// Example: Triangle mesh processing and conversion.
    /// Demonstrates mesh loading, sampling, and metric computation.
    /// </summary>
    public static void MeshProcessingExample()
    {
        Console.WriteLine("=== Triangle Mesh Processing ===\n");

        // 1. Create a simple mesh (icosahedron approximation)
        Console.WriteLine("Creating icosahedron mesh...");

        // Icosahedron vertices (12 vertices, 20 faces)
        float t = (1 + (float)Math.Sqrt(5)) / 2; // Golden ratio

        var vertices = new List<float[]>
        {
            new[] { -1, t, 0 }, new[] { 1, t, 0 }, new[] { -1, -t, 0 }, new[] { 1, -t, 0 },
            new[] { 0, -1, t }, new[] { 0, 1, t }, new[] { 0, -1, -t }, new[] { 0, 1, -t },
            new[] { t, 0, -1 }, new[] { t, 0, 1 }, new[] { -t, 0, -1 }, new[] { -t, 0, 1 }
        };

        // Normalize to unit sphere
        for (int i = 0; i < vertices.Count; i++)
        {
            float len = (float)Math.Sqrt(vertices[i][0] * vertices[i][0] +
                                         vertices[i][1] * vertices[i][1] +
                                         vertices[i][2] * vertices[i][2]);
            vertices[i][0] /= len;
            vertices[i][1] /= len;
            vertices[i][2] /= len;
        }

        // Some triangle faces (simplified)
        var faces = new int[][]
        {
            new[] { 0, 11, 5 }, new[] { 0, 5, 1 }, new[] { 0, 1, 7 },
            new[] { 0, 7, 10 }, new[] { 0, 10, 11 }, new[] { 1, 5, 9 }
        };

        Console.WriteLine($"Vertices: {vertices.Count}");
        Console.WriteLine($"Faces: {faces.Length}\n");

        // 2. Describe mesh operations
        Console.WriteLine("Available mesh operations:");
        Console.WriteLine("- Compute vertex normals");
        Console.WriteLine("- Build adjacency (vertex/edge/face neighbors)");
        Console.WriteLine("- Loop subdivision");
        Console.WriteLine("- Simplification (edge collapse)");
        Console.WriteLine("- Sample points from surface\n");

        // 3. Sample points from mesh surface
        Console.WriteLine("Sampling 1000 points from mesh surface...");
        var random = RandomHelper.CreateSeededRandom(42);
        int numSamples = 1000;
        var sampledPoints = new float[numSamples * 3];

        for (int i = 0; i < numSamples; i++)
        {
            // Pick random face
            int faceIdx = random.Next(faces.Length);
            var face = faces[faceIdx];

            // Random barycentric coordinates
            float u = (float)random.NextDouble();
            float v = (float)random.NextDouble();
            if (u + v > 1)
            {
                u = 1 - u;
                v = 1 - v;
            }
            float w = 1 - u - v;

            // Interpolate vertex positions
            var v0 = vertices[face[0]];
            var v1 = vertices[face[1]];
            var v2 = vertices[face[2]];

            sampledPoints[i * 3] = u * v0[0] + v * v1[0] + w * v2[0];
            sampledPoints[i * 3 + 1] = u * v0[1] + v * v1[1] + w * v2[1];
            sampledPoints[i * 3 + 2] = u * v0[2] + v * v1[2] + w * v2[2];
        }

        Console.WriteLine($"Sampled {numSamples} points\n");

        // 4. Compare sampled points to original vertices
        var vertexArray = new float[vertices.Count * 3];
        for (int i = 0; i < vertices.Count; i++)
        {
            vertexArray[i * 3] = vertices[i][0];
            vertexArray[i * 3 + 1] = vertices[i][1];
            vertexArray[i * 3 + 2] = vertices[i][2];
        }

        var vertexTensor = new Tensor<float>(new[] { vertices.Count, 3 }, new Vector<float>(vertexArray));
        var sampleTensor = new Tensor<float>(new[] { numSamples, 3 }, new Vector<float>(sampledPoints));

        var chamfer = new ChamferDistance<float>();
        float cd = chamfer.Compute(sampleTensor, vertexTensor);
        Console.WriteLine($"Chamfer Distance (samples vs vertices): {cd:E4}");

        Console.WriteLine("\n=== Mesh Processing Complete ===\n");
    }

    #endregion

    /// <summary>
    /// Runs all 3D AI examples.
    /// </summary>
    public static void RunAllExamples()
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║              AiDotNet 3D AI Capabilities Demo              ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════╝\n");

        try
        {
            PointCloudClassificationExample();
            PointCloudSegmentationExample();
            NeRFRenderingExample();
            ReconstructionEvaluationExample();
            VoxelDetectionExample();
            MeshProcessingExample();

            Console.WriteLine("╔════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║                 All Examples Completed!                    ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════╝");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"\n❌ Example failed with error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
}
