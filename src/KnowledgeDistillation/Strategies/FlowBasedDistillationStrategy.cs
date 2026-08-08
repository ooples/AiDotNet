using System;
using System.Linq;

using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Flow-based distillation that matches the information flow between layers.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected. arXiv 1710.01878 is "To prune, or not to prune: exploring the efficacy of
// pruning for model compression" (Zhu & Gupta) — a real paper, but a different one about pruning, not
// distillation. This work appeared at CVPR 2017 (pp. 7130-7138) and is not on arXiv, so the canonical
// DOI replaces the wrong id.
[ResearchPaper("A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning",
    "https://doi.org/10.1109/CVPR.2017.754",
    Year = 2017,
    Authors = "Junho Yim, Donggyu Joo, Jihoon Bae, Junmo Kim")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class FlowBasedDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _flowWeight;

    public FlowBasedDistillationStrategy(
        double flowWeight = 0.5,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        _flowWeight = flowWeight;
    }

    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        T totalLoss = NumOps.Zero;

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            var studentSoft = DistillationHelper<T>.Softmax(studentRow, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherRow, Temperature);
            var softLoss = DistillationHelper<T>.KLDivergence(teacherSoft, studentSoft);
            softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

            T sampleLoss = softLoss;

            if (labelRow != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentRow, 1.0);
                var hardLoss = DistillationHelper<T>.CrossEntropy(studentProbs, labelRow);
                sampleLoss = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
            }

            totalLoss = NumOps.Add(totalLoss, sampleLoss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;
        int outputDim = studentBatchOutput.Columns;
        var batchGradient = new Matrix<T>(batchSize, outputDim);

        for (int r = 0; r < batchSize; r++)
        {
            Vector<T> studentRow = studentBatchOutput.GetRow(r);
            Vector<T> teacherRow = teacherBatchOutput.GetRow(r);
            Vector<T>? labelRow = trueLabelsBatch?.GetRow(r);

            int n = studentRow.Length;
            var gradient = new Vector<T>(n);
            var studentSoft = DistillationHelper<T>.Softmax(studentRow, Temperature);
            var teacherSoft = DistillationHelper<T>.Softmax(teacherRow, Temperature);

            for (int i = 0; i < n; i++)
            {
                var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature));
            }

            if (labelRow != null)
            {
                var studentProbs = DistillationHelper<T>.Softmax(studentRow, 1.0);

                for (int i = 0; i < n; i++)
                {
                    var hardGrad = NumOps.Subtract(studentProbs[i], labelRow[i]);
                    gradient[i] = NumOps.Add(
                        NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                        NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
                }
            }

            for (int c = 0; c < outputDim; c++)
            {
                batchGradient[r, c] = gradient[c];
            }
        }

        var batchScale = NumOps.FromDouble(batchSize);
        for (int r = 0; r < batchSize; r++)
        {
            for (int c = 0; c < outputDim; c++)
            {
                batchGradient[r, c] = NumOps.Divide(batchGradient[r, c], batchScale);
            }
        }

        return batchGradient;
    }

    /// <summary>
    /// Computes flow loss by matching flow matrices between layers.
    /// </summary>
    /// <param name="studentFeatures">Student features from multiple layers.</param>
    /// <param name="teacherFeatures">Teacher features from multiple layers.</param>
    /// <returns>Flow loss value.</returns>
    public T ComputeFlowLoss(Vector<T>[] studentFeatures, Vector<T>[] teacherFeatures)
    {
        if (studentFeatures == null || teacherFeatures == null)
            throw new ArgumentNullException("Features cannot be null");
        if (studentFeatures.Length != teacherFeatures.Length)
            throw new ArgumentException("studentFeatures and teacherFeatures must have the same number of layers");
        if (studentFeatures.Length < 2 || teacherFeatures.Length < 2)
            throw new ArgumentException("studentFeatures and teacherFeatures must have at least 2 layers to compute flow matrices");

        // Validate all vectors have consistent dimensions
        if (studentFeatures.Length > 0)
        {
            int studentDim = studentFeatures[0].Length;
            int teacherDim = teacherFeatures[0].Length;

            for (int i = 0; i < studentFeatures.Length; i++)
            {
                if (studentFeatures[i].Length != studentDim)
                    throw new ArgumentException($"All student features must have same dimension. Expected {studentDim}, got {studentFeatures[i].Length} at layer {i}");
                if (teacherFeatures[i].Length != teacherDim)
                    throw new ArgumentException($"All teacher features must have same dimension. Expected {teacherDim}, got {teacherFeatures[i].Length} at layer {i}");
                if (studentFeatures[i].Length != teacherFeatures[i].Length)
                    throw new ArgumentException($"Student and teacher features must have matching dimensions. Got student={studentFeatures[i].Length}, teacher={teacherFeatures[i].Length} at layer {i}");
                if (studentFeatures[i].Length == 0)
                    throw new ArgumentException($"Feature vectors cannot be empty at layer {i}");
            }
        }

        T totalLoss = NumOps.Zero;
        int flowCount = 0;

        // One FSP matrix per consecutive layer pair, for both networks, then the squared Frobenius
        // norm of their difference — equation (2) of the paper.
        for (int i = 0; i < studentFeatures.Length - 1; i++)
        {
            var studentFsp = ComputeFspMatrix(studentFeatures[i], studentFeatures[i + 1]);
            var teacherFsp = ComputeFspMatrix(teacherFeatures[i], teacherFeatures[i + 1]);

            totalLoss = NumOps.Add(totalLoss, SquaredFrobeniusDistance(studentFsp, teacherFsp));
            flowCount++;
        }

        var avgLoss = flowCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(flowCount)) : NumOps.Zero;
        return NumOps.Multiply(avgLoss, NumOps.FromDouble(_flowWeight));
    }

    /// <summary>
    /// Computes the FSP ("flow of solution procedure") loss from structured feature MAPS, which is the
    /// form the paper defines.
    /// </summary>
    /// <param name="studentFeatures">Student feature maps per layer, each <c>[channels, height, width]</c>
    /// or <c>[batch, channels, height, width]</c>.</param>
    /// <param name="teacherFeatures">Teacher feature maps, same layer count. Spatial dimensions may
    /// differ from the student's; channel counts at each layer must match, because the two FSP matrices
    /// have to be the same size to be compared.</param>
    /// <remarks>
    /// Prefer this overload. The <see cref="Vector{T}"/> overload cannot see channel or spatial
    /// structure, so it treats each layer as a single spatial position — correct, but a degenerate case
    /// of what the paper describes.
    /// </remarks>
    public T ComputeFlowLoss(Tensor<T>[] studentFeatures, Tensor<T>[] teacherFeatures)
    {
        if (studentFeatures is null) throw new ArgumentNullException(nameof(studentFeatures));
        if (teacherFeatures is null) throw new ArgumentNullException(nameof(teacherFeatures));
        if (studentFeatures.Length != teacherFeatures.Length)
        {
            throw new ArgumentException(
                "studentFeatures and teacherFeatures must have the same number of layers");
        }

        if (studentFeatures.Length < 2)
        {
            throw new ArgumentException(
                "At least 2 layers are required: an FSP matrix describes the flow BETWEEN two layers.");
        }

        T totalLoss = NumOps.Zero;
        int flowCount = 0;

        for (int i = 0; i < studentFeatures.Length - 1; i++)
        {
            var studentFsp = ComputeFspMatrix(studentFeatures[i], studentFeatures[i + 1]);
            var teacherFsp = ComputeFspMatrix(teacherFeatures[i], teacherFeatures[i + 1]);

            if (studentFsp.Shape[0] != teacherFsp.Shape[0] || studentFsp.Shape[1] != teacherFsp.Shape[1])
            {
                throw new ArgumentException(
                    $"FSP matrices at layer pair {i} have different shapes " +
                    $"({studentFsp.Shape[0]}x{studentFsp.Shape[1]} vs " +
                    $"{teacherFsp.Shape[0]}x{teacherFsp.Shape[1]}). " +
                    "The paper requires matching channel counts at the distilled layers so the two " +
                    "matrices are comparable.");
            }

            totalLoss = NumOps.Add(totalLoss, SquaredFrobeniusDistance(studentFsp, teacherFsp));
            flowCount++;
        }

        var avgLoss = flowCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(flowCount)) : NumOps.Zero;
        return NumOps.Multiply(avgLoss, NumOps.FromDouble(_flowWeight));
    }

    /// <summary>
    /// The FSP matrix between two layers' flattened features, treating each as a single spatial
    /// position: <c>G[i, j] = a[i] * b[j]</c>.
    /// </summary>
    /// <remarks>
    /// This replaced a single scalar inner product. The FSP matrix is what carries the paper's
    /// information: entry (i, j) records how strongly channel i of the first layer co-activates with
    /// channel j of the second, so the matrix describes the DIRECTION of the transformation between
    /// layers. Summing it to one number discards all of that and leaves only overall magnitude, which
    /// two very different transformations can share — so the distillation signal was largely absent.
    /// </remarks>
    private Tensor<T> ComputeFspMatrix(Vector<T> layerI, Vector<T> layerJ)
    {
        int m = layerI.Length;
        int n = layerJ.Length;

        // The outer product a (m x 1) * b (1 x n) is a single GEMM. Writing it as a double loop over
        // NumOps would cost m*n virtual dispatches for what one matmul does in a blocked kernel.
        var a = Engine.Reshape(Tensor<T>.FromVector(layerI), [m, 1]);
        var b = Engine.Reshape(Tensor<T>.FromVector(layerJ), [1, n]);
        return Engine.TensorMatMul(a, b);
    }

    /// <summary>
    /// The FSP matrix between two feature maps:
    /// <c>G[i, j] = sum over spatial positions of F1[i, s, t] * F2[j, s, t], divided by (h * w)</c>.
    /// </summary>
    /// <remarks>
    /// The two maps may have different spatial sizes — a student is often coarser than its teacher — so
    /// the smaller extent is used and the sum is normalized by the positions actually paired. Without
    /// that normalization the loss would scale with resolution and the per-pair weights would mean
    /// something different at every layer.
    /// </remarks>
    private Tensor<T> ComputeFspMatrix(Tensor<T> layerI, Tensor<T> layerJ)
    {
        var (ci, hi, wi) = DescribeMap(layerI, nameof(layerI));
        var (cj, hj, wj) = DescribeMap(layerJ, nameof(layerJ));

        // The FSP sum over spatial positions IS a matrix product once each map is flattened to
        // [channels, positions]:  G = F1 * F2^T / positions.
        //
        // Written as nested loops over channels and positions this costs ci*cj*h*w scalar NumOps calls
        // — for a 256-channel pair at 32x32 that is ~67 million virtually-dispatched operations, versus
        // one blocked GEMM. The loop form is also not a recorded op, so it would sever the tape.
        var flatI = layerI;
        var flatJ = layerJ;

        // The paper aligns a pair whose two layers differ spatially before forming the matrix. Resample
        // the second map onto the first's grid so the shared "positions" axis is genuinely shared.
        if (hi != hj || wi != wj)
        {
            var asNchw = Engine.Reshape(layerJ, [1, cj, hj, wj]);
            var resized = Engine.Interpolate(
                asNchw, [hi, wi], InterpolateMode.Bilinear, alignCorners: false);
            flatJ = Engine.Reshape(resized, [cj, hi, wi]);
        }

        int positions = hi * wi;
        if (positions <= 0)
        {
            throw new ArgumentException(
                "Feature maps have no spatial positions, so no FSP matrix is defined.");
        }

        var a = Engine.Reshape(flatI, [ci, positions]);
        var b = Engine.Reshape(flatJ, [cj, positions]);

        var gram = Engine.TensorMatMul(a, Engine.TensorTranspose(b));   // [ci, cj]
        return Engine.TensorMultiplyScalar(gram, NumOps.FromDouble(1.0 / positions));
    }

    /// <summary>
    /// Extracts (channels, height, width) from a rank-3 <c>[C,H,W]</c> or single-sample rank-4
    /// <c>[1,C,H,W]</c> feature map.
    /// </summary>
    private static (int Channels, int Height, int Width) DescribeMap(Tensor<T> map, string paramName)
    {
        if (map is null) throw new ArgumentNullException(paramName);

        var s = map.Shape;
        if (s.Length == 3) return (s[0], s[1], s[2]);
        if (s.Length == 4)
        {
            if (s[0] != 1)
            {
                throw new ArgumentException(
                    $"Batched feature maps must be passed one sample at a time; got batch {s[0]}. The " +
                    "paper averages the FSP loss over samples, so batching here would conflate them.",
                    paramName);
            }

            return (s[1], s[2], s[3]);
        }

        throw new ArgumentException(
            $"Feature map must be rank-3 [C,H,W] or rank-4 [1,C,H,W]; got rank {s.Length}.", paramName);
    }

    /// <summary>
    /// Squared Frobenius norm of the difference between two equally-shaped matrices:
    /// <c>sum over all entries of (a - b)^2</c>.
    /// </summary>
    private T SquaredFrobeniusDistance(Tensor<T> a, Tensor<T> b)
    {
        // Elementwise subtract, square, then reduce — three vectorized engine ops rather than a double
        // loop of scalar NumOps calls over every matrix entry.
        var diff = Engine.TensorSubtract(a, b);
        var squared = Engine.TensorMultiply(diff, diff);
        var axes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        var total = Engine.ReduceSum(squared, axes, keepDims: false);
        return total[0];
    }
}
