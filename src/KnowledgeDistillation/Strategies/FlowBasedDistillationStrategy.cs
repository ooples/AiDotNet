using System;

using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Flow-based distillation that matches the information flow between layers.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
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
                gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
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

        // Compute flow matrices between consecutive layers
        for (int i = 0; i < studentFeatures.Length - 1; i++)
        {
            var studentFlow = ComputeFlowMatrix(studentFeatures[i], studentFeatures[i + 1]);
            var teacherFlow = ComputeFlowMatrix(teacherFeatures[i], teacherFeatures[i + 1]);

            // MSE between flow matrices
            T flowDiff = NumOps.Subtract(studentFlow, teacherFlow);
            T squaredDiff = NumOps.Multiply(flowDiff, flowDiff);
            totalLoss = NumOps.Add(totalLoss, squaredDiff);
            flowCount++;
        }

        var avgLoss = flowCount > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(flowCount)) : NumOps.Zero;
        return NumOps.Multiply(avgLoss, NumOps.FromDouble(_flowWeight));
    }

    private T ComputeFlowMatrix(Vector<T> layerI, Vector<T> layerJ)
    {
        // Simplified flow computation: inner product between consecutive layers
        T flow = NumOps.Zero;
        int n = Math.Min(layerI.Length, layerJ.Length);

        for (int i = 0; i < n; i++)
        {
            flow = NumOps.Add(flow, NumOps.Multiply(layerI[i], layerJ[i]));
        }

        return flow;
    }
}
