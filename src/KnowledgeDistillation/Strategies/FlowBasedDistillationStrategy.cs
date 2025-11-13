using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Flow-based distillation that matches the information flow between layers.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
public class FlowBasedDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
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

    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);
            return NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
        }

        return softLoss;
    }

    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
            gradient[i] = NumOps.Multiply(diff, NumOps.FromDouble(Temperature * Temperature));
        }

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);

            for (int i = 0; i < n; i++)
            {
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);
                gradient[i] = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), gradient[i]));
            }
        }

        return gradient;
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
