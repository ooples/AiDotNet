using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Implements Flow of Solution Procedure (FSP) distillation which transfers knowledge about
/// how information flows through the network by matching layer-pair flow matrices.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> FSP distillation teaches the student network how information
/// should flow between layers by matching the "flow matrices" between corresponding layer pairs
/// in the teacher and student networks.</para>
///
/// <para><b>Real-world Analogy:</b>
/// Imagine learning to solve a complex math problem from an expert mathematician. Instead of just
/// copying their final answer (standard distillation), you learn the flow of their reasoning:
/// "First identify the key variables, then apply this transformation, then simplify..."
/// FSP distillation captures this step-by-step transformation process.</para>
///
/// <para><b>Key Insight:</b>
/// The way activations transform from one layer to another (the "flow") contains important
/// knowledge about how the network processes information. By matching these flows, the student
/// learns not just what to predict, but how to process information like the teacher.</para>
///
/// <para><b>Mathematical Foundation:</b>
/// For layer pair (i, j) where i is before j:
/// - Compute Gram matrix G = F_i^T * F_j where F_i and F_j are feature maps
/// - This captures correlations between channels in layers i and j
/// - Match student and teacher Gram matrices: ||G_student - G_teacher||^2</para>
///
/// <para><b>Flow Matrix Computation:</b>
/// Given feature maps from layers i and j:
/// - F_i has shape (batch, channels_i, spatial_dims)
/// - F_j has shape (batch, channels_j, spatial_dims)
/// - Flatten spatial dims: F_i → (batch, channels_i, H*W)
/// - Compute flow: G = (1/HW) * Σ F_i^T * F_j for each sample
/// - Average over batch</para>
///
/// <para><b>Benefits:</b>
/// - **Captures Layer Interactions**: Preserves how layers work together
/// - **Architecture Invariant**: Works even if student/teacher have different depths
/// - **Complementary to Standard Distillation**: Can combine with output matching
/// - **Improved Generalization**: Student learns processing flow, not just outputs
/// - **Efficient**: Matrix computation is parallelizable</para>
///
/// <para><b>When to Use:</b>
/// - Student and teacher have similar but not identical architectures
/// - You want to transfer procedural knowledge (how to solve), not just answers
/// - Target task requires multi-step reasoning or hierarchical features
/// - Combining with other distillation methods for maximum performance</para>
///
/// <para><b>Research Foundation:</b>
/// Based on "A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and
/// Transfer Learning" (Yim et al., CVPR 2017). The paper shows FSP matrices capture the
/// "flow of solution procedure" - how the network transforms information step by step.</para>
/// </remarks>
public class FlowBasedDistillationStrategy<T> : DistillationStrategyBase<T>
{
    private readonly double _flowWeight;

    /// <summary>
    /// Initializes a new instance of the FlowBasedDistillationStrategy.
    /// </summary>
    /// <param name="temperature">Temperature for softening probability distributions (default: 3.0).</param>
    /// <param name="alpha">Weight balancing distillation loss and true label loss (default: 0.7).</param>
    /// <param name="flowWeight">Weight for flow matrix matching loss (default: 1.0).</param>
    public FlowBasedDistillationStrategy(
        double temperature = 3.0,
        double alpha = 0.7,
        double flowWeight = 1.0)
        : base(temperature, alpha)
    {
        if (flowWeight < 0)
            throw new ArgumentOutOfRangeException(nameof(flowWeight), "Flow weight must be non-negative");

        _flowWeight = flowWeight;
    }

    /// <summary>
    /// Computes the combined distillation loss (uses standard KL divergence).
    /// </summary>
    /// <remarks>
    /// <para>For flow-based distillation, call <see cref="ComputeFlowLoss"/> separately
    /// with intermediate features and add it to this loss.</para>
    /// </remarks>
    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabel = default)
    {
        var klLoss = ComputeKLDivergence(studentOutput, teacherOutput);
        
        if (trueLabel != null && trueLabel.Length > 0)
        {
            var hardLoss = ComputeCrossEntropy(studentOutput, trueLabel);
            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);
            
            return NumOps.Add(
                NumOps.Multiply(alphaT, klLoss),
                NumOps.Multiply(oneMinusAlpha, hardLoss));
        }

        return klLoss;
    }

    /// <summary>
    /// Computes the gradient for backpropagation.
    /// </summary>
    /// <remarks>
    /// <para>For flow-based distillation, compute flow gradients separately using
    /// intermediate features and combine with this gradient.</para>
    /// </remarks>
    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabel = default)
    {
        var softTeacher = Softmax(teacherOutput, Temperature);
        var softStudent = Softmax(studentOutput, Temperature);
        
        var tempSquared = NumOps.FromDouble(Temperature * Temperature);
        var gradKL = new Vector<T>(studentOutput.Length);
        
        for (int i = 0; i < studentOutput.Length; i++)
        {
            var diff = NumOps.Subtract(softStudent[i], softTeacher[i]);
            gradKL[i] = NumOps.Multiply(tempSquared, diff);
        }

        if (trueLabel != null && trueLabel.Length > 0)
        {
            var gradHard = new Vector<T>(studentOutput.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            
            for (int i = 0; i < studentOutput.Length; i++)
            {
                gradHard[i] = NumOps.Subtract(studentProbs[i], trueLabel[i]);
            }

            var alphaT = NumOps.FromDouble(Alpha);
            var oneMinusAlpha = NumOps.FromDouble(1.0 - Alpha);
            
            for (int i = 0; i < gradKL.Length; i++)
            {
                gradKL[i] = NumOps.Add(
                    NumOps.Multiply(alphaT, gradKL[i]),
                    NumOps.Multiply(oneMinusAlpha, gradHard[i]));
            }
        }

        return gradKL;
    }

    /// <summary>
    /// Computes flow matrix matching loss between student and teacher layer pairs.
    /// </summary>
    /// <param name="studentFeatures">Feature maps from student network layers.</param>
    /// <param name="teacherFeatures">Feature maps from teacher network layers.</param>
    /// <returns>Flow matrix matching loss.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compares how information flows between layers
    /// in the student vs teacher networks by matching their "flow matrices" (Gram matrices).</para>
    ///
    /// <para><b>Usage:</b>
    /// <code>
    /// // During training, extract intermediate features
    /// var studentFeats = studentModel.GetIntermediateFeatures(input);
    /// var teacherFeats = teacherModel.GetIntermediateFeatures(input);
    /// var flowLoss = strategy.ComputeFlowLoss(studentFeats, teacherFeats);
    /// totalLoss += flowLoss;
    /// </code></para>
    /// </remarks>
    public T ComputeFlowLoss(Vector<Vector<T>> studentFeatures, Vector<Vector<T>> teacherFeatures)
    {
        if (studentFeatures.Length < 2 || teacherFeatures.Length < 2)
            throw new ArgumentException("Need at least 2 layers to compute flow matrices");

        T totalLoss = NumOps.Zero;
        int pairCount = 0;

        for (int i = 0; i < studentFeatures.Length - 1; i++)
        {
            var studentFlow = ComputeFlowMatrix(studentFeatures[i], studentFeatures[i + 1]);
            var teacherFlow = ComputeFlowMatrix(teacherFeatures[i], teacherFeatures[i + 1]);

            var flowDiff = ComputeMatrixFrobeniusNorm(studentFlow, teacherFlow);
            totalLoss = NumOps.Add(totalLoss, flowDiff);
            pairCount++;
        }

        if (pairCount > 0)
        {
            totalLoss = NumOps.Divide(totalLoss, NumOps.FromDouble(pairCount));
        }

        return NumOps.Multiply(totalLoss, NumOps.FromDouble(_flowWeight));
    }

    private Matrix<T> ComputeFlowMatrix(Vector<T> layerI, Vector<T> layerJ)
    {
        int dimI = layerI.Length;
        int dimJ = layerJ.Length;
        
        var flowMatrix = new Matrix<T>(dimI, dimJ);
        
        for (int i = 0; i < dimI; i++)
        {
            for (int j = 0; j < dimJ; j++)
            {
                flowMatrix[i, j] = NumOps.Multiply(layerI[i], layerJ[j]);
            }
        }

        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(dimI * dimJ));
        for (int i = 0; i < dimI; i++)
        {
            for (int j = 0; j < dimJ; j++)
            {
                flowMatrix[i, j] = NumOps.Multiply(flowMatrix[i, j], scale);
            }
        }

        return flowMatrix;
    }

    private T ComputeMatrixFrobeniusNorm(Matrix<T> studentMatrix, Matrix<T> teacherMatrix)
    {
        if (studentMatrix.Rows != teacherMatrix.Rows || studentMatrix.Columns != teacherMatrix.Columns)
            throw new ArgumentException("Matrices must have same dimensions");

        T sumSquaredDiff = NumOps.Zero;
        
        for (int i = 0; i < studentMatrix.Rows; i++)
        {
            for (int j = 0; j < studentMatrix.Columns; j++)
            {
                var diff = NumOps.Subtract(studentMatrix[i, j], teacherMatrix[i, j]);
                sumSquaredDiff = NumOps.Add(sumSquaredDiff, NumOps.Multiply(diff, diff));
            }
        }

        return sumSquaredDiff;
    }

    private T ComputeKLDivergence(Vector<T> studentOutput, Vector<T> teacherOutput)
    {
        var softTeacher = Softmax(teacherOutput, Temperature);
        var softStudent = Softmax(studentOutput, Temperature);

        T kl = NumOps.Zero;
        var epsilon = NumOps.FromDouble(Epsilon);

        for (int i = 0; i < studentOutput.Length; i++)
        {
            var teacherProb = NumOps.Add(softTeacher[i], epsilon);
            var studentProb = NumOps.Add(softStudent[i], epsilon);
            
            var ratio = NumOps.Divide(teacherProb, studentProb);
            var logRatio = NumOps.Log(ratio);
            kl = NumOps.Add(kl, NumOps.Multiply(teacherProb, logRatio));
        }

        var tempSquared = NumOps.FromDouble(Temperature * Temperature);
        return NumOps.Multiply(kl, tempSquared);
    }

    private T ComputeCrossEntropy(Vector<T> predictions, Vector<T> targets)
    {
        var probs = Softmax(predictions, 1.0);
        T loss = NumOps.Zero;
        var epsilon = NumOps.FromDouble(Epsilon);

        for (int i = 0; i < predictions.Length; i++)
        {
            var prob = NumOps.Add(probs[i], epsilon);
            var logProb = NumOps.Log(prob);
            loss = NumOps.Subtract(loss, NumOps.Multiply(targets[i], logProb));
        }

        return loss;
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        var scaled = new Vector<T>(logits.Length);
        var temp = NumOps.FromDouble(temperature);
        
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = NumOps.Divide(logits[i], temp);
        }

        var maxVal = scaled[0];
        for (int i = 1; i < scaled.Length; i++)
        {
            var comparison = NumOps.Subtract(scaled[i], maxVal);
            if (NumOps.GreaterThan(comparison, NumOps.Zero))
                maxVal = scaled[i];
        }

        var expSum = NumOps.Zero;
        var expValues = new Vector<T>(scaled.Length);
        
        for (int i = 0; i < scaled.Length; i++)
        {
            var shifted = NumOps.Subtract(scaled[i], maxVal);
            expValues[i] = NumOps.Exp(shifted);
            expSum = NumOps.Add(expSum, expValues[i]);
        }

        var result = new Vector<T>(scaled.Length);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Divide(expValues[i], expSum);
        }

        return result;
    }
}

