using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Benchmarks.Helpers;

/// <summary>
/// Provides shared utility methods for tensor and matrix benchmarks.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This helper class contains common tasks used in multiple benchmarks, 
/// such as creating random data and comparing results. Moving this logic here makes the individual 
/// benchmarks cleaner and easier to maintain.</para>
/// </remarks>
public static class BenchmarkHelper
{
    /// <summary>
    /// Creates a matrix of the specified size filled with random float values between -1 and 1.
    /// </summary>
    /// <param name="rows">The number of rows in the matrix.</param>
    /// <param name="cols">The number of columns in the matrix.</param>
    /// <param name="random">The random number generator to use.</param>
    /// <returns>A new matrix populated with random values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When testing math operations, we need data to work with. 
    /// This method fills a grid (matrix) with random numbers so we can test if the GPU 
    /// and CPU give the same answers even with unpredictable data.</para>
    /// </remarks>
    public static Matrix<float> CreateRandomMatrix(int rows, int cols, Random random)
    {
        var matrix = new Matrix<float>(rows, cols);
        var data = matrix.AsWritableSpan();
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = (float)((random.NextDouble() * 2.0) - 1.0);
        }

        return matrix;
    }

    /// <summary>
    /// Compares two result matrices and calculates error statistics.
    /// </summary>
    /// <param name="reference">The authoritative reference matrix (usually from the CPU).</param>
    /// <param name="actual">The matrix to validate (usually from the GPU).</param>
    /// <returns>A tuple containing maximum error, average error, and count of non-finite values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computers can sometimes make small mistakes in rounding, or big mistakes 
    /// if a component isn't working right. This method checks every single number in two results 
    /// to see how different they are. It helps us guarantee that our fast GPU math is just as 
    /// accurate as our reliable CPU math.</para>
    /// </remarks>
    public static (double maxError, double avgError, int nonFiniteCount) Compare(
        Matrix<float> reference,
        Matrix<float> actual)
    {
        var refSpan = reference.AsSpan();
        var actSpan = actual.AsSpan();

        double maxError = 0;
        double sumError = 0;
        int count = 0;
        int nonFiniteCount = 0;

        for (int i = 0; i < actSpan.Length; i++)
        {
            float actVal = actSpan[i];
            if (float.IsNaN(actVal) || float.IsInfinity(actVal))
            {
                nonFiniteCount++;
                continue;
            }

            double error = Math.Abs(refSpan[i] - actVal);
            sumError += error;
            if (error > maxError)
                maxError = error;
            count++;
        }

        double avgError = count > 0 ? sumError / count : double.NaN;
        return (maxError, avgError, nonFiniteCount);
    }
}
