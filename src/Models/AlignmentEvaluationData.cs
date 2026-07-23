namespace AiDotNet.Models;

using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// Contains test cases for evaluating AI alignment.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AlignmentEvaluationData<T>
{
    /// <summary>
    /// Gets or sets the test prompts or inputs.
    /// </summary>
    public Matrix<T> TestInputs { get; set; } = Matrix<T>.Empty();

    /// <summary>
    /// Gets or sets the expected/desired outputs for each test input.
    /// </summary>
    public Matrix<T> ExpectedOutputs { get; set; } = Matrix<T>.Empty();

    /// <summary>
    /// Gets or sets the evaluation criteria for each test.
    /// </summary>
    public string[] EvaluationCriteria { get; set; } = Array.Empty<string>();

    /// <summary>
    /// Gets or sets ground truth labels for classification tasks.
    /// </summary>
    public int[] Labels { get; set; } = Array.Empty<int>();

    /// <summary>
    /// Gets or sets human-annotated scores for reference.
    /// </summary>
    public double[] ReferenceScores { get; set; } = Array.Empty<double>();
}
