using System.Numerics;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.Statistics;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Regression;

public sealed class MultipleRegression : IRegression<double[], double>
{
    private double YIntercept { get; set; }
    private double[] Coefficients { get; set; } = Array.Empty<double>();
    private MatrixDecomposition MatrixDecomposition { get; set; }
    private MatrixLayout MatrixLayout { get; set; }
    private bool UseIntercept { get; set; }

    public double[] Predictions { get; private set; }
    public IMetrics Metrics { get; private set; }

    /// <summary>
    /// Performs multiple regression on the provided inputs and outputs.
    /// This handles all of the steps needed to create a trained ai model including training, normalizing, splitting, and transforming the data.
    /// </summary>
    /// <param name="inputs">The raw inputs (predicted values) to compare against the output values</param>
    /// <param name="outputs">The raw outputs (actual values) to compare against the input values</param>
    /// <param name="regressionOptions">Different options to allow full customization of the regression process</param>
    /// <exception cref="ArgumentNullException">The input array and/or output array is null</exception>
    /// <exception cref="ArgumentException">The input array or output array is either not the same length or doesn't have enough data</exception>
    public MultipleRegression(double[][] inputs, double[] outputs, MultipleRegressionOptions? regressionOptions = null)
    {
        if (inputs == null)
        {
            throw new ArgumentNullException(nameof(inputs), "Inputs can't be null");
        }

        if (outputs == null)
        {
            throw new ArgumentNullException(nameof(outputs), "Outputs can't be null");
        }

        var inputSize = inputs[0].Length;
        /*
        if (inputSize != outputs.Length)
        {
            throw new ArgumentException("Inputs and outputs must have the same length");
        }
        */

        if (inputSize < 2)
        {
            throw new ArgumentException("Inputs and outputs must have at least 2 values each");
        }

        regressionOptions ??= new MultipleRegressionOptions();
        MatrixDecomposition = regressionOptions.MatrixDecomposition;
        MatrixLayout = regressionOptions.MatrixLayout;
        UseIntercept = regressionOptions.UseIntercept;
        var trainingPctSize = regressionOptions.TrainingPctSize;

        if (trainingPctSize <= 0 || trainingPctSize >= 100)
        {
            throw new ArgumentException($"{nameof(trainingPctSize)} must be greater than 0 and less than 100", nameof(trainingPctSize));
        }

        // Set the training sizes to determine if we have enough training data to fit the model
        var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
        var outOfSampleSize = inputSize - trainingSize;
        var minSize = Math.Max(2, inputs.Length);

        /*
        if (trainingSize < minSize)
        {
            throw new ArgumentException($"Training data must contain at least {minSize} values. " +
                                        $"You either need to increase your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }

        if (outOfSampleSize < 2)
        {
            throw new ArgumentException($"Out of sample data must contain at least 2 values. " +
                                        $"You either need to decrease your {nameof(trainingPctSize)} or increase the amount of inputs and outputs data");
        }
        */

        var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
            PrepareData(inputs, outputs, trainingSize, regressionOptions.Normalization);
        Fit(trainingInputs, trainingOutputs);
        Predictions = Transform(oosInputs);
        Metrics = new Metrics(Predictions, oosOutputs, inputs.Length);
    }

    internal override void Fit(double[][] inputs, double[] outputs)
    {
        var m = Matrix<double>.Build;
        var inputMatrix = MatrixLayout == MatrixLayout.ColumnArrays ? m.DenseOfColumnArrays(inputs) : m.DenseOfRowArrays(inputs);
        var outputVector = CreateVector.Dense(outputs);
        var result = CreateVector.Dense<double>(inputs.Length + (UseIntercept ? 1 : 0));

        if (UseIntercept)
        {
            inputMatrix = MatrixLayout == MatrixLayout.ColumnArrays ? 
                inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, MathNet.Numerics.LinearAlgebra.Vector<double>.One)) :
                inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, MathNet.Numerics.LinearAlgebra.Vector<double>.One));
        }

        switch (MatrixDecomposition)
        {
            case MatrixDecomposition.Cholesky:
                inputMatrix.Cholesky().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Evd:
                inputMatrix.Evd().Solve(outputVector, result);
                break;
            case MatrixDecomposition.GramSchmidt:
                inputMatrix.GramSchmidt().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Lu:
                inputMatrix.LU().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Qr:
                inputMatrix.QR().Solve(outputVector, result);
                break;
            case MatrixDecomposition.Svd:
                inputMatrix.Svd().Solve(outputVector, result);
                break;
            default:
                inputMatrix.Solve(outputVector, result);
                break;
        }

        Coefficients = result.ToArray();
        YIntercept = 0;
    }

    internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) 
        PrepareData(double[][] inputs, double[] outputs, int trainingSize, INormalization? normalization)
    {
        return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
    }

    internal override double[] Transform(double[][] inputs)
    {
        var predictions = new double[inputs[0].Length];

        for (var i = 0; i < inputs.Length; i++)
        {
            for (var j = 0; j < inputs[j].Length; j++)
            {
                predictions[j] += YIntercept + Coefficients[i] * inputs[i][j];
            }
        }

        return predictions;
    }
}