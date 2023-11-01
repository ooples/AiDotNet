using System;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace AiDotNet.Regression
{
    /// <summary>
    /// MultipleRegression class implementing IRegression interface.
    /// </summary>
    public sealed class MultipleRegression : IRegression<double[], double>
    {
        // Properties
        private double YIntercept { get; set; }
        private double[] Coefficients { get; set; } = Array.Empty<double>();
        private MultipleRegressionOptions RegressionOptions { get; }

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
            // Validation checks
            ValidationHelper.CheckForNullItems(inputs, outputs);
            var inputSize = inputs[0].Length;
            ValidationHelper.CheckForInvalidInputSize(inputSize, outputs.Length);

            // Set RegressionOptions
            RegressionOptions = regressionOptions ?? new MultipleRegressionOptions();

            // Calculate training size
            var trainingPctSize = RegressionOptions.TrainingPctSize;
            ValidationHelper.CheckForInvalidTrainingPctSize(trainingPctSize);
            var trainingSize = (int)Math.Floor(inputSize * trainingPctSize / 100);
            ValidationHelper.CheckForInvalidTrainingSizes(trainingSize, inputSize - trainingSize, Math.Max(2, inputs.Length), trainingPctSize);

            // Prepare data
            var (trainingInputs, trainingOutputs, oosInputs, oosOutputs) =
                PrepareData(inputs, outputs, trainingSize, RegressionOptions.Normalization);
            
            // Fit model and make predictions
            Fit(trainingInputs, trainingOutputs);
            Predictions = Transform(oosInputs);

            // Calculate metrics
            Metrics = new Metrics(Predictions, oosOutputs, inputs.Length);
        }

        // Fit method
        internal override void Fit(double[][] inputs, double[] outputs)
        {
            // Build input matrix based on MatrixLayout option
            var m = Matrix<double>.Build;
            var inputMatrix = RegressionOptions.MatrixLayout switch
            {
                MatrixLayout.ColumnArrays => m.DenseOfColumnArrays(inputs),
                MatrixLayout.RowArrays => m.DenseOfRowArrays(inputs),
                _ => m.DenseOfColumnArrays(inputs)
            };
            var outputVector = CreateVector.Dense(outputs);
            var result = CreateVector.Dense<double>(inputs.Length + (RegressionOptions.UseIntercept ? 1 : 0));

            // Insert column or row based on UseIntercept option
            if (RegressionOptions.UseIntercept)
            {
                inputMatrix = RegressionOptions.MatrixLayout == MatrixLayout.ColumnArrays ? 
                    inputMatrix.InsertColumn(0, CreateVector.Dense(outputs.Length, Vector<double>.One)) :
                    inputMatrix.InsertRow(0, CreateVector.Dense(outputs.Length, Vector<double>.One));
            }

            // Choose decomposition method based on MatrixDecomposition option and solve matrix
            switch (RegressionOptions.MatrixDecomposition)
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
                    throw new ArgumentException("Invalid MatrixDecomposition option");
            }

            // Set coefficients and YIntercept
            Coefficients = result.ToArray();
            YIntercept = 0;
        }

        // PrepareData method
        internal override (double[][] trainingInputs, double[] trainingOutputs, double[][] oosInputs, double[] oosOutputs) 
            PrepareData(double[][] inputs, double[] outputs, int trainingSize, INormalization? normalization)
        {
            // Prepare data based on normalization option
            return normalization?.PrepareData(inputs, outputs, trainingSize) ?? NormalizationHelper.SplitData(inputs, outputs, trainingSize);
        }

        // Transform method
        internal override double[] Transform(double[][] inputs)
        {
            // Initialize predictions array and store lengths for optimization
            var predictions = new double[inputs[0].Length];
            var inputsLength = inputs.Length;
            var inputs0Length = inputs[0].Length;

            // Use parallel processing for large data sets
            Parallel.For(0, inputsLength, i =>
            {
                for (var j = 0; j < inputs0Length; j++)
                {
                    predictions[j] += YIntercept + Coefficients[i] * inputs[i][j];
                }
            });

            return predictions;
        }
    }
}
