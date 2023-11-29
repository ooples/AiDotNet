
namespace AiDotNetTestConsole;

internal class Program
{
    static void Main(string[] args)
    {
        var inputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var multInputs = new [] { new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 
            new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }};
        var multOutputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var outputs = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var test1 = new[] { new[] { 1.0, 4.0 }, new[] { 2.0, 5.0 }, new[] { 3.0, 2.0 }, new [] { 4.0, 5.0 } };
        var test3 = new[] { new[] { 1.0, 2.0, 3.0, 4.0 }, new[] { 4.0, 5.0, 2.0, 3.0 } };
        var test2 = new[] { 15.0, 20, 10, 15.0 };

        //var simpleRegression = new SimpleRegression(inputs, outputs);
        //var metrics1 = simpleRegression.Metrics;
        //var predictions1 = simpleRegression.Predictions;

        //var advancedSimpleRegression = new SimpleRegression(inputs, outputs, new SimpleRegressionOptions()
        //{
        //    TrainingPctSize = 20,
        //    Normalization = new DecimalNormalization()
        //});
        //var metrics2 = advancedSimpleRegression.Metrics;
        //var predictions2 = advancedSimpleRegression.Predictions;

        //var multipleRegression = new MultipleRegression(test3, test2, 
        //    new MultipleRegressionOptions() { TrainingPctSize = 99, MatrixDecomposition = MatrixDecomposition.Lu, UseIntercept = true });
        //var metrics3 = multipleRegression.Metrics;
        //var predictions3 = multipleRegression.Predictions;

        var arrayOne = new double[] { 4.4, 1.1, 3.3, 2.2 };
        int[] keySort = Enumerable.Range(0, arrayOne.Length).ToArray();
        var stringOne = new string[] {"four", "one", "three", "two" };

        // Arrange
        double expectedPredictions1 = 900.014766;
        double actualPredictions1;
        var matrix1 = new double[,] { { 8.34, -5.66, 2.58 }, { -0.74, 5.27, 8.48 }, { -3.97, -8.97, 0.1 } };

        double expectedPredictions2 = -54242311.1124654;
        double actualPredictions2;
        var matrix2 = new double[,] { { -1.29, 7.67, -9.01, 9.32, 1.01, -7.43, 8.44 }, { -2.31, -9.52, 7.22, 10.12, -5.28, 3.39, 9.81 }, { 7.28, 7.85, 5.11, 10.89, 6.21, -4.53, 4.37 }, { -4.3, 0.38, 1.55, -4.5, -0.55, 9.73, -8.36 }, { 8.35, -1.8, -8.81, 4.71, 4.12, -5.33, -7.56 }, { 0.03, 1.66, 4.95, -9.09, 2.83, 9.35, 10.25 }, { 2.06, -0.03, 8.17, 2.3, 8.05, -4.24, -3.12 } };

        double expectedPredictions3 = -861.143991;
        double actualPredictions3;
        var matrix3 = new double[,] { { -9.82, -2.15, -3.08 }, { -4.77, 8.58, 0.7 }, { -4.6, 1.54, 8.25 } };

        double expectedPredictions4 = 7514.77667572;
        double actualPredictions4;
        var matrix4 = new double[,] { { 9.36, 9.41, 10.39, -4.93 }, { 2.66, -5.9, -1.59, -7.23 }, { -4.16, -6.45, -2.85, 10.79 }, { -2, -9.54, 5.07, 2.92 } };

        double expectedPredictions5 = 1770013.15097733;
        double actualPredictions5;
        var matrix5 = new double[,] { { 5.86, 2.84, -8.43, -8.75, 5.61, 3.21, 7.46 }, { 8.32, 5.5, -0.88, -5.33, -8.88, 7.26, -8.6 }, { 8.65, 5.76, -3.95, -7.69, 2.57, 8.91, 6.03 }, { 8.66, 0.08, -3.38, -2.13, 1.11, -5.18, -8.42 }, { 10.19, 3.91, 2.53, 9.65, 0.93, -1.9, -8.44 }, { -7.79, 5.73, -3.09, 9.26, -8.82, 2.06, 2.47 }, { -6.83, -8.34, 8.95, -4.66, 5.87, 4.05, -7.5 } };

        // Act
        actualPredictions1 = CalculateDeterminantRecursive(matrix1);
        actualPredictions2 = CalculateDeterminantRecursive(matrix2);
        actualPredictions3 = CalculateDeterminantRecursive(matrix3);
        actualPredictions4 = CalculateDeterminantRecursive(matrix4);
        actualPredictions5 = CalculateDeterminantRecursive(matrix5);

        Console.WriteLine(expectedPredictions1.ToString());
        Console.WriteLine(actualPredictions1.ToString());
        Console.WriteLine(expectedPredictions2.ToString());
        Console.WriteLine(actualPredictions2.ToString());
        Console.WriteLine(expectedPredictions3.ToString());
        Console.WriteLine(actualPredictions3.ToString());
        Console.WriteLine(expectedPredictions4.ToString());
        Console.WriteLine(actualPredictions4.ToString());
        Console.WriteLine(expectedPredictions5.ToString());
        Console.WriteLine(actualPredictions5.ToString());

        //var inputsT = new double[] { -3, -2, -1, -0.2, 1, 3 };
        //var outputsT = new double[] { 0.9, 0.8, 0.4, 0.2, 0.1, 0 };

        var inputsT = new double[] { 1, 2, 3, 4, 5, 6, 8, 11, 13 };
        var outputsT = new double[] { 3, 5, 8, 11, 13, 15, 17, 20, 22 };

        double[] aTermsT;

        aTermsT = FindValues(inputsT, outputsT, 3);
        PrintToConsole(aTermsT);

        //Sort by Inputs
        //Array.Sort(arrayOne, keySort);
        //PrintToConsole(keySort);

        //Array.Sort(keySort, stringOne);
        //PrintToConsole(keySort);
    }

    static void PrintToConsole<T>(T[] input)
    {
        string tmpStr = "";
        foreach (var i in input)
        {
            tmpStr = String.Concat(tmpStr, " ", i?.ToString());
        }
        Console.WriteLine(tmpStr);
    }

    public static double[] FindValues(double[] inputs, double[] outputs, int kOrder)
    {
        //Throw exceptions
        //1. Check to see inputs.length == outputs.length
        //2. Check to see that kOrder <= inputs/outputs.length - 1
        //3. maybe check to see if inputs/outputs.length >= some minimum value

        //initialize
        int mSize = kOrder + 1;
        int sigSize = kOrder * 2 + 1; //the +1 is to include the 0th term which is just the size of the input/output array N
        int nSize = inputs.Length;
        double[,] mainMatrix = new double[mSize, mSize];
        double[,] subMatrix = new double[mSize, mSize];
        double[,] yTerms = new double[mSize, 1];
        double[] sigmaSums = new double[sigSize];
        double[] aTerms = new double[mSize];

        //Create an array of the entire list of summed terms from x[i], x[i]^2....x[i]^(2*k)
        sigmaSums[0] = nSize;
        for (var i = 0; i < nSize; i++)
        {
            sigmaSums[1] += inputs[i];
            yTerms[0, 0] += outputs[i];

            for (var p = 1; p < mSize; p++)
            {
                yTerms[p, 0] += outputs[i] * Math.Pow(inputs[i], p);
            }

            for (var j = 2; j < sigSize; j++)
            {
                sigmaSums[j] += Math.Pow(inputs[i], j);
            }
        }

        //Populate the Matrices with the summed terms
        for (var i = 0; i < mSize; i++)
        {
            for (var j = 0; j < mSize; j++)
            {
                mainMatrix[i, j] = sigmaSums[i + j];
                subMatrix[i, j] = sigmaSums[i + j];
            }
        }

        //Loop through entire aTerms array, set subMatrix = correct form calculate
        //Return subMatrix to original form and then iterate.
        for (var i = 0; i < mSize; i++)
        {
            //Now place yTerm matrix into the correct column of subMatrix
            replaceColumn(subMatrix, yTerms, i, 0);
            //Do Calculation
            aTerms[i] = CalculateDeterminantRecursive(subMatrix) / CalculateDeterminantRecursive(mainMatrix);
            //Return subMatrix into original form
            replaceColumn(subMatrix, mainMatrix, i, i);
        }

        return aTerms;
    }

    static void replaceColumn(double[,] destination, double[,] source, int destColumn, int srcColumn)
    {
        //exceptions
        //Ensure the size of source matrix column is equal to the size of destination matrix column
        //ensure destColumn is in scope of destination, ie destColumn < sizeOf Destinations rows
        //ensure srcColumn is in scope of source, ie srcColumn < sizeOf source rows

        //rows = 0 ; col = 1
        int size = source.GetLength(0);

        for (var i = 0; i < size; i++)
        {
            destination[i, destColumn] = source[i, srcColumn];
        }
    }

    public static double CalculateDeterminantRecursive(double[,] matrix)
    {
        var size = matrix.GetLength(0);

        if (size != matrix.GetLength(1))
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (size == 1)
        {
            return matrix[0, 0];
        }

        double determinant = 0;

        for (var i = 0; i < size; i++)
        {
            var subMatrix = CreateSubMatrix(matrix, 0, i);
            determinant += Math.Pow(-1, i) * matrix[0, i] * CalculateDeterminantRecursive(subMatrix);
        }

        return determinant;
    }

    private static double[,] CreateSubMatrix(double[,] matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        var size = matrix.GetLength(0);
        var subMatrix = new double[size - 1, size - 1];

        var r = 0;
        for (var i = 0; i < size; i++)
        {
            if (i == excludeRowIndex)
            {
                continue;
            }

            var c = 0;
            for (var j = 0; j < size; j++)
            {
                if (j == excludeColumnIndex)
                {
                    continue;
                }

                subMatrix[r, c] = matrix[i, j];
                c++;
            }

            r++;
        }

        return subMatrix;
    }
}