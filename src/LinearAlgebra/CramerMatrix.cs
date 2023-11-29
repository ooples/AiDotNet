using AiDotNet.Helpers;

namespace AiDotNet.LinearAlgebra;

internal class CramerMatrix : IMatrix<double, int>
{
    public double[] Coefficients { get; }

    public CramerMatrix(double[] inputs, double[] outputs, int kOrder)
    {
        //Throw exceptions
        //1. Check to see inputs.length == outputs.length
        //2. Check to see that kOrder <= inputs/outputs.length - 1
        //3. Check to see if inputs/outputs.length >= some minimum value

        //Build
        var (mainMatrix, subMatrix, yTerms) = BuildMatrix(inputs, outputs, kOrder);

        //Solve
        var aTerms = Solve(mainMatrix, subMatrix, yTerms);
        Coefficients = aTerms;
    }
    public (double[,] mainMatrix, double[,] subMatrix, double[,] yTerms) BuildMatrix(double[] inputs, double[] outputs, int kOrder)
    {
        //initialize
        var mSize = kOrder + 1;
        var sigSize = kOrder * 2 + 1; //the +1 is to include the 0th term which is just the size of the input/output array N
        var nSize = inputs.Length;
        var mainMatrix = new double[mSize, mSize];
        var subMatrix = new double[mSize, mSize];
        var yTerms = new double[mSize, 1];
        var sigmaSums = new double[sigSize];

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

        return (mainMatrix, subMatrix, yTerms);
    }

    public double[] Solve(double[,] mMatrix, double[,] sMatrix, double[,] yTerms)
    {
        var size = mMatrix.GetLength(0);
        var aTerms = new double[size];

        for (var i = 0; i < size; i++)
        {
            //Now place yTerm matrix into the correct column of subMatrix
            MatrixHelper.ReplaceColumn(sMatrix, yTerms, i, 0);              //replaceColumn(subMatrix, yTerms, i, 0);

            //Calculate the Coefficients of Solution Fit
            aTerms[i] = MatrixHelper.CalculateDeterminantRecursive(sMatrix) / MatrixHelper.CalculateDeterminantRecursive(mMatrix);

            //Return subMatrix into original form
            MatrixHelper.ReplaceColumn(sMatrix, mMatrix, i , i);             //replaceColumn(subMatrix, mainMatrix, i, i);
        }

        return aTerms;
    }

}