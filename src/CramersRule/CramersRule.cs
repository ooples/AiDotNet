namespace AiDotNet.Regression;

public class CramersRule
{
    internal double[] FindValues(double[] inputs, double[] outputs,int kOrder)
    {
        //Throw exceptions
        //1. Check to see inputs.length == outputs.length
        //2. Check to see that kOrder <= inputs/outputs.length - 1
        //3. maybe check to see if inputs/outputs.length >= some minimum value

        //initialize
        int mSize = kOrder + 1;
        int sigSize = kOrder * 2;
        int nSize = inputs.Length;
        double[,] mainMatrix = new double[mSize, mSize];
        double[,] subMatrix = new double[mSize, mSize];
        double[,] yTerms = new double[mSize, 1];
        double[] sigmaSums = new double[sigSize];
        double[] aTerms = new double[mSize];

        //Create an array of the entire list of summed terms from x[i], x[i]^2....x[i]^(2*k)
        for (var i = 0; i < nSize; i++)
        {
            sigmaSums[0] += inputs[i];
            yTerms[0, 0] += outputs[i];

            for (var p = 1; p < mSize; p++)
            {
                yTerms[p, 0] += outputs[i] * Math.Pow(inputs[i], p + 1);
            }

            for (var j = 1; j < sigSize; j++)
            {
                sigmaSums[j] += Math.Pow(inputs[i], j + 1);
            }
        }

        //Populate the Matrices with the summed terms
        for (var i = 0; i < mSize; i++)
        {
            for (var j = 0; j < mSize; j++)
            {
                mainMatrix[i, j] = i == j && i == 0 ? nSize : sigmaSums[i + j];
                subMatrix[i, j] = i == j && i == 0 ? nSize : sigmaSums[i + j];
            }
        }

        //Loop through entire aTerms array, set subMatrix = correct form calculate
        //Return subMatrix to original form and then iterate.
        for (var i = 0; i < mSize; i++)
        {
            //Now place yTerm matrix into the correct column of subMatrix
            replaceColumn(subMatrix, yTerms, i, 0);
            //Do Calculation
            aTerms[i] = findDeterminant(subMatrix) / findDeterminant(mainMatrix);
            //Return subMatrix into original form
            replaceColumn(subMatrix, mainMatrix, i, i);
        }

        return aTerms;
    }

    internal void replaceColumn(double[,] destination, double[,] source, int destColumn, int srcColumn)
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

    internal double findDeterminant(double[,] matrix)
    {
        return 1.0;
    }
}
