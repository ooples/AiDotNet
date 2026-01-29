using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Simple utility for splitting data into train/validation/test sets.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Before training a model, you need to split your data:
/// - <b>Training set:</b> The data your model learns from (typically 70-80%)
/// - <b>Validation set:</b> Used to tune hyperparameters and prevent overfitting (typically 10-15%)
/// - <b>Test set:</b> Final evaluation of model performance (typically 10-15%)
/// </para>
/// </remarks>
public static class DataSplitter
{
    /// <summary>
    /// Splits data into training, validation, and test sets.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <typeparam name="TInput">The input data type.</typeparam>
    /// <typeparam name="TOutput">The output data type.</typeparam>
    /// <param name="X">The feature data.</param>
    /// <param name="y">The label data.</param>
    /// <param name="trainRatio">Proportion for training (default 0.7).</param>
    /// <param name="validationRatio">Proportion for validation (default 0.15).</param>
    /// <param name="shuffle">Whether to shuffle before splitting (default true).</param>
    /// <param name="randomSeed">Random seed for reproducibility (default 42).</param>
    /// <returns>Tuple of (XTrain, yTrain, XVal, yVal, XTest, yTest).</returns>
    public static (TInput XTrain, TOutput yTrain, TInput XVal, TOutput yVal, TInput XTest, TOutput yTest)
        Split<T, TInput, TOutput>(
            TInput X,
            TOutput y,
            double trainRatio = 0.7,
            double validationRatio = 0.15,
            bool shuffle = true,
            int randomSeed = 42)
    {
        if (X is Matrix<T> xMatrix && y is Vector<T> yVector)
        {
            var result = SplitMatrix<T>(xMatrix, yVector, trainRatio, validationRatio, shuffle, randomSeed);
            return (
                (TInput)(object)result.XTrain,
                (TOutput)(object)result.yTrain,
                (TInput)(object)result.XVal,
                (TOutput)(object)result.yVal,
                (TInput)(object)result.XTest,
                (TOutput)(object)result.yTest
            );
        }
        else if (X is Tensor<T> xTensor && y is Tensor<T> yTensor)
        {
            var result = SplitTensor<T>(xTensor, yTensor, trainRatio, validationRatio, shuffle, randomSeed);
            return (
                (TInput)(object)result.XTrain,
                (TOutput)(object)result.yTrain,
                (TInput)(object)result.XVal,
                (TOutput)(object)result.yVal,
                (TInput)(object)result.XTest,
                (TOutput)(object)result.yTest
            );
        }

        throw new NotSupportedException(
            $"Unsupported data types: X={typeof(TInput).Name}, y={typeof(TOutput).Name}. " +
            "Supported: (Matrix<T>, Vector<T>) or (Tensor<T>, Tensor<T>).");
    }

    private static (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, Matrix<T> XTest, Vector<T> yTest)
        SplitMatrix<T>(
            Matrix<T> X,
            Vector<T> y,
            double trainRatio,
            double validationRatio,
            bool shuffle,
            int randomSeed)
    {
        int totalSamples = X.Rows;
        int trainSize = (int)(totalSamples * trainRatio);
        int validationSize = (int)(totalSamples * validationRatio);
        int testSize = totalSamples - trainSize - validationSize;

        var indices = Enumerable.Range(0, totalSamples).ToList();
        if (shuffle)
        {
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            indices = [.. indices.OrderBy(_ => random.Next())];
        }

        // Create output matrices and vectors
        var XTrain = new Matrix<T>(trainSize, X.Columns);
        var yTrain = new Vector<T>(trainSize);
        var XVal = new Matrix<T>(validationSize, X.Columns);
        var yVal = new Vector<T>(validationSize);
        var XTest = new Matrix<T>(testSize, X.Columns);
        var yTest = new Vector<T>(testSize);

        // Copy training data
        for (int i = 0; i < trainSize; i++)
        {
            XTrain.SetRow(i, X.GetRow(indices[i]));
            yTrain[i] = y[indices[i]];
        }

        // Copy validation data
        for (int i = 0; i < validationSize; i++)
        {
            XVal.SetRow(i, X.GetRow(indices[i + trainSize]));
            yVal[i] = y[indices[i + trainSize]];
        }

        // Copy test data
        for (int i = 0; i < testSize; i++)
        {
            XTest.SetRow(i, X.GetRow(indices[i + trainSize + validationSize]));
            yTest[i] = y[indices[i + trainSize + validationSize]];
        }

        return (XTrain, yTrain, XVal, yVal, XTest, yTest);
    }

    private static (Tensor<T> XTrain, Tensor<T> yTrain, Tensor<T> XVal, Tensor<T> yVal, Tensor<T> XTest, Tensor<T> yTest)
        SplitTensor<T>(
            Tensor<T> X,
            Tensor<T> y,
            double trainRatio,
            double validationRatio,
            bool shuffle,
            int randomSeed)
    {
        int totalSamples = X.Shape[0];
        int trainSize = (int)(totalSamples * trainRatio);
        int validationSize = (int)(totalSamples * validationRatio);
        int testSize = totalSamples - trainSize - validationSize;

        var indices = Enumerable.Range(0, totalSamples).ToList();
        if (shuffle)
        {
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            indices = [.. indices.OrderBy(_ => random.Next())];
        }

        // Create output tensors
        int[] xTrainShape = (int[])X.Shape.Clone();
        xTrainShape[0] = trainSize;
        var XTrain = new Tensor<T>(xTrainShape);

        int[] xValShape = (int[])X.Shape.Clone();
        xValShape[0] = validationSize;
        var XVal = new Tensor<T>(xValShape);

        int[] xTestShape = (int[])X.Shape.Clone();
        xTestShape[0] = testSize;
        var XTest = new Tensor<T>(xTestShape);

        int[] yTrainShape = (int[])y.Shape.Clone();
        yTrainShape[0] = trainSize;
        var yTrain = new Tensor<T>(yTrainShape);

        int[] yValShape = (int[])y.Shape.Clone();
        yValShape[0] = validationSize;
        var yVal = new Tensor<T>(yValShape);

        int[] yTestShape = (int[])y.Shape.Clone();
        yTestShape[0] = testSize;
        var yTest = new Tensor<T>(yTestShape);

        // Copy data using recursive sample copying
        for (int i = 0; i < trainSize; i++)
        {
            CopySample(X, XTrain, indices[i], i);
            CopySample(y, yTrain, indices[i], i);
        }

        for (int i = 0; i < validationSize; i++)
        {
            CopySample(X, XVal, indices[i + trainSize], i);
            CopySample(y, yVal, indices[i + trainSize], i);
        }

        for (int i = 0; i < testSize; i++)
        {
            CopySample(X, XTest, indices[i + trainSize + validationSize], i);
            CopySample(y, yTest, indices[i + trainSize + validationSize], i);
        }

        return (XTrain, yTrain, XVal, yVal, XTest, yTest);
    }

    private static void CopySample<T>(Tensor<T> source, Tensor<T> destination, int sourceIndex, int destIndex)
    {
        CopySampleRecursive(source, destination, sourceIndex, destIndex, 1, new int[source.Rank]);
    }

    private static void CopySampleRecursive<T>(
        Tensor<T> source,
        Tensor<T> destination,
        int sourceIndex,
        int destIndex,
        int currentDim,
        int[] indices)
    {
        if (currentDim == source.Rank)
        {
            indices[0] = sourceIndex;
            T value = source[indices];
            indices[0] = destIndex;
            destination[indices] = value;
        }
        else
        {
            for (int i = 0; i < source.Shape[currentDim]; i++)
            {
                indices[currentDim] = i;
                CopySampleRecursive(source, destination, sourceIndex, destIndex, currentDim + 1, indices);
            }
        }
    }
}
