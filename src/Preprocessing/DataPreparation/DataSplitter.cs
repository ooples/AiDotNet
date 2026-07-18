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
    /// <param name="embargo">
    /// Number of rows to DROP as a gap at each chronological split boundary (train/val and val/test), sized
    /// from the label/forecast horizon. Prevents a horizon-<paramref name="embargo"/> label from straddling a
    /// boundary and leaking future information across partitions. Only applies on the CHRONOLOGICAL path
    /// (<paramref name="shuffle"/> = false); ignored when shuffling (i.i.d. data has no temporal boundary).
    /// Default 0 (no gap) preserves the historical behavior exactly.
    /// </param>
    /// <returns>Tuple of (XTrain, yTrain, XVal, yVal, XTest, yTest).</returns>
    public static (TInput XTrain, TOutput yTrain, TInput XVal, TOutput yVal, TInput XTest, TOutput yTest)
        Split<T, TInput, TOutput>(
            TInput X,
            TOutput y,
            double trainRatio = 0.7,
            double validationRatio = 0.15,
            bool shuffle = true,
            int randomSeed = 42,
            int embargo = 0)
    {
        if (X is Matrix<T> xMatrix && y is Vector<T> yVector)
        {
            var result = SplitMatrix<T>(xMatrix, yVector, trainRatio, validationRatio, shuffle, randomSeed, embargo);
            return (
                (TInput)(object)result.XTrain,
                (TOutput)(object)result.yTrain,
                (TInput)(object)result.XVal,
                (TOutput)(object)result.yVal,
                (TInput)(object)result.XTest,
                (TOutput)(object)result.yTest
            );
        }
        else if (X is Matrix<T> xMatrixMulti && y is Matrix<T> yMatrix)
        {
            // Multi-output (n×H) regression: X is n×features, y is n×H. Split by the SAME shuffled row
            // partition so features and every horizon column stay aligned.
            var result = SplitMatrixMatrix<T>(xMatrixMulti, yMatrix, trainRatio, validationRatio, shuffle, randomSeed, embargo);
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
            var result = SplitTensor<T>(xTensor, yTensor, trainRatio, validationRatio, shuffle, randomSeed, embargo);
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

    /// <summary>
    /// Computes train/validation/test partition sizes that always sum to
    /// <paramref name="totalSamples"/> and never leave a requested partition empty
    /// when the data can afford it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The split ratios are fractions, so multiplying them by a
    /// small row count and rounding down (<c>floor</c>) can accidentally produce a
    /// partition with zero rows — for example one row at 70/15/15 rounds to
    /// "0 train, 0 validation, 1 test", and any fewer than seven rows leaves the
    /// validation set empty. An empty <i>training</i> set is the worst case: the model
    /// never trains and its layers stay at their random initial values. This method
    /// guarantees a usable split by giving training priority and then handing each
    /// other requested partition at least one row once there are enough rows to go
    /// around, reclaiming those rows from the largest partition so the totals still
    /// add up. For datasets large enough that <c>floor</c> already fills every
    /// partition, the result is identical to the plain ratio split.
    /// </para>
    /// </remarks>
    private static (int trainSize, int validationSize, int testSize) ComputeSplitSizes(
        int totalSamples, double trainRatio, double validationRatio)
    {
        if (totalSamples <= 0)
        {
            return (0, 0, 0);
        }

        bool wantValidation = validationRatio > 0.0;
        bool wantTest = (1.0 - trainRatio - validationRatio) > 0.0;
        int wantedPartitions = 1 + (wantValidation ? 1 : 0) + (wantTest ? 1 : 0);

        // Fewer rows than requested partitions (1 or 2 rows): training takes
        // priority — a model cannot learn from an empty set — then test, then
        // validation gets whatever (if anything) is left.
        if (totalSamples < wantedPartitions)
        {
            int tinyTrain = 1;
            int tinyTest = (wantTest && totalSamples >= 2) ? 1 : 0;
            return (tinyTrain, totalSamples - tinyTrain - tinyTest, tinyTest);
        }

        int trainSize = (int)(totalSamples * trainRatio);
        int validationSize = (int)(totalSamples * validationRatio);
        int testSize = totalSamples - trainSize - validationSize;

        // Integer floor() can still starve a single partition to zero even when the
        // data could afford it. Promote each starved-but-requested partition to one
        // row, reclaiming it from the larger of the other two so the totals still sum
        // to totalSamples and the dominant (training) share is preferred on ties.
        if (trainSize < 1)
        {
            if (validationSize >= testSize) { validationSize--; }
            else { testSize--; }
            trainSize++;
        }
        if (wantValidation && validationSize < 1)
        {
            if (trainSize > testSize) { trainSize--; }
            else { testSize--; }
            validationSize++;
        }
        if (wantTest && testSize < 1)
        {
            if (trainSize > validationSize) { trainSize--; }
            else { validationSize--; }
            testSize++;
        }

        return (trainSize, validationSize, testSize);
    }

    /// <summary>
    /// Chronological partition layout with an EMBARGO gap of <paramref name="embargo"/> rows dropped at each
    /// internal boundary (train→val and val→test). Returns the size of each partition and the START index of
    /// val/test within the ordered [0, totalSamples) sequence, so a horizon-<paramref name="embargo"/> label
    /// can never straddle a boundary (the last train row's label lands inside the dropped gap, not in val).
    /// Falls back to the plain contiguous layout when the data is too small to afford the gaps.
    /// </summary>
    private static (int trainSize, int valStart, int valSize, int testStart, int testSize) ComputeEmbargoedLayout(
        int totalSamples, double trainRatio, double validationRatio, int embargo)
    {
        embargo = Math.Max(0, embargo);
        int usable = totalSamples - 2 * embargo;
        if (embargo == 0 || usable < 3)
        {
            // No gap (or too small to afford one): contiguous train|val|test.
            var (tr, va, te) = ComputeSplitSizes(totalSamples, trainRatio, validationRatio);
            return (tr, tr, va, tr + va, te);
        }

        var (trainSize, valSize, testSize) = ComputeSplitSizes(usable, trainRatio, validationRatio);
        int valStart = trainSize + embargo;                 // drop [trainSize, trainSize+embargo)
        int testStart = valStart + valSize + embargo;       // drop [valStart+valSize, testStart)
        return (trainSize, valStart, valSize, testStart, testSize);
    }

    private static (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XVal, Vector<T> yVal, Matrix<T> XTest, Vector<T> yTest)
        SplitMatrix<T>(
            Matrix<T> X,
            Vector<T> y,
            double trainRatio,
            double validationRatio,
            bool shuffle,
            int randomSeed,
            int embargo = 0)
    {
        int totalSamples = X.Rows;
        var indices = Enumerable.Range(0, totalSamples).ToList();

        int trainSize, valStart, validationSize, testStart, testSize;
        if (shuffle)
        {
            // i.i.d. path: shuffle, then contiguous partitions (embargo is meaningless without temporal order).
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            indices = [.. indices.OrderBy(_ => random.Next())];
            (trainSize, validationSize, testSize) = ComputeSplitSizes(totalSamples, trainRatio, validationRatio);
            valStart = trainSize;
            testStart = trainSize + validationSize;
        }
        else
        {
            // Chronological path: honor the embargo gap so horizon-`embargo` labels never cross a boundary.
            (trainSize, valStart, validationSize, testStart, testSize) =
                ComputeEmbargoedLayout(totalSamples, trainRatio, validationRatio, embargo);
        }

        var XTrain = new Matrix<T>(trainSize, X.Columns);
        var yTrain = new Vector<T>(trainSize);
        var XVal = new Matrix<T>(validationSize, X.Columns);
        var yVal = new Vector<T>(validationSize);
        var XTest = new Matrix<T>(testSize, X.Columns);
        var yTest = new Vector<T>(testSize);

        for (int i = 0; i < trainSize; i++)
        {
            XTrain.SetRow(i, X.GetRow(indices[i]));
            yTrain[i] = y[indices[i]];
        }

        for (int i = 0; i < validationSize; i++)
        {
            XVal.SetRow(i, X.GetRow(indices[valStart + i]));
            yVal[i] = y[indices[valStart + i]];
        }

        for (int i = 0; i < testSize; i++)
        {
            XTest.SetRow(i, X.GetRow(indices[testStart + i]));
            yTest[i] = y[indices[testStart + i]];
        }

        return (XTrain, yTrain, XVal, yVal, XTest, yTest);
    }

    private static (Matrix<T> XTrain, Matrix<T> yTrain, Matrix<T> XVal, Matrix<T> yVal, Matrix<T> XTest, Matrix<T> yTest)
        SplitMatrixMatrix<T>(
            Matrix<T> X,
            Matrix<T> y,
            double trainRatio,
            double validationRatio,
            bool shuffle,
            int randomSeed,
            int embargo = 0)
    {
        int totalSamples = X.Rows;
        var indices = Enumerable.Range(0, totalSamples).ToList();

        int trainSize, valStart, validationSize, testStart, testSize;
        if (shuffle)
        {
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            indices = [.. indices.OrderBy(_ => random.Next())];
            (trainSize, validationSize, testSize) = ComputeSplitSizes(totalSamples, trainRatio, validationRatio);
            valStart = trainSize;
            testStart = trainSize + validationSize;
        }
        else
        {
            (trainSize, valStart, validationSize, testStart, testSize) =
                ComputeEmbargoedLayout(totalSamples, trainRatio, validationRatio, embargo);
        }

        var XTrain = new Matrix<T>(trainSize, X.Columns);
        var yTrain = new Matrix<T>(trainSize, y.Columns);
        var XVal = new Matrix<T>(validationSize, X.Columns);
        var yVal = new Matrix<T>(validationSize, y.Columns);
        var XTest = new Matrix<T>(testSize, X.Columns);
        var yTest = new Matrix<T>(testSize, y.Columns);

        for (int i = 0; i < trainSize; i++)
        {
            XTrain.SetRow(i, X.GetRow(indices[i]));
            yTrain.SetRow(i, y.GetRow(indices[i]));
        }

        for (int i = 0; i < validationSize; i++)
        {
            XVal.SetRow(i, X.GetRow(indices[valStart + i]));
            yVal.SetRow(i, y.GetRow(indices[valStart + i]));
        }

        for (int i = 0; i < testSize; i++)
        {
            XTest.SetRow(i, X.GetRow(indices[testStart + i]));
            yTest.SetRow(i, y.GetRow(indices[testStart + i]));
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
            int randomSeed,
            int embargo = 0)
    {
        int totalSamples = X.Shape[0];
        var indices = Enumerable.Range(0, totalSamples).ToList();

        int trainSize, valStart, validationSize, testStart, testSize;
        if (shuffle)
        {
            var random = RandomHelper.CreateSeededRandom(randomSeed);
            indices = [.. indices.OrderBy(_ => random.Next())];
            (trainSize, validationSize, testSize) = ComputeSplitSizes(totalSamples, trainRatio, validationRatio);
            valStart = trainSize;
            testStart = trainSize + validationSize;
        }
        else
        {
            (trainSize, valStart, validationSize, testStart, testSize) =
                ComputeEmbargoedLayout(totalSamples, trainRatio, validationRatio, embargo);
        }

        // Create output tensors — MUST clone shape arrays because modifying
        // the first element would mutate the source tensor's internal _shape.
        int[] xTrainShape = X.Shape.ToArray();
        xTrainShape[0] = trainSize;
        var XTrain = new Tensor<T>(xTrainShape);

        int[] xValShape = X.Shape.ToArray();
        xValShape[0] = validationSize;
        var XVal = new Tensor<T>(xValShape);

        int[] xTestShape = X.Shape.ToArray();
        xTestShape[0] = testSize;
        var XTest = new Tensor<T>(xTestShape);

        int[] yTrainShape = y.Shape.ToArray();
        yTrainShape[0] = trainSize;
        var yTrain = new Tensor<T>(yTrainShape);

        int[] yValShape = y.Shape.ToArray();
        yValShape[0] = validationSize;
        var yVal = new Tensor<T>(yValShape);

        int[] yTestShape = y.Shape.ToArray();
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
            CopySample(X, XVal, indices[valStart + i], i);
            CopySample(y, yVal, indices[valStart + i], i);
        }

        for (int i = 0; i < testSize; i++)
        {
            CopySample(X, XTest, indices[testStart + i], i);
            CopySample(y, yTest, indices[testStart + i], i);
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
