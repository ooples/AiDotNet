namespace AiDotNet.Helpers;

public static class NeuralNetworkHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    // Activation functions
    public static T ReLU(T x) => MathHelper.Max(x, NumOps.Zero);
    public static T ReLUDerivative(T x) => NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Zero;

    public static T Sigmoid(T x) => NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(x))));
    public static T SigmoidDerivative(T x)
    {
        T sigmoid = Sigmoid(x);
        return NumOps.Multiply(sigmoid, NumOps.Subtract(NumOps.One, sigmoid));
    }

    public static T TanH(T x)
    {
        T exp2x = NumOps.Exp(NumOps.Multiply(x, NumOps.FromDouble(2)));
        return NumOps.Divide(NumOps.Subtract(exp2x, NumOps.One), NumOps.Add(exp2x, NumOps.One));
    }
    public static T TanHDerivative(T x)
    {
        T tanh = TanH(x);
        return NumOps.Subtract(NumOps.One, NumOps.Multiply(tanh, tanh));
    }

    public static T Linear(T x) => x;
    public static T LinearDerivative(T x) => NumOps.One;

    // Loss functions
    public static T MeanSquaredError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanSquaredError(predicted, actual);
    }
    public static Vector<T> MeanSquaredErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => NumOps.Multiply(NumOps.FromDouble(2), x)).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T MeanAbsoluteError(Vector<T> predicted, Vector<T> actual)
    {
        return StatisticsHelper<T>.CalculateMeanAbsoluteError(predicted, actual);
    }
    public static Vector<T> MeanAbsoluteErrorDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Transform(x => NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One)).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T BinaryCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            sum = NumOps.Add(sum, NumOps.Add(
                NumOps.Multiply(actual[i], NumOps.Log(p)),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, actual[i]), NumOps.Log(NumOps.Subtract(NumOps.One, p)))
            ));
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }
    public static Vector<T> BinaryCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        Vector<T> derivative = new Vector<T>(predicted.Length, NumOps);
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            derivative[i] = NumOps.Divide(
                NumOps.Subtract(p, actual[i]),
                NumOps.Multiply(p, NumOps.Subtract(NumOps.One, p))
            );
        }
        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T CategoricalCrossEntropy(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T p = MathHelper.Clamp(predicted[i], NumOps.FromDouble(1e-15), NumOps.FromDouble(1 - 1e-15));
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(p)));
        }
        return NumOps.Negate(sum);
    }

    public static Vector<T> CategoricalCrossEntropyDerivative(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        return predicted.Subtract(actual).Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T HuberLoss(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? NumOps.FromDouble(1.0);

        T sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(predicted[i], actual[i]));
            if (NumOps.LessThanOrEquals(diff, delta))
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(diff, diff)));
            }
            else
            {
                sum = NumOps.Add(sum, NumOps.Subtract(
                    NumOps.Multiply(delta, diff),
                    NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(delta, delta))
                ));
            }
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> HuberLossDerivative(Vector<T> predicted, Vector<T> actual, T? delta = default)
    {
        if (predicted.Length != actual.Length)
            throw new ArgumentException("Predicted and actual vectors must have the same length.");

        delta = delta ?? NumOps.FromDouble(1.0);

        Vector<T> derivative = new(predicted.Length, NumOps);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            if (NumOps.LessThanOrEquals(NumOps.Abs(diff), delta))
            {
                derivative[i] = diff;
            }
            else
            {
                derivative[i] = NumOps.Multiply(delta, NumOps.GreaterThan(diff, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One));
            }
        }

        return derivative.Divide(NumOps.FromDouble(predicted.Length));
    }

    public static T LogCoshLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = NumOps.Subtract(predicted[i], actual[i]);
            var logCosh = NumOps.Log(NumOps.Add(NumOps.Exp(diff), NumOps.Exp(NumOps.Negate(diff))));
            sum = NumOps.Add(sum, logCosh);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> LogCoshLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length, NumOps);

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = NumOps.Subtract(predicted[i], actual[i]);
            derivative[i] = MathHelper.Tanh(diff);
        }

        return derivative;
    }

    public static T QuantileLoss(Vector<T> predicted, Vector<T> actual, T quantile)
    {
        var sum = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = NumOps.Subtract(actual[i], predicted[i]);
            var loss = NumOps.GreaterThan(diff, NumOps.Zero)
                ? NumOps.Multiply(quantile, diff)
                : NumOps.Multiply(NumOps.Subtract(NumOps.One, quantile), NumOps.Negate(diff));
            sum = NumOps.Add(sum, loss);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> QuantileLossDerivative(Vector<T> predicted, Vector<T> actual, T quantile)
    {
        var derivative = new Vector<T>(predicted.Length, NumOps);

        for (int i = 0; i < predicted.Length; i++)
        {
            var diff = NumOps.Subtract(actual[i], predicted[i]);
            derivative[i] = NumOps.GreaterThan(diff, NumOps.Zero)
                ? NumOps.Negate(quantile)
                : NumOps.Subtract(NumOps.One, quantile);
        }

        return derivative;
    }

    public static T CrossEntropyLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(predicted[i])));
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }

    public static Vector<T> CrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Divide(NumOps.Negate(actual[i]), predicted[i]);
        }

        return derivative;
    }

    public static T BinaryCrossEntropyLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, 
                NumOps.Add(
                    NumOps.Multiply(actual[i], NumOps.Log(predicted[i])),
                    NumOps.Multiply(NumOps.Subtract(NumOps.One, actual[i]), NumOps.Log(NumOps.Subtract(NumOps.One, predicted[i])))
                )
            );
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Length)));
    }

    public static Vector<T> BinaryCrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Divide(
                NumOps.Subtract(predicted[i], actual[i]),
                NumOps.Multiply(predicted[i], NumOps.Subtract(NumOps.One, predicted[i]))
            );
        }

        return derivative;
    }

    public static T CategoricalCrossEntropyLoss(Matrix<T> predicted, Matrix<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Rows; i++)
        {
            for (int j = 0; j < predicted.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(actual[i, j], NumOps.Log(predicted[i, j])));
            }
        }

        return NumOps.Negate(NumOps.Divide(sum, NumOps.FromDouble(predicted.Rows)));
    }

    public static Matrix<T> CategoricalCrossEntropyLossDerivative(Matrix<T> predicted, Matrix<T> actual)
    {
        var derivative = new Matrix<T>(predicted.Rows, predicted.Columns);
        for (int i = 0; i < predicted.Rows; i++)
        {
            for (int j = 0; j < predicted.Columns; j++)
            {
                derivative[i, j] = NumOps.Divide(NumOps.Negate(actual[i, j]), predicted[i, j]);
            }
        }

        return derivative;
    }

    public static T HingeLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, MathHelper.Max(NumOps.Zero, NumOps.Subtract(NumOps.One, NumOps.Multiply(actual[i], predicted[i]))));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> HingeLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.LessThan(NumOps.Multiply(actual[i], predicted[i]), NumOps.One) ? NumOps.Negate(actual[i]) : NumOps.Zero;
        }

        return derivative;
    }

    public static T KullbackLeiblerDivergence(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(actual[i], NumOps.Log(NumOps.Divide(actual[i], predicted[i]))));
        }

        return sum;
    }

    public static Vector<T> KullbackLeiblerDivergenceDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Negate(NumOps.Divide(actual[i], predicted[i]));
        }

        return derivative;
    }

    public static T PoissonLoss(Vector<T> predicted, Vector<T> actual)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Subtract(predicted[i], NumOps.Multiply(actual[i], NumOps.Log(predicted[i]))));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> PoissonLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Subtract(NumOps.One, NumOps.Divide(actual[i], predicted[i]));
        }

        return derivative;
    }

    public static T CosineSimilarityLoss(Vector<T> predicted, Vector<T> actual)
    {
        var dotProduct = NumOps.Zero;
        var normPredicted = NumOps.Zero;
        var normActual = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(predicted[i], actual[i]));
            normPredicted = NumOps.Add(normPredicted, NumOps.Multiply(predicted[i], predicted[i]));
            normActual = NumOps.Add(normActual, NumOps.Multiply(actual[i], actual[i]));
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(dotProduct, NumOps.Multiply(NumOps.Sqrt(normPredicted), NumOps.Sqrt(normActual))));
    }

    public static Vector<T> CosineSimilarityLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var dotProduct = NumOps.Zero;
        var normPredicted = NumOps.Zero;
        var normActual = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            dotProduct = NumOps.Add(dotProduct, NumOps.Multiply(predicted[i], actual[i]));
            normPredicted = NumOps.Add(normPredicted, NumOps.Multiply(predicted[i], predicted[i]));
            normActual = NumOps.Add(normActual, NumOps.Multiply(actual[i], actual[i]));
        }
        
        var derivative = new Vector<T>(predicted.Length);
        var normProduct = NumOps.Multiply(NumOps.Sqrt(normPredicted), NumOps.Sqrt(normActual));
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Divide(
                NumOps.Subtract(
                    NumOps.Multiply(actual[i], normPredicted),
                    NumOps.Multiply(predicted[i], dotProduct)
                ),
                NumOps.Multiply(normProduct, normPredicted)
            );
        }

        return derivative;
    }

    private static T EuclideanDistance(Vector<T> v1, Vector<T> v2)
    {
        var sum = NumOps.Zero;
        for (int i = 0; i < v1.Length; i++)
        {
            var diff = NumOps.Subtract(v1[i], v2[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    public static T FocalLoss(Vector<T> predicted, Vector<T> actual, T gamma, T alpha)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T pt = NumOps.Equals(actual[i], NumOps.One) ? predicted[i] : NumOps.Subtract(NumOps.One, predicted[i]);
            T alphaT = NumOps.Equals(actual[i], NumOps.One) ? alpha : NumOps.Subtract(NumOps.One, alpha);
            loss = NumOps.Add(loss, NumOps.Multiply(NumOps.Negate(alphaT), 
                NumOps.Multiply(NumOps.Power(NumOps.Subtract(NumOps.One, pt), gamma), NumOps.Log(pt))));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(predicted.Length));
    }

    public static Vector<T> FocalLossDerivative(Vector<T> predicted, Vector<T> actual, T gamma, T alpha)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T pt = NumOps.Equals(actual[i], NumOps.One) ? predicted[i] : NumOps.Subtract(NumOps.One, predicted[i]);
            T alphaT = NumOps.Equals(actual[i], NumOps.One) ? alpha : NumOps.Subtract(NumOps.One, alpha);
            T term1 = NumOps.Multiply(NumOps.Negate(alphaT), NumOps.Power(NumOps.Subtract(NumOps.One, pt), NumOps.Subtract(gamma, NumOps.One)));
            T term2 = NumOps.Subtract(NumOps.Multiply(gamma, NumOps.Subtract(NumOps.One, pt)), pt);
            derivative[i] = NumOps.Multiply(term1, term2);
        }

        return derivative;
    }

    public static T TripletLoss(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative, T margin)
    {
        var batchSize = anchor.Rows;
        var totalLoss = NumOps.Zero;

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

            var loss = MathHelper.Max(NumOps.Zero, NumOps.Add(NumOps.Subtract(positiveDistance, negativeDistance), margin));
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    public static (Matrix<T> AnchorGradient, Matrix<T> PositiveGradient, Matrix<T> NegativeGradient) TripletLossDerivative(Matrix<T> anchor, Matrix<T> positive, Matrix<T> negative, T margin)
    {
        var batchSize = anchor.Rows;
        var featureCount = anchor.Columns;

        var anchorGradient = new Matrix<T>(batchSize, featureCount);
        var positiveGradient = new Matrix<T>(batchSize, featureCount);
        var negativeGradient = new Matrix<T>(batchSize, featureCount);

        for (int i = 0; i < batchSize; i++)
        {
            var anchorSample = anchor.GetRow(i);
            var positiveSample = positive.GetRow(i);
            var negativeSample = negative.GetRow(i);

            var positiveDistance = EuclideanDistance(anchorSample, positiveSample);
            var negativeDistance = EuclideanDistance(anchorSample, negativeSample);

            var loss = NumOps.Subtract(NumOps.Add(positiveDistance, margin), negativeDistance);

            if (NumOps.GreaterThan(loss, NumOps.Zero))
            {
                for (int j = 0; j < featureCount; j++)
                {
                    var anchorPositiveDiff = NumOps.Subtract(anchorSample[j], positiveSample[j]);
                    var anchorNegativeDiff = NumOps.Subtract(anchorSample[j], negativeSample[j]);

                    anchorGradient[i, j] = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Subtract(anchorPositiveDiff, anchorNegativeDiff));
                    positiveGradient[i, j] = NumOps.Multiply(NumOps.FromDouble(-2), anchorPositiveDiff);
                    negativeGradient[i, j] = NumOps.Multiply(NumOps.FromDouble(2), anchorNegativeDiff);
                }
            }
            else
            {
                // If the triplet loss is zero or negative, the gradients are zero
                for (int j = 0; j < featureCount; j++)
                {
                    anchorGradient[i, j] = NumOps.Zero;
                    positiveGradient[i, j] = NumOps.Zero;
                    negativeGradient[i, j] = NumOps.Zero;
                }
            }
        }

        return (anchorGradient, positiveGradient, negativeGradient);
    }

    public static T ContrastiveLoss(Vector<T> output1, Vector<T> output2, T similarityLabel, T margin)
    {
        T distance = EuclideanDistance(output1, output2);
        T similarTerm = NumOps.Multiply(similarityLabel, NumOps.Power(distance, NumOps.FromDouble(2)));
        T dissimilarTerm = NumOps.Multiply(NumOps.Subtract(NumOps.One, similarityLabel), 
            NumOps.Power(MathHelper.Max(NumOps.Zero, NumOps.Subtract(margin, distance)), NumOps.FromDouble(2)));

        return NumOps.Add(similarTerm, dissimilarTerm);
    }

    public static (Vector<T>, Vector<T>) ContrastiveLossDerivative(Vector<T> output1, Vector<T> output2, T similarityLabel, T margin)
    {
        T distance = EuclideanDistance(output1, output2);
        Vector<T> grad1 = new Vector<T>(output1.Length);
        Vector<T> grad2 = new Vector<T>(output2.Length);

        for (int i = 0; i < output1.Length; i++)
        {
            T diff = NumOps.Subtract(output1[i], output2[i]);
            if (NumOps.Equals(similarityLabel, NumOps.One))
            {
                grad1[i] = NumOps.Multiply(NumOps.FromDouble(2), diff);
                grad2[i] = NumOps.Multiply(NumOps.FromDouble(-2), diff);
            }
            else
            {
                if (NumOps.LessThan(distance, margin))
                {
                    grad1[i] = NumOps.Multiply(NumOps.FromDouble(-2), NumOps.Multiply(NumOps.Subtract(margin, distance), diff));
                    grad2[i] = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(NumOps.Subtract(margin, distance), diff));
                }
            }
        }

        return (grad1, grad2);
    }

    public static T DiceLoss(Vector<T> predicted, Vector<T> actual)
    {
        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        T diceCoefficient = NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), intersection), 
            NumOps.Add(sumPredicted, sumActual));

        return NumOps.Subtract(NumOps.One, diceCoefficient);
    }

    public static Vector<T> DiceLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T intersection = NumOps.Zero;
        T sumPredicted = NumOps.Zero;
        T sumActual = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, NumOps.Multiply(predicted[i], actual[i]));
            sumPredicted = NumOps.Add(sumPredicted, predicted[i]);
            sumActual = NumOps.Add(sumActual, actual[i]);
        }

        T denominator = NumOps.Power(NumOps.Add(sumPredicted, sumActual), NumOps.FromDouble(2));

        for (int i = 0; i < predicted.Length; i++)
        {
            T numerator = NumOps.Subtract(
                NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(actual[i], NumOps.Add(sumPredicted, sumActual))),
                NumOps.Multiply(NumOps.FromDouble(2), NumOps.Multiply(intersection, NumOps.FromDouble(2)))
            );
            derivative[i] = NumOps.Divide(numerator, denominator);
        }

        return derivative;
    }

    public static T JaccardLoss(Vector<T> predicted, Vector<T> actual)
    {
        T intersection = NumOps.Zero;
        T union = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));
            union = NumOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }

        return NumOps.Subtract(NumOps.One, NumOps.Divide(intersection, union));
    }

    public static Vector<T> JaccardLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        T intersection = NumOps.Zero;
        T union = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            intersection = NumOps.Add(intersection, MathHelper.Min(predicted[i], actual[i]));
            union = NumOps.Add(union, MathHelper.Max(predicted[i], actual[i]));
        }

        for (int i = 0; i < predicted.Length; i++)
        {
            if (NumOps.GreaterThan(predicted[i], actual[i]))
            {
                derivative[i] = NumOps.Divide(NumOps.Subtract(union, intersection), NumOps.Power(union, NumOps.FromDouble(2)));
            }
            else if (NumOps.LessThan(predicted[i], actual[i]))
            {
                derivative[i] = NumOps.Divide(NumOps.Negate(NumOps.Subtract(union, intersection)), NumOps.Power(union, NumOps.FromDouble(2)));
            }
            else
            {
                derivative[i] = NumOps.Zero;
            }
        }

        return derivative;
    }

    public static T WeightedCrossEntropyLoss(Vector<T> predicted, Vector<T> actual, Vector<T> weights)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss = NumOps.Add(loss, NumOps.Multiply(weights[i], 
                NumOps.Multiply(actual[i], NumOps.Log(predicted[i]))));
        }

        return NumOps.Negate(loss);
    }

    public static Vector<T> WeightedCrossEntropyLossDerivative(Vector<T> predicted, Vector<T> actual, Vector<T> weights)
    {
        var derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Multiply(
                weights[i],
                NumOps.Divide(
                    NumOps.Subtract(predicted[i], actual[i]),
                    NumOps.Multiply(predicted[i], NumOps.Subtract(NumOps.One, predicted[i]))
                )
            );
        }

        return derivative;
    }

    public static T OrdinalRegressionLoss(Vector<T> predicted, Vector<T> actual, int numClasses)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            for (int j = 0; j < numClasses - 1; j++)
            {
                T indicator = NumOps.GreaterThan(actual[i], NumOps.FromDouble(j)) ? NumOps.One : NumOps.Zero;
                loss = NumOps.Add(loss, NumOps.Log(NumOps.Add(NumOps.One, NumOps.Exp(NumOps.Negate(NumOps.Multiply(indicator, predicted[i]))))));
            }
        }

        return loss;
    }

    public static Vector<T> OrdinalRegressionLossDerivative(Vector<T> predicted, Vector<T> actual, int numClasses)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < numClasses - 1; j++)
            {
                T indicator = NumOps.GreaterThan(actual[i], NumOps.FromDouble(j)) ? NumOps.One : NumOps.Zero;
                T expTerm = NumOps.Exp(NumOps.Negate(NumOps.Multiply(indicator, predicted[i])));
                sum = NumOps.Add(sum, NumOps.Divide(NumOps.Negate(NumOps.Multiply(indicator, expTerm)), NumOps.Add(NumOps.One, expTerm)));
            }
            derivative[i] = sum;
        }

        return derivative;
    }

    public static T ExponentialLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss = NumOps.Add(loss, NumOps.Exp(NumOps.Negate(NumOps.Multiply(actual[i], predicted[i]))));
        }

        return loss;
    }

    public static Vector<T> ExponentialLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            derivative[i] = NumOps.Multiply(
                NumOps.Negate(actual[i]),
                NumOps.Exp(NumOps.Negate(NumOps.Multiply(actual[i], predicted[i])))
            );
        }

        return derivative;
    }

    public static T SquaredHingeLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = NumOps.Subtract(NumOps.One, NumOps.Multiply(actual[i], predicted[i]));
            loss = NumOps.Add(loss, NumOps.Power(MathHelper.Max(NumOps.Zero, margin), NumOps.FromDouble(2)));
        }

        return loss;
    }

    public static Vector<T> SquaredHingeLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T margin = NumOps.Subtract(NumOps.One, NumOps.Multiply(actual[i], predicted[i]));
            if (NumOps.GreaterThan(margin, NumOps.Zero))
            {
                derivative[i] = NumOps.Multiply(NumOps.FromDouble(-2), NumOps.Multiply(actual[i], margin));
            }
            else
            {
                derivative[i] = NumOps.Zero;
            }
        }

        return derivative;
    }

    public static T ModifiedHuberLoss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T z = NumOps.Multiply(actual[i], predicted[i]);
            if (NumOps.GreaterThanOrEquals(z, NumOps.FromDouble(-1)))
            {
                loss = NumOps.Add(loss, NumOps.Power(MathHelper.Max(NumOps.Zero, NumOps.Subtract(NumOps.One, z)), NumOps.FromDouble(2)));
            }
            else
            {
                loss = NumOps.Add(loss, NumOps.Negate(NumOps.Multiply(NumOps.FromDouble(4), z)));
            }
        }

        return loss;
    }

    public static Vector<T> ModifiedHuberLossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T z = NumOps.Multiply(actual[i], predicted[i]);
            if (NumOps.GreaterThanOrEquals(z, NumOps.FromDouble(-1)))
            {
                if (NumOps.LessThan(z, NumOps.One))
                {
                    derivative[i] = NumOps.Multiply(NumOps.FromDouble(-2), NumOps.Multiply(actual[i], NumOps.Subtract(NumOps.One, z)));
                }
                else
                {
                    derivative[i] = NumOps.Zero;
                }
            }
            else
            {
                derivative[i] = NumOps.Multiply(NumOps.FromDouble(-4), actual[i]);
            }
        }

        return derivative;
    }

    public static T SmoothL1Loss(Vector<T> predicted, Vector<T> actual)
    {
        T loss = NumOps.Zero;
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Abs(NumOps.Subtract(predicted[i], actual[i]));
            if (NumOps.LessThan(diff, NumOps.One))
            {
                loss = NumOps.Add(loss, NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Power(diff, NumOps.FromDouble(2))));
            }
            else
            {
                loss = NumOps.Add(loss, NumOps.Subtract(diff, NumOps.FromDouble(0.5)));
            }
        }

        return loss;
    }

    public static Vector<T> SmoothL1LossDerivative(Vector<T> predicted, Vector<T> actual)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T diff = NumOps.Subtract(predicted[i], actual[i]);
            if (NumOps.LessThan(NumOps.Abs(diff), NumOps.One))
            {
                derivative[i] = diff;
            }
            else
            {
                derivative[i] = NumOps.SignOrZero(diff);
            }
        }

        return derivative;
    }

    public static T ElasticNetLoss(Vector<T> predicted, Vector<T> actual, T l1Ratio, T alpha)
    {
        T mseLoss = MeanSquaredError(predicted, actual);
        T l1Regularization = NumOps.Zero;
        T l2Regularization = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i++)
        {
            l1Regularization = NumOps.Add(l1Regularization, NumOps.Abs(predicted[i]));
            l2Regularization = NumOps.Add(l2Regularization, NumOps.Power(predicted[i], NumOps.FromDouble(2)));
        }

        T l1Term = NumOps.Multiply(NumOps.Multiply(alpha, l1Ratio), l1Regularization);
        T l2Term = NumOps.Multiply(NumOps.Multiply(NumOps.Multiply(alpha, NumOps.Subtract(NumOps.One, l1Ratio)), NumOps.FromDouble(0.5)), l2Regularization);

        return NumOps.Add(NumOps.Add(mseLoss, l1Term), l2Term);
    }

    public static Vector<T> ElasticNetLossDerivative(Vector<T> predicted, Vector<T> actual, T l1Ratio, T alpha)
    {
        Vector<T> derivative = new Vector<T>(predicted.Length);
        for (int i = 0; i < predicted.Length; i++)
        {
            T mseGradient = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Subtract(predicted[i], actual[i]));
            T l1Gradient = NumOps.Multiply(alpha, NumOps.Multiply(l1Ratio, NumOps.SignOrZero(predicted[i])));
            T l2Gradient = NumOps.Multiply(NumOps.Multiply(alpha, NumOps.Subtract(NumOps.One, l1Ratio)), predicted[i]);
            
            derivative[i] = NumOps.Add(NumOps.Add(mseGradient, l1Gradient), l2Gradient);
        }

        return derivative;
    }
}