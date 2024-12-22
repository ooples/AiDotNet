using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class GaussianProcessFitDetector<T> : FitDetectorBase<T>
{
    private readonly GaussianProcessFitDetectorOptions _options;

    public GaussianProcessFitDetector(GaussianProcessFitDetectorOptions? options = null)
    {
        _options = options ?? new GaussianProcessFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var (meanPrediction, variancePrediction) = PerformGaussianProcessRegression(evaluationData);

        var averageUncertainty = variancePrediction.Average();
        var rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(evaluationData.ModelStats.Actual, meanPrediction);

        if (_numOps.LessThan(rmse, _numOps.FromDouble(_options.GoodFitThreshold)) &&
            _numOps.LessThan(averageUncertainty, _numOps.FromDouble(_options.LowUncertaintyThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (_numOps.GreaterThan(rmse, _numOps.FromDouble(_options.OverfitThreshold)) &&
                 _numOps.LessThan(averageUncertainty, _numOps.FromDouble(_options.LowUncertaintyThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.GreaterThan(rmse, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.GreaterThan(averageUncertainty, _numOps.FromDouble(_options.HighUncertaintyThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var (meanPrediction, variancePrediction) = PerformGaussianProcessRegression(evaluationData);

        var averageUncertainty = variancePrediction.Average();
        var rmse = StatisticsHelper<T>.CalculateRootMeanSquaredError(evaluationData.ModelStats.Actual, meanPrediction);

        // Normalize confidence level to [0, 1]
        var uncertaintyFactor = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, averageUncertainty));
        var rmseFactor = _numOps.Divide(_numOps.One, _numOps.Add(_numOps.One, rmse));

        return _numOps.Multiply(uncertaintyFactor, rmseFactor);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The Gaussian Process model appears to be well-fitted. Consider:");
                recommendations.Add("1. Validating the model on new, unseen data to ensure generalization.");
                recommendations.Add("2. Exploring different kernel functions to potentially improve performance.");
                recommendations.Add("3. Investigating areas of higher uncertainty in the predictions.");
                break;
            case FitType.Overfit:
                recommendations.Add("The Gaussian Process model may be overfitting. Consider:");
                recommendations.Add("1. Increasing the noise parameter in the kernel to reduce sensitivity to individual data points.");
                recommendations.Add("2. Using a simpler kernel function to reduce model complexity.");
                recommendations.Add("3. Collecting more training data, especially in areas where the model shows high confidence but poor performance.");
                break;
            case FitType.Underfit:
                recommendations.Add("The Gaussian Process model may be underfitting. Consider:");
                recommendations.Add("1. Using a more complex kernel function to capture more intricate patterns in the data.");
                recommendations.Add("2. Reducing the noise parameter in the kernel if you believe your data is relatively noise-free.");
                recommendations.Add("3. Feature engineering or adding more relevant features to provide more information to the model.");
                break;
            case FitType.Unstable:
                recommendations.Add("The Gaussian Process model appears unstable. Consider:");
                recommendations.Add("1. Checking for numerical instabilities in the kernel matrix calculations.");
                recommendations.Add("2. Normalizing or standardizing your input features.");
                recommendations.Add("3. Using a different optimization method for hyperparameter tuning.");
                break;
        }

        return recommendations;
    }

    private (Vector<T>, Vector<T>) PerformGaussianProcessRegression(ModelEvaluationData<T> evaluationData)
    {
        var X = evaluationData.ModelStats.FeatureMatrix;
        var y = evaluationData.ModelStats.Actual;

        // Hyperparameter optimization
        var optimizedHyperparameters = OptimizeHyperparameters(X, y);
        _options.LengthScale = optimizedHyperparameters.LengthScale;
        _options.NoiseVariance = optimizedHyperparameters.NoiseVariance;

        // Calculate kernel matrix with optimized hyperparameters
        var K = CalculateKernelMatrix(X, X);

        // Add noise to the diagonal for numerical stability
        for (int i = 0; i < K.Rows; i++)
        {
            K[i, i] = _numOps.Add(K[i, i], _numOps.FromDouble(_options.NoiseVariance));
        }

        // Perform Cholesky decomposition
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);

        // Solve for alpha: K * alpha = y
        var alpha = choleskyDecomposition.Solve(y);

        // Prepare for predictions (in this case, we're predicting on the same points as the training data)
        var XStar = X;
        var KStar = CalculateKernelMatrix(X, XStar);

        // Calculate mean predictions
        var meanPrediction = KStar.Transpose().Multiply(alpha);

        // Calculate variance predictions
        var variancePrediction = new Vector<T>(XStar.Rows);
        for (int i = 0; i < XStar.Rows; i++)
        {
            var kStarStar = CalculateRBFKernel(XStar.GetRow(i), XStar.GetRow(i));
            var v = choleskyDecomposition.Solve(KStar.GetColumn(i));
            variancePrediction[i] = _numOps.Subtract(kStarStar, v.DotProduct(v));
        }

        return (meanPrediction, variancePrediction);
    }

    private (double LengthScale, double NoiseVariance) OptimizeHyperparameters(Matrix<T> X, Vector<T> y)
    {
        // Implement a simple grid search for hyperparameter optimization
        double bestLengthScale = 1.0;
        double bestNoiseVariance = 0.1;
        T bestLogLikelihood = _numOps.MinValue;

        var lengthScales = new[] { 0.1, 0.5, 1.0, 2.0, 5.0 };
        var noiseVariances = new[] { 0.01, 0.1, 0.5, 1.0 };

        foreach (var ls in lengthScales)
        {
            foreach (var nv in noiseVariances)
            {
                _options.LengthScale = ls;
                _options.NoiseVariance = nv;

                var K = CalculateKernelMatrix(X, X);
                for (int i = 0; i < K.Rows; i++)
                {
                    K[i, i] = _numOps.Add(K[i, i], _numOps.FromDouble(nv));
                }

                var logLikelihood = CalculateLogLikelihood(K, y);

                if (_numOps.GreaterThan(logLikelihood, bestLogLikelihood))
                {
                    bestLogLikelihood = logLikelihood;
                    bestLengthScale = ls;
                    bestNoiseVariance = nv;
                }
            }
        }

        return (bestLengthScale, bestNoiseVariance);
    }

    private T CalculateLogLikelihood(Matrix<T> K, Vector<T> y)
    {
        var choleskyDecomposition = new CholeskyDecomposition<T>(K);
        var L = choleskyDecomposition.L;
        var alpha = choleskyDecomposition.Solve(y);
    
        var dataFit = _numOps.Multiply(_numOps.FromDouble(-0.5), y.DotProduct(alpha));
        var complexity = _numOps.Multiply(_numOps.FromDouble(-1), L.LogDeterminant());
        var normalization = _numOps.Multiply(_numOps.FromDouble(-0.5 * K.Rows), _numOps.Log(_numOps.FromDouble(2 * Math.PI)));

        return _numOps.Add(_numOps.Add(dataFit, complexity), normalization);
    }

    private Matrix<T> CalculateKernelMatrix(Matrix<T> X1, Matrix<T> X2)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);
        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = CalculateRBFKernel(X1.GetRow(i), X2.GetRow(j));
            }
        }

        return K;
    }

    private Vector<T> CalculateKernelVector(Matrix<T> X, Vector<T> x)
    {
        var k = new Vector<T>(X.Rows);
        for (int i = 0; i < X.Rows; i++)
        {
            k[i] = CalculateRBFKernel(X.GetRow(i), x);
        }

        return k;
    }

    private T CalculateRBFKernel(Vector<T> x1, Vector<T> x2)
    {
        // RBF Kernel
        var diff = x1 - x2;
        var squaredDist = diff.DotProduct(diff);

        return _numOps.Exp(_numOps.Negate(_numOps.Divide(squaredDist, _numOps.FromDouble(2 * _options.LengthScale * _options.LengthScale))));
    }
}