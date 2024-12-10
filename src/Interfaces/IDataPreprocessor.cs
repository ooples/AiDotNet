namespace AiDotNet.Interfaces;

public interface IDataPreprocessor
{
    (Matrix<double> X, Vector<double> y, NormalizationInfo normInfo) PreprocessData(Matrix<double> X, Vector<double> y);
    
    (Matrix<double> XTrain, Vector<double> yTrain, Matrix<double> XValidation, Vector<double> yValidation, Matrix<double> XTest, Vector<double> yTest) 
        SplitData(Matrix<double> X, Vector<double> y);
}