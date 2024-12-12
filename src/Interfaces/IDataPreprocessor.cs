namespace AiDotNet.Interfaces;

public interface IDataPreprocessor<T>
{
    (Matrix<T> X, Vector<T> y, NormalizationInfo<T> normInfo) PreprocessData(Matrix<T> X, Vector<T> y);
    
    (Matrix<T> XTrain, Vector<T> yTrain, Matrix<T> XValidation, Vector<T> yValidation, Matrix<T> XTest, Vector<T> yTest) 
        SplitData(Matrix<T> X, Vector<T> y);
}