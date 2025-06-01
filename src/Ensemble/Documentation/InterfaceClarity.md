# Interface Clarity: IModelSelector vs IDynamicModelSelector

## Overview

During the ensemble implementation, we discovered two interfaces with the same name `IModelSelector` but different purposes. We've resolved this by renaming the ensemble-specific interface to `IDynamicModelSelector`.

## The Two Interfaces

### 1. IModelSelector<T, TInput, TOutput> (Original)
- **Location**: `/src/Interfaces/IModelSelector.cs`
- **Purpose**: Automatic model selection and recommendation based on data characteristics
- **Use Case**: Helps users choose the right type of model for their data
- **Key Methods**:
  - `SelectModel(TInput sampleX, TOutput sampleY)` - Automatically selects the best model type
  - `GetModelRecommendations(TInput sampleX, TOutput sampleY)` - Provides ranked recommendations

### 2. IDynamicModelSelector<T> (Renamed from IModelSelector)
- **Location**: `/src/Ensemble/Interfaces/IDynamicModelSelector.cs`
- **Purpose**: Dynamic model selection within ensemble methods
- **Use Case**: Selects which models from an ensemble to use for specific inputs
- **Key Methods**:
  - `SelectModelsForInput<TInput, TOutput>(TInput input, IReadOnlyList<IFullModel<T, TInput, TOutput>> models)`
  - `UpdatePerformance<TInput, TOutput>(TInput input, List<TOutput> predictions, TOutput actual, List<int> modelIndices)`

## Why This Matters

1. **No Naming Conflicts**: Having two interfaces with the same name would cause compilation errors and confusion
2. **Clear Separation of Concerns**: 
   - `IModelSelector` is about choosing what TYPE of model to use
   - `IDynamicModelSelector` is about choosing WHICH models from an ensemble to use
3. **Better Developer Experience**: Clear, distinct names make the codebase easier to understand

## Usage Examples

### Using IModelSelector (Model Type Selection)
```csharp
// This helps you choose between different model types (e.g., neural network vs regression)
var modelSelector = new DefaultModelSelector<double, Matrix<double>, Vector<double>>();
var bestModel = modelSelector.SelectModel(trainingData, targetValues);
```

### Using IDynamicModelSelector (Ensemble Model Selection)
```csharp
// This helps you choose which models from your ensemble to use for a specific input
var dynamicSelector = new KNearestCompetenceSelector<double>();
var selectedIndices = dynamicSelector.SelectModelsForInput(input, ensembleModels);
```

## Conclusion

By renaming the ensemble-specific interface to `IDynamicModelSelector`, we've:
- Eliminated potential naming conflicts
- Made the codebase clearer and more maintainable
- Preserved the functionality of both interfaces
- Made it easier for developers to understand which interface to use for their specific needs