# Ensemble Feature Organization Structure

## Overview

The ensemble feature has been organized to follow the same structure as the rest of the AiDotNet project. This ensures consistency and makes the codebase easier to navigate.

## Current Organization

### Interfaces (Moved to `/src/Interfaces/`)
- `IEnsembleModel.cs` - Main interface for ensemble models
- `ICombinationStrategy.cs` - Interface for combination strategies
- `IDynamicModelSelector.cs` - Interface for dynamic model selection (renamed from IModelSelector to avoid conflicts)

### Models (Moved to `/src/Models/`)
- `EnsembleModelBase.cs` - Base class for ensemble models
- `VotingEnsemble.cs` - Voting-based ensemble implementation

### Options (Moved to `/src/Models/Options/`)
- `EnsembleOptions.cs` - Configuration options for ensemble models

### Strategies (Remain in `/src/Ensemble/Strategies/`)
- All combination strategy implementations (AdaBoost, Stacking, Blending, etc.)
- These remain in the Ensemble folder as they are specific to ensemble functionality

### Documentation (Remains in `/src/Ensemble/Documentation/`)
- Following the pattern of other features like ReinforcementLearning

## Benefits of This Organization

1. **Consistency**: Follows the same structure as other features in the project
2. **Discoverability**: Interfaces are easy to find in the main Interfaces folder
3. **No Naming Conflicts**: Resolved the IModelSelector conflict by renaming to IDynamicModelSelector
4. **Clear Separation**: Core components (interfaces, models, options) are in standard locations while specialized components (strategies) remain grouped

## Namespace Updates

All moved files have been updated to use the correct namespaces:
- Interfaces: `AiDotNet.Interfaces`
- Models: `AiDotNet.Models`
- Options: `AiDotNet.Models.Options`
- Strategies: `AiDotNet.Ensemble.Strategies` (unchanged)

## Usage Example

```csharp
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Ensemble.Strategies;

// Create an ensemble model
var options = new EnsembleOptions<double>
{
    Strategy = EnsembleStrategy.Stacking,
    EnableParallelProcessing = true
};

var ensemble = new VotingEnsemble<double, Matrix<double>, Vector<double>>(options);
```

## Future Considerations

When adding new ensemble-related components:
- Interfaces go in `/src/Interfaces/`
- Model classes go in `/src/Models/`
- Options go in `/src/Models/Options/`
- Strategies and specialized components remain in `/src/Ensemble/`