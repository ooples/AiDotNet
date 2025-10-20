# Placeholder Implementations - Quick Reference

## Total: 73 NotImplementedException instances

## Critical Priority (20 instances) - SaveModel/LoadModel

These are **REQUIRED** for production use:

- ModelIndividual.cs
- ExpressionTree.cs
- NeuralNetworkModel.cs
- VectorModel.cs
- OptimizerBase.cs
- DecisionTreeAsyncRegressionBase.cs
- DecisionTreeRegressionBase.cs
- NonLinearRegressionBase.cs
- RegressionBase.cs
- TimeSeriesModelBase.cs

**Implementation**: Use System.Text.Json, follow existing patterns

## High Priority (24 instances) - IParameterizable & IFeatureAware

**IParameterizable** (12 instances):
- GetParameters()
- SetParameters()
- ParameterCount
- WithParameters()

**IFeatureAware** (12 instances):
- GetFeatureImportance()
- GetActiveFeatureIndices()
- SetActiveFeatureIndices()
- IsFeatureUsed()

## Medium Priority (14 instances) - Clone/Train

- Clone/DeepCopy implementations (8)
- Training method placeholders (6)

## Low Priority (15 instances) - Advanced Features

- Interpretability helpers
- Activation function factory
- Advanced neural network types (ESN, ELM, GNN, etc.)

## Recommended Approach

1. **This week**: Implement SaveModel/LoadModel for all models (~2-3 days with agents)
2. **Next week**: Implement IParameterizable and IFeatureAware (~1-2 days)
3. **Then**: Merge to master for v1.0.0 release
4. **Later**: Medium/Low priority in future releases

## Use Agent Coordination

```bash
# Generate targeted user stories
~/.claude/scripts/create-user-stories.sh --focus "SaveModel and LoadModel implementations"

# Or use agent coordination directly
/agent-coordination
```

See ACTION-PLAN.md for full details and CI/CD setup instructions.
