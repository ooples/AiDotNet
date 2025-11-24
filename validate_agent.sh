#!/bin/bash
# Agent Validation Script
# Usage: ./validate_agent.sh <AGENT_NUM> <FEATURE_NAME>

AGENT_NUM=$1
FEATURE_NAME=$2
ERROR_BASELINE=144

if [ -z "$AGENT_NUM" ] || [ -z "$FEATURE_NAME" ]; then
  echo "Usage: ./validate_agent.sh <AGENT_NUM> <FEATURE_NAME>"
  echo "Example: ./validate_agent.sh 0 Sparsemax"
  exit 1
fi

echo "================================================="
echo "Validating Agent $AGENT_NUM: $FEATURE_NAME"
echo "================================================="
echo ""

# Check 1: Build
echo "[1/4] Build Check..."
dotnet build --no-restore 2>&1 | tee "build_agent_${AGENT_NUM}.txt" > /dev/null
ERROR_COUNT=$(grep -c "error CS" "build_agent_${AGENT_NUM}.txt" || echo "0")

if [ $ERROR_COUNT -gt $ERROR_BASELINE ]; then
  echo "❌ FAILED: Introduced $((ERROR_COUNT - ERROR_BASELINE)) new errors"
  echo "   Current: $ERROR_COUNT errors, Baseline: $ERROR_BASELINE"
  exit 1
fi
echo "✅ Build passed ($ERROR_COUNT errors, baseline $ERROR_BASELINE)"
echo ""

# Check 2: JIT support flag
echo "[2/4] JIT Support Check..."
JIT_FOUND=$(grep -r "SupportsJitCompilation => true" src/ActivationFunctions/ | grep -i "${FEATURE_NAME}Activation.cs" || echo "")
if [ -z "$JIT_FOUND" ]; then
  echo "❌ FAILED: JIT support not enabled for $FEATURE_NAME"
  echo "   Expected: 'SupportsJitCompilation => true' in src/ActivationFunctions/${FEATURE_NAME}Activation.cs"
  exit 1
fi
echo "✅ JIT support enabled in ${FEATURE_NAME}Activation.cs"
echo ""

# Check 3: Required methods
echo "[3/4] Required Methods Check..."

# TensorOperations method
if ! grep -q "public static ComputationNode<T> $FEATURE_NAME" src/Autodiff/TensorOperations.cs; then
  echo "❌ FAILED: TensorOperations method missing"
  echo "   Expected: 'public static ComputationNode<T> $FEATURE_NAME' in src/Autodiff/TensorOperations.cs"
  exit 1
fi
echo "✅ TensorOperations.$FEATURE_NAME() exists"

# IEngine method
if ! grep -q "Tensor<T> $FEATURE_NAME" src/Engines/IEngine.cs; then
  echo "❌ FAILED: IEngine method missing"
  echo "   Expected: 'Tensor<T> $FEATURE_NAME' in src/Engines/IEngine.cs"
  exit 1
fi
echo "✅ IEngine.$FEATURE_NAME() exists"

# CpuEngine implementation
if ! grep -q "public Tensor<T> $FEATURE_NAME" src/Engines/CpuEngine.cs; then
  echo "❌ FAILED: CpuEngine implementation missing"
  echo "   Expected: 'public Tensor<T> $FEATURE_NAME' in src/Engines/CpuEngine.cs"
  exit 1
fi
echo "✅ CpuEngine.$FEATURE_NAME() implemented"

# GpuEngine implementation
if ! grep -q "public Tensor<T> $FEATURE_NAME" src/Engines/GpuEngine.cs; then
  echo "❌ FAILED: GpuEngine implementation missing"
  echo "   Expected: 'public Tensor<T> $FEATURE_NAME' in src/Engines/GpuEngine.cs"
  exit 1
fi
echo "✅ GpuEngine.$FEATURE_NAME() implemented"

# ApplyToGraph implementation
if ! grep -q "ApplyToGraph" "src/ActivationFunctions/${FEATURE_NAME}Activation.cs"; then
  echo "⚠️  Warning: ApplyToGraph() override not found (may use base class)"
else
  echo "✅ ApplyToGraph() override exists"
fi
echo ""

# Check 4: No placeholders
echo "[4/4] Code Quality Check..."
NOTIMPL_COUNT=$(grep -r "NotImplementedException" src/ActivationFunctions/${FEATURE_NAME}Activation.cs 2>/dev/null | wc -l)
TODO_COUNT=$(grep -r "TODO.*JIT\|TODO.*Change.*SupportsJit" src/ActivationFunctions/${FEATURE_NAME}Activation.cs 2>/dev/null | wc -l)

if [ $NOTIMPL_COUNT -gt 0 ]; then
  echo "❌ FAILED: Found NotImplementedException placeholders"
  exit 1
fi

if [ $TODO_COUNT -gt 0 ]; then
  echo "❌ FAILED: Found TODO comments about JIT"
  exit 1
fi
echo "✅ No placeholders or TODO comments"
echo ""

echo "================================================="
echo "✅✅✅ Agent $AGENT_NUM VALIDATION PASSED ✅✅✅"
echo "================================================="
echo ""
echo "Summary:"
echo "  - Build: $ERROR_COUNT errors (baseline: $ERROR_BASELINE)"
echo "  - JIT Support: Enabled"
echo "  - Methods: TensorOperations, IEngine, CpuEngine, GpuEngine"
echo "  - Quality: No placeholders"
echo ""
echo "Ready to commit!"
exit 0
