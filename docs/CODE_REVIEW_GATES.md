# JIT Compilation - Code Review and Validation Gates

**Purpose**: Ensure all agent PRs meet quality standards before merging to master.
**Reviewer**: Agent 8 (Code Reviewer - Quality Gate)

---

## Automated Build Validation Script

```bash
#!/bin/bash
# File: validate-pr.sh
# Usage: ./validate-pr.sh <PR_NUMBER>

PR_NUMBER=$1

if [ -z "$PR_NUMBER" ]; then
    echo "Usage: $0 <PR_NUMBER>"
    exit 1
fi

echo "========================================="
echo "PR #${PR_NUMBER} Validation Report"
echo "========================================="
echo ""

# Clone PR branch
echo "[1/9] Fetching PR branch..."
git fetch origin pull/${PR_NUMBER}/head:pr-${PR_NUMBER}
git checkout pr-${PR_NUMBER}

# Build validation
echo "[2/9] Building net462..."
dotnet build src/AiDotNet.csproj -c Release -f net462
if [ $? -ne 0 ]; then
    echo "FAIL: net462 build failed"
    exit 1
fi

echo "[3/9] Building net471..."
dotnet build src/AiDotNet.csproj -c Release -f net471
if [ $? -ne 0 ]; then
    echo "FAIL: net471 build failed"
    exit 1
fi

echo "[4/9] Building netstandard2.0..."
dotnet build src/AiDotNet.csproj -c Release -f netstandard2.0
if [ $? -ne 0 ]; then
    echo "FAIL: netstandard2.0 build failed"
    exit 1
fi

# Run tests
echo "[5/9] Running tests..."
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj
if [ $? -ne 0 ]; then
    echo "WARNING: Some tests failed"
fi

# Code quality checks
echo "[6/9] Checking for null-forgiving operator..."
NULL_FORGIVING=$(grep -r "!" src/ | grep -v "!=" | grep -v "xml" | grep -v "!string" | grep -v "IsNullOrEmpty" | wc -l)
if [ $NULL_FORGIVING -gt 0 ]; then
    echo "FAIL: Found $NULL_FORGIVING instances of null-forgiving operator (!)"
    grep -r "!" src/ | grep -v "!=" | grep -v "xml" | grep -v "!string" | grep -v "IsNullOrEmpty"
    exit 1
fi

echo "[7/9] Checking for System.Text.Json usage..."
SYSTEM_TEXT_JSON=$(grep -r "System.Text.Json" src/ | wc -l)
if [ $SYSTEM_TEXT_JSON -gt 0 ]; then
    echo "FAIL: Found System.Text.Json usage (use Newtonsoft.Json instead)"
    grep -r "System.Text.Json" src/
    exit 1
fi

echo "[8/9] Checking for KeyValuePair deconstruction..."
KVP_DECON=$(grep -r "var (.*,.*) in" src/ | wc -l)
if [ $KVP_DECON -gt 0 ]; then
    echo "WARNING: Found potential KeyValuePair deconstruction (not supported in net462)"
    grep -r "var (.*,.*) in" src/
fi

echo "[9/9] Checking for investigation files..."
INVESTIGATION_FILES=$(ls *REPORT* *FINDINGS* *INVESTIGATION* 2>/dev/null | wc -l)
if [ $INVESTIGATION_FILES -gt 0 ]; then
    echo "FAIL: Found investigation/report files that should not be committed"
    ls *REPORT* *FINDINGS* *INVESTIGATION*
    exit 1
fi

echo ""
echo "========================================="
echo "VALIDATION PASSED"
echo "========================================="
```

---

## PowerShell Validation Script (Windows)

```powershell
# File: Validate-PR.ps1
# Usage: .\Validate-PR.ps1 -PRNumber 123

param(
    [Parameter(Mandatory=$true)]
    [int]$PRNumber
)

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "PR #$PRNumber Validation Report" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Clone PR branch
Write-Host "[1/9] Fetching PR branch..." -ForegroundColor Yellow
git fetch origin pull/$PRNumber/head:pr-$PRNumber
git checkout pr-$PRNumber

# Build validation
Write-Host "[2/9] Building net462..." -ForegroundColor Yellow
dotnet build src/AiDotNet.csproj -c Release -f net462
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: net462 build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[3/9] Building net471..." -ForegroundColor Yellow
dotnet build src/AiDotNet.csproj -c Release -f net471
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: net471 build failed" -ForegroundColor Red
    exit 1
}

Write-Host "[4/9] Building netstandard2.0..." -ForegroundColor Yellow
dotnet build src/AiDotNet.csproj -c Release -f netstandard2.0
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: netstandard2.0 build failed" -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "[5/9] Running tests..." -ForegroundColor Yellow
dotnet test tests/AiDotNet.Tests/AiDotNetTests.csproj
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some tests failed" -ForegroundColor Yellow
}

# Code quality checks
Write-Host "[6/9] Checking for null-forgiving operator..." -ForegroundColor Yellow
$NullForgiv = Get-ChildItem -Path src -Recurse -Filter *.cs |
    Select-String -Pattern "!" |
    Where-Object {
        $_.Line -notmatch "!=" -and
        $_.Line -notmatch "IsNullOrEmpty" -and
        $_.Line -notmatch "!string" -and
        $_.Line -match "\w+!" # Match word followed by ! (null-forgiving)
    }
if ($NullForgiv) {
    Write-Host "FAIL: Found null-forgiving operator (!)" -ForegroundColor Red
    $NullForgiv | ForEach-Object { Write-Host "$($_.Path):$($_.LineNumber) - $($_.Line)" }
    exit 1
}

Write-Host "[7/9] Checking for System.Text.Json usage..." -ForegroundColor Yellow
$SystemTextJson = Get-ChildItem -Path src -Recurse -Filter *.cs |
    Select-String -Pattern "System.Text.Json"
if ($SystemTextJson) {
    Write-Host "FAIL: Found System.Text.Json usage (use Newtonsoft.Json instead)" -ForegroundColor Red
    $SystemTextJson | ForEach-Object { Write-Host "$($_.Path):$($_.LineNumber) - $($_.Line)" }
    exit 1
}

Write-Host "[8/9] Checking for KeyValuePair deconstruction..." -ForegroundColor Yellow
$KVPDecon = Get-ChildItem -Path src -Recurse -Filter *.cs |
    Select-String -Pattern "var \([^,]+,[^)]+\) in"
if ($KVPDecon) {
    Write-Host "WARNING: Found potential KeyValuePair deconstruction (not supported in net462)" -ForegroundColor Yellow
    $KVPDecon | ForEach-Object { Write-Host "$($_.Path):$($_.LineNumber) - $($_.Line)" }
}

Write-Host "[9/9] Checking for investigation files..." -ForegroundColor Yellow
$InvestFiles = Get-ChildItem -Recurse -Filter "*REPORT*","*FINDINGS*","*INVESTIGATION*" -ErrorAction SilentlyContinue
if ($InvestFiles) {
    Write-Host "FAIL: Found investigation/report files that should not be committed" -ForegroundColor Red
    $InvestFiles | ForEach-Object { Write-Host $_.FullName }
    exit 1
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "VALIDATION PASSED" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
```

---

## Manual Review Checklist

### Critical Items (Must Pass)

- [ ] **Build Success**
  - [ ] net462 build succeeds
  - [ ] net471 build succeeds
  - [ ] netstandard2.0 build succeeds

- [ ] **No Null-Forgiving Operators**
  - [ ] No use of `!` operator to suppress nullable warnings
  - [ ] All parameters have proper null checks
  - [ ] Use `is not null` pattern instead

- [ ] **Framework Compatibility**
  - [ ] Only Newtonsoft.Json used (no System.Text.Json)
  - [ ] No KeyValuePair deconstruction in net462 code paths
  - [ ] No C# 9+ features unless conditional compilation used

- [ ] **No Investigation Files**
  - [ ] No *REPORT*.md files
  - [ ] No *FINDINGS*.md files
  - [ ] No *INVESTIGATION*.md files
  - [ ] No temp-*.ps1 or debug-*.ps1 scripts

### High Priority Items

- [ ] **Tests Pass**
  - [ ] All existing tests continue to pass
  - [ ] New functionality has unit tests
  - [ ] Integration tests added where appropriate

- [ ] **IEngine Integration**
  - [ ] All operations use IEngine where methods exist
  - [ ] Engine instance validated before use (not null)
  - [ ] Consistent pattern across all operations

- [ ] **Error Handling**
  - [ ] Proper exception types used
  - [ ] Meaningful error messages
  - [ ] No swallowing of exceptions

- [ ] **Commit Message Format**
  - [ ] Follows conventional commits: `type(scope): description`
  - [ ] Subject line is lowercase
  - [ ] Body lines are <= 100 characters
  - [ ] Breaking changes noted in footer

### Medium Priority Items

- [ ] **XML Documentation**
  - [ ] All public methods have XML comments
  - [ ] Parameters documented
  - [ ] Return values documented
  - [ ] Exceptions documented with `<exception>` tags

- [ ] **Code Quality**
  - [ ] Follows existing code patterns
  - [ ] No duplicate code
  - [ ] Meaningful variable names
  - [ ] Appropriate use of constants

- [ ] **Performance**
  - [ ] No obvious performance regressions
  - [ ] Efficient algorithms used
  - [ ] No unnecessary allocations

### Nice to Have

- [ ] **Documentation**
  - [ ] README updated if needed
  - [ ] Pattern guide updated (for foundational changes)
  - [ ] Examples added for new features

---

## Story-Specific Review Criteria

### Story 1: IEngine Integration (Agent 1)

**Additional checks:**
- [ ] `TensorOperations.MatrixMultiply` uses `IEngine.TensorMatMul`
- [ ] `TensorOperations.Transpose` uses `IEngine.TensorTranspose`
- [ ] Backward pass still computes gradients correctly
- [ ] No performance regression vs previous implementation
- [ ] ComputationNode structure unchanged

**Test Coverage:**
- [ ] Test with null engine (should throw ArgumentNullException)
- [ ] Test with mismatched engines (a and b have different engines)
- [ ] Test gradient computation matches previous version
- [ ] Test large tensors (10000x10000)

---

### Story 2-4: IR Operations (Agents 2-4)

**Additional checks:**
- [ ] Each IR operation class follows naming convention: `{Activation}Op`
- [ ] All inherit from `IROp` interface
- [ ] Forward() method implemented correctly
- [ ] Backward() method computes correct gradient
- [ ] Uses IEngine methods where available (GELU, ELU, Mish, Swish, SiLU)
- [ ] Parameterized activations (PReLU, RReLU, LeakyReLU) accept parameters correctly

**Test Coverage:**
- [ ] Test Forward() with known inputs/outputs
- [ ] Test Backward() gradient matches numerical gradient
- [ ] Test with edge cases (NaN, Inf, very large/small values)
- [ ] Test parameterized activations with different parameter values

**Gradient Verification:**
```csharp
// Numerical gradient check
float epsilon = 1e-5f;
float numericalGrad = (Forward(x + epsilon) - Forward(x - epsilon)) / (2 * epsilon);
float analyticalGrad = Backward(x, gradOutput);
Assert.AreEqual(numericalGrad, analyticalGrad, 1e-3f);
```

---

### Story 5: TensorOperations Methods (Agent 5)

**Additional checks:**
- [ ] All 37 activation functions have TensorOperations methods
- [ ] Each method returns `ComputationNode<T>`
- [ ] Delegates to IEngine where methods exist
- [ ] Backward function implemented for each
- [ ] Parameterized activations have overloads (default + custom parameter)
- [ ] Follows existing pattern from ReLU, Sigmoid, Tanh

**Test Coverage:**
- [ ] Test each activation with known input/output
- [ ] Test backward pass computes gradients
- [ ] Test with different numeric types (float, double)
- [ ] Test parameterized activations with various parameters
- [ ] Test gradient accumulation (multiple backward passes)

**Required Methods:**
```
GELU, ELU, SELU, CELU, LeakyReLU, PReLU, RReLU, ThresholdedReLU,
Swish, SiLU, Mish, HardSigmoid, HardTanh, ScaledTanh, Softplus,
SoftSign, BentIdentity, Identity, Linear, Softmin, LogSoftmax,
LogSoftmin, Sparsemax, SphericalSoftmax, GumbelSoftmax, TaylorSoftmax,
HierarchicalSoftmax, Maxout, Sign, Gaussian, ISRU, LiSHT, SQRBF,
Squash, BinarySpikingActivation
```

---

### Story 6: DenseLayer Production Ready (Agent 6)

**Additional checks:**
- [ ] ExportComputationGraph applies activation function
- [ ] ApplyActivationToGraph helper method implemented
- [ ] CanActivationBeJitted helper method implemented
- [ ] SupportsJitCompilation returns true when activation supported
- [ ] Symbolic batch dimension used (-1 instead of 1)
- [ ] Comprehensive null checks for weights, biases, input
- [ ] Throws NotSupportedException for unsupported activations with clear message
- [ ] Graph output matches Forward() output exactly

**Test Coverage:**
- [ ] Test with ReLU activation - graph matches Forward()
- [ ] Test with Sigmoid activation - graph matches Forward()
- [ ] Test with Tanh activation - graph matches Forward()
- [ ] Test with GELU activation - graph matches Forward()
- [ ] Test with unsupported activation - throws NotSupportedException
- [ ] Test CanActivationBeJitted for all supported activations
- [ ] Test with null weights - throws InvalidOperationException
- [ ] Test with different batch sizes (symbolic dimension)

**Critical Validation:**
```csharp
// Graph output must match Forward() exactly
var layer = new DenseLayer<float>(10, 5, new ReLUActivation<float>());
layer.Initialize();
var input = new Tensor<float>(new int[] { 32, 10 }); // batch=32
var forwardOutput = layer.Forward(input);
var graphOutput = ExecuteGraph(layer.ExportComputationGraph(...), input);
AssertTensorsEqual(forwardOutput, graphOutput, epsilon: 1e-6f);
```

---

### Story 7: Pattern Documentation (Agent 7)

**Additional checks:**
- [ ] Pattern guide is clear and comprehensive
- [ ] Code examples are complete and compilable
- [ ] Activation mapping reference lists all 37 activations
- [ ] Helper methods added to LayerBase.cs
- [ ] Unit tests for DenseLayer JIT compilation pass
- [ ] Integration tests with real workloads pass
- [ ] Troubleshooting guide covers common issues
- [ ] Examples show how to replicate pattern for other layers

**Test Coverage:**
- [ ] Test DenseLayer JIT matches Forward() on MNIST data
- [ ] Test multiple activation functions (ReLU, GELU, Tanh)
- [ ] Test with real training workload
- [ ] Performance benchmark: JIT vs regular Forward()
- [ ] Test pattern on at least one other layer type (e.g., ConvolutionalLayer stub)

---

## Approval Workflow

### Step 1: Automated Validation
1. Run validation script (Bash or PowerShell)
2. If FAIL, reject PR and provide feedback
3. If PASS, proceed to manual review

### Step 2: Manual Code Review
1. Review code changes in GitHub PR
2. Check story-specific criteria
3. Verify test coverage
4. Run tests locally if needed
5. Provide feedback if issues found

### Step 3: Approval Decision

**APPROVED - Ready to Merge**
- All automated checks pass
- All manual checklist items pass
- Tests are comprehensive
- Code quality is high
- Documentation is complete

**CHANGES REQUESTED**
- Some issues found (documented in feedback)
- Agent must address feedback and re-submit
- Re-run validation after changes

**REJECTED - Major Rework Needed**
- Critical issues found (null-forgiving operators, build failures, etc.)
- Design problems or architectural concerns
- Agent must rework implementation significantly

### Step 4: Merge
1. Squash commits if needed (for cleaner history)
2. Ensure commit message follows conventional commits
3. Merge to master
4. Delete feature branch
5. Notify dependent agents (if applicable)

---

## Common Issues and Solutions

### Issue: Null-Forgiving Operator Found

**Problem:**
```csharp
string value = nullableString!;  // WRONG
```

**Solution:**
```csharp
if (nullableString is not null)
{
    string value = nullableString;  // Compiler knows it's not null
    // Use value
}
```

### Issue: System.Text.Json Used

**Problem:**
```csharp
using System.Text.Json;
var doc = JsonDocument.Parse(json);
```

**Solution:**
```csharp
using Newtonsoft.Json.Linq;
var obj = JObject.Parse(json);
```

### Issue: KeyValuePair Deconstruction

**Problem:**
```csharp
foreach (var (key, value) in dictionary)  // WRONG in net462
```

**Solution:**
```csharp
foreach (var kvp in dictionary)
{
    string key = kvp.Key;
    int value = kvp.Value;
}
```

### Issue: Missing XML Documentation

**Problem:**
```csharp
public static ComputationNode<T> GELU(ComputationNode<T> input)  // No docs
```

**Solution:**
```csharp
/// <summary>
/// Applies GELU (Gaussian Error Linear Unit) activation function element-wise.
/// GELU(x) = x * Φ(x) where Φ is the CDF of standard normal distribution.
/// </summary>
/// <param name="input">Input computation node</param>
/// <returns>Computation node with GELU applied</returns>
/// <exception cref="ArgumentNullException">Thrown when input is null</exception>
public static ComputationNode<T> GELU(ComputationNode<T> input)
```

### Issue: Test Failure

**Problem:**
```
Test failed: Expected 0.5, Actual 0.49999
```

**Solution:**
- Use epsilon-based comparison for floating point
- Increase epsilon if needed (1e-6f for float, 1e-12 for double)
- Check for numerical stability issues
- Verify algorithm implementation

---

## Metrics and Reporting

### Per-PR Metrics

Track for each PR:
- Build time (all 3 frameworks)
- Test execution time
- Number of files changed
- Lines of code added/removed
- Number of review iterations
- Time from PR creation to merge

### Overall Project Metrics

Track for the epic:
- Total PRs merged
- Average review time
- Build success rate
- Test pass rate
- Code coverage change
- Performance improvements (JIT speedup achieved)

---

## Emergency Rollback Procedure

If a merged PR causes critical issues:

1. **Identify the problem** - What broke?
2. **Assess impact** - Is master broken? Are other agents blocked?
3. **Quick fix or revert?**
   - If quick fix possible (<30 min), do it
   - Otherwise, revert the merge commit
4. **Revert command**:
   ```bash
   git revert -m 1 <merge-commit-hash>
   git push origin master
   ```
5. **Notify the team** - Post in coordination channel
6. **Root cause analysis** - Why did this slip through review?
7. **Update review process** - Add checks to prevent recurrence

---

## Success Criteria

Epic is complete when:
- [ ] All 7 agent PRs approved and merged
- [ ] Master build succeeds on all frameworks
- [ ] All tests pass
- [ ] DenseLayer JIT compilation is production-ready
- [ ] Pattern documentation enables other layers to be implemented
- [ ] No critical or high severity issues outstanding
- [ ] Performance target achieved (5-10x speedup with JIT)

