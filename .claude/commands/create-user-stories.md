# Create User Stories Command

This command analyzes the codebase and generates user stories based on build errors, code quality issues, or manual investigation requests.

## Usage

### Automatic Mode (Default)
Analyzes build errors and code quality automatically:
```
/create-user-stories
```

### Manual Investigation Mode
Investigate specific areas or problems:
```
/create-user-stories --investigate "Problem description" --context "Additional context"
```

## How It Works

1. **Gather Codebase Context**
   - Run `dotnet build` to capture errors and warnings
   - Use `find` to get directory structure
   - Use `grep` to search for specific patterns if needed

2. **Send to Gemini**
   - Use `gemini-2.5-flash` model (2M+ token context)
   - Provide full codebase context
   - Request structured user story output

3. **Output Format**
   - User stories in standard format (US-XXX-YYY)
   - Acceptance criteria
   - Technical considerations
   - Dependencies and blockers

## Implementation

When this command is invoked:

### Step 1: Detect Mode
- Check if `--investigate` flag is present
- If yes: Manual investigation mode
- If no: Automatic build error analysis mode

### Step 2: Automatic Mode (Default)
```bash
# Capture build output
dotnet build 2>&1 > /tmp/build-output.txt

# Get directory structure
find . -type f -name "*.cs" | head -100 > /tmp/codebase-structure.txt

# Combine for Gemini
cat /tmp/build-output.txt /tmp/codebase-structure.txt | gemini -m gemini-2.5-flash "Analyze this .NET codebase build output and structure. Generate user stories for fixing build errors, addressing code quality issues, and implementing missing features. Format output as:

US-XXX-YYY: [Title]
Priority: High/Medium/Low
Type: Bug/Feature/Refactor

Description:
[What needs to be done]

Acceptance Criteria:
- [ ] Criterion 1
- [ ] Criterion 2

Technical Notes:
[Implementation guidance]

Dependencies:
[Other user stories or tasks]
"
```

### Step 3: Manual Investigation Mode
When `--investigate` flag is provided:

```bash
# Parse arguments
INVESTIGATE_TOPIC="$1"  # e.g., "Missing namespace organization"
CONTEXT="$2"            # e.g., "Example files reference AiDotNet.FederatedLearning..."

# Get comprehensive codebase dump
find . -type f -name "*.cs" -exec head -50 {} \; > /tmp/codebase-sample.txt

# Get specific commit if referenced
git show COMMIT_SHA:path/to/file >> /tmp/historical-context.txt 2>/dev/null || true

# Search for related code
grep -r "FederatedLearning\|ReinforcementLearning\|Deployment\|Pipeline" --include="*.cs" . > /tmp/related-code.txt

# Combine everything
cat /tmp/codebase-sample.txt /tmp/historical-context.txt /tmp/related-code.txt | gemini -m gemini-2.5-flash "
Investigation Topic: ${INVESTIGATE_TOPIC}

Context: ${CONTEXT}

Task: Analyze this codebase to:
1. Find where related functionality currently exists (may be under wrong namespace)
2. Identify what's missing vs. what's misorganized
3. Determine proper namespace structure
4. Create a migration/implementation plan
5. Generate multiple properly-scoped user stories

For each user story, specify:
- What exists now (if anything)
- What needs to change
- Where code should be moved/created
- Dependencies between stories
- Estimated complexity

Format as standard user stories with US-XXX-YYY identifiers.
"
```

## Arguments

- `--investigate "topic"`: Manual investigation mode with specific topic
- `--context "details"`: Additional context for manual investigations
- `--commit SHA`: Include specific historical commit in analysis

## Examples

### Example 1: Automatic Build Error Analysis
```bash
/create-user-stories
```
Analyzes current build errors and generates user stories to fix them.

### Example 2: Namespace Organization Investigation
```bash
/create-user-stories \
  --investigate "US-NF-004: Missing namespace organization" \
  --context "Example files reference AiDotNet.FederatedLearning, ReinforcementLearning, Deployment, Pipeline, ProductionMonitoring, Interpretability. These likely exist but are misorganized. Need to: 1) Find where this code currently lives, 2) Determine proper namespace structure, 3) Create migration plan, 4) Restore example files" \
  --commit c8c5acf
```

### Example 3: Feature Gap Analysis
```bash
/create-user-stories \
  --investigate "Transfer Learning implementation gaps" \
  --context "Copilot identified 6 issues where source/target data is confused. Need comprehensive review of transfer learning architecture."
```

## Output Location

User stories are output directly to the conversation. DO NOT create files unless explicitly requested.

## Notes

- Use `gemini-2.5-flash` model (NOT gemini-2.0-flash-thinking-exp)
- Include enough context for Gemini to understand codebase structure
- For manual investigations, search the codebase thoroughly before asking Gemini
- Generate properly scoped user stories (not too big, not too small)
- Include technical details and implementation guidance
