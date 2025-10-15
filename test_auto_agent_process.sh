#!/bin/bash

# Test Auto Agent Process for AiDotNet Project
# Demonstrates the full automated build fix process using BuildFixAgents

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
PROJECT_DIR="/home/ooples/AiDotNet"
AGENTS_DIR="$PROJECT_DIR/BuildFixAgents"
BUILD_OUTPUT="$PROJECT_DIR/build_output.txt"
PROCESS_LOG="$PROJECT_DIR/auto_agent_process.log"

# Initialize logging
exec > >(tee -a "$PROCESS_LOG")
exec 2>&1

echo -e "${BLUE}=== AiDotNet Auto Agent Build Fix Process ===${NC}"
echo "Timestamp: $(date)"
echo "Project: $PROJECT_DIR"
echo

# Step 1: Initial Build Status
echo -e "${CYAN}Step 1: Checking Initial Build Status${NC}"
cd "$PROJECT_DIR"

# Run build and capture output
echo "Running initial build..."
if dotnet build > "$BUILD_OUTPUT" 2>&1; then
    echo -e "${GREEN}✓ Build succeeded - no fixes needed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Build failed - starting auto fix process${NC}"
    echo "Errors found: $(grep -c "error" "$BUILD_OUTPUT" || echo "0")"
fi

# Step 2: Analyze Build Errors
echo -e "\n${CYAN}Step 2: Analyzing Build Errors${NC}"
cd "$AGENTS_DIR"

# Copy build output to where generic_build_analyzer expects it
cp "$BUILD_OUTPUT" "$AGENTS_DIR/build_output.txt"

# Run the generic build analyzer
echo "Running build analyzer..."
./generic_build_analyzer.sh analyze

# Show error summary from the analyzer output
echo -e "\n${YELLOW}Error Summary:${NC}"
echo "Total errors: $(grep -c "error" "$BUILD_OUTPUT" || echo "0")"
echo "Error analysis complete - check error_analysis.json for details"

# Step 3: Initialize BuildFixAgents System
echo -e "\n${CYAN}Step 3: Initializing BuildFixAgents System${NC}"

# Initialize core features
echo "Initializing knowledge management..."
./knowledge_management_system.sh init >/dev/null 2>&1

echo "Initializing advanced caching..."
./advanced_caching_system.sh init >/dev/null 2>&1

echo "Initializing security scanning..."
./advanced_security_suite.sh init >/dev/null 2>&1

# Step 4: Run Auto Fix Process
echo -e "\n${CYAN}Step 4: Running Automated Fix Process${NC}"

# Use the autofix script with knowledge from previous fixes
echo "Starting intelligent auto-fix..."
./autofix.sh auto

# Step 5: Multi-Agent Coordination
echo -e "\n${CYAN}Step 5: Running Multi-Agent Coordination${NC}"

# Bootstrap the multi-agent system
echo "Bootstrapping multi-agent system..."
if [[ -f "./bootstrap_multi_agent_system.sh" ]]; then
    ./bootstrap_multi_agent_system.sh
else
    # Run agents individually if bootstrap doesn't exist
    echo "Running specialized agents..."
    
    # Agent 1: Duplicate Resolution
    echo -e "\n${YELLOW}Agent 1: Duplicate Resolver${NC}"
    ./agent1_duplicate_resolver.sh
    
    # Agent 2: Constraints Specialist
    echo -e "\n${YELLOW}Agent 2: Constraints Specialist${NC}"
    ./agent2_constraints_specialist.sh
    
    # Agent 3: Inheritance Specialist
    echo -e "\n${YELLOW}Agent 3: Inheritance Specialist${NC}"
    ./agent3_inheritance_specialist.sh
fi

# Step 6: Verify Fixes
echo -e "\n${CYAN}Step 6: Verifying Fixes${NC}"
cd "$PROJECT_DIR"

# Run build again
echo "Running build after fixes..."
if dotnet build > "${BUILD_OUTPUT}.fixed" 2>&1; then
    echo -e "${GREEN}✓ Build succeeded after fixes!${NC}"
    FIXED=true
else
    echo -e "${YELLOW}⚠ Build still has errors${NC}"
    FIXED=false
    
    # Count remaining errors
    REMAINING_ERRORS=$(grep -c "error" "${BUILD_OUTPUT}.fixed" || echo "0")
    echo "Remaining errors: $REMAINING_ERRORS"
fi

# Step 7: Learn from Process
echo -e "\n${CYAN}Step 7: Learning from Fix Process${NC}"
cd "$AGENTS_DIR"

# Capture knowledge from this fix session
echo "Capturing knowledge from fix process..."
./knowledge_management_system.sh learn "$PROJECT_DIR" "$BUILD_OUTPUT"

# Record successful patterns
if [[ "$FIXED" == "true" ]]; then
    ./knowledge_management_system.sh solution \
        "AiDotNet build errors with ModelMetadata" \
        "Used multi-agent system to resolve interface and constraint issues" \
        "build_errors" \
        "high"
fi

# Step 8: Generate Report
echo -e "\n${CYAN}Step 8: Generating Fix Report${NC}"

cat > "$PROJECT_DIR/auto_fix_report.md" <<EOF
# Auto Agent Fix Report

## Summary
- **Project**: AiDotNet
- **Date**: $(date)
- **Initial Errors**: $(grep -c "error" "$BUILD_OUTPUT" || echo "0")
- **Remaining Errors**: $(grep -c "error" "${BUILD_OUTPUT}.fixed" 2>/dev/null || echo "0")
- **Status**: $([ "$FIXED" == "true" ] && echo "✅ FIXED" || echo "⚠️ PARTIAL FIX")

## Process Steps Executed
1. ✓ Initial build analysis
2. ✓ Error categorization and prioritization
3. ✓ Multi-agent coordination
4. ✓ Automated fix application
5. ✓ Verification build
6. ✓ Knowledge capture

## Key Fixes Applied
$(grep "Fixed:" "$AGENTS_DIR"/*.log 2>/dev/null | head -10 || echo "- See agent logs for details")

## Agents Involved
- Duplicate Resolver Agent
- Constraints Specialist Agent
- Inheritance Specialist Agent
- Build Checker Agent

## Next Steps
$(if [ "$FIXED" != "true" ]; then
    echo "- Review remaining errors in ${BUILD_OUTPUT}.fixed"
    echo "- Consider running targeted agents for specific error types"
    echo "- Manual intervention may be required for complex issues"
else
    echo "- All build errors resolved"
    echo "- Consider running tests to verify functionality"
    echo "- Create PR with fixes"
fi)
EOF

echo -e "\n${GREEN}Report saved to: $PROJECT_DIR/auto_fix_report.md${NC}"

# Step 9: Optional - Create PR
if [[ "$FIXED" == "true" ]]; then
    echo -e "\n${CYAN}Step 9: Creating Pull Request (Optional)${NC}"
    echo "Would you like to create a PR with the fixes? (Requires GitHub CLI)"
    echo -e "${YELLOW}Run: cd $PROJECT_DIR && gh pr create --title 'Auto-fix build errors' --body '@auto_fix_report.md'${NC}"
fi

# Summary
echo -e "\n${BLUE}=== Process Summary ===${NC}"
echo "Total time: $SECONDS seconds"
echo "Logs saved to: $PROCESS_LOG"
echo "Build outputs: $BUILD_OUTPUT, ${BUILD_OUTPUT}.fixed"

if [[ "$FIXED" == "true" ]]; then
    echo -e "\n${GREEN}✅ SUCCESS: All build errors have been automatically fixed!${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️ PARTIAL SUCCESS: Some errors were fixed, but manual intervention needed.${NC}"
    exit 1
fi