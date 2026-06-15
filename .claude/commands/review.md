---
description: Review code for quality, bugs, and improvements
argument-hint: [optional: file path]
allowed-tools: Read, Grep, Glob, Bash(git diff:*)
---

Review code and provide feedback.

Target: $ARGUMENTS (or current git diff if not specified)

## Review Focus Areas

1. **Bugs & Errors**: Logic errors, null checks, edge cases
2. **Security**: Input validation, injection risks, sensitive data
3. **Performance**: Obvious inefficiencies, N+1 queries
4. **Readability**: Naming, complexity, documentation needs

## Steps

1. If file path provided, read that file
2. If no path, run `git diff` to see current changes
3. Analyze the code against the focus areas
4. Provide structured feedback

## Output Format

```markdown
## Code Review

### Summary
[One sentence overall assessment]

### Issues Found

#### Critical (Must Fix)
- [issue]: [location] - [brief explanation]

#### Warnings (Should Fix)
- [issue]: [location] - [brief explanation]

#### Suggestions (Nice to Have)
- [suggestion]: [location] - [brief explanation]

### What's Good
- [positive observation]
```

## Guidelines

- Be specific about locations (file:line if possible)
- Provide actionable feedback
- Don't nitpick style unless it impacts readability
- Acknowledge good patterns you see
