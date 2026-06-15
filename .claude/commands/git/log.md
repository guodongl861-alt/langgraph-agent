---
description: Show recent git history with summary
argument-hint: [optional: number of commits, default 5]
allowed-tools: Bash(git:*)
model: haiku
---

Show recent git commit history.

Number of commits: $ARGUMENTS (default: 5 if not specified)

## Steps

1. Run `git log --oneline -n [count]`
2. Provide a brief summary

## Output Format

```
## Recent Commits

| Hash | Message | Author | Time |
|------|---------|--------|------|
| abc123 | feat: add login | John | 2h ago |
| def456 | fix: null check | Jane | 5h ago |
...

### Summary
- Total: [n] commits shown
- Most active: [most frequent commit type]
- Recent focus: [what recent work seems to be about]
```

Keep it concise and scannable.
