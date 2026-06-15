---
description: Add a TODO comment to code
argument-hint: [todo message, use ! for high priority]
allowed-tools: Read, Edit
---

Add a TODO comment based on: $ARGUMENTS

## Priority Detection

- `!` at start → HIGH priority: `// TODO [HIGH]: message`
- `?` at start → DISCUSS: `// TODO [DISCUSS]: message`
- No marker → Normal: `// TODO: message`

## Comment Format by Language

Detect the file type and use appropriate comment syntax:

- JavaScript/TypeScript: `// TODO: message`
- Python: `# TODO: message`
- HTML: `<!-- TODO: message -->`
- CSS: `/* TODO: message */`
- Shell: `# TODO: message`

## Steps

1. Identify the file context (currently open or recently edited)
2. Parse the message and detect priority
3. Format the TODO comment appropriately
4. Add at a logical location:
   - Near related code if context is clear
   - At the top of the function/block otherwise
5. Confirm the addition

## Examples

Input: `/todo fix null check`
Output: `// TODO: fix null check`

Input: `/todo ! critical security fix`
Output: `// TODO [HIGH]: critical security fix`

Input: `/todo ? should we use async here`
Output: `// TODO [DISCUSS]: should we use async here`

## Output

Confirm briefly:
```
✓ Added TODO at [file]:[line]
```
