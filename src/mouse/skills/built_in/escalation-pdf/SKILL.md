---
name: escalation-pdf
description: >
  Create a polished PDF report from escalation data. Use when the user asks
  to generate, export, summarize, or format escalation data into a PDF,
  report, downloadable document, or shareable summary. Be especially likely
  to use this skill when the request mentions escalations, incidents,
  risk items, open issues, or asks for a PDF of results.
triggers:
  - keywords: [escalation, escalations, pdf, report, export, export to pdf, downloadable, summary, incident, incidents, risk, risks]
  - keywords: [generate pdf, create pdf, make pdf, build pdf, create report, export report]
---

# Escalation PDF

Use this skill when the user wants escalation data turned into a PDF report.

## Goal

Produce a clean, professional, and easy-to-scan PDF that helps a human
review escalation data quickly. The report should emphasize readability,
clarity, and actionability over raw density.

## Recommended report structure

1. **Title page or header**
   - Report title
   - Organization / group name if available
   - Date generated
   - Time range or source context if available

2. **Executive summary**
   - Total escalation count
   - Breakdown by severity / status if present
   - 2–5 sentence summary of the main patterns

3. **Escalation detail table**
   - ID / reference
   - Group / client / conversation
   - Status
   - Severity / risk level
   - Short reason / summary
   - Created / updated timestamp

4. **Notable items**
   - Highlight the most urgent, open, or repeated escalations
   - Call out trends, duplicates, or unresolved issues

5. **Appendix**
   - Raw data or expanded notes if needed

## Formatting guidance

- Use clear headings and plenty of whitespace.
- Keep tables compact, but do not truncate critical issue text.
- Use consistent labels for status and severity.
- Prefer concise bullets for narrative summaries.
- If the source data is large, paginate cleanly and repeat column headers.
- If a PDF library or generator is available in the project, use that.
- If not, explain the limitation and provide the best available export path.

## Practical behavior

When generating the PDF, do the following:

- Inspect the incoming escalation data before formatting it.
- Preserve important identifiers like `gid`, `pid`, `proj_id`, `Status`,
  and `Escalations` when they are available.
- Group related escalations together if that improves readability.
- If the user asks for a client-specific report, filter to that client or group.
- If the user asks for “the escalations of X,” make sure the report is scoped
  to X rather than all data.

## Output expectations

The end result should usually be one of:

- a `.pdf` file saved to disk, or
- a clearly structured export that can be converted to PDF

If the PDF is being generated from conversational data, prefer a summary that
is fit for sharing with a human stakeholder.

## Implementation notes

If you need to create support files for this skill, keep them alongside this
SKILL.md inside the skill directory.

If the user has not yet specified formatting preferences, default to:

- a title
- a short summary section
- a table of escalations
- a closing section with recommended next actions
