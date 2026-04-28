# Result Format Registry

This directory defines reusable result formats. A result format is a contract
for artifact layout and table semantics.

Format IDs use dotted names:

```text
benchmark.table.v1
analysis.note.v1
```

The filename replaces dots with underscores:

```text
benchmark_table_v1.md
analysis_note_v1.md
```

When adding a format, include:

- purpose;
- required files;
- required artifact directories;
- required CSV columns, if any;
- required LaTeX files, if any;
- validation commands or review expectations.
