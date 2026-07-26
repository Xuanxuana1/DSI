# English runtime messages

Translate every user-facing Chinese runtime string in the Python codebase,
including logs, exceptions, warnings, command-line help, and test output.
Preserve interpolation placeholders, technical names, metric names, and
program behavior. Chinese strings that identify source-data columns or values
remain unchanged because translating them would alter dataset parsing.

Validation requires:

- no Chinese characters in user-facing Python messages;
- successful Python bytecode compilation;
- passing repository tests.
