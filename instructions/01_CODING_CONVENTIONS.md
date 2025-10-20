# Coding Conventions

1.  **Language:** All code will be written in Python 3.9+.
2.  **Formatting:** Code will be formatted using the Black code style.
3.  **Type Hinting:** All function signatures must include type hints for clarity and static analysis.
4.  **Docstrings:** All modules and functions must have clear, concise docstrings explaining their purpose, arguments, and return values.
5.  **Modularity:** Core logic (like D-IQM calculations) should be kept in the `src/diqa` directory. Scripts in the `scripts/` directory should import this logic and handle the execution pipeline.
6.  **Dependencies:** All external dependencies must be listed in `requirements.txt`. The project should remain lightweight, relying only on the libraries specified.
