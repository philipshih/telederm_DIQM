# Development & Contribution Guidelines

This document outlines the coding conventions and standards for this project to ensure code quality, readability, and maintainability.

## 1. General
- **Language:** All code must be written in Python 3.9+.
- **Formatting:** All Python code must be formatted using the `black` code formatter with default settings.
- **Imports:** Imports should be organized into three groups: standard library, third-party libraries, and local application imports, separated by a blank line.

## 2. Documentation
- **Type Hinting:** All function and method signatures must include type hints for all arguments and return values.
- **Docstrings:** All modules, classes, and functions must have clear, Google-style docstrings explaining their purpose, arguments (`Args:`), and return values (`Returns:`).

## 3. Code Structure & Modularity
- **Core Logic vs. Scripts:**
    - **`src/diqa/`**: This directory is for the core, reusable logic of the project. It should contain application-agnostic functions and classes.
    - **`scripts/`**: This directory is for the executable pipeline scripts that handle tasks like data processing and model training. They act as the "runners" of the project, importing logic from `src/diqa/`.
- **Configuration:** Avoid hardcoding paths or parameters. Use command-line arguments (`argparse`) for scripts and define constants where appropriate.

## 4. Error Handling & Logging
- **Robustness:** Functions that interact with the filesystem must handle potential errors gracefully (e.g., `FileNotFoundError`).
- **Logging:** Use the `logging` module for status updates or error messages in library code. `print()` is acceptable in the `scripts/` for reporting user-facing progress.

## 5. Dependencies
- **Minimalism:** The project must remain lightweight. All external dependencies must be explicitly listed in `requirements.txt`.
