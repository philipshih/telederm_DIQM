# Coding Conventions (V2)

## 1. General
- **Language:** All code must be written in Python 3.9+.
- **Formatting:** All Python code must be formatted using the `black` code formatter with default settings.
- **Imports:** Imports should be organized into three groups: standard library, third-party libraries, and local application imports, separated by a blank line.

## 2. Documentation
- **Type Hinting:** All function and method signatures **must** include type hints for all arguments and return values. Use the `typing` module where necessary.
- **Docstrings:** All modules, classes, and functions must have Google-style docstrings. They must clearly explain the purpose, arguments (`Args:`), and return values (`Returns:`).

## 3. Code Structure & Modularity
- **Core Logic vs. Scripts:**
    - **`src/diqa/`**: This is for the core, reusable logic of the project. It should contain functions and classes that are application-agnostic (e.g., the D-IQM calculation functions). This code should be importable and usable in other projects.
    - **`scripts/`**: This is for the executable pipeline scripts. These scripts should handle tasks like parsing command-line arguments, reading data from disk, calling the core logic from `src/diqa/`, and saving results. They are the "runners" of the project.
- **No Hardcoding:** Avoid hardcoding paths, filenames, or magic numbers. Use command-line arguments (`argparse`) for scripts and define constants at the top of files where appropriate.

## 4. Error Handling & Logging
- **Robustness:** Functions that interact with the filesystem (e.g., reading images) must handle potential errors gracefully (e.g., file not found, corrupted image). A function should not crash the entire script.
- **Logging:** Use the `logging` module for any status updates or error messages. Do not use `print()` statements in the `src/diqa` library code. `print()` is acceptable in the `scripts/` for reporting progress to the console.

## 5. Dependencies
- **Minimalism:** The project must remain lightweight. All external dependencies must be explicitly listed in `requirements.txt`.
- **No Unused Libraries:** Do not import libraries that are not used.
