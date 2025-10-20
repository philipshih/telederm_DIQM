# Development Best Practices

This document outlines a strict set of rules for project contributions to ensure consistency and maintainability.

1.  **Verify File Paths:**
    - Before writing to or reading from a path, ensure it exists or is handled correctly.
    - Do not make assumptions about the contents of a directory.

2.  **Adhere to the Project Structure:**
    - Core, reusable, application-agnostic logic **only** goes in `src/diqa/`.
    - Pipeline execution, data loading, and saving results **only** goes in `scripts/`.
    - These concerns should not be mixed.

3.  **Scope Changes Appropriately:**
    - Only make changes to files relevant to the current, explicit task.
    - Do not perform broad refactoring or "clean up" code outside the scope of the immediate goal. Execute the requested task and nothing more.

4.  **Manage Dependencies:**
    - Do not introduce any new third-party libraries without discussion.
    - The project must remain lightweight. All dependencies must be listed in `requirements.txt`.

5.  **Implement to Specification:**
    - Do not add extra features, parameters, or metrics that were not explicitly defined in the project brief or the current task.
    - Adhere to the established project plan.

6.  **Clarify Ambiguity:**
    - If a task is ambiguous or contradicts a previous instruction, seek clarification before proceeding.
    - Do not make assumptions about intent.
