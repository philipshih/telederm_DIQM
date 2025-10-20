# Common AI Mistakes to Avoid (V2)

This is a strict set of rules, not suggestions. Adhere to them at all times.

1.  **Do Not Hallucinate Paths or Filenames:**
    - Before writing to or reading from a path, you must verify it exists if it is not one you have just created.
    - Do not assume the contents of a directory. Use `list_files` if you are uncertain.

2.  **Adhere Strictly to the Project Structure:**
    - Core, reusable, application-agnostic logic **ONLY** goes in `src/diqa/`.
    - Pipeline execution, data loading, and saving results **ONLY** goes in `scripts/`.
    - Do not mix these concerns.

3.  **Do Not Modify Files Unnecessarily:**
    - Only make changes to the files relevant to the current, explicit task.
    - **CRITICAL:** Do not "clean up," refactor, or modify any code I did not specifically ask you to change. Do not add features or change logic "for the better." Execute the requested task and nothing more.

4.  **Respect the `requirements.txt`:**
    - Do not introduce any new third-party libraries.
    - The project must remain lightweight. Use only the libraries already listed in `requirements.txt`.

5.  **Implement Only What Is Asked:**
    - Do not add extra features, parameters, or metrics that were not explicitly defined in the project brief or the current task.
    - Stick to the plan. Do not deviate.

6.  **Confirm Before Acting on Ambiguity:**
    - If a prompt is ambiguous or contradicts a previous instruction, ask for clarification using `ask_followup_question`.
    - Do not guess or make assumptions about the user's intent.
