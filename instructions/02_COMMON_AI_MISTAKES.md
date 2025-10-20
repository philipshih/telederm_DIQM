# Common AI Mistakes to Avoid

1.  **Do Not Hallucinate Paths or Filenames:** Only reference files and directories that you have confirmed exist. Do not assume the contents of a directory.
2.  **Adhere Strictly to the Project Structure:** Place new code in the correct location. Core, reusable functions go in `src/diqa/`. Execution scripts go in `scripts/`.
3.  **Do Not Modify Files Unnecessarily:** Only make changes to the files relevant to the current task. Do not refactor or "clean up" code that is outside the scope of the immediate goal.
4.  **Respect the `requirements.txt`:** Do not introduce new third-party libraries. The project must remain lightweight and use only the approved dependencies.
5.  **Implement Only What Is Asked:** Do not add extra features or metrics that were not explicitly defined in the project brief. Stick to the plan.
