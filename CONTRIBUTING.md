# Contributing to CVIP

Thank you for considering contributing to CVIP! We welcome contributions from everyone.

## How to Contribute

1. Fork the repository and clone it to your local machine.
2. Create a new branch for your contribution: `git checkout -b feature/new-feature`.
3. Make your changes and ensure they adhere to the project's coding standards (check `Coding standarts` section below)
4. Test your changes locally to ensure they work as expected.
5. Commit your changes: `git commit -m "Add new feature"`
6. Push to your forked repository: `git push origin feature/new-feature`
7. Open a pull request on the production repository's `dev` branch.

## Reporting Bugs

If you encounter a bug or issue with the project, please follow these steps to report it:

1. Check if the issue has already been reported by searching the issues tab.
2. If it hasn't been reported, open a new issue and provide a detailed description of the problem.
3. Include any error messages or relevant information to help diagnose the issue.
4. If possible, provide steps to reproduce the issue.

## Feature Requests

If you have an idea for a new feature or enhancement, we'd love to hear it! Follow these steps to submit a feature request:

1. Check if the feature has already been requested by searching the issues tab.
2. If it hasn't been requested, open a new issue and describe the feature you'd like to see.
3. Provide any additional context or use cases that may be helpful in implementing the feature.

## Coding Standards

To maintain consistency and readability across the project, please adhere to the following coding standards:

### Integrating New Filter

To integrate a new filter into the project, follow these steps:

1. **Create Filter Class:**
   - Create a new Python class for your filter under `src/filters`.
   - Ensure the class inherits from the base `Filter` class.

2. **Video Processing Support:**
   - If your filter can process videos:
     - Navigate to `src/filters/VideoEditor.py`.
     - Add your filter to the list of filters that accept video processing.

3. **Update Processor Mapping:**
   - Navigate to `src/Processor.py`.
   - Add your filter to the `class_map` dictionary to enable its usage in processing pipelines.

4. **Update Argument Verification:**
   - Navigate to `src/VerifyArgs.py`.
   - Add a case for your filter to handle argument verification.

5. **Update Query Verification (Optional):**
   - If necessary, navigate to `src/VerifyQuery.py`.
   - Add support for your filter in query verification.

6. **Import Filter in CVIP Module:**
   - Navigate to `CVIP.py`.
   - Add an import statement for your filter at the top of the file.

By following these steps, your new filter will be integrated into the project and ready for use. Don't forget to test your changes locally before opening a pull request.

### Python Style Guide

- Follow PEP 8 guidelines for Python code.
- Use meaningful variable and function names.
- Write clear and concise docstrings for classes, methods, and functions.
- Use four spaces for indentation.
- Try to limit lines to 79 characters.
- Use snake_case for variable and function names.
- Use descriptive comments to explain complex logic or algorithms.

### Type Annotations

- Use type annotations for function arguments and return values.
- Follow PEP 484 and PEP 526 guidelines for type annotations.

### Documentation

- Document all public classes, methods, and functions using docstrings.
- Provide examples of usage in the docstrings when applicable.
- Update documentation to reflect changes in code functionality.

### Version Control

- Use meaningful commit messages that describe the purpose of the changes.
- Reference relevant issues or pull requests in commit messages where applicable.
- Keep commits small and focused on a single task or feature.
- Squash commits before opening a pull request if necessary.

### Dependency Management

- Specify project dependencies in a `requirements.txt` file.
- Pin dependencies to specific versions to ensure consistency across environments.

By following these coding standards, we can maintain a clean and efficient codebase that is easy to understand and contribute to.

## Questions

If you have any questions about the project or contributing, feel free to reach out to maintainers:
[Valeria Yakovleva](melarozz18@gmail.com)
[Maksim Kotenkov](example@gmail.com)
[Kirill Makanin](example@gmail.com)

We appreciate your contributions to CVIP!
