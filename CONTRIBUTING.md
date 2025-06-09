# Contributing to JSSEnv

We love your input! We want to make contributing to JSSEnv as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally
3. Install development dependencies:

```bash
pip install -e ".[dev]"
```

This will install the package in development mode along with all development dependencies.

### Running Tests

Run the full test suite:

```bash
pytest
```

With coverage:

```bash
pytest --cov=JSSEnv tests/
```

### Pull Requests

1. Update the README.md with details of changes to the interface, if applicable.
2. Update the tests as appropriate.
3. The PR should work for Python 3.8+.
4. Ensure all tests pass.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.