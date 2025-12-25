# Contributing to Threat Plane

Thank you for your interest in contributing to the Threat Plane project! This document provides guidelines and information for contributors.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,full]"
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use Black for formatting: `black src/`
- Use isort for import sorting: `isort src/`
- Add type hints to all functions
- Write docstrings for all public functions and classes

### Testing

- Write tests for new features
- Ensure all tests pass: `pytest tests/`
- Aim for >80% code coverage

### Documentation

- Update docstrings when changing functionality
- Update README.md if adding new features
- Add examples for new capabilities

## Areas of Interest

We're particularly interested in contributions in these areas:

### Algorithms & Methods
- Additional dimensionality reduction techniques
- New anomaly detection methods
- Improved attack path algorithms
- Causal discovery implementations
- Hyperbolic embedding methods

### Integrations
- Security tool connectors (Splunk, Elastic, etc.)
- Threat intelligence feed parsers
- CMDB/asset management integrations
- SOAR platform integrations

### Visualization
- Three.js component improvements
- New visualization overlays
- VR/AR support
- Real-time streaming updates

### Performance
- GPU acceleration
- Distributed computing support
- Incremental update algorithms
- Memory optimization

## Pull Request Process

1. **Update documentation** for any changes
2. **Add tests** for new functionality
3. **Ensure CI passes** all checks
4. **Request review** from maintainers
5. **Address feedback** promptly

### PR Title Format
```
[TYPE] Brief description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Adding tests
- perf: Performance improvement
```

## Reporting Issues

When reporting issues, please include:

- **Description**: Clear description of the issue
- **Steps to reproduce**: How to trigger the issue
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Environment**: Python version, OS, package versions
- **Logs/Screenshots**: Any relevant error messages

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the technical merits
- Help others learn and grow

## Questions?

- Open a GitHub Discussion for general questions
- Use Issues for bug reports and feature requests
- Reach out to maintainers for sensitive matters

Thank you for contributing to making security more accessible through geometric intelligence!
