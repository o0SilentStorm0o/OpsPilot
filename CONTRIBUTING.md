# Contributing to OpsPilot

Thank you for your interest in contributing to OpsPilot! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Node version, etc.)
- Relevant logs or screenshots

### Suggesting Features

Feature suggestions are welcome! Please:
- Check existing issues first
- Provide clear use case
- Explain expected benefit
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/opspilot.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow code style guidelines
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   npm test
   npm run lint
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "feat: add new incident category"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Code Style

### TypeScript
- Use strict TypeScript mode
- Prefer interfaces over types for object shapes
- Use explicit return types for functions
- Follow ESLint rules

```typescript
// Good
export async function analyzeLogs(logs: LogEntry[]): Promise<LogAnalysisResult> {
  // Implementation
}

// Avoid
export async function analyzeLogs(logs) {
  // Implementation
}
```

### Python
- Follow PEP 8
- Use type hints
- Use Black for formatting
- Maximum line length: 100 characters

```python
# Good
def generate_incident(category: str) -> Dict[str, Any]:
    """Generate a single incident"""
    pass

# Avoid
def generate_incident(category):
    pass
```

## 🧪 Testing

### Backend Tests
```bash
npm test
```

### ML Tests
```bash
cd ml
pytest
```

## 📚 Documentation

- Update README.md for user-facing changes
- Update ARCHITECTURE.md for design changes
- Add JSDoc/docstrings for new functions
- Update TRAINING_GUIDE.md for new features

## 🔄 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Ensure CI passes** (lint, tests, build)
4. **Request review** from maintainers
5. **Address feedback** promptly

### PR Title Format
```
type: short description

Examples:
- feat: add performance incident category
- fix: resolve classification accuracy issue
- docs: update installation guide
- refactor: optimize log analysis workflow
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

## 🎯 Areas for Contribution

### Easy (Good First Issues)
- Documentation improvements
- Adding more synthetic data patterns
- UI enhancements
- Additional test cases

### Medium
- New incident categories
- Performance optimizations
- Additional metrics
- Integration tests

### Advanced
- RAG integration
- Multi-modal support
- Model quantization improvements
- Advanced workflow orchestration

## 💡 Development Tips

### Local Setup
```bash
# Backend
npm install
npm run dev

# ML
cd ml
pip install -r requirements.txt
python generate_dataset.py
```

### Debugging
- Enable debug logging: `LOG_LEVEL=debug`
- Use VS Code debugger
- Check docker logs: `docker-compose logs -f`

### Testing Workflows
```bash
# Test single workflow
npm test -- workflows/analyzeLogs.test.ts

# With coverage
npm test -- --coverage
```

## 🚀 Release Process

Maintainers will:
1. Review and merge PRs
2. Update version in package.json
3. Create GitHub release
4. Build and push Docker images

## 📋 Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] CI passes
- [ ] No security vulnerabilities
- [ ] Performance considered
- [ ] Error handling implemented

## 🛡️ Security

- Report security issues privately to security@opspilot.example.com
- Do not create public issues for security vulnerabilities
- Follow responsible disclosure

## ❓ Questions?

- Check existing documentation
- Search closed issues
- Create a discussion topic
- Join community chat (if available)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for making OpsPilot better!** 🎉
