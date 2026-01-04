# Copilot Instructions

## Code Organization and Structure

### Source Code (`src/` directory)
- **Single Responsibility Principle**: Each Python file must focus on one specific functionality
- **Line Limit**: Maximum 150 lines per file (excluding blank lines)
- **Modularity**: Break down complex features into multiple focused modules

### Scripts (`scripts/` directory)
- **Purpose**: Each script should accomplish one specific task or workflow
- **Composition**: Scripts may import and combine multiple functions from `src/` modules
- **Line Limit**: No strict line limit, but maintain readability and clarity

## Code Quality Standards

### Output and Logging
- **Print Statements**: Only use alphanumeric characters (A-Z, a-z, 0-9) and standard punctuation marks
- **Prohibited**: No special Unicode symbols, emoji, or ASCII art in `print()` statements
- **Example**: 
  - ✅ `print("Processing completed successfully.")`
  - ❌ `print("Processing completed ✓")`
  - ❌ `print("===> Starting process")`

### Documentation

#### Code Comments
- **Requirement**: Every module, class, function, and complex statement block must have comments
- **Style**: Comments should be concise yet complete
- **Language**: Use clear, descriptive language
- **Format**: Follow PEP 257 docstring conventions for Python

#### README Files
- **Placement**: Every directory must contain a `README.md` file
- **Content**: Provide clear, concise documentation about the directory's purpose and contents
- **Standards**: Follow [Markdownlint](https://github.com/DavidAnson/markdownlint) rules
- **Include**:
  - Directory purpose and overview
  - File descriptions
  - Usage examples (where applicable)
  - Dependencies (if any)

## General Guidelines

- Prioritize code readability and maintainability
- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Keep functions small and focused
- Write self-documenting code supplemented with comments

## Example Structure

project/
├── src/
│ ├── README.md
│ ├── data_processor.py # Max 150 lines
│ └── validator.py # Max 150 lines
└── scripts/
├── README.md
└── run_pipeline.py # Combines src modules, no line limit