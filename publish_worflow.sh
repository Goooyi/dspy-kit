#!/bin/bash
# Publishing workflow for dspy-kit

set -e  # Exit on any error

echo "🚀 Publishing dspy-kit to PyPI"

# Step 1: Pre-publish checks
echo "📋 Running pre-publish checks..."

# Check if on main branch
current_branch=$(git branch --show-current)
if [ "$current_branch" != "main" ]; then
    echo "❌ Not on main branch. Current branch: $current_branch"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "❌ Uncommitted changes detected. Please commit all changes first."
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
uv run pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Please fix tests before publishing."
    exit 1
fi

# Run linting
echo "🔍 Running linting..."
uv run ruff check dspy_evals tests
if [ $? -ne 0 ]; then
    echo "❌ Linting failed. Please fix linting issues before publishing."
    exit 1
fi

# Step 2: Version management
echo "📝 Current version information:"
current_version=$(grep 'version = ' pyproject.toml | cut -d'"' -f2)
echo "Current version: $current_version"

read -p "Enter new version (or press enter to keep $current_version): " new_version
if [ -n "$new_version" ]; then
    # Update version in pyproject.toml
    sed -i "s/version = \"$current_version\"/version = \"$new_version\"/" pyproject.toml

    # Update version in __init__.py
    sed -i "s/__version__ = \"$current_version\"/__version__ = \"$new_version\"/" dspy_evals/__init__.py

    # Commit version update
    git add pyproject.toml dspy_evals/__init__.py
    git commit -m "Bump version to $new_version"
    git tag "v$new_version"

    echo "✅ Version updated to $new_version"
    current_version=$new_version
fi

# Step 3: Build package
echo "📦 Building package..."
uv build

# Step 4: Test on TestPyPI first
echo "🧪 Publishing to TestPyPI first..."
read -p "Publish to TestPyPI? (y/N): " test_publish
if [ "$test_publish" = "y" ] || [ "$test_publish" = "Y" ]; then
    uv publish --repository testpypi
    echo "✅ Published to TestPyPI"
    echo "🔗 Check: https://test.pypi.org/project/dspy-kit/"
    echo ""
    echo "Test installation with:"
    echo "pip install --index-url https://test.pypi.org/simple/ dspy-kit==$current_version"
    echo ""
    read -p "Test installation successful? Continue to real PyPI? (y/N): " continue_publish
    if [ "$continue_publish" != "y" ] && [ "$continue_publish" != "Y" ]; then
        echo "❌ Aborting PyPI publication"
        exit 1
    fi
fi

# Step 5: Publish to real PyPI
echo "🌍 Publishing to PyPI..."
read -p "Are you sure you want to publish to PyPI? This cannot be undone! (y/N): " final_confirm
if [ "$final_confirm" = "y" ] || [ "$final_confirm" = "Y" ]; then
    uv publish
    echo "🎉 Successfully published to PyPI!"
    echo "🔗 Package: https://pypi.org/project/dspy-kit/"

    # Push git changes
    git push origin main
    git push origin "v$current_version"
    echo "✅ Git changes pushed"

else
    echo "❌ Publication cancelled"
    exit 1
fi

echo ""
echo "🎉 Publication complete!"
echo "📦 Version $current_version is now available on PyPI"
echo "💿 Install with: pip install dspy-kit==$current_version"
