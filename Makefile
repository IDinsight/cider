# Makes a fresh install of the repository environment
fresh-env :
	uv sync --no-install-project && uv run pre-commit install && cd ..

# Runs all tests
test:
	uv run pytest tests/


# Clears results from jupyter notebooks; results should not be commited as they contain binary blobs which bloat/obscure git history
clear-nb:
	@if [ "$(DIRS)" = "" ]; then \
		find . -not -path '*/\.*' -type f -name "*.ipynb" -exec uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +; \
	else \
		find $(DIRS) -not -path '*/\.*' -type f -name "*.ipynb" -exec uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +; \
	fi

# Delete Python cache files
clear-pycache:
	find .. -type f -name "*.py[co]" -delete
	find .. -type d -name "__pycache__" -delete
	find .. -type d -name ".pytest_cache" -delete
