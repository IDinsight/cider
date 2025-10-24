

# Runs all tests
test:
	uv run ./check_for_unmarked_tests.sh
	uv run pytest $(filter-out $@,$(MAKECMDGOALS))


# Clears results from jupyter notebooks; results should not be commited as they contain binary blobs which bloat/obscure git history
clear-nb:
	find $(filter-out $@,$(MAKECMDGOALS)) -not -path '*/\.*' -type f -name "*.ipynb" -execdir jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +

# Dummy command, needed to interpret multiple words as args rather than commands. See https://stackoverflow.com/questions/6273608/how-to-pass-argument-to-makefile-from-command-line
%:
    @:
