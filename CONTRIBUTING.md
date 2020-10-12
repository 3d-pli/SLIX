# Contributing to SLIX (Scattered Light Imaging ToolboX)

We would love your input to this repository! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

The development of SLIX is done through GitHub. A copy of the repository is also hosted on the [Forschungszentrum Jülich GitLab](https://jugit.fz-juelich.de/j.reuter/slix). All images shown in the [README.md](https://github.com/3d-pli/SLIX/blob/master/README.md) are hosted there.

Pull requests, issues and feature requests are accepted via GitHub. If you plan to contribute to SLIX please follow the guidelines below.

## Seek support
The Scattered Light Imaging ToolboX is maintained by Jan André Reuter and Miriam Menzel. For bug reports, feature requests and pull requests, please read the instructions below. For further support, you can contact both per mail.

| Person           | Mail adress            |
| ---------------- | ---------------------- |
| Jan André Reuter | j.reuter@fz-juelich.de |
|    Miriam Menzel | m.menzel@fz-juelich.de |


## Bug reports

We use GitHub issues to track public bugs. Report a bug by opening a new issue [here](https://github.com/3d-pli/SLIX/issues).
Write bug reports with detail, background, and add sample code if possible.

Great Bug Reports tend to have:

- A quick summary and/or background
- Steps to reproduce
    - Be specific!
    - Give sample code if you can. 
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Pull requests

Pull requests are the best way to propose changes to the codebase. When proposing a pull request, please follow these guidelines:

- Fork the repo and create your branch from master.
- If you've added code that should be tested, add tests.
- Ensure the test suite passes.
- Make sure your code lints.
- Issue that pull request!

## Testing

SLIX does use [pytest](https://docs.pytest.org/en/stable/) to test the code for errors. In addition [flake8](https://flake8.pycqa.org/en/latest/) and [pylint](https://www.pylint.org/) are used for linting. 

Pull requests and commits to the master branch should be automatically tested using GitHub actions with a simple workflow. If you want to test your code locally follow the next steps:

1. Change your directory to the root of SLIX
2. If not done yet, install pytest, flake8 and via pip (or conda in an Anaconda environment)
```bash 
pip install flake8 pytest
```
or 
```bash
conda install flake8 pytest
```
3. First, run `pylint` and `flake8` for linting. If there are some issues, fix them before testing your code.
4. Run `pytest` and check if there are any errors.

## Merging Pull Requests

This section describes how to cleanly merge your pull requests.

### 1. Rebase

From your project repository, merge from master and rebase your commits
(replace `pull-request-branch-name` as appropriate):

```
git fetch origin
git checkout -b pull-request-branch-name origin/pull-request-branch-name
git rebase master
```

If there are conflicts:

```
git mergetool
git rebase --continue
```

### 2. Push

Update branch with the rebased history:

```
git push origin pull-request-branch-name --force
```

The following steps are intended for the maintainers:

### 3. Merge

```
git checkout master
git merge --no-ff pull-request-branch-name
```

### 4. Test

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
pytest
```

### 5. Version

Modify version in `setup.py`:

```
git add setup.py
git commit 
```

### 6. Push to master

```
git push origin master
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md) and [tqdm](https://github.com/tqdm/tqdm/blob/830cd7f9cb3e6fe9b1c3f601ff451debf9509916/CONTRIBUTING.md)