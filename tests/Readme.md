#### Running Tests

This project uses [pytest](https://docs.pytest.org/en/stable/) for testing. 

- To run the tests, simply run the following command in the root directory of the project:
```bash
pytest
```

- To run a specific test file, you can run the following command:
```bash
pytest tests/test_file.py
```
 Don't forget to replace `test_file.py` with the name of the test file you want to run.

- To run a specific test function, you can run the following command:
```bash
pytest tests/test_file.py::test_function
```

Some of the tests requires some examples files to be present in the `tests/examples` directory. If you want to run all the tests, you can download the examples from this [github repo](https://github.com/OpendTect/OpendTect-ML-Dev/raw/main/webinars/2021-04-22/Examples/data) and place them in the `tests/examples` directory.
