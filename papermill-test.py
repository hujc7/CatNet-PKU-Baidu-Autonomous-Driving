import papermill as pm
from datetime import datetime

if __name__ == "__main__":
    pm.execute_notebook(
        "papermill-test.ipynb",
        "papermill-test-output.ipynb",
        log_output=True,
    )