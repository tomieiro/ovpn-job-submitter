FROM nvcr.io/nvidia/pytorch:26.05-py3

WORKDIR /workspace

COPY runner/requirements.txt /opt/notebook-runner/requirements.txt

# The container runs as root by design; the warning pip prints about it is
# noise in the job log, both here and in the notebook's own %pip cells.
ENV PIP_ROOT_USER_ACTION=ignore

RUN python -m pip install \
        --no-cache-dir \
        --requirement /opt/notebook-runner/requirements.txt

COPY runner/execute_notebook.py \
     /opt/notebook-runner/execute_notebook.py

COPY payload/ /workspace/

RUN mkdir -p /workspace/outputs

ENV PYTHONUNBUFFERED=1

CMD ["python", "/opt/notebook-runner/execute_notebook.py", "/workspace/notebook.ipynb", "/workspace/outputs/notebook.executed.ipynb"]
