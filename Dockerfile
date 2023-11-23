# test

FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -y pandas=2.1.2 \
    scikit-learn=1.3.2 \
    altair=5.1.2 \
    imbalanced-learn \
    ipykernel=6.26.0 \
    jupyter_contrib_nbextensions=0.7.0 \
    matplotlib \
    notebook=6.5.4 \
    python \
    requests=2.31.0 \
    scikit-learn=1.3.2 \
    vegafusion-jupyter=1.4.3 \
    vegafusion-python-embed=1.4.3 \
    vegafusion=1.4.3 \
    vl-convert-python=1.0.1 \
    pytest=7.4.3