
# Use Jupyter's minimal-notebook as base image
FROM quay.io/jupyter/minimal-notebook:notebook-7.0.6

# install necessary packages for analysis and prediction
RUN conda install -y pandas=2.1.2 \
    scikit-learn=1.3.2 \
    altair=5.1.2 \
    imbalanced-learn \
    ipykernel=6.26.0 \
    jupyter_contrib_nbextensions=0.7.0 \
    matplotlib \
    notebook=7.0.6 \
    python=3.11.6 \
    requests=2.31.0 \
    vegafusion-jupyter=1.4.3 \
    vegafusion-python-embed=1.4.3 \
    vegafusion=1.4.3 \
    pytest=7.4.3 \
    vl-convert-python=1.1.0 