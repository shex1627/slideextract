FROM public.ecr.aws/lambda/python:3.9

# Install the function's dependencies using file requirements.txt
# from your project folder.
ENV LAMBDA_TASK_ROOT=/var/task
ENV LAMBDA_RUNTIME_DIR=/var/runtime
RUN yum install mesa-libGL -y
RUN yum install vim-enhanced -y 

COPY requirements.txt  .
#RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
RUN pip install -r requirements.txt 
RUN python -m spacy download en_core_web_sm

COPY slide_segmentation-1.0.0-py3-none-any.whl .
RUN pip install slide_segmentation-1.0.0-py3-none-any.whl
#RUN pip install streamlit
RUN pip install git+https://github.com/pytube/pytube
RUN pip uninstall streamlit -y
RUN pip install streamlit
RUN mkdir /var/video

# Copy function code
# COPY app.py ${LAMBDA_TASK_ROOT}
COPY app.py ${LAMBDA_TASK_ROOT}
# for testing
COPY test_basic_slide_extractor.py ${LAMBDA_TASK_ROOT}
RUN mkdir ${LAMBDA_TASK_ROOT}/streamlit_app
ADD ./streamlit_app ${LAMBDA_TASK_ROOT}/streamlit_app

WORKDIR ${LAMBDA_TASK_ROOT}/streamlit_app

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
ENTRYPOINT ["streamlit", "run"]

CMD ["--server.port", "8080", "app.py"]