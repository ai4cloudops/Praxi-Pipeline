# Use the official AWS Lambda Python 3.10 base image for container functions
FROM public.ecr.aws/lambda/python:3.10

# 1. Copy requirements & install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy handler code
COPY src/ ${LAMBDA_TASK_ROOT}/src/

# 3. Copy your models into /opt/models
#    (You can also mount EFS at runtime instead, if you prefer)
COPY models/ /opt/models/

# 4. Set handler
#    Your code entry point is src/lambda_function.lambda_handler
ENV MODELS_PATH=/opt/models
CMD ["src/lambda_function.lambda_handler"]
