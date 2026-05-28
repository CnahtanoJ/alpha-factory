# Use the official AWS Lambda Python 3.11 base image
FROM public.ecr.aws/lambda/python:3.11

# Set environment variables for Python stability and pathing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}"

# Install system dependencies (needed for LightGBM/XGBoost)
RUN yum install -y libgomp gcc gcc-c++ && \
    yum clean all && \
    rm -rf /var/cache/yum

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy your code folders (Package structure)
COPY bot/ ${LAMBDA_TASK_ROOT}/bot/
COPY data_pipeline/ ${LAMBDA_TASK_ROOT}/data_pipeline/
COPY analytics/ ${LAMBDA_TASK_ROOT}/analytics/

# Set the handler (Path to your executor_handler)
CMD [ "bot.bot_executor.executor_handler" ]
