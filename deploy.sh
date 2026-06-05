#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -eo pipefail

# Configuration with sensible defaults (overrideable via Environment Variables)
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-}"
ECR_REPOSITORY="${ECR_REPOSITORY:-alpha-factory-executor}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-alpha-factory-bot}"

# Helper function to print usage instructions
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --account-id ID       AWS Account ID (required if AWS_ACCOUNT_ID env var is not set)"
    echo "  --region REGION       AWS Region (default: $AWS_REGION)"
    echo "  --repo REPOSITORY     ECR Repository Name (default: $ECR_REPOSITORY)"
    echo "  --tag TAG             Docker Image Tag (default: $IMAGE_TAG)"
    echo "  --lambda FUNCTION     AWS Lambda Function Name (default: $LAMBDA_FUNCTION_NAME)"
    echo "  -h, --help            Show this help message"
    echo ""
    exit 1
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account-id)
            AWS_ACCOUNT_ID="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --repo)
            ECR_REPOSITORY="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA_FUNCTION_NAME="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate AWS Account ID
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "Error: AWS Account ID must be specified via --account-id or AWS_ACCOUNT_ID environment variable."
    exit 1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

echo "============================================="
echo " Starting Alpha Factory Lambda Deployment"
echo "============================================="
echo "AWS Region:      ${AWS_REGION}"
echo "AWS Account ID:  ${AWS_ACCOUNT_ID}"
echo "ECR Registry:    ${ECR_REGISTRY}"
echo "ECR Repo:        ${ECR_REPOSITORY}"
echo "Image Tag:       ${IMAGE_TAG}"
echo "Lambda Function: ${LAMBDA_FUNCTION_NAME}"
echo "============================================="

# 1. Verify Prerequisites
echo "Checking prerequisites..."
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI is not installed or not in PATH."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or daemon is not running."
    exit 1
fi

# 2. Authenticate to Amazon ECR
echo "Authenticating to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# 3. Build Docker Image
echo "Building Docker Image..."
# Note: Using build-arg or local file context if needed. 
# Target is AWS Lambda environment, typically linux/amd64
docker build --platform linux/amd64 -t "${ECR_REPOSITORY}:${IMAGE_TAG}" .

# 4. Tag Docker Image
echo "Tagging Docker Image..."
docker tag "${ECR_REPOSITORY}:${IMAGE_TAG}" "${FULL_IMAGE_NAME}"

# 5. Push to Amazon ECR
echo "Pushing Docker Image to ECR..."
docker push "${FULL_IMAGE_NAME}"

# 6. Update Lambda Function Code
echo "Updating AWS Lambda Function..."
aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --image-uri "${FULL_IMAGE_NAME}" \
    --region "${AWS_REGION}" \
    > /dev/null

echo "Waiting for Lambda function update to complete..."
aws lambda wait function-updated \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}"

echo "============================================="
echo " Deployment Successful!"
echo " Lambda Updated with Image: ${FULL_IMAGE_NAME}"
echo "============================================="
