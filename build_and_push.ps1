# Alpha Factory - AWS ECR Deployment Script
# Usage: .\build_and_push.ps1 -AccountNumber 123456789012 -Region ap-southeast-1

param (
    [Parameter(Mandatory=$true)]
    [string]$AccountNumber,
    [Parameter(Mandatory=$true)]
    [string]$Region
)

$RepoName = "alpha-factory"
$ImageTag = "latest"
$FullRepoUri = "$AccountNumber.dkr.ecr.$Region.amazonaws.com/$RepoName"

Write-Host "Starting Build and Push for $RepoName..." -ForegroundColor Cyan

# 0. Ensure Repository Exists
Write-Host "Checking if ECR repository exists..."
aws ecr describe-repositories --repository-names $RepoName --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repository does not exist. Creating it now..."
    aws ecr create-repository --repository-name $RepoName --region $Region
}

# 1. Login to ECR
Write-Host "Logging in to AWS ECR..."
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $FullRepoUri
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to login to ECR"; exit }

# 2. Build the image
Write-Host "Building Docker image (forcing single-platform manifest)..."
$env:DOCKER_DEFAULT_PLATFORM = "linux/amd64"
# Using --provenance=false to avoid the "Image Index" issue in Lambda
docker build --platform linux/amd64 --provenance=false -t "${RepoName}" .
if ($LASTEXITCODE -ne 0) { Write-Error "Docker build failed"; exit }

# 3. Tag the image
Write-Host "Tagging image..."
docker tag "${RepoName}:latest" "${FullRepoUri}:${ImageTag}"

# 4. Push to ECR
Write-Host "Pushing to AWS ECR (this may take a few minutes)..."
docker push "${FullRepoUri}:${ImageTag}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Deployment Complete!" -ForegroundColor Green
    Write-Host "Now go to the Lambda Console and update the Image URI to: ${FullRepoUri}:$ImageTag" -ForegroundColor Yellow
} else {
    Write-Error "Push failed."
}
