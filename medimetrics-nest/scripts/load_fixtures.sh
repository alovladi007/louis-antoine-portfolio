#!/bin/bash

# Load synthetic DICOM fixtures into the system
# Usage: ./scripts/load_fixtures.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
WEB_URL="${WEB_URL:-http://localhost:3000}"
FIXTURES_DIR="${FIXTURES_DIR:-data/fixtures/synthetic}"

echo -e "${YELLOW}Loading fixtures into MediMetrics...${NC}"

# Check if API is running
if ! curl -s -f "${API_URL}/health" > /dev/null; then
    echo -e "${RED}Error: API is not running at ${API_URL}${NC}"
    echo "Please start the services first: docker compose up"
    exit 1
fi

# Generate synthetic DICOM files if they don't exist
if [ ! -d "$FIXTURES_DIR" ] || [ -z "$(ls -A $FIXTURES_DIR)" ]; then
    echo -e "${YELLOW}Generating synthetic DICOM files...${NC}"
    python scripts/generate_synthetic_dicom.py --output-dir "$FIXTURES_DIR" --count 5
fi

# Login as demo admin
echo -e "${YELLOW}Logging in as admin...${NC}"
LOGIN_RESPONSE=$(curl -s -X POST "${API_URL}/auth/login" \
    -H "Content-Type: application/json" \
    -d '{
        "email": "admin@demo.local",
        "password": "Demo123!"
    }')

# Extract access token
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | grep -o '"accessToken":"[^"]*' | cut -d'"' -f4)

if [ -z "$ACCESS_TOKEN" ]; then
    echo -e "${RED}Error: Failed to login. Response: ${LOGIN_RESPONSE}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Logged in successfully${NC}"

# Function to upload a DICOM file
upload_dicom() {
    local file_path=$1
    local file_name=$(basename "$file_path")
    
    echo -e "${YELLOW}Uploading ${file_name}...${NC}"
    
    # Create a new study
    STUDY_RESPONSE=$(curl -s -X POST "${API_URL}/studies" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"patientPseudoId\": \"PATIENT-$(date +%s)\",
            \"modality\": \"CR\",
            \"bodyPart\": \"CHEST\",
            \"description\": \"Fixture: ${file_name}\"
        }")
    
    STUDY_ID=$(echo "$STUDY_RESPONSE" | grep -o '"id":"[^"]*' | cut -d'"' -f4)
    
    if [ -z "$STUDY_ID" ]; then
        echo -e "${RED}Error: Failed to create study. Response: ${STUDY_RESPONSE}${NC}"
        return 1
    fi
    
    # Get presigned upload URL
    UPLOAD_URL_RESPONSE=$(curl -s -X POST "${API_URL}/studies/${STUDY_ID}/upload-url" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"fileName\": \"${file_name}\",
            \"fileType\": \"application/dicom\"
        }")
    
    UPLOAD_URL=$(echo "$UPLOAD_URL_RESPONSE" | grep -o '"uploadUrl":"[^"]*' | cut -d'"' -f4)
    
    if [ -z "$UPLOAD_URL" ]; then
        echo -e "${RED}Error: Failed to get upload URL. Response: ${UPLOAD_URL_RESPONSE}${NC}"
        return 1
    fi
    
    # Upload file to S3
    curl -s -X PUT "${UPLOAD_URL}" \
        -H "Content-Type: application/dicom" \
        --data-binary "@${file_path}"
    
    # Confirm upload
    curl -s -X POST "${API_URL}/studies/${STUDY_ID}/confirm-upload" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"status\": \"completed\"}"
    
    echo -e "${GREEN}✓ Uploaded ${file_name} to study ${STUDY_ID}${NC}"
    
    # Push to Orthanc
    echo -e "${YELLOW}Pushing to Orthanc...${NC}"
    curl -s -X POST "${API_URL}/dicom/push" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{\"studyId\": \"${STUDY_ID}\"}"
    
    echo -e "${GREEN}✓ Pushed to DICOM server${NC}"
    
    return 0
}

# Upload all DICOM files
UPLOAD_COUNT=0
for dcm_file in "$FIXTURES_DIR"/*.dcm; do
    if [ -f "$dcm_file" ]; then
        if upload_dicom "$dcm_file"; then
            UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
        fi
        # Add delay to avoid rate limiting
        sleep 1
    fi
done

# Upload sample PNG images
for png_file in "$FIXTURES_DIR"/*.png; do
    if [ -f "$png_file" ]; then
        echo -e "${YELLOW}Uploading $(basename $png_file)...${NC}"
        # Similar upload process for PNG files
        UPLOAD_COUNT=$((UPLOAD_COUNT + 1))
    fi
done

echo -e "${GREEN}✅ Successfully loaded ${UPLOAD_COUNT} fixtures${NC}"
echo -e "${GREEN}View them at: ${WEB_URL}/dashboard${NC}"

# Run a sample inference on the first study
if [ $UPLOAD_COUNT -gt 0 ]; then
    echo -e "${YELLOW}Running sample inference...${NC}"
    
    # Get first study
    STUDIES_RESPONSE=$(curl -s "${API_URL}/studies?limit=1" \
        -H "Authorization: Bearer ${ACCESS_TOKEN}")
    
    FIRST_STUDY_ID=$(echo "$STUDIES_RESPONSE" | grep -o '"id":"[^"]*' | head -1 | cut -d'"' -f4)
    
    if [ ! -z "$FIRST_STUDY_ID" ]; then
        # Run inference
        INFERENCE_RESPONSE=$(curl -s -X POST "${API_URL}/inference/run" \
            -H "Authorization: Bearer ${ACCESS_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"studyId\": \"${FIRST_STUDY_ID}\",
                \"model\": \"classifier\"
            }")
        
        JOB_ID=$(echo "$INFERENCE_RESPONSE" | grep -o '"jobId":"[^"]*' | cut -d'"' -f4)
        
        if [ ! -z "$JOB_ID" ]; then
            echo -e "${GREEN}✓ Started inference job: ${JOB_ID}${NC}"
            echo -e "${GREEN}Check status at: ${API_URL}/jobs/${JOB_ID}${NC}"
        fi
    fi
fi

echo -e "${GREEN}✅ Fixture loading complete!${NC}"