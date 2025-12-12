#!/bin/bash
# Quick script to generate datasets for ABD and DED tasks in range 20000-23302

set -e  # Exit on error

echo "========================================"
echo "Affine Agent Dataset Generation"
echo "========================================"
echo ""

# Configuration
START_ID=20000
END_ID=23302
SAVE_INTERVAL=5000
OUTPUT_DIR="generated_datasets"
BASE_URL="http://89.221.67.132:17628/v1/"

# # Check if API key is set
# if [ -z "$OPENAI_API_KEY" ]; then
#     echo "Error: OPENAI_API_KEY environment variable not set"
#     echo "Please set it with: export OPENAI_API_KEY=your_key_here"
#     exit 1
# fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Task range: $START_ID - $END_ID"
echo "  Save interval: $SAVE_INTERVAL"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Ask which environment
echo "Which environment do you want to generate?"
echo "  1) ABD only"
echo "  2) DED only"
echo "  3) Both ABD and DED"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "Generating ABD dataset..."
        python cli.py generate-dataset \
            --env abd \
            --base-url $BASE_URL \
            --start-id $START_ID \
            --end-id $END_ID \
            --save-interval $SAVE_INTERVAL \
            --output-dir "$OUTPUT_DIR" \
            --verbose
        ;;
    2)
        echo ""
        echo "Generating DED dataset..."
        python cli.py generate-dataset \
            --env ded \
            --base-url $BASE_URL \
            --start-id $START_ID \
            --end-id $END_ID \
            --save-interval $SAVE_INTERVAL \
            --output-dir "$OUTPUT_DIR" \
            --verbose
        ;;
    3)
        echo ""
        echo "Generating ABD dataset..."
        python cli.py generate-dataset \
            --env abd \
            --base-url $BASE_URL \
            --start-id $START_ID \
            --end-id $END_ID \
            --save-interval $SAVE_INTERVAL \
            --output-dir "$OUTPUT_DIR" \
            --verbose
        
        echo ""
        echo "Generating DED dataset..."
        python cli.py generate-dataset \
            --env ded \
            --base-url $BASE_URL \
            --start-id $START_ID \
            --end-id $END_ID \
            --save-interval $SAVE_INTERVAL \
            --output-dir "$OUTPUT_DIR" \
            --verbose
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Dataset generation complete!"
echo "Check $OUTPUT_DIR/ for output files"
echo "========================================"
