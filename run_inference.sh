#!/bin/bash

# DocDiff Inference Runner Script
# Usage: ./run_inference.sh [input_path] [output_path] [options]

# Set default values
CONFIG_FILE="conf.yml"
INIT_MODEL="checksave/init.pth"
DENOISER_MODEL="checksave/denoiser.pth"
SAVE_INTERMEDIATE=""

# Help function
show_help() {
    echo "DocDiff Inference Runner"
    echo "Usage: ./run_inference.sh [input_path] [output_path] [options]"
    echo ""
    echo "Options:"
    echo "  -c, --config      Configuration file path (default: conf.yml)"
    echo "  -i, --init        Initial predictor model path (default: checksave/init.pth)"
    echo "  -d, --denoiser    Denoiser model path (default: checksave/denoiser.pth)"
    echo "  -s, --save        Save intermediate results"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_inference.sh input.jpg output.png"
    echo "  ./run_inference.sh ./input_folder ./output_folder"
    echo "  ./run_inference.sh input.jpg output.png --save"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -i|--init)
            INIT_MODEL="$2"
            shift 2
            ;;
        -d|--denoiser)
            DENOISER_MODEL="$2"
            shift 2
            ;;
        -s|--save)
            SAVE_INTERMEDIATE="--save_intermediate"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$INPUT_PATH" ]]; then
                INPUT_PATH="$1"
            elif [[ -z "$OUTPUT_PATH" ]]; then
                OUTPUT_PATH="$1"
            else
                echo "Unknown option: $1"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if required arguments are provided
if [[ -z "$INPUT_PATH" ]] || [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: Input and output paths are required."
    echo ""
    show_help
    exit 1
fi

# Check if input exists
if [[ ! -e "$INPUT_PATH" ]]; then
    echo "Error: Input path '$INPUT_PATH' does not exist."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "Error: Python is not installed or not in PATH."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check if inference.py exists
if [[ ! -f "inference.py" ]]; then
    echo "Error: inference.py not found in current directory."
    exit 1
fi

# Run inference
echo "Running DocDiff inference..."
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Config: $CONFIG_FILE"
echo "Init Model: $INIT_MODEL"
echo "Denoiser Model: $DENOISER_MODEL"
echo ""

$PYTHON_CMD inference.py \
    --config "$CONFIG_FILE" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    --init_model "$INIT_MODEL" \
    --denoiser_model "$DENOISER_MODEL" \
    $SAVE_INTERMEDIATE

# Check if inference was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_PATH"
else
    echo ""
    echo "Inference failed. Please check the error messages above."
    exit 1
fi 