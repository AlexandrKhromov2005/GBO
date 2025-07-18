BUILD_DIR := build
EXECUTABLE := main
# Set DEBUG=1 when calling make to enable verbose debug logging (passes ENABLE_DEBUG_LOG=ON to CMake)
DEBUG ?= 0

.PHONY: all run test clean rm_images clear_dataset build_dataset help total_clean

all: $(BUILD_DIR)/$(EXECUTABLE)

$(BUILD_DIR)/$(EXECUTABLE):
	cmake -B $(BUILD_DIR) -S . -DENABLE_DEBUG_LOG=$(DEBUG)
	$(MAKE) -C $(BUILD_DIR)

run: all
	@echo "Running the executable..."
	./$(BUILD_DIR)/$(EXECUTABLE)

test: all
	@echo "Running tests..."
	cd $(BUILD_DIR) && ctest --output-on-failure

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)

rm_images:
	@echo "Removing non-original images..."
	@find images -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
		! -name 'airplane.png' \
		! -name 'baboon.png' \
		! -name 'boat.png' \
		! -name 'bridge.png' \
		! -name 'earth_from_space.png' \
		! -name 'lake.png' \
		! -name 'lenna.png' \
		! -name 'pepper.png' \
		! -name 'watermark.png' \
		-exec rm -v {} \;

clear_dataset:
	@echo "Clearing dataset directory..."
	rm -rf dataset/*
	@echo "Dataset directory cleared."

build_dataset:
	@echo "Building dataset..."
	./build/main --build-dataset

help:
	@echo "Makefile commands:"
	@echo "  all          - Build the project"
	@echo "  run          - Run the executable"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean the build directory"
	@echo "  rm_images    - Remove non-original images from the images directory"
	@echo "  clear_dataset - Clear the dataset directory"
	@echo "  build_dataset - Build the dataset"
	@echo "  help         - Show this help message"
	@echo "  total_clean  - Clean everything including images and dataset"

total_clean: clean rm_images clear_dataset
	@echo "Total clean completed."
	@echo "All temporary files and directories have been removed."