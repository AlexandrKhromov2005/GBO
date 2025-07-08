BUILD_DIR := build
EXECUTABLE := main

.PHONY: all run test clean rm_images clear_dataset build_dataset help

all: $(BUILD_DIR)/$(EXECUTABLE)

$(BUILD_DIR)/$(EXECUTABLE):
	cmake -B $(BUILD_DIR) -S .
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