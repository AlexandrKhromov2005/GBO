BUILD_DIR := build
EXECUTABLE := main

.PHONY: all run test clean

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
