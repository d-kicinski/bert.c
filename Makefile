.PHONY: build run

BUILD_DIR = build

run: build
	./$(BUILD_DIR)/bert
	
build:
	gcc bert.c -o $(BUILD_DIR)/bert -DBERT_IMPLEMENTATION

