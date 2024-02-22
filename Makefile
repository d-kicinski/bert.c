.PHONY: build run

BUILD_DIR = build

all: build run

run: build
	./$(BUILD_DIR)/bert
	
build:
	mkdir -p $(BUILD_DIR)
	gcc bert.c -o $(BUILD_DIR)/bert -DBERT_IMPLEMENTATION

