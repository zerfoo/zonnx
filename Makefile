.PHONY: all test lint build clean format lint-fix download download-url

all: test lint build

test:
	go test ./...

lint:
	golangci-lint run

lint-fix:
	golangci-lint run --fix

format:
	go fmt ./...

build:
	mkdir -p bin
	go build -o bin/train cmd/train/main.go
	go build -o bin/predict cmd/predict/main.go
	go build -o bin/process_features cmd/process_features/main.go
	go build -o bin/download cmd/download/main.go

download:
	go run cmd/download/main.go -f v5.0/train.parquet -d .

download-url:
ifndef URL
	$(error URL is required. Usage: make download-url URL=<url>)
endif
	go run cmd/download/main.go -u $(URL) -f numerai_training_data.parquet -d .

clean:
	rm -rf bin