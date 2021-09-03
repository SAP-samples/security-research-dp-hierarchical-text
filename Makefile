LOG_PATH=./logs


.PHONY: install uninstall test
all: install

install:
	# Download and install HEMKit (Hierarchical Evaluation Metrics Toolkit)
	mkdir -p tools
	curl -o ./tools/HEMKit.zip http://nlp.cs.aueb.gr/software_and_datasets/HEMKit.zip
	unzip -u ./tools/HEMKit.zip -d tools
	(cd ./tools/HEMKit/software; make)

	# Create Folders and install DPH
	mkdir -p logs
	mkdir -p models
	mkdir -p experiments
	pip install -e ./

uninstall:
	pip uninstall dph
	rmdir tools
